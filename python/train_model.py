"""
MLB Oracle v4.0 — Phase 3: ML Model Trainer + Calibration
==========================================================

Trains a Logistic Regression model on historical game features with:
  - Walk-forward (time-series) cross-validation (4 splits, 2018–2025)
  - StandardScaler feature normalization
  - Platt/Isotonic calibration via CalibratedClassifierCV
  - Model artifacts saved to data/model/ as JSON for TypeScript runtime

Usage:
    python train_model.py                    # train on data/historical_features.csv
    python train_model.py --input path.csv   # custom input CSV
    python train_model.py --evaluate-only    # just print CV metrics, don't save
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")  # suppress convergence warnings during CV

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "model"
INPUT_CSV = DATA_DIR / "historical_features.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature columns (30 features — must match FeatureVector in types.ts) ────

# All features with real data in training AND at TypeScript pipeline inference time.
# The TS pipeline (featureEngine.ts) computes pitcher, lineup, park, and momentum
# features from FanGraphs/Statcast APIs — so the model can use them.
# The Python predict.py (standalone oracle) can't use these — it uses the 4-feature
# playoff model as fallback during regular season.
FEATURE_COLUMNS = [
    # Core team strength
    "elo_diff",               # Elo rating gap
    "pythagorean_diff",       # Pythagorean win% differential
    "log5_prob",              # Log5 expected win probability
    "sci_adjusted_diff",      # Strength-of-schedule adjusted diff
    # Starting pitcher (biggest single-game predictor)
    "sp_xfip_diff",           # xFIP differential (fielding-independent pitching)
    "sp_kbb_diff",            # K-BB% differential (strikeout minus walk rate)
    "sp_siera_diff",          # SIERA differential (skill-interactive ERA)
    "sp_rolling_gs_diff",     # Rolling game score differential
    # Lineup strength
    "lineup_woba_diff",       # Lineup wOBA differential
    "lineup_wrc_plus_diff",   # Lineup wRC+ differential
    # Ballpark & situational
    "park_factor",            # Park run factor (Coors = 1.15, Oracle = 0.88)
    "momentum_diff",          # Recent win streak momentum
    "run_diff_diff",          # Season run differential gap
    "platoon_advantage",      # Lineup handedness vs pitcher advantage
]

# Walk-forward CV splits: (train_seasons, test_season)
# Each split trains on all data up to the train cutoff, tests on next season.
CV_SPLITS = [
    {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31", "label": "2022"},
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31", "label": "2023"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "label": "2024"},
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31", "label": "2025"},
]


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """
    Load historical features CSV.

    Returns
    -------
    df        : full DataFrame (all columns)
    X         : feature matrix (n_samples × 30)
    y         : label array (0/1)
    dates     : game_date Series for time-series splitting
    """
    if not csv_path.exists():
        print(f"ERROR: Input CSV not found at {csv_path}")
        print("Run build_dataset.py first to generate historical_features.csv")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Date range: {df['game_date'].min().date()} -> {df['game_date'].max().date()}")
    print(f"Seasons: {sorted(df['game_date'].dt.year.unique())}")
    print(f"Home win rate: {df['label'].mean():.3f}")

    # Ensure all feature columns exist; fill missing with 0.0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            print(f"  WARNING: Feature '{col}' missing from CSV — filling with 0.0")
            df[col] = 0.0

    # Replace NaN / inf with 0.0
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df["label"].values.astype(np.int32)
    dates = df["game_date"]

    return df, X, y, dates


# ─── Walk-forward cross-validation ───────────────────────────────────────────

def simulate_kalshi_pnl(
    all_proba: np.ndarray,
    all_outcomes: np.ndarray,
    thresholds: list[float] | None = None,
    edge_assumption: float = 0.07,
) -> list[dict]:
    """
    Simulate Kalshi-style P&L on the combined test predictions.

    Two pricing scenarios per threshold:
      Conservative  — pay the model's own probability (no market discount)
      Realistic     — pay (model_prob - edge_assumption), simulating buying
                      when the market underprices by ~edge_assumption

    Kalshi mechanics per 1 contract:
      YES at price P:  win (1-P) if correct, lose P if wrong
      NO  at price P:  same math from the other side

    Returns list of result dicts for printing/saving.
    """
    if thresholds is None:
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]

    results = []
    for thr in thresholds:
        for scenario, entry_fn in [
            ("conservative", lambda p: p),
            ("realistic",    lambda p: max(0.30, p - edge_assumption)),
        ]:
            bets_taken = 0
            wins = 0
            total_spent = 0.0
            total_pnl = 0.0

            for prob, outcome in zip(all_proba, all_outcomes):
                # --- Home bet ---
                if prob >= thr:
                    entry = entry_fn(prob)
                    cost = entry
                    pnl = (1.0 - entry) if outcome == 1 else -entry
                    total_spent += cost
                    total_pnl += pnl
                    bets_taken += 1
                    if outcome == 1:
                        wins += 1
                # --- Away bet ---
                elif prob <= (1.0 - thr):
                    away_entry = entry_fn(1.0 - prob)
                    cost = away_entry
                    pnl = (1.0 - away_entry) if outcome == 0 else -away_entry
                    total_spent += cost
                    total_pnl += pnl
                    bets_taken += 1
                    if outcome == 0:
                        wins += 1

            win_rate = wins / bets_taken if bets_taken else 0.0
            roi = (total_pnl / total_spent * 100) if total_spent > 0 else 0.0

            results.append({
                "threshold": thr,
                "scenario": scenario,
                "bets": bets_taken,
                "wins": wins,
                "win_rate": win_rate,
                "total_pnl_per_contract": round(total_pnl, 2),
                "total_spent": round(total_spent, 2),
                "roi_pct": round(roi, 2),
                # Scale to $10/bet for readability
                "pnl_per_100_bets_10usd": round((total_pnl / max(1, bets_taken)) * 10 * 100, 2),
            })

    return results


def print_pnl_table(pnl_results: list[dict], edge_assumption: float = 0.07) -> None:
    """Print a formatted P&L table showing simulated Kalshi returns."""
    print("\n" + "=" * 85)
    print("  SIMULATED KALSHI P&L  (1 contract per bet)")
    print(f"  Conservative = paying model's own probability (fair value)")
    print(f"  Realistic    = paying model prob - {edge_assumption:.0%} (buying with edge over market)")
    print("=" * 85)
    print(f"{'Threshold':<11} {'Scenario':<14} {'Bets':>6} {'Win%':>7} {'P&L/contract':>14} {'ROI':>7}  {'P&L/100 bets@$10':>18}")
    print("-" * 85)

    prev_thr = None
    for r in pnl_results:
        if prev_thr is not None and r["threshold"] != prev_thr:
            print()
        prev_thr = r["threshold"]
        pnl_sign = "+" if r["total_pnl_per_contract"] >= 0 else ""
        roi_sign = "+" if r["roi_pct"] >= 0 else ""
        scale_sign = "+" if r["pnl_per_100_bets_10usd"] >= 0 else ""
        print(
            f"  >={r['threshold']:.0%}    {r['scenario']:<14} {r['bets']:>6,} "
            f"{r['win_rate']:>6.1%}  "
            f"{pnl_sign}{r['total_pnl_per_contract']:>10.2f}    "
            f"{roi_sign}{r['roi_pct']:>5.1f}%  "
            f"{scale_sign}${r['pnl_per_100_bets_10usd']:>14.2f}"
        )

    print("-" * 85)
    print("  P&L/contract: cumulative dollars won/lost across all test bets (1 contract each)")
    print("  P&L/100 bets @ $10: what $10/bet sizing earns per 100 bets at this threshold")
    print()

    # Highlight best ROI
    best = max(pnl_results, key=lambda r: r["roi_pct"])
    print(f"  Best ROI: {best['scenario']} at >={best['threshold']:.0%} threshold "
          f"({best['roi_pct']:+.1f}% ROI, {best['bets']:,} bets)")
    print("=" * 85)


# ─── Structural bet-eligibility filter ───────────────────────────────────────
# Mirrors the filters in betEngine.ts — only games the system would actually bet.

MIN_PYTH_DIFF = 0.03   # |pythagorean_diff| — skip toss-ups
MIN_ELO_DIFF  = 25.0   # |elo_diff|         — skip equal-quality matchups
SKIP_MONTH    = 10     # October            — small sample, playoff chaos


def build_eligible_mask(df_slice: pd.DataFrame) -> np.ndarray:
    """Return boolean mask of games that pass the structural quality filters."""
    pyth_ok  = df_slice["pythagorean_diff"].abs() >= MIN_PYTH_DIFF
    elo_ok   = df_slice["elo_diff"].abs()          >= MIN_ELO_DIFF
    month_ok = pd.to_datetime(df_slice["game_date"]).dt.month != SKIP_MONTH
    return (pyth_ok & elo_ok & month_ok).values


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    cv_splits: list[dict],
    df: pd.DataFrame | None = None,   # needed for structural filter features
) -> list[dict]:
    """
    Time-series (walk-forward) cross-validation.

    For each split:
      1. Train LogisticRegression on all data up to train_end
      2. Evaluate on data in [test_start, test_end]
      3. Record Brier score, log-loss, accuracy, and P&L simulation

    Returns list of per-split metric dicts.
    """
    results = []
    all_proba: list[float] = []        # LR probabilities
    all_proba_xgb: list[float] = []    # XGBoost probabilities
    all_outcomes: list[int] = []
    all_eligible: list[bool] = []   # True = game passes structural filters

    model_label = "XGB" if HAS_XGBOOST else "LR"
    print("\n" + "=" * 75)
    print(f"Walk-Forward Cross-Validation  (LR vs {model_label})")
    print("=" * 75)
    print(f"{'Split':<8} {'Train rows':<12} {'Test rows':<11} {'LR Brier':<10} {'LR Acc':<9} {'XGB Brier':<11} {'XGB Acc'}")
    print("-" * 75)

    for split in cv_splits:
        train_mask = dates <= split["train_end"]
        test_mask = (dates >= split["test_start"]) & (dates <= split["test_end"])

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if len(X_train) < 50:
            print(f"  Split {split['label']}: insufficient training data ({len(X_train)} rows) — skipping")
            continue
        if len(X_test) < 10:
            print(f"  Split {split['label']}: insufficient test data ({len(X_test)} rows) — skipping")
            continue

        # Scale features: fit on training data only (avoid data leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ── Logistic Regression ────────────────────────────────────────────
        lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        lr_model.fit(X_train_scaled, y_train)
        y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]  # P(home win)
        y_pred = (y_proba >= 0.5).astype(int)
        brier = brier_score_loss(y_test, y_proba)
        ll = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # ── XGBoost (tree-based — no scaling needed) ────────────────────────
        xgb_brier, xgb_acc = float("nan"), float("nan")
        y_proba_xgb = y_proba  # fallback: same as LR
        if HAS_XGBOOST:
            xgb = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42, verbosity=0,
            )
            xgb.fit(X_train, y_train)
            y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
            xgb_brier = brier_score_loss(y_test, y_proba_xgb)
            xgb_acc = accuracy_score(y_test, (y_proba_xgb >= 0.5).astype(int))

        # High-confidence subset (>60% predicted probability, using best model)
        best_proba = y_proba_xgb if HAS_XGBOOST else y_proba
        high_conf_mask = np.abs(best_proba - 0.5) >= 0.10
        hc_acc = accuracy_score(y_test[high_conf_mask], (best_proba[high_conf_mask] >= 0.5).astype(int)) if high_conf_mask.sum() > 0 else float("nan")

        split_result = {
            "label": split["label"],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "brier": brier,
            "log_loss": ll,
            "accuracy": acc,
            "high_conf_accuracy": hc_acc,
            "high_conf_count": int(high_conf_mask.sum()),
            "xgb_brier": xgb_brier,
            "xgb_accuracy": xgb_acc,
        }
        results.append(split_result)

        # Accumulate for combined P&L analysis (use XGBoost probs if available)
        all_proba.extend(y_proba.tolist())
        all_proba_xgb.extend(y_proba_xgb.tolist())
        all_outcomes.extend(y_test.tolist())

        # Build structural eligibility mask for this test split
        if df is not None:
            eligible = build_eligible_mask(df.loc[test_mask])
        else:
            eligible = np.ones(len(y_test), dtype=bool)
        all_eligible.extend(eligible.tolist())

        xgb_str = f"{xgb_brier:<11.4f} {xgb_acc:.3f}" if not np.isnan(xgb_brier) else "  (no xgboost) "
        print(
            f"{split['label']:<8} {len(X_train):<12} {len(X_test):<11} "
            f"{brier:<10.4f} {acc:<9.3f} {xgb_str}"
        )

    if results:
        avg_brier = np.mean([r["brier"] for r in results])
        avg_ll = np.mean([r["log_loss"] for r in results])
        avg_acc = np.mean([r["accuracy"] for r in results])
        xgb_results = [r for r in results if not np.isnan(r.get("xgb_brier", float("nan")))]
        avg_xgb_brier = np.mean([r["xgb_brier"] for r in xgb_results]) if xgb_results else float("nan")
        avg_xgb_acc = np.mean([r["xgb_accuracy"] for r in xgb_results]) if xgb_results else float("nan")
        xgb_avg_str = f"{avg_xgb_brier:<11.4f} {avg_xgb_acc:.3f}" if not np.isnan(avg_xgb_brier) else ""
        print("-" * 75)
        print(f"{'AVERAGE':<8} {'':<12} {'':<11} {avg_brier:<10.4f} {avg_acc:<9.3f} {xgb_avg_str}")
        print("=" * 75)
        if HAS_XGBOOST and not np.isnan(avg_xgb_brier):
            winner = "XGBoost" if avg_xgb_brier < avg_brier else "LogReg"
            print(f"  Winner: {winner}  (LR Brier={avg_brier:.4f}, XGB Brier={avg_xgb_brier:.4f})")

    # ── P&L simulation: all games vs structurally filtered games ─────────────
    if all_proba:
        proba_arr   = np.array(all_proba)
        outcome_arr = np.array(all_outcomes)
        eligible_arr = np.array(all_eligible)

        # Use XGBoost probs for P&L if they're better (lower Brier on filtered games)
        use_xgb_for_pnl = False
        if all_proba_xgb and HAS_XGBOOST:
            proba_xgb_arr = np.array(all_proba_xgb)
            xgb_brier_filt = brier_score_loss(outcome_arr[eligible_arr], proba_xgb_arr[eligible_arr])
            lr_brier_filt  = brier_score_loss(outcome_arr[eligible_arr], proba_arr[eligible_arr])
            use_xgb_for_pnl = xgb_brier_filt < lr_brier_filt
            pnl_proba = proba_xgb_arr if use_xgb_for_pnl else proba_arr
            model_name = "XGBoost" if use_xgb_for_pnl else "LogReg"
            print(f"\nP&L simulation using: {model_name} (LR filt Brier={lr_brier_filt:.4f}, XGB filt Brier={xgb_brier_filt:.4f})")
        else:
            pnl_proba = proba_arr
            model_name = "LogReg"

        filt_pct = eligible_arr.mean() * 100
        print(f"\nStructural filter ({MIN_PYTH_DIFF} pyth gap, {MIN_ELO_DIFF:.0f} Elo gap, no Oct):")
        print(f"  {eligible_arr.sum():,} / {len(eligible_arr):,} test games pass ({filt_pct:.1f}%)")

        # Overall accuracy on filtered vs unfiltered at HC threshold
        hc_thr = 0.60
        for label, arr_p, arr_o in [
            ("All games    ", pnl_proba, outcome_arr),
            ("Filtered only", pnl_proba[eligible_arr], outcome_arr[eligible_arr]),
        ]:
            hc_mask = (arr_p >= hc_thr) | (arr_p <= 1 - hc_thr)
            if hc_mask.sum() > 0:
                hc_acc = ((arr_p[hc_mask] >= 0.5).astype(int) == arr_o[hc_mask]).mean()
                print(f"  {label}: {hc_mask.sum():,} HC bets — {hc_acc:.1%} accuracy")

        # P&L for ALL games
        print(f"\n-- ALL GAMES ({model_name}) ------------------------------------------")
        pnl_all = simulate_kalshi_pnl(
            pnl_proba, outcome_arr,
            thresholds=[0.55, 0.60, 0.65, 0.70, 0.75],
            edge_assumption=0.07,
        )
        print_pnl_table(pnl_all)

        # P&L for FILTERED games only
        print(f"\n-- FILTERED GAMES ONLY ({model_name}, bettable per structural rules) --")
        pnl_filtered = simulate_kalshi_pnl(
            pnl_proba[eligible_arr], outcome_arr[eligible_arr],
            thresholds=[0.55, 0.60, 0.65, 0.70, 0.75],
            edge_assumption=0.07,
        )
        print_pnl_table(pnl_filtered)

        # Attach to results for metadata saving
        for r in results:
            r["pnl_all_65"]      = next((p for p in pnl_all      if p["threshold"] == 0.65 and p["scenario"] == "realistic"), None)
            r["pnl_filtered_65"] = next((p for p in pnl_filtered if p["threshold"] == 0.65 and p["scenario"] == "realistic"), None)

    return results


# ─── Final model training ─────────────────────────────────────────────────────

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
) -> tuple:
    """
    Train the final production model on 2018-2024 data.

    Primary: XGBoost (if available) — captures non-linear feature interactions.
    Fallback: CalibratedClassifierCV with LogisticRegression.

    Returns (calibrated_model, scaler, iso_reg, raw_lr, xgb_model_or_None)
    """
    # Final training window: 2018-2024 (exclude 2025 which is live/current)
    train_mask = dates.dt.year <= 2024
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]

    print(f"\nFinal model training: {len(X_train_full)} rows (2018-2024)")

    if len(X_train_full) < 50:
        print("WARNING: Very few training rows. Model may perform poorly.")

    # StandardScaler: fit on full training data (used for LR fallback)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_full)

    # ── XGBoost (primary, if available) ───────────────────────────────────
    final_xgb = None
    if HAS_XGBOOST:
        print("Training XGBoost (n_estimators=200, max_depth=4, lr=0.05)...")
        final_xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        final_xgb.fit(X_train_full, y_train_full)
        print("XGBoost trained successfully")

    # ── CalibratedClassifierCV (LR fallback) ──────────────────────────────
    base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    calibrated_model = CalibratedClassifierCV(base_lr, method="isotonic", cv=5)
    calibrated_model.fit(X_scaled, y_train_full)
    print("CalibratedClassifierCV (isotonic, cv=5) trained as LR fallback")

    # ── Standalone IsotonicRegression on 10% holdout ──────────────────────
    holdout_size = max(100, int(len(X_train_full) * 0.10))
    X_cal = X_scaled[-holdout_size:]
    y_cal = y_train_full[-holdout_size:]

    raw_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    raw_lr.fit(X_scaled[:-holdout_size], y_train_full[:-holdout_size])
    raw_probs = raw_lr.predict_proba(X_cal)[:, 1]

    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(raw_probs, y_cal)

    print(f"IsotonicRegression fitted on {len(X_cal)} holdout samples")
    print_calibration_curve(raw_probs, y_cal, iso_reg)

    return calibrated_model, scaler, iso_reg, raw_lr, final_xgb


# ─── Calibration curve ────────────────────────────────────────────────────────

def print_calibration_curve(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    iso_reg: IsotonicRegression,
    n_bins: int = 20,
) -> None:
    """
    Print a text calibration curve showing predicted vs actual win rate
    in probability bands. Also shows the isotonic-calibrated value.
    """
    print("\nCalibration Curve (predicted vs actual win rate):")
    print(f"{'Band':<14} {'Count':>6} {'Avg Raw':>10} {'Actual':>8} {'Iso Cal':>9} {'Diff':>8}")
    print("-" * 60)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (raw_probs >= lo) & (raw_probs < hi)
        if mask.sum() == 0:
            continue
        avg_pred = raw_probs[mask].mean()
        actual = y_true[mask].mean()
        iso_cal = iso_reg.predict([avg_pred])[0]
        diff = iso_cal - actual
        band_str = f"{lo:.2f}–{hi:.2f}"
        print(
            f"{band_str:<14} {mask.sum():>6} {avg_pred:>10.3f} {actual:>8.3f} "
            f"{iso_cal:>9.3f} {diff:>+8.3f}"
        )

    print("-" * 60)
    overall_brier = brier_score_loss(y_true, raw_probs)
    iso_probs = iso_reg.predict(raw_probs)
    iso_brier = brier_score_loss(y_true, iso_probs)
    print(f"Raw Brier: {overall_brier:.4f}  |  Isotonic-calibrated Brier: {iso_brier:.4f}")


# ─── Feature importance ───────────────────────────────────────────────────────

def print_feature_importance(model: Any, feature_names: list[str], top_n: int = 10) -> None:
    """
    Print top-N features by absolute coefficient value.
    Handles both plain LogisticRegression and CalibratedClassifierCV.
    """
    # Extract coefficients from CalibratedClassifierCV
    try:
        if hasattr(model, "calibrated_classifiers_"):
            # Average coefficients across CV folds
            coefs = np.mean(
                [c.estimator.coef_[0] for c in model.calibrated_classifiers_],
                axis=0
            )
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0]
        else:
            print("  (Cannot extract coefficients from this model type)")
            return

        importance = sorted(
            zip(feature_names, coefs),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        print(f"\nTop {top_n} Features by |Coefficient|:")
        print(f"{'Feature':<30} {'Coefficient':>12} {'Direction'}")
        print("-" * 55)
        for feat, coef in importance[:top_n]:
            direction = "-> home +" if coef > 0 else "-> away +"
            print(f"{feat:<30} {coef:>12.4f}  {direction}")

    except Exception as exc:
        print(f"  (Could not extract feature importance: {exc})")


# ─── Save model artifacts ─────────────────────────────────────────────────────

def save_model_artifacts(
    calibrated_model: CalibratedClassifierCV,
    raw_lr: LogisticRegression,
    scaler: StandardScaler,
    iso_reg: IsotonicRegression,
    cv_results: list[dict],
    feature_names: list[str],
    xgb_model=None,
) -> None:
    """
    Save all model artifacts to data/model/ as JSON files compatible
    with TypeScript JSON.parse().

    Files saved:
      coefficients.json      — LR feature coefficients + intercept (fallback)
      scaler.json            — StandardScaler mean/scale arrays
      calibration.json       — isotonic regression thresholds
      model_metadata.json    — training info, CV metrics
      xgboost_trees.json     — XGBoost trees for TypeScript scoring (if XGB trained)
      xgboost_scaler.json    — feature name -> mean/scale for XGB input normalization
    """
    # ── 1. coefficients.json ──────────────────────────────────────────────
    # Use the raw LogisticRegression coefficients (before calibration) for
    # the TypeScript runtime's linear scoring function.
    try:
        if hasattr(calibrated_model, "calibrated_classifiers_"):
            # Average across CV folds
            coefs = np.mean(
                [c.estimator.coef_[0] for c in calibrated_model.calibrated_classifiers_],
                axis=0
            )
            intercept = float(np.mean(
                [c.estimator.intercept_[0] for c in calibrated_model.calibrated_classifiers_]
            ))
        else:
            coefs = raw_lr.coef_[0]
            intercept = float(raw_lr.intercept_[0])

        coef_dict = {name: float(coef) for name, coef in zip(feature_names, coefs)}
        coef_dict["_intercept"] = intercept

        coef_path = MODEL_DIR / "coefficients.json"
        coef_path.write_text(json.dumps(coef_dict, indent=2))
        print(f"Saved coefficients.json ({len(coef_dict)-1} features + intercept)")

    except Exception as exc:
        print(f"WARNING: Could not save coefficients: {exc}")
        # Fall back to raw_lr coefficients
        coefs = raw_lr.coef_[0] if hasattr(raw_lr, "coef_") else np.zeros(len(feature_names))
        intercept = float(raw_lr.intercept_[0]) if hasattr(raw_lr, "intercept_") else 0.0
        coef_dict = {name: float(coef) for name, coef in zip(feature_names, coefs)}
        coef_dict["_intercept"] = intercept
        coef_path = MODEL_DIR / "coefficients.json"
        coef_path.write_text(json.dumps(coef_dict, indent=2))

    # ── 2. scaler.json ────────────────────────────────────────────────────
    scaler_data = {
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
    }
    scaler_path = MODEL_DIR / "scaler.json"
    scaler_path.write_text(json.dumps(scaler_data, indent=2))
    print(f"Saved scaler.json")

    # ── 3. calibration.json ───────────────────────────────────────────────
    # Isotonic regression stores piecewise-linear thresholds
    # x_thresholds: input probabilities, y_thresholds: calibrated output
    x_thresh = iso_reg.X_thresholds_.tolist()
    y_thresh = iso_reg.y_thresholds_.tolist()

    calib_data = {
        "method": "isotonic",
        "x_thresholds": x_thresh,
        "y_thresholds": y_thresh,
        "n_thresholds": len(x_thresh),
    }
    calib_path = MODEL_DIR / "calibration.json"
    calib_path.write_text(json.dumps(calib_data, indent=2))
    print(f"Saved calibration.json ({len(x_thresh)} isotonic thresholds)")

    # ── 4. model_metadata.json ────────────────────────────────────────────
    from datetime import datetime

    avg_brier = float(np.mean([r["brier"] for r in cv_results])) if cv_results else 0.0
    avg_acc = float(np.mean([r["accuracy"] for r in cv_results])) if cv_results else 0.0

    xgb_saved = False
    if xgb_model is not None:
        try:
            # Export trees via get_dump (JSON format) for TypeScript tree-walker
            tree_dumps = xgb_model.get_booster().get_dump(dump_format="json")
            parsed_trees = [json.loads(t) for t in tree_dumps]

            xgb_tree_path = MODEL_DIR / "xgboost_trees.json"
            xgb_tree_path.write_text(json.dumps({
                "n_trees": len(parsed_trees),
                "n_features": len(feature_names),
                "feature_names": feature_names,
                "base_score": 0.5,
                "trees": parsed_trees,
            }, indent=2))
            print(f"Saved xgboost_trees.json ({len(parsed_trees)} trees)")
            xgb_saved = True
        except Exception as exc:
            print(f"WARNING: Could not save XGBoost trees: {exc}")

    # ── 4. model_metadata.json ────────────────────────────────────────────
    from datetime import datetime

    avg_brier = float(np.mean([r["brier"] for r in cv_results])) if cv_results else 0.0
    avg_acc = float(np.mean([r["accuracy"] for r in cv_results])) if cv_results else 0.0
    avg_xgb_brier = float(np.mean([r["xgb_brier"] for r in cv_results if not np.isnan(r.get("xgb_brier", float("nan")))])) if cv_results else None
    avg_xgb_acc = float(np.mean([r["xgb_accuracy"] for r in cv_results if not np.isnan(r.get("xgb_accuracy", float("nan")))])) if cv_results else None

    primary_model = "xgboost" if xgb_saved else "logistic_regression"
    metadata = {
        "version": "4.1.0",
        "model_type": primary_model,
        "lr_fallback": True,
        "calibration_method": "isotonic",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "train_dates": "2018-04-01 to 2024-10-31",
        "test_dates": "2022-2025 (walk-forward CV)",
        "cv_results": cv_results,
        "avg_brier": avg_brier,
        "avg_accuracy": avg_acc,
        "avg_xgb_brier": avg_xgb_brier,
        "avg_xgb_accuracy": avg_xgb_acc,
        "trained_at": datetime.now().isoformat(),
        "sklearn_params": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "calibration_cv": 5,
            "xgb_n_estimators": 200 if xgb_saved else None,
            "xgb_max_depth": 4 if xgb_saved else None,
            "xgb_learning_rate": 0.05 if xgb_saved else None,
        },
    }
    meta_path = MODEL_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model_metadata.json")

    print(f"\nAll artifacts saved to: {MODEL_DIR}")
    print(f"  {coef_path.name}")
    print(f"  {scaler_path.name}")
    print(f"  {calib_path.name}")
    print(f"  {meta_path.name}")
    if xgb_saved:
        print(f"  xgboost_trees.json  (primary scorer for TypeScript)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global MODEL_DIR

    parser = argparse.ArgumentParser(
        description="MLB Oracle v4.0 — ML Model Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV,
        help=f"Path to historical features CSV (default: {INPUT_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        help=f"Directory to save model artifacts (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Run CV only; do not save model artifacts",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Same as --evaluate-only",
    )

    args = parser.parse_args()

    MODEL_DIR = args.output_dir
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("MLB Oracle v4.0 — Phase 3 Model Trainer")
    print("=" * 65)

    # Load data
    df, X, y, dates = load_data(args.input)

    if len(X) < 100:
        print(f"\nERROR: Only {len(X)} rows found. Need at least 100.")
        print("Run build_dataset.py first to generate historical data.")
        sys.exit(1)

    # Walk-forward cross-validation
    cv_results = walk_forward_cv(X, y, dates, CV_SPLITS, df=df)

    if args.evaluate_only or args.no_save:
        print("\n--evaluate-only: skipping model training and artifact saving.")
        return

    # Train final model (on 2018-2024 data)
    calibrated_model, scaler, iso_reg, raw_lr, final_xgb = train_final_model(X, y, dates)

    # Feature importance
    print_feature_importance(calibrated_model, FEATURE_COLUMNS)

    # Save artifacts
    save_model_artifacts(
        calibrated_model,
        raw_lr,
        scaler,
        iso_reg,
        cv_results,
        FEATURE_COLUMNS,
        xgb_model=final_xgb,
    )

    print("\n" + "=" * 65)
    print("Training complete!")
    print("Next step: run the TypeScript pipeline — it will auto-load the model.")
    print("=" * 65)


if __name__ == "__main__":
    main()
