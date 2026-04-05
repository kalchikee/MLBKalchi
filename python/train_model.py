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

warnings.filterwarnings("ignore")  # suppress convergence warnings during CV

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "model"
INPUT_CSV = DATA_DIR / "historical_features.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature columns (30 features — must match FeatureVector in types.ts) ────

FEATURE_COLUMNS = [
    "elo_diff",
    "sp_xfip_diff",
    "sp_kbb_diff",
    "sp_siera_diff",
    "sp_csw_diff",
    "sp_rolling_gs_diff",
    "bullpen_strength_diff",
    "lineup_woba_diff",
    "lineup_wrc_plus_diff",
    "team_10d_woba_diff",
    "team_10d_fip_diff",
    "pythagorean_diff",
    "log5_prob",
    "drs_diff",
    "catcher_framing_diff",
    "park_factor",
    "wind_out_cf",
    "wind_in_cf",
    "temperature",
    "umpire_run_factor",
    "rest_days_diff",
    "travel_tz_shift",
    "day_after_night",
    "is_home",
    "statcast_xba_diff",
    "statcast_barrel_diff",
    "statcast_hardhit_diff",
    "statcast_ev_diff",
    "gb_rate_diff",
    "sci_adjusted_diff",
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

def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    cv_splits: list[dict],
) -> list[dict]:
    """
    Time-series (walk-forward) cross-validation.

    For each split:
      1. Train LogisticRegression on all data up to train_end
      2. Evaluate on data in [test_start, test_end]
      3. Record Brier score, log-loss, accuracy

    Returns list of per-split metric dicts.
    """
    results = []

    print("\n" + "=" * 65)
    print("Walk-Forward Cross-Validation")
    print("=" * 65)
    print(f"{'Split':<8} {'Train rows':<12} {'Test rows':<11} {'Brier':<8} {'LogLoss':<10} {'Accuracy':<10}")
    print("-" * 65)

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

        # Train Logistic Regression
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_proba = model.predict_proba(X_test_scaled)[:, 1]  # P(home win)
        y_pred = (y_proba >= 0.5).astype(int)

        brier = brier_score_loss(y_test, y_proba)
        ll = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # High-confidence subset (>60% predicted probability)
        high_conf_mask = np.abs(y_proba - 0.5) >= 0.10
        hc_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask]) if high_conf_mask.sum() > 0 else float("nan")

        split_result = {
            "label": split["label"],
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "brier": brier,
            "log_loss": ll,
            "accuracy": acc,
            "high_conf_accuracy": hc_acc,
            "high_conf_count": int(high_conf_mask.sum()),
        }
        results.append(split_result)

        hc_str = f"  (HC: {hc_acc:.3f}, n={high_conf_mask.sum()})" if high_conf_mask.sum() > 0 else ""
        print(
            f"{split['label']:<8} {len(X_train):<12} {len(X_test):<11} "
            f"{brier:<8.4f} {ll:<10.4f} {acc:<10.3f}{hc_str}"
        )

    if results:
        avg_brier = np.mean([r["brier"] for r in results])
        avg_ll = np.mean([r["log_loss"] for r in results])
        avg_acc = np.mean([r["accuracy"] for r in results])
        print("-" * 65)
        print(f"{'AVERAGE':<8} {'':<12} {'':<11} {avg_brier:<8.4f} {avg_ll:<10.4f} {avg_acc:<10.3f}")
        print("=" * 65)
        print(f"\nBrier reference: 0.2500 (random 50/50)")
        print(f"LogLoss reference: 0.6931 (random 50/50)")

    return results


# ─── Final model training ─────────────────────────────────────────────────────

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
) -> tuple[CalibratedClassifierCV, StandardScaler, IsotonicRegression]:
    """
    Train the final production model on 2018–2024 data.

    Steps:
    1. Filter to training window (2018–2024)
    2. Fit StandardScaler
    3. Train CalibratedClassifierCV with LogisticRegression base + isotonic calibration
    4. Fit a standalone IsotonicRegression on a holdout for calibration curve plotting
    5. Return (calibrated_model, scaler, iso_reg)
    """
    # Final training window: 2018–2024 (exclude 2025 which is live/current)
    train_mask = dates.dt.year <= 2024
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]

    print(f"\nFinal model training: {len(X_train_full)} rows (2018–2024)")

    if len(X_train_full) < 50:
        print("WARNING: Very few training rows. Model may perform poorly.")

    # StandardScaler: fit on full training data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_full)

    # ── CalibratedClassifierCV (primary approach) ──────────────────────────
    # Uses cross-validation internally to fit isotonic calibration, preventing
    # overfitting the calibration to training data.
    base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    calibrated_model = CalibratedClassifierCV(
        base_lr,
        method="isotonic",
        cv=5,           # 5-fold CV for calibration fitting
    )
    calibrated_model.fit(X_scaled, y_train_full)
    print("CalibratedClassifierCV (isotonic, cv=5) trained successfully")

    # ── Standalone IsotonicRegression on 10% holdout ──────────────────────
    # Also fit a standalone isotonic regression for calibration curve plotting
    # and for saving explicit x_thresholds/y_thresholds to JSON.
    holdout_size = max(100, int(len(X_train_full) * 0.10))
    X_cal = X_scaled[-holdout_size:]
    y_cal = y_train_full[-holdout_size:]

    # Get raw logistic regression probabilities on holdout
    raw_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
    raw_lr.fit(X_scaled[:-holdout_size], y_train_full[:-holdout_size])
    raw_probs = raw_lr.predict_proba(X_cal)[:, 1]

    # Fit isotonic regression: maps raw probabilities -> calibrated probabilities
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(raw_probs, y_cal)

    print(f"IsotonicRegression fitted on {len(X_cal)} holdout samples")

    # Print calibration curve (predicted vs actual in 5% probability bands)
    print_calibration_curve(raw_probs, y_cal, iso_reg)

    return calibrated_model, scaler, iso_reg, raw_lr


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
) -> None:
    """
    Save all model artifacts to data/model/ as JSON files compatible
    with TypeScript JSON.parse().

    Files saved:
      coefficients.json   — feature coefficients + intercept
      scaler.json         — StandardScaler mean/scale arrays
      calibration.json    — isotonic regression thresholds
      model_metadata.json — training info, CV metrics
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

    metadata = {
        "version": "4.0.0",
        "model_type": "logistic_regression_calibrated",
        "calibration_method": "isotonic",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "train_dates": "2018-04-01 to 2024-10-31",
        "test_dates": "2022–2025 (walk-forward CV)",
        "cv_results": cv_results,
        "avg_brier": avg_brier,
        "avg_accuracy": avg_acc,
        "trained_at": datetime.now().isoformat(),
        "sklearn_params": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "calibration_cv": 5,
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
    cv_results = walk_forward_cv(X, y, dates, CV_SPLITS)

    if args.evaluate_only or args.no_save:
        print("\n--evaluate-only: skipping model training and artifact saving.")
        return

    # Train final model (on 2018–2024 data)
    calibrated_model, scaler, iso_reg, raw_lr = train_final_model(X, y, dates)

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
    )

    print("\n" + "=" * 65)
    print("Training complete!")
    print("Next step: run the TypeScript pipeline — it will auto-load the model.")
    print("=" * 65)


if __name__ == "__main__":
    main()
