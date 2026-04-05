"""
MLB Oracle v4.0 — Segment Accuracy Analysis
============================================
Uses the trained ML model to generate predictions on the 2022-2025 test set,
then slices accuracy and P&L by meaningful game segments to find where
the model has the strongest (and weakest) edge.

Segments analyzed:
  1. Month of season (April = early data, September = mature model)
  2. Park type (pitcher's park, neutral, hitter's park)
  3. Temperature band (cold / normal / hot)
  4. SP quality gap (big favourite has clearly better pitcher)
  5. Team dominance (pythagorean win% gap large vs close)
  6. Each home team (which stadiums/franchises is model sharpest on)
  7. Inter- vs intra-division (proxy via common opponents)

Usage:
    python python/segment_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "model"
INPUT_CSV = DATA_DIR / "historical_features.csv"

FEATURE_COLUMNS = [
    "elo_diff", "sp_xfip_diff", "sp_kbb_diff", "sp_siera_diff", "sp_csw_diff",
    "sp_rolling_gs_diff", "bullpen_strength_diff", "lineup_woba_diff",
    "lineup_wrc_plus_diff", "team_10d_woba_diff", "team_10d_fip_diff",
    "pythagorean_diff", "log5_prob", "drs_diff", "catcher_framing_diff",
    "park_factor", "wind_out_cf", "wind_in_cf", "temperature",
    "umpire_run_factor", "rest_days_diff", "travel_tz_shift", "day_after_night",
    "is_home", "statcast_xba_diff", "statcast_barrel_diff", "statcast_hardhit_diff",
    "statcast_ev_diff", "gb_rate_diff", "sci_adjusted_diff",
]

MIN_SEGMENT_SIZE = 50   # skip segments with fewer games (too noisy)
HC_THRESHOLD = 0.60     # "high conviction" prediction threshold


# ─── Load model + generate test predictions ───────────────────────────────────

def load_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward: for each test year 2022-2025, train on prior years,
    predict on test year. Returns df with 'model_prob' column added.
    """
    df = df.copy()
    df["model_prob"] = np.nan

    splits = [
        {"train_end": 2021, "test_year": 2022},
        {"train_end": 2022, "test_year": 2023},
        {"train_end": 2023, "test_year": 2024},
        {"train_end": 2024, "test_year": 2025},
    ]

    for split in splits:
        train_mask = df["season"] <= split["train_end"]
        test_mask  = df["season"] == split["test_year"]

        X_train = df.loc[train_mask, FEATURE_COLUMNS].values.astype(float)
        y_train = df.loc[train_mask, "label"].values.astype(int)
        X_test  = df.loc[test_mask,  FEATURE_COLUMNS].values.astype(float)

        if len(X_train) < 50 or len(X_test) < 10:
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        model.fit(X_train_s, y_train)

        proba = model.predict_proba(X_test_s)[:, 1]
        df.loc[test_mask, "model_prob"] = proba

    return df[df["model_prob"].notna()].copy()


# ─── Segment reporting helpers ────────────────────────────────────────────────

def segment_metrics(group: pd.DataFrame, prob_col: str = "model_prob",
                    label_col: str = "label") -> dict:
    """
    Compute accuracy, high-conviction accuracy, and P&L metrics for a group.
    P&L uses 'realistic' scenario: assume 7% edge over market.
    """
    probs = group[prob_col].values
    labels = group[label_col].values
    n = len(group)

    # Overall accuracy
    preds = (probs >= 0.5).astype(int)
    acc = (preds == labels).mean()

    # High-conviction subset
    hc_mask = (probs >= HC_THRESHOLD) | (probs <= 1 - HC_THRESHOLD)
    hc_n = hc_mask.sum()
    if hc_n >= 5:
        hc_preds = (probs[hc_mask] >= 0.5).astype(int)
        hc_acc = (hc_preds == labels[hc_mask]).mean()
    else:
        hc_acc = float("nan")

    # P&L simulation (realistic: entry = model_prob - 7%)
    EDGE = 0.07
    pnl_total = 0.0
    bet_count = 0
    bet_wins = 0
    for p, outcome in zip(probs, labels):
        if p >= HC_THRESHOLD:
            entry = max(0.30, p - EDGE)
            pnl_total += (1.0 - entry) if outcome == 1 else -entry
            bet_count += 1
            bet_wins += int(outcome == 1)
        elif p <= 1 - HC_THRESHOLD:
            away_p = 1.0 - p
            entry = max(0.30, away_p - EDGE)
            pnl_total += (1.0 - entry) if outcome == 0 else -entry
            bet_count += 1
            bet_wins += int(outcome == 0)

    roi = (pnl_total / (bet_count * 0.60) * 100) if bet_count > 0 else float("nan")

    return {
        "n_games": n,
        "accuracy": round(acc, 3),
        "hc_n": int(hc_n),
        "hc_accuracy": round(hc_acc, 3) if not np.isnan(hc_acc) else None,
        "bet_count": bet_count,
        "bet_wins": bet_wins,
        "bet_win_rate": round(bet_wins / bet_count, 3) if bet_count else None,
        "pnl_total": round(pnl_total, 2),
        "roi_pct": round(roi, 1) if not np.isnan(roi) else None,
    }


def print_segment_table(title: str, rows: list[tuple[str, dict]],
                         sort_by: str = "hc_accuracy") -> None:
    """Print a formatted segment analysis table."""
    # Filter to segments with enough games
    rows = [(label, m) for label, m in rows if m["n_games"] >= MIN_SEGMENT_SIZE]
    if not rows:
        return

    # Sort
    def sort_key(x):
        v = x[1].get(sort_by)
        return v if v is not None else -999
    rows.sort(key=sort_key, reverse=True)

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Segment':<28} {'Games':>6} {'Acc':>6} {'HC Acc':>8} {'HC Bets':>8} {'Win%':>6} {'ROI':>7}")
    print(f"  {'-'*70}")

    for label, m in rows:
        hc_acc_str = f"{m['hc_accuracy']:.1%}" if m["hc_accuracy"] is not None else "  —  "
        win_str = f"{m['bet_win_rate']:.1%}" if m["bet_win_rate"] is not None else "  —  "
        roi_str = f"{m['roi_pct']:+.1f}%" if m["roi_pct"] is not None else "  —  "
        print(
            f"  {label:<28} {m['n_games']:>6,} {m['accuracy']:>5.1%} "
            f"{hc_acc_str:>8} {m['bet_count']:>8,} {win_str:>6} {roi_str:>7}"
        )

    print(f"  {'='*70}")


# ─── Segment definitions ──────────────────────────────────────────────────────

def analyze_by_month(df: pd.DataFrame) -> None:
    df["month"] = pd.to_datetime(df["game_date"]).dt.month
    month_names = {4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October"}
    rows = []
    for m, name in month_names.items():
        g = df[df["month"] == m]
        if len(g) >= MIN_SEGMENT_SIZE:
            rows.append((name, segment_metrics(g)))
    print_segment_table("ACCURACY BY MONTH", rows)


def analyze_by_park(df: pd.DataFrame) -> None:
    bins = [
        ("Pitcher's park  (PF < 0.97)", df["park_factor"] < 0.97),
        ("Neutral park    (0.97–1.03)", (df["park_factor"] >= 0.97) & (df["park_factor"] <= 1.03)),
        ("Hitter's park   (PF > 1.03)", df["park_factor"] > 1.03),
    ]
    rows = [(label, segment_metrics(df[mask])) for label, mask in bins]
    print_segment_table("ACCURACY BY PARK TYPE", rows)


def analyze_by_temperature(df: pd.DataFrame) -> None:
    bins = [
        ("Cold  (< 55°F)",     df["temperature"] < 55),
        ("Cool  (55–68°F)",   (df["temperature"] >= 55) & (df["temperature"] < 68)),
        ("Normal (68–80°F)",  (df["temperature"] >= 68) & (df["temperature"] < 80)),
        ("Hot   (>= 80°F)",    df["temperature"] >= 80),
    ]
    rows = [(label, segment_metrics(df[mask])) for label, mask in bins]
    print_segment_table("ACCURACY BY GAME TEMPERATURE", rows)


def analyze_by_sp_quality(df: pd.DataFrame) -> None:
    """Split by the size of the starting pitcher advantage."""
    gs_diff = df["sp_rolling_gs_diff"]
    bins = [
        ("Big home SP edge  (diff >10)",   gs_diff >  10),
        ("Small home SP edge (0–10)",      (gs_diff >= 0) & (gs_diff <= 10)),
        ("Small away SP edge (0–10)",      (gs_diff < 0)  & (gs_diff >= -10)),
        ("Big away SP edge  (diff < -10)", gs_diff < -10),
    ]
    rows = [(label, segment_metrics(df[mask])) for label, mask in bins]
    print_segment_table("ACCURACY BY STARTING PITCHER QUALITY GAP", rows)


def analyze_by_team_strength_gap(df: pd.DataFrame) -> None:
    """Split by how lopsided the pythagorean win% gap is."""
    pyth = df["pythagorean_diff"]
    bins = [
        ("Strong home fav  (diff > 0.08)",   pyth >  0.08),
        ("Moderate home fav (0.03–0.08)",    (pyth >= 0.03) & (pyth <= 0.08)),
        ("Toss-up           (|diff| < 0.03)", pyth.abs() < 0.03),
        ("Moderate away fav (-0.08–-0.03)", (pyth <= -0.03) & (pyth >= -0.08)),
        ("Strong away fav  (diff < -0.08)",  pyth < -0.08),
    ]
    rows = [(label, segment_metrics(df[mask])) for label, mask in bins]
    print_segment_table("ACCURACY BY TEAM STRENGTH GAP (Pythagorean)", rows)


def analyze_by_home_team(df: pd.DataFrame) -> None:
    rows = []
    for team in sorted(df["home_team"].unique()):
        g = df[df["home_team"] == team]
        rows.append((team, segment_metrics(g)))
    print_segment_table("ACCURACY BY HOME TEAM (model predicting home results)", rows,
                         sort_by="hc_accuracy")


def analyze_by_season(df: pd.DataFrame) -> None:
    rows = []
    for season in sorted(df["season"].unique()):
        g = df[df["season"] == season]
        rows.append((str(season), segment_metrics(g)))
    print_segment_table("ACCURACY BY SEASON (test years only)", rows, sort_by="accuracy")


def analyze_by_elo_gap(df: pd.DataFrame) -> None:
    """Elo difference reflects accumulated team quality over the season."""
    elo = df["elo_diff"]
    bins = [
        ("Home clear fav  (Elo diff > 75)",   elo >  75),
        ("Home slight fav (25–75)",           (elo >= 25) & (elo <= 75)),
        ("Even match      (|diff| < 25)",      elo.abs() < 25),
        ("Away slight fav (-75–-25)",         (elo <= -25) & (elo >= -75)),
        ("Away clear fav  (Elo diff < -75)",  elo < -75),
    ]
    rows = [(label, segment_metrics(df[mask])) for label, mask in bins]
    print_segment_table("ACCURACY BY ELO RATING GAP", rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not INPUT_CSV.exists():
        print(f"ERROR: {INPUT_CSV} not found. Run build_dataset.py first.")
        sys.exit(1)

    print("=" * 80)
    print("  MLB Oracle v4.0 — Segment Accuracy Analysis")
    print("  Test years: 2022–2025 (walk-forward CV, no lookahead)")
    print(f"  High-conviction threshold: >={HC_THRESHOLD:.0%}")
    print(f"  P&L uses realistic scenario: model_prob - 7% entry price")
    print("=" * 80)

    # Load + clean
    df = pd.read_csv(INPUT_CSV, parse_dates=["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    print(f"\nLoaded {len(df):,} games ({df['season'].min()}–{df['season'].max()})")

    # Generate walk-forward predictions on test set (2022–2025 only)
    print("Generating walk-forward predictions on 2022–2025 test set...")
    test_df = load_and_predict(df)
    print(f"Test set: {len(test_df):,} games with model predictions\n")

    # Run all segment analyses
    analyze_by_season(test_df)
    analyze_by_month(test_df)
    analyze_by_park(test_df)
    analyze_by_temperature(test_df)
    analyze_by_sp_quality(test_df)
    analyze_by_team_strength_gap(test_df)
    analyze_by_elo_gap(test_df)
    analyze_by_home_team(test_df)

    print("\nDone. Look for segments with high HC Acc + positive ROI for best betting spots.")


if __name__ == "__main__":
    main()
