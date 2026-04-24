#!/usr/bin/env python3
"""
MLB Oracle v4.0 — Walk-Forward Backtest Engine
Uses historical_features.csv (17k real games, 2018-2025 skip 2020).
Walk-forward: train on seasons 1..N-1, test on season N.
Compares Logistic Regression vs XGBoost vs Ensemble.

Usage:
    python python/backtest.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "historical_features.csv"

FEATURE_NAMES = [
    "elo_diff", "sp_xfip_diff", "sp_kbb_diff", "sp_siera_diff",
    "sp_rolling_gs_diff", "lineup_woba_diff", "lineup_wrc_plus_diff",
    "pythagorean_diff", "log5_prob", "park_factor",
    "wind_out_cf", "wind_in_cf", "temperature", "umpire_run_factor",
    "rest_days_diff", "travel_tz_shift", "day_after_night",
    "statcast_xba_diff", "statcast_barrel_diff", "statcast_hardhit_diff",
    "statcast_ev_diff", "gb_rate_diff", "vegas_home_prob",
    "momentum_diff", "run_diff_diff", "platoon_advantage",
]


def main():
    print("MLB Oracle v4.0 -- Walk-Forward Backtest (Real Data)")
    print("=" * 55)

    if not CSV_PATH.exists():
        print(f"No dataset at {CSV_PATH}")
        print("Run: python python/build_dataset.py")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} real games")

    label_col = "label"
    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    print(f"Features: {len(feature_cols)}")

    seasons = sorted(df["season"].unique())
    print(f"Seasons: {list(seasons)}")
    print(f"Home win rate: {df[label_col].mean():.3f}")
    print()

    print(f"  {'Season':>6}  {'N':>5}  {'LR':>6}  {'XGB':>6}  {'Ens':>6}  {'Brier(LR)':>10}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")

    lr_results, xgb_results, ens_results = {}, {}, {}

    for i, test_season in enumerate(seasons[1:], 1):
        train_df = df[df["season"].isin(seasons[:i])]
        test_df = df[df["season"] == test_season]

        if len(train_df) < 50 or len(test_df) < 20:
            continue

        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df[label_col].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Logistic Regression
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_train_s, y_train)
        lr_preds = np.clip(lr.predict_proba(X_test_s)[:, 1], 0.01, 0.99)

        # XGBoost
        xgb_preds = None
        if HAS_XGB:
            xgb = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5,
                eval_metric='logloss', verbosity=0,
            )
            xgb.fit(X_train, y_train)
            xgb_preds = np.clip(xgb.predict_proba(X_test)[:, 1], 0.01, 0.99)

        ens_preds = (lr_preds + xgb_preds) / 2 if xgb_preds is not None else lr_preds

        def score(preds, y):
            acc = accuracy_score(y, preds >= 0.5)
            brier = brier_score_loss(y, preds)
            hc_mask = (preds >= 0.65) | (preds <= 0.35)
            hc_acc = accuracy_score(y[hc_mask], preds[hc_mask] >= 0.5) if hc_mask.sum() > 0 else None
            return acc, brier, hc_acc

        lr_acc, lr_brier, lr_hc = score(lr_preds, y_test)
        lr_results[str(test_season)] = {"accuracy": lr_acc, "brier": lr_brier, "hc_acc": lr_hc, "n": len(test_df)}

        xgb_str = "  N/A "
        ens_str = "  N/A "
        if xgb_preds is not None:
            xgb_acc, xgb_brier, xgb_hc = score(xgb_preds, y_test)
            xgb_results[str(test_season)] = {"accuracy": xgb_acc, "brier": xgb_brier, "hc_acc": xgb_hc, "n": len(test_df)}
            ens_acc, ens_brier, ens_hc = score(ens_preds, y_test)
            ens_results[str(test_season)] = {"accuracy": ens_acc, "brier": ens_brier, "hc_acc": ens_hc, "n": len(test_df)}
            xgb_str = f"{xgb_acc:.3f}"
            ens_str = f"{ens_acc:.3f}"

        print(f"  {test_season:>6}  {len(test_df):>5}  {lr_acc:.3f}  {xgb_str}  {ens_str}  {lr_brier:>10.4f}")

    def summarize(results, label):
        if not results:
            return
        accs = [r["accuracy"] for r in results.values()]
        briers = [r["brier"] for r in results.values()]
        hc_accs = [r["hc_acc"] for r in results.values() if r["hc_acc"] is not None]
        avg_hc = np.mean(hc_accs) if hc_accs else None
        print(f"\n{label}:")
        print(f"  Accuracy:            {np.mean(accs):.4f}  (binary baseline ~0.529)")
        print(f"  Brier score:         {np.mean(briers):.4f}  (lower = better, naive ~0.249)")
        if avg_hc:
            print(f"  HC accuracy (>=65%): {avg_hc:.4f}")

    summarize(lr_results, "LR Baseline")
    if xgb_results:
        summarize(xgb_results, "XGBoost")
    if ens_results:
        summarize(ens_results, "Ensemble (LR+XGB)")

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
