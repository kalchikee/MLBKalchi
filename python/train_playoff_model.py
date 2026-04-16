#!/usr/bin/env python3
"""
MLB Playoff Model Trainer — combines regular season + playoff data (3x weight).
Output: data/model/playoff_coefficients.json

Usage: python python/train_playoff_model.py
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

REG_CSV     = DATA_DIR / "historical_features.csv"
PLAYOFF_CSV = DATA_DIR / "playoff_data.csv"
PLAYOFF_WEIGHT = 3.0

FEATURE_NAMES = [
    "elo_diff", "win_pct_diff", "rpg_diff", "rapg_diff",
    "pythagorean_diff", "log5_prob",
    # Starting pitcher (single biggest signal in MLB postseason)
    "sp_era_diff", "sp_whip_diff", "sp_k9_diff", "sp_bb9_diff",
    # Series context
    "series_game_num", "series_deficit", "is_elimination_game",
]


def main():
    print("MLB Playoff Model Trainer")
    print("=" * 40)

    if not PLAYOFF_CSV.exists():
        print("No playoff data — run fetch_playoff_data.py first.")
        sys.exit(1)

    reg = pd.read_csv(REG_CSV); reg["is_playoff"] = 0; reg["weight"] = 1.0
    po  = pd.read_csv(PLAYOFF_CSV); po["is_playoff"] = 1; po["weight"] = PLAYOFF_WEIGHT

    feat_cols = [c for c in FEATURE_NAMES]
    for c in feat_cols:
        if c not in reg.columns: reg[c] = 0.0
        if c not in po.columns:  po[c] = 0.0

    keep = feat_cols + ["label", "season", "is_playoff", "weight"]
    df = pd.concat([reg[keep], po[keep]], ignore_index=True)
    print(f"Combined: {len(df)} rows ({int((df['is_playoff']==0).sum())} reg + {int((df['is_playoff']==1).sum())} playoff)")

    po_seasons = sorted(po["season"].unique())
    print(f"Playoff seasons: {po_seasons}\n")

    print("Walk-forward CV (test = playoff games only):")
    print(f"  {'Season':>6}  {'N':>4}  {'LR':>6}  {'XGB':>6}  {'Ens':>6}")
    lr_accs, xgb_accs, ens_accs = [], [], []

    for i, ts in enumerate(po_seasons):
        train = df[(df["is_playoff"]==0) | ((df["is_playoff"]==1) & df["season"].isin(po_seasons[:i]))]
        test  = df[(df["is_playoff"]==1) & (df["season"]==ts)]
        if len(train) < 50 or len(test) < 4: continue

        X_tr = train[feat_cols].fillna(0).values; y_tr = train["label"].values; w_tr = train["weight"].values
        X_te = test[feat_cols].fillna(0).values;  y_te = test["label"].values
        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_tr_s, y_tr, sample_weight=w_tr)
        lr_p = np.clip(lr.predict_proba(X_te_s)[:, 1], 0.01, 0.99)
        lr_acc = accuracy_score(y_te, lr_p >= 0.5); lr_accs.append(lr_acc)

        xgb_str = ens_str = "  N/A"
        if HAS_XGB:
            xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                 min_child_weight=5, verbosity=0, eval_metric='logloss')
            xgb.fit(X_tr, y_tr, sample_weight=w_tr)
            xp = np.clip(xgb.predict_proba(X_te)[:, 1], 0.01, 0.99)
            ep = (lr_p + xp) / 2
            xa = accuracy_score(y_te, xp >= 0.5); ea = accuracy_score(y_te, ep >= 0.5)
            xgb_accs.append(xa); ens_accs.append(ea)
            xgb_str = f"{xa:.3f}"; ens_str = f"{ea:.3f}"

        print(f"  {ts:>6}  {len(test):>4}  {lr_acc:.3f}  {xgb_str}  {ens_str}")

    print(f"\nSummary:")
    if lr_accs:  print(f"  LR:       {np.mean(lr_accs):.4f}")
    if xgb_accs: print(f"  XGBoost:  {np.mean(xgb_accs):.4f}")
    if ens_accs: print(f"  Ensemble: {np.mean(ens_accs):.4f}")

    X_all = df[feat_cols].fillna(0).values; y_all = df["label"].values; w_all = df["weight"].values
    sc_f = StandardScaler(); X_all_s = sc_f.fit_transform(X_all)
    lr_f = LogisticRegression(C=1.0, max_iter=1000); lr_f.fit(X_all_s, y_all, sample_weight=w_all)

    (MODEL_DIR / "playoff_coefficients.json").write_text(json.dumps({
        "intercept": float(lr_f.intercept_[0]), "coefficients": lr_f.coef_[0].tolist(), "feature_names": feat_cols,
    }, indent=2))
    (MODEL_DIR / "playoff_scaler.json").write_text(json.dumps({
        "mean": sc_f.mean_.tolist(), "scale": sc_f.scale_.tolist(), "feature_names": feat_cols,
    }, indent=2))
    (MODEL_DIR / "playoff_metadata.json").write_text(json.dumps({
        "sport": "MLB", "playoff_weight": PLAYOFF_WEIGHT,
        "cv_accuracy_lr": float(np.mean(lr_accs)) if lr_accs else None,
        "feature_names": feat_cols, "playoff_seasons": [int(s) for s in po_seasons],
    }, indent=2))
    print(f"\nSaved to {MODEL_DIR}/playoff_*.json")


if __name__ == "__main__":
    main()
