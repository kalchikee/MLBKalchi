#!/usr/bin/env python3
"""Refit MLB isotonic calibration using 2026 in-season graded picks.

The training-fit calibration (data/model/calibration.json) was learned
on 2018-2025 game outcomes. Production 2026 grading_history.json shows
the model is overconfident in every bucket:

    Bucket   declared  actual  miscal
    50-60%   59%       58%     +1pp   ✓
    60-70%   64%       55%    +9pp   🟡
    70-80%   74%       55%   +19pp   🔴
    80-90%   82%       62%   +20pp   🔴

The 70-80% bucket has 67 picks at 55% hit rate — exactly where Kelly
sizes up most aggressively. This script composes a new isotonic step
on top of the existing calibration so that overconfident raw outputs
get pushed back toward the empirical hit rate.

Compose: new_y(raw_prob) = ir(old_cal(raw_prob))
  - Fit ir on (modelProb, won) pairs from grading_history.json (these
    are POST-old-calibration predictions, which is what's tracked).
  - At each existing x_threshold x_i, the old calibration emits y_i.
    Evaluate the new isotonic at y_i to get the composed output.
  - Shrink toward the old y_threshold using a Gaussian-kernel ESS
    weighting so sparse buckets stay anchored.

Usage:
    python python/refit_calibration.py [--dry-run] [--prior STRENGTH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).parent.parent
HISTORY = ROOT / "data" / "grading_history.json"
CALIB_PATH = ROOT / "data" / "model" / "calibration.json"


def load_pairs(history_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (home_prob_predicted, home_won) pairs from graded picks.

    The predictions JSON stores modelProb for the PICKED team; convert
    to a home-side framing so the isotonic fit is symmetric over [0, 1]
    (matches the training-fit calibration's framing)."""
    with open(history_path) as f:
        h = json.load(f)
    xs, ys = [], []
    for r in h.get("graded", []):
        p_pick = float(r.get("modelProb") or 0)
        if p_pick == 0:
            continue
        won = 1 if r.get("correct") else 0
        picked_home = r.get("pickedTeam") == r.get("home")
        if picked_home:
            p_home, home_won = p_pick, won
        else:
            p_home, home_won = 1.0 - p_pick, 1 - won
        xs.append(p_home)
        ys.append(home_won)
    return np.array(xs), np.array(ys)


def refit_composed(
    xs: np.ndarray,
    ys: np.ndarray,
    old_x: np.ndarray,
    old_y: np.ndarray,
    prior_strength: float = 15.0,
    kernel_width: float = 0.04,
) -> np.ndarray:
    """Compose a new isotonic step on top of the existing calibration.

    The new calibration maps raw_prob -> empirical hit rate by
    evaluating the in-season isotonic AT old_y values (the existing
    calibrated outputs). This is the rigorous composition: new(raw) =
    ir(old_cal(raw)). Shrinkage at each x_threshold pulls the new
    value toward old_y when local sample density is thin."""
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(xs, ys)
    # Evaluate the new isotonic at the OLD calibrated outputs, not at
    # the raw x_thresholds — the in-season data lives in post-cal space.
    data_y = ir.predict(old_y)
    # Effective sample size near each old_y value via Gaussian kernel
    # weighting. Buckets with lots of nearby observed picks dominate;
    # sparse buckets stay near the prior (old calibration).
    ess = np.array([
        np.exp(-((xs - y) ** 2) / (2.0 * kernel_width ** 2)).sum()
        for y in old_y
    ])
    blended = (ess * data_y + prior_strength * old_y) / (ess + prior_strength)
    # Enforce monotonic non-decreasing
    for i in range(1, len(blended)):
        if blended[i] < blended[i - 1]:
            blended[i] = blended[i - 1]
    return blended


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print deltas without writing calibration.json")
    parser.add_argument("--prior", type=float, default=15.0,
                        help="Prior strength (higher = trust old fit more)")
    parser.add_argument("--kernel-width", type=float, default=0.04,
                        help="Gaussian kernel width for ESS calculation")
    args = parser.parse_args()

    xs, ys = load_pairs(HISTORY)
    if len(xs) == 0:
        print("[refit] No graded picks in history.")
        return 1
    print(f"[refit] graded picks n={len(xs)}  mean p={xs.mean():.3f}  "
          f"home hit rate={ys.mean():.3f}")

    with open(CALIB_PATH) as f:
        old = json.load(f)
    old_x = np.array(old["x_thresholds"])
    old_y = np.array(old["y_thresholds"])

    new_y = refit_composed(xs, ys, old_x, old_y,
                           prior_strength=args.prior,
                           kernel_width=args.kernel_width)

    # Show deltas at key raw-input values by interpolating both old and
    # new calibration tables.
    def interp(xs_t, ys_t, x):
        if x <= xs_t[0]:  return ys_t[0]
        if x >= xs_t[-1]: return ys_t[-1]
        for i in range(len(xs_t) - 1):
            if xs_t[i] <= x <= xs_t[i + 1]:
                t = (x - xs_t[i]) / (xs_t[i + 1] - xs_t[i])
                return ys_t[i] + t * (ys_t[i + 1] - ys_t[i])
        return ys_t[-1]

    print("[refit] raw_prob -> calibrated (old -> new):")
    for raw in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        old_v = interp(old_x, old_y, raw)
        new_v = interp(old_x, new_y, raw)
        print(f"  raw={raw:.2f}  {old_v:.3f} -> {new_v:.3f}  "
              f"(delta {new_v-old_v:+.3f})")

    if args.dry_run:
        print("[refit] --dry-run: not writing calibration.json")
        return 0

    out = {
        "method": "isotonic",
        "x_thresholds": old["x_thresholds"],
        "y_thresholds": new_y.tolist(),
        "refit_meta": {
            "source": "2026 in-season grading_history.json + composition",
            "n_picks": int(len(xs)),
            "prior_strength": args.prior,
            "kernel_width": args.kernel_width,
        },
    }
    CALIB_PATH.write_text(json.dumps(out, indent=2))
    print(f"[refit] wrote {CALIB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
