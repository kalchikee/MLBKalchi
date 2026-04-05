#!/usr/bin/env python3
"""
MLB Oracle v4.0 — Walk-Forward Backtest Engine
Walk-forward cross-validation across 2018–2025 (skip 2020).

Splits:
  1: Train 2018–2021, test 2022
  2: Train 2018–2022, test 2023
  3: Train 2018–2023, test 2024
  4: Train 2018–2024, test 2025

For each split:
  - Day-by-day predictions with sequential Elo updates (no lookahead bias)
  - Rolling 10-day stats computed only from prior games
  - Metrics: Brier score, log loss, accuracy, high-conviction accuracy (65%+), calibration

Usage:
    python python/backtest.py
    python python/backtest.py --output data/backtest_results.json
"""

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "data" / "mlb_oracle.db"
OUTPUT_PATH = REPO_ROOT / "data" / "backtest_results.json"

# ─── Elo system ───────────────────────────────────────────────────────────────

ELO_K = 20.0
ELO_REGRESSION = 0.60   # regress to mean at season start
ELO_MEAN = 1500.0
HOME_ADVANTAGE = 35.0   # Elo points for home team

class EloModel:
    def __init__(self):
        self.ratings: Dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, ELO_MEAN)

    def set(self, team: str, rating: float):
        self.ratings[team] = rating

    def win_probability(self, home: str, away: str) -> float:
        """Home team win probability."""
        h_elo = self.get(home) + HOME_ADVANTAGE
        a_elo = self.get(away)
        return 1.0 / (1.0 + 10 ** ((a_elo - h_elo) / 400.0))

    def update(self, home: str, away: str, home_won: bool):
        """Update Elo ratings after a game."""
        prob = self.win_probability(home, away)
        outcome = 1.0 if home_won else 0.0
        delta = ELO_K * (outcome - prob)
        self.ratings[home] = self.get(home) + delta
        self.ratings[away] = self.get(away) - delta

    def regress_to_mean(self, fraction: float = ELO_REGRESSION):
        """Apply mean regression at season start."""
        for team in list(self.ratings.keys()):
            self.ratings[team] = ELO_MEAN + (self.ratings[team] - ELO_MEAN) * (1 - fraction)

# ─── Rolling stats (10-day) ───────────────────────────────────────────────────

class RollingStats:
    """10-day rolling stats per team — only uses prior games (no lookahead)."""

    def __init__(self, window: int = 10):
        self.window = window
        # team -> list of (date, runs_scored, runs_allowed)
        self.game_log: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)

    def add_game(self, team: str, date: str, runs_scored: int, runs_allowed: int):
        self.game_log[team].append((date, runs_scored, runs_allowed))

    def get_stats(self, team: str, as_of_date: str) -> Dict[str, float]:
        cutoff = datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=self.window)
        games = [
            g for g in self.game_log[team]
            if datetime.strptime(g[0], "%Y-%m-%d") <= datetime.strptime(as_of_date, "%Y-%m-%d")
            and datetime.strptime(g[0], "%Y-%m-%d") > cutoff
        ]

        if not games:
            return {"rpg": 4.5, "rapg": 4.5, "run_diff": 0.0}

        rpg = sum(g[1] for g in games) / len(games)
        rapg = sum(g[2] for g in games) / len(games)
        return {"rpg": rpg, "rapg": rapg, "run_diff": rpg - rapg}

# ─── Simplified probability model ────────────────────────────────────────────

def compute_win_probability(
    home_team: str,
    away_team: str,
    elo: EloModel,
    rolling: RollingStats,
    game_date: str,
) -> float:
    """
    Combines Elo + rolling 10-day form into a win probability estimate.
    This mirrors the spirit of the full feature engine without needing
    the live API — suitable for historical backtest.
    """
    elo_prob = elo.win_probability(home_team, away_team)

    home_stats = rolling.get_stats(home_team, game_date)
    away_stats = rolling.get_stats(away_team, game_date)

    # Run differential per game as a tiebreaker / adjustment
    home_rdiff = home_stats["run_diff"]
    away_rdiff = away_stats["run_diff"]
    net_rdiff = home_rdiff - away_rdiff

    # Each 1-run-per-game differential ≈ 5% win probability shift
    rdiff_adjustment = net_rdiff * 0.025
    rdiff_adjustment = max(-0.10, min(0.10, rdiff_adjustment))

    raw_prob = elo_prob + rdiff_adjustment
    return max(0.05, min(0.95, raw_prob))

# ─── Metric calculations ──────────────────────────────────────────────────────

def brier_score(probs: List[float], outcomes: List[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)

def log_loss(probs: List[float], outcomes: List[int]) -> float:
    if not probs:
        return float("nan")
    eps = 1e-10
    return -sum(
        o * math.log(max(eps, p)) + (1 - o) * math.log(max(eps, 1 - p))
        for p, o in zip(probs, outcomes)
    ) / len(probs)

def accuracy(probs: List[float], outcomes: List[int]) -> float:
    if not probs:
        return float("nan")
    correct = sum(1 for p, o in zip(probs, outcomes) if (p >= 0.5) == bool(o))
    return correct / len(probs)

def high_conviction_accuracy(probs: List[float], outcomes: List[int], threshold: float = 0.65) -> float:
    """Accuracy on picks where |prob - 0.5| >= threshold - 0.5."""
    hc = [(p, o) for p, o in zip(probs, outcomes) if p >= threshold or p <= (1 - threshold)]
    if not hc:
        return float("nan")
    correct = sum(1 for p, o in hc if (p >= 0.5) == bool(o))
    return correct / len(hc)

def calibration_curve(probs: List[float], outcomes: List[int]) -> List[Dict]:
    """Bucket predictions into 5% bins and compare to actual win rates."""
    buckets: Dict[int, Tuple[List[float], List[int]]] = defaultdict(lambda: ([], []))

    for p, o in zip(probs, outcomes):
        bucket = int(p * 20) * 5  # 0, 5, 10, ..., 95
        bucket = min(95, max(0, bucket))
        buckets[bucket][0].append(p)
        buckets[bucket][1].append(o)

    result = []
    for lo in range(50, 100, 5):
        ps, os_ = buckets[lo]
        result.append({
            "bucket": f"{lo}-{lo+5}%",
            "predicted_mean": sum(ps) / len(ps) if ps else None,
            "actual_win_rate": sum(os_) / len(os_) if os_ else None,
            "count": len(ps),
        })
    return result

# ─── Load historical game data ────────────────────────────────────────────────

def load_games_from_db(db_path: Path) -> List[Dict]:
    """Load completed game results from the SQLite DB."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT game_id, date, home_team, away_team, home_score, away_score
            FROM game_results
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date ASC
        """)
        games = [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        games = []
    finally:
        conn.close()

    return games

def load_predictions_from_db(db_path: Path) -> List[Dict]:
    """Load stored predictions from the SQLite DB."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT game_pk, game_date, home_team, away_team,
                   mc_win_pct, calibrated_prob, vegas_prob, actual_winner
            FROM predictions
            ORDER BY game_date ASC
        """)
        preds = [dict(row) for row in cur.fetchall()]
    except sqlite3.OperationalError:
        preds = []
    finally:
        conn.close()

    return preds

# ─── Synthetic historical data generator ─────────────────────────────────────

def generate_synthetic_games(
    start_year: int = 2018,
    end_year: int = 2025,
    skip_years: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Generate synthetic game-by-game results for backtesting when real
    historical data is not in the DB.

    Uses team quality tiers to create semi-realistic outcomes.
    """
    if skip_years is None:
        skip_years = [2020]

    # Team quality tier (approximate win% for 2018–2025 blended)
    team_quality: Dict[str, float] = {
        "LAD": 0.620, "HOU": 0.595, "ATL": 0.580, "NYY": 0.565,
        "BOS": 0.555, "CLE": 0.550, "TB":  0.545, "SF":  0.540,
        "SD":  0.535, "SEA": 0.530, "MIL": 0.528, "CHC": 0.525,
        "MIN": 0.520, "PHI": 0.515, "TOR": 0.512, "STL": 0.510,
        "BAL": 0.508, "NYM": 0.505, "ARI": 0.500, "CIN": 0.498,
        "TEX": 0.495, "PIT": 0.492, "DET": 0.488, "KC":  0.485,
        "MIA": 0.482, "WSH": 0.480, "CWS": 0.478, "COL": 0.470,
        "OAK": 0.468, "LAA": 0.460,
    }

    teams = list(team_quality.keys())
    import random
    rng = random.Random(42)

    games = []
    game_id = 1

    for year in range(start_year, end_year + 1):
        if year in skip_years:
            continue

        # MLB season: April 1 – September 30 (roughly 162 games per team)
        season_start = datetime(year, 4, 1)
        season_end = datetime(year, 9, 30)
        current = season_start

        # Generate ~15 games per day (30 teams / 2)
        while current <= season_end:
            shuffled = teams[:]
            rng.shuffle(shuffled)
            matchups = list(zip(shuffled[:15], shuffled[15:]))

            for home, away in matchups:
                # Home win probability based on team quality + home advantage
                home_q = team_quality.get(home, 0.500)
                away_q = team_quality.get(away, 0.500)

                # Log5 with home advantage
                home_q_adj = min(0.90, home_q * 1.04)
                log5_num = home_q_adj - home_q_adj * away_q
                log5_den = home_q_adj + away_q - 2 * home_q_adj * away_q
                base_prob = log5_num / log5_den if abs(log5_den) > 0.001 else 0.54

                # Add Gaussian noise for realism
                prob = min(0.90, max(0.10, base_prob + rng.gauss(0, 0.05)))

                home_won = rng.random() < prob
                home_score = rng.randint(2, 8) if home_won else rng.randint(1, 6)
                away_score = rng.randint(1, 6) if home_won else rng.randint(2, 8)

                games.append({
                    "game_id": game_id,
                    "date": current.strftime("%Y-%m-%d"),
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "true_home_prob": prob,
                })
                game_id += 1

            current += timedelta(days=1)

    return games

# ─── Walk-forward backtest ────────────────────────────────────────────────────

SPLITS = [
    {"name": "Split 1", "train_years": list(range(2018, 2022)), "test_year": 2022},
    {"name": "Split 2", "train_years": list(range(2018, 2023)), "test_year": 2023},
    {"name": "Split 3", "train_years": list(range(2018, 2024)), "test_year": 2024},
    {"name": "Split 4", "train_years": list(range(2018, 2025)), "test_year": 2025},
]

SKIP_YEARS = [2020]


def run_backtest_split(
    all_games: List[Dict],
    train_years: List[int],
    test_year: int,
    split_name: str,
) -> Dict:
    """Run a single walk-forward split."""
    print(f"\n{'─'*60}")
    print(f"  {split_name}: Train {train_years[0]}–{train_years[-1]}, Test {test_year}")
    print(f"{'─'*60}")

    elo = EloModel()
    rolling = RollingStats(window=10)

    # ── Training phase: build Elo and rolling stats ──────────────────────────
    train_games = [
        g for g in all_games
        if int(g["date"][:4]) in train_years and int(g["date"][:4]) not in SKIP_YEARS
    ]
    train_games.sort(key=lambda g: g["date"])

    prev_year = None
    for game in train_games:
        year = int(game["date"][:4])

        # Regress to mean at season start
        if prev_year is not None and year != prev_year:
            elo.regress_to_mean()
        prev_year = year

        home = game["home_team"]
        away = game["away_team"]
        home_won = game["home_score"] > game["away_score"]

        # Update rolling stats
        rolling.add_game(home, game["date"], game["home_score"], game["away_score"])
        rolling.add_game(away, game["date"], game["away_score"], game["home_score"])

        elo.update(home, away, home_won)

    print(f"  Training complete: {len(train_games)} games processed")

    # ── Test phase: day-by-day predictions ────────────────────────────────────
    test_games = [
        g for g in all_games
        if int(g["date"][:4]) == test_year
    ]
    test_games.sort(key=lambda g: g["date"])

    # Regress at season start
    elo.regress_to_mean()

    probs: List[float] = []
    outcomes: List[int] = []
    dates_seen = set()

    for game in test_games:
        home = game["home_team"]
        away = game["away_team"]
        home_won = game["home_score"] > game["away_score"]
        game_date = game["date"]

        # Predict BEFORE updating (no lookahead)
        prob = compute_win_probability(home, away, elo, rolling, game_date)

        probs.append(prob)
        outcomes.append(1 if home_won else 0)
        dates_seen.add(game_date)

        # Update AFTER prediction
        rolling.add_game(home, game_date, game["home_score"], game["away_score"])
        rolling.add_game(away, game_date, game["away_score"], game["home_score"])
        elo.update(home, away, home_won)

    bs = brier_score(probs, outcomes)
    ll = log_loss(probs, outcomes)
    acc = accuracy(probs, outcomes)
    hc_acc = high_conviction_accuracy(probs, outcomes, threshold=0.65)
    cal = calibration_curve(probs, outcomes)

    hc_games = sum(1 for p in probs if p >= 0.65 or p <= 0.35)

    print(f"  Test games:       {len(probs)}")
    print(f"  Test days:        {len(dates_seen)}")
    print(f"  Brier score:      {bs:.4f}  (lower = better; 0.25 = random)")
    print(f"  Log loss:         {ll:.4f}")
    print(f"  Accuracy:         {acc:.1%}")
    print(f"  High-conv acc:    {hc_acc:.1%} ({hc_games} games ≥65%)")

    print("\n  Calibration curve:")
    print(f"  {'Bucket':<10} {'Predicted':>10} {'Actual':>10} {'Count':>7}")
    print(f"  {'─'*40}")
    for row in cal:
        pred = f"{row['predicted_mean']:.1%}" if row['predicted_mean'] else "N/A"
        actual = f"{row['actual_win_rate']:.1%}" if row['actual_win_rate'] else "N/A"
        print(f"  {row['bucket']:<10} {pred:>10} {actual:>10} {row['count']:>7}")

    return {
        "split": split_name,
        "train_years": train_years,
        "test_year": test_year,
        "n_games": len(probs),
        "n_days": len(dates_seen),
        "brier_score": round(bs, 4) if not math.isnan(bs) else None,
        "log_loss": round(ll, 4) if not math.isnan(ll) else None,
        "accuracy": round(acc, 4) if not math.isnan(acc) else None,
        "high_conv_accuracy": round(hc_acc, 4) if not math.isnan(hc_acc) else None,
        "high_conv_games": hc_games,
        "calibration_curve": cal,
    }


def run_backtest(output_path: Optional[Path] = None) -> Dict:
    print("=" * 60)
    print("  MLB Oracle v4.0 — Walk-Forward Backtest")
    print("=" * 60)

    # Load real data from DB if available
    real_games = load_games_from_db(DB_PATH)
    print(f"\nLoaded {len(real_games)} real game results from DB")

    # Supplement with synthetic data for years not in DB
    synthetic_games = generate_synthetic_games(2018, 2025, skip_years=SKIP_YEARS)
    print(f"Generated {len(synthetic_games)} synthetic games for historical backtest")

    # Prefer real games over synthetic (merge by game_id)
    real_ids = {g["game_id"] for g in real_games}
    merged = real_games + [g for g in synthetic_games if g.get("game_id") not in real_ids]
    merged.sort(key=lambda g: g["date"])

    print(f"Total games for backtest: {len(merged)}")

    # Run each split
    split_results = []
    for split in SPLITS:
        result = run_backtest_split(
            all_games=merged,
            train_years=split["train_years"],
            test_year=split["test_year"],
            split_name=split["name"],
        )
        split_results.append(result)

    # Aggregate metrics
    valid = [r for r in split_results if r["brier_score"] is not None]
    avg_brier = sum(r["brier_score"] for r in valid) / len(valid) if valid else None
    avg_ll = sum(r["log_loss"] for r in valid) / len(valid) if valid else None
    avg_acc = sum(r["accuracy"] for r in valid) / len(valid) if valid else None
    hc_valid = [r for r in valid if r["high_conv_accuracy"] is not None]
    avg_hc_acc = sum(r["high_conv_accuracy"] for r in hc_valid) / len(hc_valid) if hc_valid else None

    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS (Average Across All Splits)")
    print("=" * 60)
    print(f"  Avg Brier score:      {avg_brier:.4f}" if avg_brier else "  Avg Brier score:      N/A")
    print(f"  Avg Log loss:         {avg_ll:.4f}" if avg_ll else "  Avg Log loss:         N/A")
    print(f"  Avg Accuracy:         {avg_acc:.1%}" if avg_acc else "  Avg Accuracy:         N/A")
    print(f"  Avg High-conv acc:    {avg_hc_acc:.1%}" if avg_hc_acc else "  Avg High-conv acc:    N/A")
    print()

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": "4.0.0",
        "backtest_years": "2018-2025 (skip 2020)",
        "splits": split_results,
        "aggregate": {
            "avg_brier_score": round(avg_brier, 4) if avg_brier else None,
            "avg_log_loss": round(avg_ll, 4) if avg_ll else None,
            "avg_accuracy": round(avg_acc, 4) if avg_acc else None,
            "avg_high_conv_accuracy": round(avg_hc_acc, 4) if avg_hc_acc else None,
        },
    }

    # Save results
    out_path = output_path or OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")

    return summary


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB Oracle Walk-Forward Backtest")
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"SQLite DB path (default: {DB_PATH})"
    )
    args = parser.parse_args()

    if args.db != DB_PATH:
        DB_PATH = args.db  # type: ignore

    run_backtest(output_path=args.output)
