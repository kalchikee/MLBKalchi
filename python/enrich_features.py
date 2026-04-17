#!/usr/bin/env python3
"""
MLB Feature Enrichment — backfills missing columns in historical_features.csv
using the MLB Stats API (statsapi.mlb.com) and game-date computations.

Adds: statcast proxies, bullpen strength, rest days, rolling team stats,
      fielding quality, ground ball rates.

Usage: python python/enrich_features.py
"""
import sys, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_CSV    = DATA_DIR / "historical_features.csv"
OUTPUT_CSV   = DATA_DIR / "historical_features.csv"
BACKUP_CSV   = DATA_DIR / "historical_features_backup.csv"

MLB_BASE = "https://statsapi.mlb.com/api/v1"
HEADERS  = {"User-Agent": "MLB-Oracle/4.1"}


def mlb_get(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"    FAIL: {e}")
    return {}


def fetch_team_stats(year: int) -> dict:
    """Fetch batting, pitching, and fielding stats from MLB Stats API."""
    stats = {}

    # Team batting
    data = mlb_get(f"{MLB_BASE}/teams/stats?season={year}&group=hitting&stats=season&sportIds=1&hydrate=team")
    for split in data.get("stats", [{}])[0].get("splits", []):
        abbr = split.get("team", {}).get("abbreviation", "")
        if not abbr:
            continue
        s = split.get("stat", {})
        ab = float(s.get("atBats", 1) or 1)
        go = float(s.get("groundOuts", 0) or 0)
        ao = float(s.get("airOuts", 0) or 0)
        stats.setdefault(abbr, {}).update({
            "avg":     float(s.get("avg", ".250").replace(".", "0.", 1) if isinstance(s.get("avg"), str) and not s["avg"].startswith("0") else s.get("avg", 0.250)),
            "obp":     float(s.get("obp", 0.320) or 0.320),
            "slg":     float(s.get("slg", 0.400) or 0.400),
            "ops":     float(s.get("ops", 0.720) or 0.720),
            "hr_rate": float(s.get("homeRuns", 0) or 0) / ab if ab > 0 else 0.03,
            "gb_rate": go / (go + ao) if (go + ao) > 0 else 0.43,
        })

    # Team pitching
    data = mlb_get(f"{MLB_BASE}/teams/stats?season={year}&group=pitching&stats=season&sportIds=1&hydrate=team")
    for split in data.get("stats", [{}])[0].get("splits", []):
        abbr = split.get("team", {}).get("abbreviation", "")
        if not abbr:
            continue
        s = split.get("stat", {})
        ip_str = str(s.get("inningsPitched", "1") or "1")
        ip_parts = ip_str.split(".")
        ip = float(ip_parts[0]) + int(ip_parts[1]) / 3.0 if len(ip_parts) == 2 else float(ip_str)
        stats.setdefault(abbr, {}).update({
            "team_era":  float(s.get("era", 4.00) or 4.00),
            "team_whip": float(s.get("whip", 1.25) or 1.25),
            "team_k9":   float(s.get("strikeOuts", 0) or 0) * 9 / ip if ip > 0 else 8.5,
            "team_bb9":  float(s.get("baseOnBalls", 0) or 0) * 9 / ip if ip > 0 else 3.0,
        })

    # Team fielding
    data = mlb_get(f"{MLB_BASE}/teams/stats?season={year}&group=fielding&stats=season&sportIds=1&hydrate=team")
    for split in data.get("stats", [{}])[0].get("splits", []):
        abbr = split.get("team", {}).get("abbreviation", "")
        if not abbr:
            continue
        s = split.get("stat", {})
        gp = float(s.get("gamesPlayed", 162) or 162)
        errors = float(s.get("errors", 0) or 0)
        stats.setdefault(abbr, {}).update({
            "fielding_pct":    float(s.get("fielding", 0.985) or 0.985),
            "range_factor":    float(s.get("rangeFactorPerGame", 3.8) or 3.8),
            "errors_per_game": errors / gp if gp > 0 else 0.5,
        })

    return stats


def compute_rest_days(df: pd.DataFrame) -> pd.Series:
    """Compute rest days difference (home - away) for each game."""
    df_sorted = df.sort_values("game_date")
    last_game = {}
    rest_diffs = []

    for _, row in df_sorted.iterrows():
        h, a = row["home_team"], row["away_team"]
        gd = pd.to_datetime(row["game_date"], errors="coerce")

        rest_h = (gd - last_game[h]).days if h in last_game and pd.notna(gd) else 2
        rest_a = (gd - last_game[a]).days if a in last_game and pd.notna(gd) else 2
        rest_diffs.append(rest_h - rest_a)

        if pd.notna(gd):
            last_game[h] = gd
            last_game[a] = gd

    return pd.Series(rest_diffs, index=df_sorted.index)


def compute_rolling_stats(df: pd.DataFrame, window: int = 10) -> tuple:
    """Compute rolling 10-game team offense/defense proxies."""
    df_sorted = df.sort_values("game_date")
    team_runs_for = defaultdict(list)
    team_runs_ag = defaultdict(list)
    woba_diffs, fip_diffs = [], []

    for _, row in df_sorted.iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = row.get("home_score", 4.5), row.get("away_score", 4.5)

        hrf = team_runs_for[h][-window:] or [4.5]
        hra = team_runs_ag[h][-window:] or [4.5]
        arf = team_runs_for[a][-window:] or [4.5]
        ara = team_runs_ag[a][-window:] or [4.5]

        woba_diffs.append(np.mean(hrf) / 4.5 - np.mean(arf) / 4.5)
        fip_diffs.append(np.mean(hra) - np.mean(ara))

        team_runs_for[h].append(hs)
        team_runs_ag[h].append(as_)
        team_runs_for[a].append(as_)
        team_runs_ag[a].append(hs)

    return (
        pd.Series(woba_diffs, index=df_sorted.index),
        pd.Series(fip_diffs, index=df_sorted.index),
    )


def main():
    print("MLB Feature Enrichment (MLB Stats API)")
    print("=" * 50)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")
    df.to_csv(BACKUP_CSV, index=False)
    print(f"Backup saved to {BACKUP_CSV}")

    seasons = sorted(df["season"].unique())
    print(f"Seasons: {seasons}")

    # ── 1. Rest days ─────────────────────────────────────────────────────
    print("\n[1/3] Computing rest days...")
    df = df.sort_values("game_date")
    df["rest_days_diff"] = compute_rest_days(df)
    print(f"  rest_days_diff: std={df['rest_days_diff'].std():.3f}")

    # ── 2. Rolling 10-game stats ─────────────────────────────────────────
    print("\n[2/3] Computing rolling team stats...")
    woba_roll, fip_roll = compute_rolling_stats(df)
    df["team_10d_woba_diff"] = woba_roll
    df["team_10d_fip_diff"] = fip_roll
    print(f"  team_10d_woba_diff: std={df['team_10d_woba_diff'].std():.3f}")
    print(f"  team_10d_fip_diff:  std={df['team_10d_fip_diff'].std():.3f}")

    # ── 3. MLB Stats API team stats ──────────────────────────────────────
    print("\n[3/3] Fetching MLB Stats API team stats...")
    for year in seasons:
        print(f"\n  Season {year}...")
        stats = fetch_team_stats(int(year))
        print(f"    Got stats for {len(stats)} teams")

        mask = df["season"] == year
        for idx in df[mask].index:
            h, a = df.at[idx, "home_team"], df.at[idx, "away_team"]
            hs, as_ = stats.get(h, {}), stats.get(a, {})

            # Statcast proxies from MLB batting stats
            df.at[idx, "statcast_xba_diff"]    = hs.get("avg", 0.250) - as_.get("avg", 0.250)
            df.at[idx, "statcast_barrel_diff"]  = hs.get("hr_rate", 0.03) - as_.get("hr_rate", 0.03)
            df.at[idx, "statcast_hardhit_diff"] = hs.get("slg", 0.400) - as_.get("slg", 0.400)
            df.at[idx, "gb_rate_diff"]          = hs.get("gb_rate", 0.43) - as_.get("gb_rate", 0.43)

            # Bullpen strength (team ERA differential — lower = better)
            df.at[idx, "bullpen_strength_diff"] = as_.get("team_era", 4.0) - hs.get("team_era", 4.0)

            # Defense (fielding quality proxy — range factor differential)
            df.at[idx, "drs_diff"] = hs.get("range_factor", 3.8) - as_.get("range_factor", 3.8)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Enriched feature summary:")
    enriched = [
        "rest_days_diff", "team_10d_woba_diff", "team_10d_fip_diff",
        "statcast_xba_diff", "statcast_barrel_diff", "statcast_hardhit_diff",
        "gb_rate_diff", "bullpen_strength_diff", "drs_diff",
    ]
    for col in enriched:
        s = df[col].std() if col in df.columns else 0
        status = "✓" if s > 0.001 else "✗ still zero"
        print(f"  {col:30s}  std={s:.4f}  {status}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
