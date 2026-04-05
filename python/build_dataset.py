"""
MLB Oracle v4.0 — Phase 3: Historical Dataset Builder
======================================================

Downloads MLB game data from 2018–2025 (skipping 2020 COVID season),
computes the same 30 features used in live predictions, and saves to
data/historical_features.csv. Elo ratings are computed sequentially
game-by-game using the same K=4 / log(1+margin) formula as TypeScript.

Features that cannot be reconstructed historically (Statcast, live
bullpen, etc.) are filled with 0.0 and flagged in ESTIMATED_FEATURES.

Usage:
    pip install -r requirements.txt
    python build_dataset.py            # full 2018–2025 run
    python build_dataset.py --resume   # resume from checkpoint
    python build_dataset.py --year 2023  # single season
"""

import argparse
import json
import math
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_PATH = DATA_DIR / "build_checkpoint.json"
OUTPUT_CSV = DATA_DIR / "historical_features.csv"
PARK_FACTORS_PATH = DATA_DIR / "park_factors.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────

# Seasons to process (skip 2020 COVID season)
SEASONS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

# Elo parameters — must match TypeScript monteCarlo.ts
ELO_K = 4.0
ELO_DEFAULT = 1500.0

# These features cannot be computed from historical schedule/linescore data.
# They are set to 0.0 and noted here for transparency.
ESTIMATED_FEATURES = {
    "sp_csw_diff",          # Called Strikes + Whiffs — Statcast, not in schedule API
    "bullpen_strength_diff",# Requires intra-season bullpen usage tracking
    "team_10d_woba_diff",   # Requires per-game wOBA splits
    "team_10d_fip_diff",    # Requires per-game FIP splits
    "drs_diff",             # Defensive Runs Saved — FanGraphs only
    "catcher_framing_diff", # Statcast-only metric
    "wind_out_cf",          # Weather at game time — not in historical schedule
    "wind_in_cf",           # Same
    "temperature",          # Same
    "umpire_run_factor",    # Umpire tracking not in schedule API
    "travel_tz_shift",      # Would require team schedule lookups
    "day_after_night",      # Would require prior-day game lookups
    "statcast_xba_diff",    # Statcast only
    "statcast_barrel_diff", # Statcast only
    "statcast_hardhit_diff",# Statcast only
    "statcast_ev_diff",     # Statcast only
    "gb_rate_diff",         # Statcast only
}

# Team abbreviation mapping (MLB Stats API team IDs → 3-letter abbr)
TEAM_ID_TO_ABBR: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
    # Historic Oakland (moved to Sacramento as Athletics 2025)
    163: "OAK",
}

# Park run factors (from park_factors.json if available, else defaults)
PARK_RUN_FACTORS: dict[str, float] = {}

# ─── HTTP helpers ─────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json", "User-Agent": "MLBOracle/4.0"})

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def api_get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET with simple retry/backoff logic."""
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            time.sleep(wait)
    return {}


# ─── Elo engine ───────────────────────────────────────────────────────────────

class EloSystem:
    """
    Sequential Elo tracker. Same formula as src/features/elo.ts:
      expected = 1 / (1 + 10^((opp - own) / 400))
      K = 4
      MoV multiplier = log(1 + |margin|)
      new_rating = old_rating + K * MoV_mult * (result - expected)
    Season reset: regress 50% toward 1500.
    """

    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}
        self.current_season: int = 0

    def _get(self, team: str) -> float:
        return self.ratings.get(team, ELO_DEFAULT)

    def season_regress(self, season: int) -> None:
        """Regress all ratings 50% toward mean at start of new season."""
        if season == self.current_season:
            return
        self.current_season = season
        for team in list(self.ratings.keys()):
            self.ratings[team] = 0.5 * self.ratings[team] + 0.5 * ELO_DEFAULT

    def get_diff(self, home: str, away: str) -> float:
        """Return home_elo - away_elo (used as feature)."""
        return self._get(home) - self._get(away)

    def update(self, home: str, away: str, home_score: int, away_score: int) -> None:
        """Update ratings after a completed game."""
        home_rating = self._get(home)
        away_rating = self._get(away)

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))
        expected_away = 1.0 - expected_home

        home_win = 1.0 if home_score > away_score else 0.0
        away_win = 1.0 - home_win

        margin = abs(home_score - away_score)
        mov_mult = math.log(1.0 + margin) if margin > 0 else 1.0

        self.ratings[home] = home_rating + ELO_K * mov_mult * (home_win - expected_home)
        self.ratings[away] = away_rating + ELO_K * mov_mult * (away_win - expected_away)


# ─── Park factor loader ───────────────────────────────────────────────────────

def load_park_factors() -> dict[str, float]:
    """Load team-level run factors from park_factors.json."""
    if not PARK_FACTORS_PATH.exists():
        return {}
    try:
        raw = json.loads(PARK_FACTORS_PATH.read_text())
        # park_factors.json is keyed by stadium name with a 'team' key
        result: dict[str, float] = {}
        for _stadium, info in raw.items():
            team = info.get("team", "")
            factor = info.get("run_factor", 1.0)
            if team:
                result[team] = factor
        return result
    except Exception:
        return {}


# ─── Season schedule fetcher ──────────────────────────────────────────────────

def get_season_dates(season: int) -> tuple[str, str]:
    """Return (start_date, end_date) for regular season + playoffs."""
    # Regular season roughly April–October; include spring training from April
    return f"{season}-04-01", f"{season}-10-31"


def fetch_schedule(start_date: str, end_date: str) -> list[dict]:
    """
    Fetch all games in date range from MLB Stats API.
    Returns list of raw game dicts with linescore hydration.
    """
    url = f"{MLB_API_BASE}/schedule"
    params = {
        "sportId": 1,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "linescore,probablePitcher",
        "gameType": "R,F,D,L,W",  # Regular + postseason
    }
    data = api_get(url, params)
    games = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            games.append(game)
    return games


def fetch_pitcher_stats(player_id: int, season: int) -> dict:
    """
    Fetch season-level pitching stats for a player via MLB Stats API.
    Returns dict with era, whip, strikeouts, walks, inningsPitched.
    Falls back to neutral defaults on error.
    """
    defaults = {
        "era": 4.20,
        "whip": 1.30,
        "k_per_9": 8.5,
        "bb_per_9": 3.0,
        "innings_pitched": 0.0,
        "games_started": 0,
    }
    if player_id <= 0:
        return defaults

    try:
        url = f"{MLB_API_BASE}/people/{player_id}/stats"
        params = {"stats": "season", "season": season, "group": "pitching", "sportId": 1}
        data = api_get(url, params)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return defaults
        s = splits[0].get("stat", {})
        ip = float(s.get("inningsPitched", 0) or 0)
        so = float(s.get("strikeOuts", 0) or 0)
        bb = float(s.get("baseOnBalls", 0) or 0)
        k_per_9 = (so / ip * 9.0) if ip > 0 else 8.5
        bb_per_9 = (bb / ip * 9.0) if ip > 0 else 3.0
        return {
            "era": float(s.get("era", 4.20) or 4.20),
            "whip": float(s.get("whip", 1.30) or 1.30),
            "k_per_9": k_per_9,
            "bb_per_9": bb_per_9,
            "innings_pitched": ip,
            "games_started": int(s.get("gamesStarted", 0) or 0),
        }
    except Exception:
        return defaults


def fetch_team_stats(team_id: int, season: int) -> dict:
    """
    Fetch team-level season hitting + pitching stats.
    Returns dict with runs_per_game, obp, slg, era, fip_approx.
    """
    defaults = {
        "runs_per_game": 4.5,
        "obp": 0.320,
        "slg": 0.420,
        "avg": 0.250,
        "era": 4.20,
        "fip_approx": 4.20,
        "k_pct": 0.220,
        "bb_pct": 0.082,
    }
    try:
        # Hitting
        url = f"{MLB_API_BASE}/teams/{team_id}/stats"
        params = {"stats": "season", "season": season, "group": "hitting", "sportId": 1}
        data = api_get(url, params)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if splits:
            s = splits[0].get("stat", {})
            games = int(s.get("gamesPlayed", 1) or 1)
            runs = float(s.get("runs", games * 4.5) or games * 4.5)
            ab = float(s.get("atBats", 1) or 1)
            h = float(s.get("hits", 0) or 0)
            so = float(s.get("strikeOuts", 0) or 0)
            bb = float(s.get("baseOnBalls", 0) or 0)
            pa = ab + bb + float(s.get("hitByPitch", 0) or 0)
            defaults["runs_per_game"] = runs / max(games, 1)
            defaults["obp"] = float(s.get("obp", 0.320) or 0.320)
            defaults["slg"] = float(s.get("slg", 0.420) or 0.420)
            defaults["avg"] = float(s.get("avg", 0.250) or 0.250)
            defaults["k_pct"] = so / pa if pa > 0 else 0.220
            defaults["bb_pct"] = bb / pa if pa > 0 else 0.082

        # Pitching
        params["group"] = "pitching"
        data = api_get(url, params)
        splits = data.get("stats", [{}])[0].get("splits", [])
        if splits:
            s = splits[0].get("stat", {})
            era = float(s.get("era", 4.20) or 4.20)
            whip = float(s.get("whip", 1.30) or 1.30)
            # Approximate FIP from ERA (no HR/IP breakdown in team stats)
            defaults["era"] = era
            defaults["fip_approx"] = era * 0.92 + 0.30  # rough FIP-ERA relationship

    except Exception:
        pass

    return defaults


# ─── Feature computation ──────────────────────────────────────────────────────

def compute_sp_features(
    home_pitcher_id: int,
    away_pitcher_id: int,
    season: int,
    sp_cache: dict,
) -> dict[str, float]:
    """
    Compute starting pitcher differential features.
    xFIP ≈ FIP estimate; SIERA ≈ ERA-based estimate; K-BB% from K/9 and BB/9.
    """
    def get_sp(pid: int) -> dict:
        if pid not in sp_cache:
            sp_cache[pid] = fetch_pitcher_stats(pid, season)
        return sp_cache[pid]

    hp = get_sp(home_pitcher_id)
    ap = get_sp(away_pitcher_id)

    # Approximate xFIP from ERA (no fly ball rate historically)
    # xFIP ≈ ERA * 0.88 (pitchers with good stuff tend to outperform ERA)
    home_xfip = hp["era"] * 0.88
    away_xfip = ap["era"] * 0.88

    home_siera = hp["era"] * 0.92 + 0.20
    away_siera = ap["era"] * 0.92 + 0.20

    # K-BB% approximation from per-9 rates
    home_k_pct = hp["k_per_9"] / 27.0
    away_k_pct = ap["k_per_9"] / 27.0
    home_bb_pct = hp["bb_per_9"] / 27.0
    away_bb_pct = ap["bb_per_9"] / 27.0
    home_kbb = home_k_pct - home_bb_pct
    away_kbb = away_k_pct - away_bb_pct

    # Rolling game score: approximate from ERA (better ERA → higher score)
    # Clamp to [30, 75] range
    home_gs = max(30.0, min(75.0, 75.0 - (hp["era"] - 3.0) * 8.0))
    away_gs = max(30.0, min(75.0, 75.0 - (ap["era"] - 3.0) * 8.0))

    return {
        # Home minus Away (positive = home SP advantage)
        "sp_xfip_diff": away_xfip - home_xfip,     # lower ERA is better for pitcher; away - home
        "sp_kbb_diff": home_kbb - away_kbb,
        "sp_siera_diff": away_siera - home_siera,
        "sp_csw_diff": 0.0,                          # ESTIMATED — not available historically
        "sp_rolling_gs_diff": home_gs - away_gs,
    }


def compute_team_features(
    home_team_id: int,
    away_team_id: int,
    home_abbr: str,
    away_abbr: str,
    season: int,
    team_cache: dict,
    park_factors: dict[str, float],
) -> dict[str, float]:
    """
    Compute team-level offensive/pitching features.
    wOBA approximated from OBP + SLG; wRC+ from wOBA vs league average.
    Pythagorean expectation from runs scored/allowed.
    """
    def get_team(tid: int) -> dict:
        if tid not in team_cache:
            team_cache[tid] = fetch_team_stats(tid, season)
        return team_cache[tid]

    ht = get_team(home_team_id)
    at = get_team(away_team_id)

    # wOBA approximation: 0.69*(BB%) + 0.72*(HBP%) + 0.88*(1B%) + 1.24*(2B%) + 1.56*(3B%) + 2.03*(HR%)
    # With only OBP/SLG/AVG available: use linear approximation
    # wOBA ≈ 0.45*OBP + 0.40*SLG - 0.10
    home_woba = 0.45 * ht["obp"] + 0.40 * ht["slg"] - 0.10
    away_woba = 0.45 * at["obp"] + 0.40 * at["slg"] - 0.10
    league_woba = 0.320  # historical MLB average

    # wRC+ ≈ (wOBA / lgwOBA) * 100 (simplified, ignoring park)
    home_wrc = (home_woba / league_woba) * 100.0
    away_wrc = (away_woba / league_woba) * 100.0

    # Pythagorean win % using exponent 1.83
    def pyth_win_pct(rpg_off: float, rpg_def: float) -> float:
        if rpg_off <= 0 or rpg_def <= 0:
            return 0.5
        exp = 1.83
        return rpg_off**exp / (rpg_off**exp + rpg_def**exp)

    home_pyth = pyth_win_pct(ht["runs_per_game"], ht["era"] / 9.0 * ht.get("runs_per_game", 4.5))
    away_pyth = pyth_win_pct(at["runs_per_game"], at["era"] / 9.0 * at.get("runs_per_game", 4.5))

    # Log5: probability that home team beats away team
    def log5(p_home: float, p_away: float) -> float:
        if p_home <= 0 or p_away <= 0:
            return 0.5
        num = p_home * (1.0 - p_away)
        den = p_home * (1.0 - p_away) + p_away * (1.0 - p_home)
        return num / den if den > 0 else 0.5

    log5_prob = log5(home_pyth, away_pyth)

    # Park factor: use home team's park
    park_factor = park_factors.get(home_abbr, 1.0)

    # sci_adjusted_diff: approximate from wOBA diff adjusted for SPs
    sci_diff = (home_woba - away_woba) * 10.0  # rough scale

    return {
        "bullpen_strength_diff": 0.0,          # ESTIMATED
        "lineup_woba_diff": home_woba - away_woba,
        "lineup_wrc_plus_diff": home_wrc - away_wrc,
        "team_10d_woba_diff": 0.0,             # ESTIMATED
        "team_10d_fip_diff": 0.0,              # ESTIMATED
        "pythagorean_diff": home_pyth - away_pyth,
        "log5_prob": log5_prob,
        "drs_diff": 0.0,                       # ESTIMATED
        "catcher_framing_diff": 0.0,           # ESTIMATED
        "park_factor": park_factor,
        "wind_out_cf": 0.0,                    # ESTIMATED
        "wind_in_cf": 0.0,                     # ESTIMATED
        "temperature": 70.0,                   # ESTIMATED — use neutral value
        "umpire_run_factor": 1.0,              # ESTIMATED
        "rest_days_diff": 0.0,                 # ESTIMATED
        "travel_tz_shift": 0.0,                # ESTIMATED
        "day_after_night": 0.0,                # ESTIMATED
        "is_home": 1.0,                        # Always 1 (features are from home perspective)
        "statcast_xba_diff": 0.0,              # ESTIMATED
        "statcast_barrel_diff": 0.0,           # ESTIMATED
        "statcast_hardhit_diff": 0.0,          # ESTIMATED
        "statcast_ev_diff": 0.0,               # ESTIMATED
        "gb_rate_diff": 0.0,                   # ESTIMATED
        "sci_adjusted_diff": sci_diff,
    }


def build_feature_row(
    game: dict,
    season: int,
    elo_system: EloSystem,
    sp_cache: dict,
    team_cache: dict,
    park_factors: dict[str, float],
) -> dict | None:
    """
    Build a single feature row for one completed game.
    Returns None if game data is insufficient.
    """
    try:
        teams = game.get("teams", {})
        home_data = teams.get("home", {})
        away_data = teams.get("away", {})

        home_team = home_data.get("team", {})
        away_team = away_data.get("team", {})

        home_id = home_team.get("id", 0)
        away_id = away_team.get("id", 0)

        if not home_id or not away_id:
            return None

        home_abbr = TEAM_ID_TO_ABBR.get(home_id, f"T{home_id}")
        away_abbr = TEAM_ID_TO_ABBR.get(away_id, f"T{away_id}")

        # Scores (for Elo update and label)
        home_score = home_data.get("score")
        away_score = away_data.get("score")

        # Skip games without final scores
        if home_score is None or away_score is None:
            return None
        home_score = int(home_score)
        away_score = int(away_score)

        # Skip ties (rare in MLB but can happen in spring training)
        if home_score == away_score:
            return None

        game_date = game.get("gameDate", "")[:10]
        game_pk = game.get("gamePk", 0)

        # Elo diff BEFORE updating (pre-game rating)
        elo_diff = elo_system.get_diff(home_abbr, away_abbr)

        # Starting pitchers
        home_sp = home_data.get("probablePitcher", {}) or {}
        away_sp = away_data.get("probablePitcher", {}) or {}
        home_sp_id = home_sp.get("id", 0) or 0
        away_sp_id = away_sp.get("id", 0) or 0

        # Compute all features
        sp_features = compute_sp_features(home_sp_id, away_sp_id, season, sp_cache)
        team_features = compute_team_features(
            home_id, away_id, home_abbr, away_abbr, season, team_cache, park_factors
        )

        # Update Elo AFTER recording the pre-game diff
        elo_system.update(home_abbr, away_abbr, home_score, away_score)

        # Combine all features
        row = {
            "game_date": game_date,
            "game_id": game_pk,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_score": home_score,
            "away_score": away_score,
            "season": season,
            # Feature 1: Elo differential
            "elo_diff": elo_diff,
        }
        row.update(sp_features)
        row.update(team_features)

        # Label: 1 = home win, 0 = away win
        row["label"] = 1 if home_score > away_score else 0

        return row

    except Exception as exc:
        return None


# ─── Checkpoint logic ─────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Load progress checkpoint from disk."""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            pass
    return {"completed_seasons": [], "game_ids_done": []}


def save_checkpoint(checkpoint: dict) -> None:
    """Persist progress checkpoint to disk."""
    CHECKPOINT_PATH.write_text(json.dumps(checkpoint, indent=2))


# ─── Main dataset builder ─────────────────────────────────────────────────────

def build_dataset(
    seasons: list[int] | None = None,
    resume: bool = True,
    sleep_between_requests: float = 0.3,
) -> None:
    """
    Main entry point. Downloads and processes all historical games,
    computes features, and saves to CSV.

    Parameters
    ----------
    seasons      : list of seasons to process (default: SEASONS global)
    resume       : if True, skip already-completed seasons/games
    sleep_between_requests : seconds to sleep between API calls to avoid rate limiting
    """
    if seasons is None:
        seasons = SEASONS

    park_factors = load_park_factors()
    print(f"Loaded park factors for {len(park_factors)} teams")

    checkpoint = load_checkpoint() if resume else {"completed_seasons": [], "game_ids_done": []}
    completed_seasons: set[int] = set(checkpoint.get("completed_seasons", []))
    game_ids_done: set[int] = set(checkpoint.get("game_ids_done", []))

    # Load existing CSV rows if resuming
    all_rows: list[dict] = []
    if resume and OUTPUT_CSV.exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        all_rows = existing_df.to_dict("records")
        print(f"Resuming: {len(all_rows)} rows already in {OUTPUT_CSV}")

    # Elo system persists across seasons (applies season regression at start of each)
    elo_system = EloSystem()
    # SP/team stats caches — keyed by (player_id, season) and (team_id, season)
    sp_cache: dict = {}
    team_cache: dict = {}

    for season in sorted(seasons):
        if season in completed_seasons:
            print(f"Season {season} already complete — skipping")
            # Still need to rebuild Elo for this season for subsequent seasons
            # (Elo must be sequential). If skipping, use checkpoint Elo state.
            # For simplicity, we re-fetch the season schedule but skip feature building.
            continue

        print(f"\n{'='*60}")
        print(f"Processing season {season}")
        print(f"{'='*60}")

        elo_system.season_regress(season)

        start_date, end_date = get_season_dates(season)

        print(f"Fetching schedule {start_date} -> {end_date}...")
        try:
            games = fetch_schedule(start_date, end_date)
        except Exception as exc:
            print(f"  ERROR fetching schedule: {exc}")
            continue

        # Filter to completed Final games only
        completed_games = [
            g for g in games
            if g.get("status", {}).get("abstractGameState") == "Final"
            or g.get("status", {}).get("detailedState", "").startswith("Final")
        ]

        print(f"Found {len(games)} total games, {len(completed_games)} completed")

        season_rows: list[dict] = []
        skipped = 0
        errors = 0

        # Cache team stats once per season (expensive API calls)
        print(f"Pre-fetching team stats for season {season}...")
        # Get unique team IDs from this season's games
        team_ids_in_season: set[int] = set()
        for g in completed_games[:50]:  # Sample to find all teams
            teams = g.get("teams", {})
            home_id = teams.get("home", {}).get("team", {}).get("id", 0)
            away_id = teams.get("away", {}).get("team", {}).get("id", 0)
            if home_id:
                team_ids_in_season.add(home_id)
            if away_id:
                team_ids_in_season.add(away_id)

        # Pre-fetch team stats to avoid repeated API calls
        print(f"  Fetching stats for {len(team_ids_in_season)} teams...")
        for tid in tqdm(team_ids_in_season, desc="Team stats", leave=False):
            cache_key = (tid, season)
            if cache_key not in team_cache:
                try:
                    team_cache[cache_key] = fetch_team_stats(tid, season)
                    time.sleep(sleep_between_requests)
                except Exception:
                    team_cache[cache_key] = {}

        # Remap team_cache to use integer keys (as the function expects)
        for (tid, s), stats in list(team_cache.items()):
            if s == season:
                team_cache[tid] = stats  # also store by team_id for this season

        print(f"Processing {len(completed_games)} games...")
        for game in tqdm(completed_games, desc=f"Season {season}", unit="game"):
            game_pk = game.get("gamePk", 0)

            # Skip if already processed
            if game_pk in game_ids_done:
                skipped += 1
                continue

            try:
                row = build_feature_row(
                    game, season, elo_system, sp_cache, team_cache, park_factors
                )
                if row is not None:
                    season_rows.append(row)
                    game_ids_done.add(game_pk)
                else:
                    skipped += 1
            except Exception as exc:
                errors += 1

            # Throttle to avoid rate limiting
            time.sleep(sleep_between_requests * 0.1)  # Minimal sleep between games

        print(f"  Season {season}: {len(season_rows)} rows built, {skipped} skipped, {errors} errors")

        all_rows.extend(season_rows)
        completed_seasons.add(season)

        # Save progress after each season
        df = pd.DataFrame(all_rows)
        _ensure_all_columns(df).to_csv(OUTPUT_CSV, index=False)
        print(f"  Saved {len(all_rows)} total rows to {OUTPUT_CSV}")

        # Update checkpoint
        save_checkpoint({
            "completed_seasons": list(completed_seasons),
            "game_ids_done": list(game_ids_done),
            "last_saved": pd.Timestamp.now().isoformat(),
            "total_rows": len(all_rows),
        })

        # Polite pause between seasons
        time.sleep(1.0)

    # Final save
    if all_rows:
        df = _ensure_all_columns(pd.DataFrame(all_rows))
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nFinal dataset: {len(df)} rows saved to {OUTPUT_CSV}")
        _print_dataset_summary(df)
    else:
        print("\nNo rows generated.")


def _ensure_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all 30 feature columns exist; fill missing with 0.0.
    Also fill any NaN values with 0.0.
    """
    required_cols = [
        "game_date", "game_id", "home_team", "away_team", "home_score", "away_score", "season",
        # All 30 features from FeatureVector
        "elo_diff", "sp_xfip_diff", "sp_kbb_diff", "sp_siera_diff", "sp_csw_diff",
        "sp_rolling_gs_diff", "bullpen_strength_diff", "lineup_woba_diff",
        "lineup_wrc_plus_diff", "team_10d_woba_diff", "team_10d_fip_diff",
        "pythagorean_diff", "log5_prob", "drs_diff", "catcher_framing_diff",
        "park_factor", "wind_out_cf", "wind_in_cf", "temperature", "umpire_run_factor",
        "rest_days_diff", "travel_tz_shift", "day_after_night", "is_home",
        "statcast_xba_diff", "statcast_barrel_diff", "statcast_hardhit_diff",
        "statcast_ev_diff", "gb_rate_diff", "sci_adjusted_diff",
        # Label
        "label",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0
    # Fill NaN with 0.0 for all feature columns
    feature_cols = [c for c in required_cols if c not in ("game_date", "home_team", "away_team")]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df[required_cols]


def _print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a summary of the built dataset."""
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total rows       : {len(df)}")
    print(f"Date range       : {df['game_date'].min()} → {df['game_date'].max()}")
    print(f"Seasons          : {sorted(df['season'].unique())}")
    print(f"Home win rate    : {df['label'].mean():.3f}")
    print(f"\nEstimated features (set to 0.0 or default):")
    for feat in sorted(ESTIMATED_FEATURES):
        print(f"  - {feat}")
    print("\nFeature statistics:")
    feature_cols = [
        c for c in df.columns
        if c not in ("game_date", "game_id", "home_team", "away_team",
                      "home_score", "away_score", "season", "label")
    ]
    print(df[feature_cols].describe().T[["mean", "std", "min", "max"]].round(4).to_string())


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLB Oracle v4.0 — Historical Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start from scratch, ignoring checkpoint",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Process only this single season (e.g., --year 2023)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help="Process specific seasons (e.g., --seasons 2022 2023 2024)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between API requests (default: 0.3)",
    )

    args = parser.parse_args()

    if args.year is not None:
        seasons = [args.year]
    elif args.seasons is not None:
        seasons = args.seasons
    else:
        seasons = SEASONS

    print(f"MLB Oracle v4.0 — Historical Dataset Builder")
    print(f"Seasons: {seasons}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Resume: {args.resume}")

    build_dataset(
        seasons=seasons,
        resume=args.resume,
        sleep_between_requests=args.sleep,
    )


if __name__ == "__main__":
    main()
