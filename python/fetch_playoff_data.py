#!/usr/bin/env python3
"""
MLB Playoff Data Fetcher — Wild Card through World Series.
Uses MLB Stats API (statsapi.mlb.com). Includes starting pitcher ERA/WHIP/K9/BB9.
Output: data/playoff_data.csv

Usage: python python/fetch_playoff_data.py
"""
import sys, json, time, requests
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR  = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "python"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

REG_CSV = DATA_DIR / "historical_features.csv"
OUT_CSV = DATA_DIR / "playoff_data.csv"

# MLB Stats API
MLB_BASE = "https://statsapi.mlb.com/api/v1"
HEADERS  = {"User-Agent": "MLB-Oracle/4.1"}

# Skip 2020: 60-game season played in empty stadiums (no meaningful home advantage)
PLAYOFF_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

K_FACTOR   = 4.0
HOME_ADV   = 35.0
LEAGUE_ELO = 1500.0

# League-average pitcher defaults (used when pitcher data unavailable)
AVG_ERA  = 3.80
AVG_WHIP = 1.20
AVG_K9   = 8.50
AVG_BB9  = 2.80


def mlb_get(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed {url}: {e}")
    return {}


def fetch_playoff_games(year: int) -> list:
    cache = CACHE_DIR / f"mlb_playoffs_{year}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        # Invalidate if game_type missing (old cache format)
        if data and "game_type" not in data[0]:
            cache.unlink()
        else:
            return data

    # MLB postseason game types: F=Wild Card, D=Division Series, L=LCS, W=World Series
    url = (f"{MLB_BASE}/schedule?sportId=1&season={year}"
           f"&gameType=F,D,L,W&hydrate=team,linescore")
    data = mlb_get(url)

    GAME_TYPE_MAX = {"F": 3, "D": 5, "L": 7, "W": 7}

    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            status = g.get("status", {}).get("detailedState", "")
            if "Final" not in status and "Completed" not in status:
                continue
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            h_abbr = home.get("team", {}).get("abbreviation", "")
            a_abbr = away.get("team", {}).get("abbreviation", "")
            h_score = int(home.get("score", 0) or 0)
            a_score = int(away.get("score", 0) or 0)
            game_type = g.get("gameType", "D")
            game_pk = str(g.get("gamePk", ""))
            if not h_abbr or not a_abbr:
                continue
            games.append({
                "game_id":    game_pk,
                "game_date":  g.get("gameDate", "")[:10],
                "home_team":  h_abbr,
                "away_team":  a_abbr,
                "home_score": h_score,
                "away_score": a_score,
                "season":     year,
                "game_type":  game_type,
                "max_games":  GAME_TYPE_MAX.get(game_type, 7),
            })

    cache.write_text(json.dumps(games, indent=2))
    return games


def fetch_game_starters(game_pk: str) -> dict:
    """Return {home_sp_id, away_sp_id} for a completed game via boxscore."""
    cache = CACHE_DIR / f"mlb_box_{game_pk}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    data = mlb_get(f"{MLB_BASE}/game/{game_pk}/boxscore")
    result = {"home_sp_id": None, "away_sp_id": None}

    for side in ("home", "away"):
        pitchers = data.get("teams", {}).get(side, {}).get("pitchers", [])
        if pitchers:
            result[f"{side}_sp_id"] = pitchers[0]  # first pitcher = starter

    cache.write_text(json.dumps(result, indent=2))
    time.sleep(0.15)
    return result


def fetch_pitcher_season_stats(person_id: int, year: int) -> dict:
    """Return ERA, WHIP, K/9, BB/9 for a pitcher's regular season."""
    cache = CACHE_DIR / f"mlb_pitcher_{person_id}_{year}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    url = (f"{MLB_BASE}/people/{person_id}/stats"
           f"?stats=season&group=pitching&season={year}&gameType=R")
    data = mlb_get(url)

    result = {"era": AVG_ERA, "whip": AVG_WHIP, "k9": AVG_K9, "bb9": AVG_BB9}
    splits = data.get("stats", [{}])[0].get("splits", []) if data.get("stats") else []
    if splits:
        s = splits[0].get("stat", {})
        ip_str = str(s.get("inningsPitched", "0") or "0")
        # IP format "72.1" means 72⅓ innings — convert to decimal
        ip_parts = ip_str.split(".")
        ip = float(ip_parts[0]) + int(ip_parts[1]) / 3.0 if len(ip_parts) == 2 else float(ip_str)

        if ip > 0:
            era_str  = str(s.get("era",  AVG_ERA))
            whip_str = str(s.get("whip", AVG_WHIP))
            k = int(s.get("strikeOuts", 0) or 0)
            bb = int(s.get("baseOnBalls", 0) or 0)
            try:
                result["era"]  = float(era_str)
                result["whip"] = float(whip_str)
            except (ValueError, TypeError):
                pass
            result["k9"]  = k  * 9 / ip
            result["bb9"] = bb * 9 / ip

    cache.write_text(json.dumps(result, indent=2))
    time.sleep(0.1)
    return result


def build_elo_and_stats(reg_df: pd.DataFrame, year: int) -> tuple:
    elo   = defaultdict(lambda: LEAGUE_ELO)
    stats = defaultdict(lambda: {"wins": 0, "games": 0, "runs_for": [], "runs_agst": []})
    s = reg_df[reg_df["season"] == year].sort_values("game_date")
    for _, row in s.iterrows():
        h, a = row["home_team"], row["away_team"]
        he = elo[h] + HOME_ADV; ae = elo[a]
        exp = 1 / (1 + 10 ** ((ae - he) / 400))
        act = int(row["label"])
        elo[h] += K_FACTOR * (act - exp)
        elo[a] += K_FACTOR * ((1 - act) - (1 - exp))
        stats[h]["wins"]      += act;   stats[a]["wins"]      += (1 - act)
        stats[h]["games"]     += 1;     stats[a]["games"]     += 1
        h_r = row.get("home_score", 4); a_r = row.get("away_score", 4)
        stats[h]["runs_for"].append(h_r);   stats[h]["runs_agst"].append(a_r)
        stats[a]["runs_for"].append(a_r);   stats[a]["runs_agst"].append(h_r)

    team_stats = {}
    for team, st in stats.items():
        g = st["games"]
        team_stats[team] = {
            "win_pct": st["wins"] / g if g else 0.5,
            "rpg":     np.mean(st["runs_for"])  if st["runs_for"]  else 4.5,
            "rapg":    np.mean(st["runs_agst"]) if st["runs_agst"] else 4.5,
        }
    return dict(elo), team_stats


def add_series_context(games: list) -> list:
    """
    Add series context to MLB playoff games.
    MLB format (since 2022 expansion to 12 teams):
      Wild Card (F): best-of-3  → first to 2 wins
      Division Series (D): best-of-5  → first to 3 wins
      League Championship (L): best-of-7  → first to 4 wins
      World Series (W): best-of-7  → first to 4 wins
    """
    WINS_TO_ADVANCE = {"F": 2, "D": 3, "L": 4, "W": 4}

    sorted_games = sorted(games, key=lambda g: g.get("game_date", ""))
    series_wins: dict = defaultdict(lambda: defaultdict(int))

    result = []
    for g in sorted_games:
        h, a = g["home_team"], g["away_team"]
        gtype = g.get("game_type", "D")
        key = (frozenset([h, a]), gtype)
        wins_needed = WINS_TO_ADVANCE.get(gtype, 4)

        h_wins = series_wins[key][h]
        a_wins = series_wins[key][a]
        game_num = h_wins + a_wins + 1

        series_deficit = h_wins - a_wins
        is_elimination = int((h_wins == wins_needed - 1) or (a_wins == wins_needed - 1))

        label = 1 if g["home_score"] > g["away_score"] else 0
        if label == 1:
            series_wins[key][h] += 1
        else:
            series_wins[key][a] += 1

        result.append({
            **g,
            "series_game_num":     game_num,
            "series_deficit":      series_deficit,
            "is_elimination_game": is_elimination,
        })
    return result


def main():
    print("MLB Playoff Data Fetcher")
    print("=" * 40)

    if not REG_CSV.exists():
        print(f"No regular season CSV at {REG_CSV}")
        sys.exit(1)

    reg_df = pd.read_csv(REG_CSV)
    reg_df["game_date"] = pd.to_datetime(reg_df["game_date"], errors="coerce")

    all_rows = []

    for year in PLAYOFF_YEARS:
        print(f"\nYear {year}")
        elo, stats = build_elo_and_stats(reg_df, year)
        games = fetch_playoff_games(year)
        games = add_series_context(games)
        by_type = {}
        for g in games:
            by_type[g.get("game_type","?")] = by_type.get(g.get("game_type","?"), 0) + 1
        print(f"  Fetched {len(games)} playoff games: {by_type}")

        for g in games:
            h, a = g["home_team"], g["away_team"]
            h_elo = elo.get(h, LEAGUE_ELO); a_elo = elo.get(a, LEAGUE_ELO)
            hs  = stats.get(h, {"win_pct": 0.5, "rpg": 4.5, "rapg": 4.5})
            as_ = stats.get(a, {"win_pct": 0.5, "rpg": 4.5, "rapg": 4.5})
            label = 1 if g["home_score"] > g["away_score"] else 0

            # Starting pitcher stats
            h_era = AVG_ERA; h_whip = AVG_WHIP; h_k9 = AVG_K9; h_bb9 = AVG_BB9
            a_era = AVG_ERA; a_whip = AVG_WHIP; a_k9 = AVG_K9; a_bb9 = AVG_BB9

            game_pk = g.get("game_id", "")
            if game_pk:
                starters = fetch_game_starters(str(game_pk))
                if starters.get("home_sp_id"):
                    ps = fetch_pitcher_season_stats(starters["home_sp_id"], year)
                    h_era, h_whip, h_k9, h_bb9 = ps["era"], ps["whip"], ps["k9"], ps["bb9"]
                if starters.get("away_sp_id"):
                    ps = fetch_pitcher_season_stats(starters["away_sp_id"], year)
                    a_era, a_whip, a_k9, a_bb9 = ps["era"], ps["whip"], ps["k9"], ps["bb9"]

            row = {
                "season":       year,
                "game_id":      g["game_id"],
                "game_date":    g["game_date"],
                "home_team":    h,
                "away_team":    a,
                "home_score":   g["home_score"],
                "away_score":   g["away_score"],
                "label":        label,
                "is_playoff":   1,
                "game_type":    g.get("game_type", "D"),
                "max_games":    g.get("max_games", 7),
                "elo_diff":         h_elo - a_elo,
                "win_pct_diff":     hs["win_pct"] - as_["win_pct"],
                "rpg_diff":         hs["rpg"]  - as_["rpg"],
                "rapg_diff":        hs["rapg"] - as_["rapg"],
                "pythagorean_diff": (hs["win_pct"] - as_["win_pct"]) * 0.3,
                "log5_prob":        hs["win_pct"] / (hs["win_pct"] + as_["win_pct"]) if (hs["win_pct"] + as_["win_pct"]) > 0 else 0.5,
                # Pitcher features (single biggest signal in MLB)
                "sp_era_diff":   h_era  - a_era,    # positive = home SP has higher ERA (disadvantage)
                "sp_whip_diff":  h_whip - a_whip,
                "sp_k9_diff":    h_k9   - a_k9,     # positive = home SP strikes out more (advantage)
                "sp_bb9_diff":   h_bb9  - a_bb9,
                # Series context (round-aware: WC=best-of-3, DS=best-of-5, LCS/WS=best-of-7)
                "series_game_num":     g["series_game_num"],
                "series_deficit":      g["series_deficit"],
                "is_elimination_game": g["is_elimination_game"],
            }
            all_rows.append(row)

            # Update Elo through series
            exp = 1 / (1 + 10 ** ((a_elo - (h_elo + HOME_ADV)) / 400))
            elo[h] = h_elo + K_FACTOR * (label - exp)
            elo[a] = a_elo + K_FACTOR * ((1 - label) - (1 - exp))

    if not all_rows:
        print("\nNo playoff data fetched.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} playoff games to {OUT_CSV}")
    print(f"Seasons: {df['season'].unique().tolist()}")
    print(f"Home win rate: {df['label'].mean():.3f}")
    # Report pitcher data coverage
    has_sp = (df["sp_era_diff"] != 0).sum()
    print(f"Games with pitcher data: {has_sp}/{len(df)} ({has_sp/len(df)*100:.0f}%)")


if __name__ == "__main__":
    main()
