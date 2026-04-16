#!/usr/bin/env python3
"""
MLB Oracle v4.1 — Live Predictions
Fetches today's MLB games from ESPN, rebuilds Elo from history,
and runs the trained model. Auto-switches to the playoff model
from Wild Card (early October) through World Series (late October/early November).

Usage:
  python python/predict.py              # today's games
  python python/predict.py --date 20261010
"""
import argparse, json, math, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = DATA_DIR / "model"
HIST_CSV  = DATA_DIR / "historical_features.csv"

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
HEADERS   = {"User-Agent": "MLB-Oracle/4.1"}

INITIAL_ELO = 1500.0
K_FACTOR    = 4.0
HOME_ADV    = 35.0


# ── Season detection ───────────────────────────────────────────────────────────

def is_playoff_season(date_str: str) -> bool:
    """MLB playoffs: early October through early November."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return False
    return (d.month == 10) or (d.month == 11 and d.day <= 10)


# ── ESPN helpers ───────────────────────────────────────────────────────────────

def fetch_json(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return {}


AVG_ERA  = 3.80
AVG_WHIP = 1.20
AVG_K9   = 8.50
AVG_BB9  = 2.80


def fetch_games(date_str: str) -> list:
    # MLB playoffs: game type determines series length
    # F=Wild Card (best-of-3, 2 wins), D=DS (best-of-5, 3 wins), L=LCS (best-of-7, 4 wins), W=WS (best-of-7, 4 wins)
    WINS_NEEDED = {"F": 2, "D": 3, "L": 4, "W": 4}
    playoff = is_playoff_season(date_str)
    season_param = "&seasontype=3" if playoff else ""
    data = fetch_json(f"{ESPN_BASE}/scoreboard?dates={date_str}&limit=20{season_param}")
    games = []
    for ev in data.get("events", []):
        status = ev.get("status", {}).get("type", {}).get("name", "")
        comp   = (ev.get("competitions") or [{}])[0]
        cs     = comp.get("competitors", [])
        home   = next((c for c in cs if c.get("homeAway") == "home"), None)
        away   = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        h_abbr = home.get("team", {}).get("abbreviation", "").upper()
        a_abbr = away.get("team", {}).get("abbreviation", "").upper()

        # Extract series wins from ESPN playoff competition object
        h_wins, a_wins = 0, 0
        game_type = comp.get("type", {}).get("abbreviation", "D") or "D"
        series = comp.get("series", {})
        if series:
            for sc in series.get("competitors", []):
                wins = int(sc.get("wins", 0) or 0)
                if sc.get("homeAway") == "home":
                    h_wins = wins
                elif sc.get("homeAway") == "away":
                    a_wins = wins

        wins_needed = WINS_NEEDED.get(game_type, 4)
        game_num   = h_wins + a_wins + 1
        deficit    = h_wins - a_wins
        elim_game  = int(h_wins == wins_needed - 1 or a_wins == wins_needed - 1)

        # Extract probable pitcher ERA from ESPN (available for scheduled games)
        h_era = AVG_ERA; h_whip = AVG_WHIP; h_k9 = AVG_K9; h_bb9 = AVG_BB9
        a_era = AVG_ERA; a_whip = AVG_WHIP; a_k9 = AVG_K9; a_bb9 = AVG_BB9
        for prob in comp.get("probables", []):
            side = prob.get("homeAway", "")
            sp_stats = {s["name"]: s.get("displayValue", "") for s in prob.get("statistics", [])}
            def _f(key, default):
                try: return float(sp_stats.get(key, default))
                except (ValueError, TypeError): return float(default)
            if side == "home":
                h_era  = _f("ERA", AVG_ERA); h_whip = _f("WHIP", AVG_WHIP)
                h_k9   = _f("K9",  AVG_K9);  h_bb9  = _f("BB9",  AVG_BB9)
            elif side == "away":
                a_era  = _f("ERA", AVG_ERA); a_whip = _f("WHIP", AVG_WHIP)
                a_k9   = _f("K9",  AVG_K9);  a_bb9  = _f("BB9",  AVG_BB9)

        games.append({
            "event_name":          ev.get("name", ""),
            "status":              status,
            "home_abbr":           h_abbr,
            "home_id":             home.get("team", {}).get("id", ""),
            "home_name":           home.get("team", {}).get("displayName", ""),
            "away_abbr":           a_abbr,
            "away_id":             away.get("team", {}).get("id", ""),
            "away_name":           away.get("team", {}).get("displayName", ""),
            "series_game_num":     game_num,
            "series_deficit":      deficit,
            "is_elimination_game": elim_game,
            "sp_era_diff":         h_era  - a_era,
            "sp_whip_diff":        h_whip - a_whip,
            "sp_k9_diff":          h_k9   - a_k9,
            "sp_bb9_diff":         h_bb9  - a_bb9,
        })
    return games


def fetch_team_record(team_id: str) -> dict:
    data  = fetch_json(f"{ESPN_BASE}/teams/{team_id}?enable=record,stats")
    items = data.get("team", {}).get("record", {}).get("items", [])
    result = {"win_pct": 0.5, "games_played": 0}
    for item in items:
        if item.get("type") == "total":
            stats = {s["name"]: s["value"] for s in item.get("stats", [])}
            gp    = stats.get("gamesPlayed", 0) or 0
            wins  = stats.get("wins", 0) or 0
            result["win_pct"] = wins / gp if gp > 0 else 0.5
            result["games_played"] = int(gp)
    return result


# ── Elo reconstruction ─────────────────────────────────────────────────────────

def build_elo_from_history() -> dict:
    if not HIST_CSV.exists():
        return {}
    df = pd.read_csv(HIST_CSV, usecols=["season", "game_date", "home_team",
                                         "away_team", "label"])
    df = df.sort_values("game_date")
    elo = defaultdict(lambda: INITIAL_ELO)
    last_season = None

    for _, row in df.iterrows():
        season = row["season"]
        if last_season and season != last_season:
            for t in list(elo.keys()):
                elo[t] = 0.70 * elo[t] + 0.30 * INITIAL_ELO
        last_season = season
        h, a   = str(row["home_team"]).upper(), str(row["away_team"]).upper()
        hw     = int(row["label"])
        rh, ra = elo[h], elo[a]
        exp_h  = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADV)) / 400.0))
        delta  = K_FACTOR * (hw - exp_h)
        elo[h] = rh + delta
        elo[a] = ra - delta

    return dict(elo)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(date_str: str) -> dict | None:
    try:
        if is_playoff_season(date_str):
            po = MODEL_DIR / "playoff_coefficients.json"
            ps = MODEL_DIR / "playoff_scaler.json"
            pm = MODEL_DIR / "playoff_metadata.json"
            if po.exists() and ps.exists() and pm.exists():
                po_data = json.loads(po.read_text())
                # Convert to dict-key format (MLB uses dict format)
                coeff = dict(zip(po_data["feature_names"], po_data["coefficients"]))
                coeff["_intercept"] = po_data["intercept"]
                meta = json.loads(pm.read_text())
                meta["feature_names"] = po_data["feature_names"]
                print("  [PLAYOFFS] Using MLB playoff model")
                return {
                    "coeff":  coeff,
                    "scaler": json.loads(ps.read_text()),
                    "calib":  {"x_thresholds": [], "y_thresholds": []},
                    "meta":   meta,
                }
        coeff  = json.loads((MODEL_DIR / "coefficients.json").read_text())
        scaler = json.loads((MODEL_DIR / "scaler.json").read_text())
        calib  = json.loads((MODEL_DIR / "calibration.json").read_text())
        meta   = json.loads((MODEL_DIR / "model_metadata.json").read_text())
        return {"coeff": coeff, "scaler": scaler, "calib": calib, "meta": meta}
    except Exception as e:
        print(f"  Model load failed: {e}")
        return None


def predict_proba(model: dict, fv: dict) -> float:
    features  = model["meta"].get("feature_names", [])
    coeff_map = model["coeff"]
    intercept = coeff_map.get("_intercept", coeff_map.get("intercept", 0.0))
    mean      = model["scaler"]["mean"]
    scale     = model["scaler"].get("scale", model["scaler"].get("var", [1.0]*len(mean)))

    x     = np.array([(fv.get(f, 0.0) - mean[i]) / (scale[i] if scale[i] != 0 else 1.0)
                      for i, f in enumerate(features)])
    coeff = np.array([coeff_map.get(f, 0.0) for f in features])
    logit = float(np.dot(coeff, x)) + intercept
    raw   = 1.0 / (1.0 + math.exp(-logit))

    calib = model["calib"]
    bins  = calib.get("x_thresholds", calib.get("bins", []))
    cals  = calib.get("y_thresholds", calib.get("calibrated", []))
    if not bins or not cals:
        p = raw
    elif raw <= bins[0]:
        p = cals[0]
    elif raw >= bins[-1]:
        p = cals[-1]
    else:
        p = raw
        for i in range(len(bins) - 1):
            if bins[i] <= raw <= bins[i + 1]:
                t = (raw - bins[i]) / (bins[i + 1] - bins[i])
                p = cals[i] + t * (cals[i + 1] - cals[i])
                break

    # Cap at 85%: no MLB regular-season game is more predictable than this
    # (avoids out-of-distribution extremes when advanced features are unavailable)
    return max(0.15, min(0.85, p))


# ── Feature builder ────────────────────────────────────────────────────────────

def _shrink_wp(win_pct: float, games_played: int, prior_games: int = 30) -> float:
    """Bayesian shrinkage toward .500 for small sample sizes.
    With prior_games=30: a 10-0 team (.500 prior) → ~0.625 not 1.000.
    A full-season team (162 games) is almost unaffected.
    """
    return (win_pct * games_played + 0.5 * prior_games) / (games_played + prior_games)


def build_features(elo_ratings: dict, h_abbr: str, a_abbr: str,
                   h_rec: dict, a_rec: dict,
                   series_game_num: int = 1, series_deficit: int = 0,
                   is_elimination_game: int = 0,
                   sp_era_diff: float = 0.0, sp_whip_diff: float = 0.0,
                   sp_k9_diff: float = 0.0, sp_bb9_diff: float = 0.0) -> dict:
    rh   = elo_ratings.get(h_abbr, INITIAL_ELO)
    ra   = elo_ratings.get(a_abbr, INITIAL_ELO)
    # Shrink win% toward .500 based on games played — prevents extreme predictions
    # early in the season when sample sizes are small
    h_wp = _shrink_wp(h_rec["win_pct"], h_rec.get("games_played", 30))
    a_wp = _shrink_wp(a_rec["win_pct"], a_rec.get("games_played", 30))
    log5 = h_wp / (h_wp + a_wp) if (h_wp + a_wp) > 0 else 0.5

    return {
        "elo_diff":              rh - ra,
        "pythagorean_diff":      h_wp - a_wp,
        "log5_prob":             log5,
        # Pitcher / advanced stats: zeroed — not available from ESPN
        "sp_xfip_diff":          0.0,
        "sp_kbb_diff":           0.0,
        "sp_siera_diff":         0.0,
        "sp_csw_diff":           0.0,
        "sp_rolling_gs_diff":    0.0,
        "bullpen_strength_diff": 0.0,
        "lineup_woba_diff":      0.0,
        "lineup_wrc_plus_diff":  0.0,
        "team_10d_woba_diff":    0.0,
        "team_10d_fip_diff":     0.0,
        "drs_diff":              0.0,
        "catcher_framing_diff":  0.0,
        "park_factor":           1.0,
        "wind_out_cf":           0.0,
        "wind_in_cf":            0.0,
        "temperature":           70.0,
        "umpire_run_factor":     1.0,
        "rest_days_diff":        0.0,
        "travel_tz_shift":       0.0,
        "day_after_night":       0.0,
        "is_home":               1.0,
        "statcast_xba_diff":     0.0,
        "statcast_barrel_diff":  0.0,
        "statcast_hardhit_diff": 0.0,
        "statcast_ev_diff":      0.0,
        "gb_rate_diff":          0.0,
        "sci_adjusted_diff":     h_wp - a_wp,
        "vegas_home_prob":       0.0,
        "momentum_diff":         0.0,
        "run_diff_diff":         0.0,
        "platoon_advantage":     0.0,
        # Playoff model features
        "win_pct_diff":          h_wp - a_wp,
        "rpg_diff":              0.0,
        "rapg_diff":             0.0,
        # Pitcher (most important signal in MLB)
        "sp_era_diff":           float(sp_era_diff),
        "sp_whip_diff":          float(sp_whip_diff),
        "sp_k9_diff":            float(sp_k9_diff),
        "sp_bb9_diff":           float(sp_bb9_diff),
        # Series context
        "series_game_num":       float(series_game_num),
        "series_deficit":        float(series_deficit),
        "is_elimination_game":   float(is_elimination_game),
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def pad(s: str, w: int) -> str:
    return s[:w].ljust(w)


def print_predictions(results: list, date_str: str) -> None:
    width = 85
    print("\n" + "=" * width)
    print(f"  MLB ORACLE v4.1  |  {date_str}  |  {len(results)} games")
    print("=" * width)
    print("  " + pad("MATCHUP", 30) + pad("HOME WIN%", 11) + pad("AWAY WIN%", 11) + "PICK")
    print("-" * width)
    for r in sorted(results, key=lambda x: -max(x["home_prob"], x["away_prob"])):
        matchup  = f"{r['home_abbr']} vs {r['away_abbr']}"
        home_pct = f"{r['home_prob']*100:.1f}%"
        away_pct = f"{r['away_prob']*100:.1f}%"
        pick     = r["home_abbr"] if r["home_prob"] >= r["away_prob"] else r["away_abbr"]
        star     = " *" if max(r["home_prob"], r["away_prob"]) >= 0.60 else ""
        print(f"  {pad(matchup, 30)}{pad(home_pct, 11)}{pad(away_pct, 11)}{pick}{star}")
    print("-" * width)
    print("* = high confidence (>= 60%)\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()
    date_str = args.date

    print(f"=== MLB Oracle v4.1 — Predictions for {date_str} ===\n")

    model = load_model(date_str)
    if not model:
        print("ERROR: No model found. Run: python python/train_model.py")
        return

    print("Loading Elo ratings from history...")
    elo = build_elo_from_history()
    print(f"  {len(elo)} teams rated")

    print(f"\nFetching games for {date_str}...")
    games = fetch_games(date_str)

    if not games:
        for offset in list(range(1, 4)) + list(range(-1, -4, -1)):
            d = (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=offset)).strftime("%Y%m%d")
            games = fetch_games(d)
            if games:
                label = "next" if offset > 0 else "most recent"
                print(f"  No games today — showing {label} games ({d})")
                date_str = d
                break

    if not games:
        print("No games found.")
        return

    scheduled = [g for g in games if "SCHEDULED" in g["status"]] or games
    print(f"  Found {len(scheduled)} game(s)\n")

    results = []
    for game in scheduled:
        h_rec  = fetch_team_record(game["home_id"])
        a_rec  = fetch_team_record(game["away_id"])
        time.sleep(0.1)

        fv     = build_features(elo, game["home_abbr"], game["away_abbr"], h_rec, a_rec,
                               series_game_num     = game["series_game_num"],
                               series_deficit      = game["series_deficit"],
                               is_elimination_game = game["is_elimination_game"],
                               sp_era_diff         = game["sp_era_diff"],
                               sp_whip_diff        = game["sp_whip_diff"],
                               sp_k9_diff          = game["sp_k9_diff"],
                               sp_bb9_diff         = game["sp_bb9_diff"])
        home_p = predict_proba(model, fv)

        results.append({
            "home_abbr": game["home_abbr"],
            "away_abbr": game["away_abbr"],
            "home_prob": home_p,
            "away_prob": 1.0 - home_p,
        })

    print_predictions(results, date_str)


if __name__ == "__main__":
    main()
