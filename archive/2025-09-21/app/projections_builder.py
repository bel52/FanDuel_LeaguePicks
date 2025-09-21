import os, sys, time, math, re, logging, argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import requests
import pandas as pd

LOG = logging.getLogger("projections_builder")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FANTASY_NERDS_KEY = os.getenv("FANTASY_NERDS_API_KEY", "").strip()
ODDS_API_KEY      = os.getenv("ODDS_API_KEY", "").strip()
USE_WEATHER       = os.getenv("USE_WEATHER","0") == "1"

INPUT_DIR  = os.getenv("INPUT_DIR", "data/input")
os.makedirs(INPUT_DIR, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------
def _safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def _norm_name(s: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', (s or "").lower()).strip()

def _write_pos_csv(pos: str, rows: List[Dict[str, object]]):
    out_path = os.path.join(INPUT_DIR, f"{pos.lower()}.csv")
    df = pd.DataFrame(rows)
    # Ensure required columns
    required = ["PLAYER NAME","TEAM","OPP","PROJ PTS","SALARY"]
    for col in required:
        if col not in df.columns:
            df[col] = "" if col in ("PLAYER NAME","TEAM","OPP") else 0
    # Keep stable column order if possible
    cols = [c for c in ["PLAYER NAME","TEAM","OPP","PROJ PTS","SALARY","PROJ ROSTER %"] if c in df.columns]
    df = df[cols]
    df.to_csv(out_path, index=False)
    LOG.info(f"Wrote {pos} -> {out_path} ({len(df)} rows)")

# ----------------------------
# Data sources
# ----------------------------
def sleeper_active_name_set() -> set:
    """Return a set of full-name lowercase strings for active players (with a team)."""
    try:
        r = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=20)
        r.raise_for_status()
        data = r.json()
        actives = set()
        for p in data.values():
            status = str(p.get("status","") or "").lower()
            team = p.get("team")
            if status == "active" and team:
                full = p.get("full_name") or f"{p.get('first_name','')} {p.get('last_name','')}".strip()
                if full:
                    actives.add(_norm_name(full))
        if not actives:
            LOG.warning("Sleeper returned no active names; skipping active filter.")
        return actives
    except Exception as e:
        LOG.warning(f"Sleeper active fetch failed: {e}")
        return set()

def espn_scoreboard():
    """Basic ESPN scoreboard for opponent mapping; no auth required."""
    try:
        r = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        LOG.warning(f"ESPN scoreboard fetch failed: {e}")
        return {}

def odds_api_totals() -> Dict[Tuple[str,str], Dict[str,float]]:
    """Fetch game totals/spreads from The Odds API; return keyed by (AWAY, HOME)."""
    if not ODDS_API_KEY:
        return {}
    try:
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {"regions":"us","markets":"totals,spreads","oddsFormat":"american","apiKey":ODDS_API_KEY}
        r = requests.get(url, params=params, timeout=25); r.raise_for_status()
        games = r.json()
        out = {}
        for g in games:
            away = (g.get("away_team") or "").upper().split()[-1][:3]
            home = (g.get("home_team") or "").upper().split()[-1][:3]
            total = None; spread_home = None
            for bk in g.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk.get("key") == "totals":
                        for o in mk.get("outcomes", []):
                            if o.get("name","").lower() == "over":
                                total = _safe_float(o.get("point"))
                    if mk.get("key") == "spreads":
                        for o in mk.get("outcomes", []):
                            # home team spread is negative if favored
                            if o.get("name","").upper().startswith(home):
                                spread_home = _safe_float(o.get("point"))
            if away and home and total:
                out[(away[:3], home[:3])] = {"total": float(total), "home_spread": float(spread_home) if spread_home is not None else 0.0}
        return out
    except Exception as e:
        LOG.warning(f"Odds API fetch failed: {e}")
        return {}

def open_meteo_adjustment_factor():
    """Very lightweight weather impact factor. 1.0 = neutral, <1 dampening passing.
       For simplicity, return a dict keyed by team abbr -> factor."""
    # To keep things dependency-free and robust, we skip stadium geocoding here.
    # We apply a conservative neutral factor.
    return {}

# ----------------------------
# FantasyNerds – free tier projections + FanDuel salaries
# ----------------------------
def fantasynerds_week(week: int) -> Dict[str, pd.DataFrame]:
    """
    Fetch per-position projections and FanDuel salaries from FantasyNerds free tier.
    NOTE: Requires FANTASY_NERDS_API_KEY in env.
    Returns dict of DataFrames keyed by POS (QB/RB/WR/TE/DST) with at least:
      PLAYER NAME, TEAM, OPP, PROJ PTS, SALARY
    """
    if not FANTASY_NERDS_KEY:
        raise RuntimeError("FANTASY_NERDS_API_KEY not set")

    base = "https://api.fantasynerds.com/v1/nfl"
    headers = {}
    dfs = {}

    # Endpoints (FN docs show separate endpoints; we normalize)
    endpoints = {
        "QB": f"{base}/weekly-projections?position=QB&week={week}&apikey={FANTASY_NERDS_KEY}",
        "RB": f"{base}/weekly-projections?position=RB&week={week}&apikey={FANTASY_NERDS_KEY}",
        "WR": f"{base}/weekly-projections?position=WR&week={week}&apikey={FANTASY_NERDS_KEY}",
        "TE": f"{base}/weekly-projections?position=TE&week={week}&apikey={FANTASY_NERDS_KEY}",
        "DST": f"{base}/weekly-projections?position=DST&week={week}&apikey={FANTASY_NERDS_KEY}",
    }
    # Salary endpoint
    salaries_url = f"{base}/dfs-salaries?site=fanduel&week={week}&apikey={FANTASY_NERDS_KEY}"

    # Fetch salaries
    try:
        sr = requests.get(salaries_url, headers=headers, timeout=25); sr.raise_for_status()
        salary_rows = sr.json() if isinstance(sr.json(), list) else sr.json().get("players", [])
        # Map by normalized player name + team for joining
        sal_map = {}
        for row in salary_rows:
            name = row.get("player") or row.get("name")
            team = (row.get("team") or "").upper()
            sal  = int(_safe_float(row.get("salary")))
            if name and team and sal:
                sal_map[(_norm_name(name), team)] = sal
    except Exception as e:
        LOG.warning(f"FantasyNerds salaries fetch failed: {e}")
        sal_map = {}

    # Fetch projections per position
    for pos, url in endpoints.items():
        try:
            r = requests.get(url, headers=headers, timeout=25); r.raise_for_status()
            rows = r.json() if isinstance(r.json(), list) else r.json().get("players", [])
            out = []
            for row in rows:
                name = row.get("player") or row.get("name")
                team = (row.get("team") or "").upper()
                opp  = (row.get("opponent") or row.get("opp") or "").upper().replace("@","").replace(" ","")
                pts  = _safe_float(row.get("projPoints") or row.get("projected_points") or row.get("points"))
                # join salary
                sal = sal_map.get((_norm_name(name), team), 0)
                out.append({
                    "PLAYER NAME": name,
                    "TEAM": team,
                    "OPP": opp,
                    "PROJ PTS": pts,
                    "SALARY": sal
                })
            df = pd.DataFrame(out)
            dfs[pos] = df
        except Exception as e:
            LOG.warning(f"FantasyNerds projections fetch failed for {pos}: {e}")
            dfs[pos] = pd.DataFrame(columns=["PLAYER NAME","TEAM","OPP","PROJ PTS","SALARY"])

    return dfs

# ----------------------------
# Simple implied total adjustment
# ----------------------------
def implied_adjust(dfs: Dict[str,pd.DataFrame], odds: Dict[Tuple[str,str], Dict[str,float]]):
    """Boost or dampen projections based on game totals/spreads."""
    if not odds:
        return
    for pos, df in dfs.items():
        if df.empty: 
            continue
        adj_pts = []
        for _, row in df.iterrows():
            team = (row.get("TEAM") or "").upper()
            opp  = (row.get("OPP") or "").upper()
            base = _safe_float(row.get("PROJ PTS"), 0.0)
            if not team or not opp:
                adj_pts.append(base); continue
            # Find matchup (either direction)
            mtch = odds.get((team[:3], opp[:3])) or odds.get((opp[:3], team[:3]))
            if not mtch:
                adj_pts.append(base); continue
            total = mtch.get("total", 44.5)
            # Scale around 44.5 baseline
            factor = 1.0 + (total - 44.5) * 0.01  # each point over baseline adds ~1%
            adj_pts.append(max(0.0, base * factor))
        df["PROJ PTS"] = adj_pts

# ----------------------------
# Main entry
# ----------------------------
def autodetect_week() -> int:
    # Lightweight: default to 1..18; here, just use 1 if unknown.
    # For production, wire a proper current-week resolver.
    return int(os.getenv("NFL_WEEK", "1"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default="auto", help="NFL week number or 'auto'")
    args = ap.parse_args()
    week = autodetect_week() if args.week == "auto" else int(args.week)

    LOG.info(f"Building projections/salaries for week {week} (no FantasyPros).")

    active_set = sleeper_active_name_set()
    odds = odds_api_totals()
    weather_factor = open_meteo_adjustment_factor() if USE_WEATHER else {}

    # Preferred path: FantasyNerds free tier (projections + FanDuel salaries)
    if not FANTASY_NERDS_KEY:
        LOG.error("FANTASY_NERDS_API_KEY is missing. Set it in .env to enable full automation.")
        sys.exit(2)

    dfs = fantasynerds_week(week)
    implied_adjust(dfs, odds)

    # Active-roster filter
    if active_set:
        for pos, df in dfs.items():
            if df.empty: 
                continue
            df["__namekey__"] = df["PLAYER NAME"].map(_norm_name)
            before = len(df)
            df = df[df["__namekey__"].isin(active_set)].copy()
            df.drop(columns=["__namekey__"], inplace=True)
            LOG.info(f"{pos}: filtered inactive {before - len(df)} -> {len(df)} remain.")
            dfs[pos] = df

    # Minimal weather dampening example skipped here (kept neutral)
    # Ensure some basic sanity filters
    for pos, df in dfs.items():
        if df.empty: continue
        df["SALARY"] = df["SALARY"].fillna(0).astype(int)
        df["PROJ PTS"] = df["PROJ PTS"].fillna(0).astype(float)
        dfs[pos] = df[(df["SALARY"] >= 3000) & (df["SALARY"] <= 15000) & (df["PROJ PTS"] >= 0)]

    # Write CSVs for the optimizer
    for pos in ["QB","RB","WR","TE","DST"]:
        rows = dfs.get(pos, pd.DataFrame()).to_dict(orient="records")
        _write_pos_csv(pos, rows)

    LOG.info("Projection build complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
