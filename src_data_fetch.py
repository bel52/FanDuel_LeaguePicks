"""
Data ingestion and external API fetches for DFS optimizer.

This module contains helper functions to load player projections from
FantasyPros CSVs and to fetch supplemental data such as Vegas odds
and weather conditions. API responses are cached to minimize network
usage. If environment variables (e.g. ``ODDS_API_KEY``, ``WEATHER_API_KEY``)
are not set or ``USE_ODDS``/``USE_WEATHER`` flags are disabled, the
corresponding fetch functions return empty data.
"""

import os
import csv
import re
import requests
from typing import List, Dict, Any, Tuple

from src.util import parse_int, parse_float, logger

# Standard library import for dates
from datetime import date
from src.cache_manager import cache

# Keys and flags from environment
ODDS_API_KEY = os.getenv('ODDS_API_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
USE_ODDS = os.getenv('USE_ODDS', '0') == '1'
USE_WEATHER = os.getenv('USE_WEATHER', '0') == '1'

# Injury and scoreboard configuration
INJURY_API_KEY = os.getenv('INJURY_API_KEY')
SCOREBOARD_API_KEY = os.getenv('SCOREBOARD_API_KEY')
USE_SCOREBOARD = os.getenv('USE_SCOREBOARD', '0') == '1'
try:
    INJURY_POLL_INTERVAL = int(os.getenv('INJURY_POLL_INTERVAL', '30'))
except ValueError:
    INJURY_POLL_INTERVAL = 30
try:
    SCORE_POLL_INTERVAL = int(os.getenv('SCORE_POLL_INTERVAL', '30'))
except ValueError:
    SCORE_POLL_INTERVAL = 30


def _parse_name_team_pos(s: str) -> Tuple[str, str, str]:
    """Split a FantasyPros name like "Joe Mixon (CIN - RB)"."""
    name = s.split('(')[0].strip()
    m = re.search(r'\(([^)]+)\)', s or '')
    team, pos = '', ''
    if m:
        inside = m.group(1)
        parts = [p.strip() for p in inside.split('-')]
        if len(parts) >= 1:
            team = parts[0]
        if len(parts) >= 2:
            pos = parts[1]
    return name, team, pos


def _normalize_opp(o: str) -> str:
    o = (o or '').strip()
    return o[1:] if o.startswith('@') else o


def _load_fp_file(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            name_raw = (r.get('PLAYER NAME') or '').strip()
            if not name_raw:
                continue
            name, team, pos = _parse_name_team_pos(name_raw)
            if not team or not pos:
                continue
            opp = _normalize_opp(r.get('OPP', ''))
            proj = parse_float(r.get('PROJ PTS'))
            sal = parse_int(r.get('SALARY'))
            out.append({
                'Name': name,
                'Team': team,
                'Pos': pos,
                'Opp': opp,
                'ProjFP': proj,
                'Salary': sal,
            })
    logger.info(f"Loaded {len(out)} players from {path}")
    return out


def load_all_players(base_dir: str = 'data/fantasypros') -> List[Dict[str, Any]]:
    """
    Load and deduplicate player projections from multiple CSV files. Files
    expected in ``base_dir``: qb.csv, rb.csv, wr.csv, te.csv, dst.csv.
    Deduplication occurs on (Name, Pos, Team) and keeps the player with
    the highest projection if duplicates exist.
    """
    players: List[Dict[str, Any]] = []
    for filename in ('qb.csv', 'rb.csv', 'wr.csv', 'te.csv', 'dst.csv'):
        path = os.path.join(base_dir, filename)
        if os.path.exists(path):
            players.extend(_load_fp_file(path))
    # Deduplicate by Name, Pos, Team keeping highest projection
    dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for p in players:
        key = (p['Name'], p['Pos'], p['Team'])
        if key not in dedup or p['ProjFP'] > dedup[key]['ProjFP']:
            dedup[key] = p
    final = list(dedup.values())
    logger.info(f"Total unique players: {len(final)}")
    return final


# ---------------- Odds (implied totals) ----------------

def fetch_odds() -> List[Dict[str, Any]]:
    """
    Fetch current NFL odds from the odds API. Returns a list of games
    with implied point totals and spreads. Results are cached based on
    the API URL. If odds usage is disabled or no API key is provided,
    an empty list is returned.
    """
    if not (USE_ODDS and ODDS_API_KEY):
        logger.info("Odds API disabled or key missing; skipping odds fetch.")
        return []
    url = (
        f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        f"?regions=us&oddsFormat=american&markets=spreads,totals&apiKey={ODDS_API_KEY}"
    )
    # Check cache first
    cached = cache.get(url)
    if cached is not None:
        return cached
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e}")
        return []
    games: List[Dict[str, Any]] = []
    for g in payload:
        home = g.get('home_team', '')
        away = g.get('away_team', '')
        bks = g.get('bookmakers') or []
        if not bks:
            continue
        mkts = bks[0].get('markets', [])
        spread = next((m for m in mkts if m.get('key') == 'spreads'), None)
        total = next((m for m in mkts if m.get('key') == 'totals'), None)
        over_under = None
        spread_pts = None
        fav = None
        if spread and spread.get('outcomes'):
            for o in spread['outcomes']:
                pts = o.get('point')
                if pts is None:
                    continue
                if float(pts) < 0:
                    fav = o.get('name')
                    spread_pts = float(pts)
        if total and total.get('outcomes'):
            over = next((o for o in total['outcomes'] if o.get('name') == 'Over'), None)
            if over and over.get('point') is not None:
                over_under = float(over['point'])
        home_it = away_it = None
        if over_under is not None:
            if spread_pts is not None and fav:
                fav_total = (over_under / 2) - (spread_pts / 2)
                dog_total = over_under - fav_total
                if fav == home:
                    home_it, away_it = fav_total, dog_total
                elif fav == away:
                    away_it, home_it = fav_total, dog_total
            else:
                home_it = away_it = over_under / 2
        games.append({
            'Home': home,
            'Away': away,
            'Total': f"{over_under:.1f}" if over_under is not None else '',
            'SpreadFav': fav or '',
            'SpreadPts': f"{spread_pts:.1f}" if spread_pts is not None else '',
            'HomeImplied': f"{home_it:.1f}" if home_it is not None else '',
            'AwayImplied': f"{away_it:.1f}" if away_it is not None else '',
        })
    cache.set(url, games)
    logger.info(f"Fetched odds for {len(games)} games.")
    return games


# ---------------- Weather (game conditions) ----------------

def fetch_weather() -> List[Dict[str, Any]]:
    """
    Fetch weather data for upcoming NFL games. This is a stub; if
    ``USE_WEATHER`` is enabled and ``WEATHER_API_KEY`` is provided, you
    should implement an actual API call here. As a fallback, this
    function will attempt to read ``data/weather/weather.json`` if it
    exists and return its contents.
    """
    if not USE_WEATHER:
        return []
    # Check if local cached file exists
    local_path = os.path.join('data', 'weather', 'weather.json')
    if os.path.exists(local_path):
        try:
            import json
            with open(local_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Failed to load local weather.json: {e}")
    # Placeholder: call your preferred weather API using WEATHER_API_KEY
    if WEATHER_API_KEY:
        logger.info("Weather API integration not yet implemented; returning empty weather list.")
    return []

# ---------------- Injuries ----------------

def fetch_injuries() -> Dict[str, str]:
    """
    Fetch current injury information for NFL players.  Returns a mapping
    from player name to injury status (e.g. "OUT", "QUESTIONABLE").  If no
    injury provider is configured (``INJURY_API_KEY`` empty), an empty dict
    is returned.

    To implement a real injury fetch, supply an API key in ``.env`` and
    extend this function to call your chosen data provider.  Caching can be
    applied via ``cache`` if desired.
    """
    if not INJURY_API_KEY:
        logger.info("Injury API key missing; returning no injuries.")
        return {}
    # Placeholder: integrate with an injury API.  For now return empty.
    try:
        # Example API call (commented out):
        # resp = requests.get(
        #     f"https://api.example.com/nfl/injuries?key={INJURY_API_KEY}", timeout=10
        # )
        # resp.raise_for_status()
        # data = resp.json()
        # Process and normalize injury data into {player_name: status}
        pass
    except Exception as e:
        logger.warning(f"Injury fetch failed: {e}")
    return {}


# ---------------- Live Scores ----------------

def fetch_live_scores() -> Dict[str, float]:
    """
    Fetch live scoring information for NFL teams.  Returns a mapping from
    team abbreviation (e.g. "BUF", "KC") to their current score in the
    ongoing week's games.  If scoreboard usage is disabled (``USE_SCOREBOARD``
    is false), this function returns an empty dict.  The current date is
    used to query the scoreboard API.  This implementation uses ESPN's
    public scoreboard endpoint, which does not require an API key.  If the
    endpoint is unreachable or its format changes, an empty result is
    returned and a warning is logged.

    NOTE: Real‑time fantasy points at the player level typically require
    paid data feeds.  This function focuses on team scores for identifying
    which games have started.  Players whose teams have non‑null scores
    can be considered "locked" and should not be swapped in mid‑game
    updates.
    """
    # Only run if scoreboard usage is enabled
    if not USE_SCOREBOARD:
        return {}
    # Determine today's date (YYYY-MM-DD) for the scoreboard query
    today = date.today().isoformat()
    # Allow override via env; default to ESPN's scoreboard endpoint
    base_url = os.getenv(
        "SCOREBOARD_API_URL",
        "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard",
    )
    url = f"{base_url}?dates={today}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Live scores fetch failed: {e}")
        return {}
    scores: Dict[str, float] = {}
    # ESPN scoreboard JSON has an "events" array; each event contains
    # competitions with competitors representing the two teams.  Extract
    # each team's short name and its current score.
    try:
        events = data.get("events", [])
        for ev in events:
            comps = ev.get("competitions", [])
            for comp in comps:
                competitors = comp.get("competitors", [])
                for comptr in competitors:
                    team = comptr.get("team", {})
                    # Use team abbreviation if available; fall back to shortDisplayName
                    abbr = team.get("abbreviation") or team.get("shortDisplayName") or team.get("name")
                    score_str = comptr.get("score", "0")
                    try:
                        score_val = float(score_str)
                    except Exception:
                        score_val = 0.0
                    if abbr:
                        scores[abbr] = score_val
    except Exception as ex:
        logger.warning(f"Error parsing scoreboard data: {ex}")
        return {}
    return scores
