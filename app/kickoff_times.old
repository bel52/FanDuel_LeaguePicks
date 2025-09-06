import os, json, datetime as dt
from zoneinfo import ZoneInfo
from .schedule_providers import fetch_kickoffs_from_espn, fetch_kickoffs_from_oddsapi_events

KICKOFF_CACHE_PATH = "data/output/kickoffs.json"
LAST_LINEUP_PATH   = "data/output/last_lineup.json"

def _ensure_dirs():
    os.makedirs(os.path.dirname(KICKOFF_CACHE_PATH), exist_ok=True)

def _save_json(path: str, data: dict):
    _ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def build_kickoff_map(players_df):
    tz = os.getenv("TIMEZONE", "America/New_York")
    # Try cache first
    cached = _load_json(KICKOFF_CACHE_PATH)
    kickoffs = {}
    try:
        kickoffs = fetch_kickoffs_from_espn(10, tz)
    except Exception:
        kickoffs = {}
    if not kickoffs:
        try:
            kickoffs = fetch_kickoffs_from_oddsapi_events(tz) or {}
        except Exception:
            kickoffs = {}

    # Filter to teams in our player pool if provided
    teams = set()
    try:
        teams = set(str(t).upper() for t in players_df['TEAM'].dropna().unique())
    except Exception:
        pass
    if teams:
        kickoffs = {k: v for k, v in kickoffs.items() if k in teams}

    # Merge with cache (prefer fresh values)
    merged = dict(cached)
    merged.update(kickoffs)
    try:
        _save_json(KICKOFF_CACHE_PATH, merged)
    except Exception:
        # non-fatal
        pass
    return merged

def _now_local():
    tz = os.getenv("TIMEZONE", "America/New_York")
    return dt.datetime.now(ZoneInfo(tz))

def _parse_local(iso_str: str):
    try:
        return dt.datetime.fromisoformat(iso_str)
    except Exception:
        return None

def auto_lock_started_players(last_lineup: list, kickoff_map: dict) -> list:
    """Return list of player names that should be auto-locked because their game has started."""
    now = _now_local()
    locked = []
    for p in last_lineup or []:
        team = str(p.get("team") or "").upper()
        name = p.get("name")
        ko = _parse_local(kickoff_map.get(team, ""))
        if ko and now >= ko:
            locked.append(name)
    return locked

def save_last_lineup(lineup_players: list):
    data = {"saved_at": _now_local().isoformat(), "lineup": lineup_players}
    try:
        _save_json(LAST_LINEUP_PATH, data)
    except Exception:
        pass

def load_last_lineup():
    data = _load_json(LAST_LINEUP_PATH)
    return data.get("lineup") or []
