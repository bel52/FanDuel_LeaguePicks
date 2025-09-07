import os
import json
import datetime
from typing import Dict, Optional, List, Any
import requests

# Files we persist to
_LAST = os.path.join("data", "output", "last_lineup.json")
_KICKOFFS = os.path.join("data", "output", "kickoffs.json")

# Optional env var to point at your internal schedule service
# Expected response examples (both supported):
#  A) [{"teams": ["BUF","KC"], "kickoff": "2025-09-07T17:00:00Z"}, ...]
#  B) {"BUF": "2025-09-07T17:00:00Z", "KC": "2025-09-07T20:25:00Z", ...}
KICKOFF_API_URL = os.getenv("KICKOFF_API_URL")  # e.g., http://your-api/schedule


# --------------------------
# Last lineup persistence
# --------------------------
def save_last_lineup(lineup: List[Dict]) -> None:
    """Persist the latest optimized lineup for league comparison."""
    os.makedirs(os.path.dirname(_LAST), exist_ok=True)
    try:
        with open(_LAST, "w") as f:
            json.dump(lineup, f, indent=2)
    except Exception as e:
        print(f"[kickoff_times] WARNING: Could not save last lineup: {e}")


def load_last_lineup() -> Optional[List[Dict]]:
    """Load previously saved lineup, or None if not available."""
    if not os.path.exists(_LAST):
        return None
    try:
        with open(_LAST, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[kickoff_times] WARNING: Could not load last lineup: {e}")
        return None


# --------------------------
# Kickoff fetching & caching
# --------------------------
def _parse_iso(dt_str: str) -> datetime.datetime:
    """Parse an ISO timestamp string into an aware datetime in UTC."""
    # Handle trailing 'Z'
    if dt_str.endswith("Z"):
        dt_str = dt_str.replace("Z", "+00:00")
    dt = datetime.datetime.fromisoformat(dt_str)
    # Ensure timezone-aware (default to UTC if naive)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc)


def _load_cached_kickoffs() -> Dict[str, str]:
    """Load cached kickoffs from disk (ISO str), or {}."""
    if not os.path.exists(_KICKOFFS):
        return {}
    try:
        with open(_KICKOFFS, "r") as f:
            data = json.load(f)
        # Normalize keys
        return {k.upper(): v for k, v in data.items()}
    except Exception as e:
        print(f"[kickoff_times] WARNING: Could not read {_KICKOFFS}: {e}")
        return {}


def _save_cached_kickoffs(map_: Dict[str, str]) -> None:
    """Persist kickoff map (ISO strings) to disk."""
    os.makedirs(os.path.dirname(_KICKOFFS), exist_ok=True)
    try:
        with open(_KICKOFFS, "w") as f:
            json.dump(map_, f, indent=2)
    except Exception as e:
        print(f"[kickoff_times] WARNING: Could not write {_KICKOFFS}: {e}")


def _fetch_kickoffs_from_api() -> Dict[str, str]:
    """
    Fetch kickoff times from an external/internal API.
    Returns a dict: {TEAM: ISO_STRING}
    """
    if not KICKOFF_API_URL:
        # No API configured
        return {}

    try:
        resp = requests.get(KICKOFF_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[kickoff_times] ERROR: fetching {KICKOFF_API_URL}: {e}")
        return {}

    # Normalize both shapes
    result: Dict[str, str] = {}
    if isinstance(data, dict):
        # Shape B: {"BUF":"...","KC":"..."}
        for team, iso_str in data.items():
            if isinstance(iso_str, str):
                result[team.upper()] = iso_str
    elif isinstance(data, list):
        # Shape A: [{"teams":["BUF","KC"], "kickoff":"..."}]
        for g in data:
            kickoff = g.get("kickoff")
            teams = g.get("teams", [])
            if isinstance(kickoff, str) and isinstance(teams, list):
                for t in teams:
                    result[str(t).upper()] = kickoff
    else:
        print("[kickoff_times] WARNING: Unknown schedule payload shape")

    return result


def build_kickoff_map(players_df: Optional[Any] = None) -> Dict[str, str]:
    """
    Build a kickoff map {TEAM: ISO_STRING}.
    Priority:
        1) Fetch from API (if KICKOFF_API_URL set)
        2) Fall back to cached file data/output/kickoffs.json
    The players_df parameter is accepted for backward-compatibility.
    """
    # Try API first if configured
    api_map = _fetch_kickoffs_from_api()
    if api_map:
        _save_cached_kickoffs(api_map)
        return api_map

    # Fallback to cache
    cache_map = _load_cached_kickoffs()
    return cache_map


# --------------------------
# Late swap auto-locking
# --------------------------
def auto_lock_started_players(last_lineup: Optional[List[Dict]], kickoff_map: Dict[str, str]) -> List[str]:
    """
    From the previously saved lineup, return a list of player NAMES whose team
    kickoff time is in the past (<= now UTC).
    """
    if not last_lineup or not kickoff_map:
        return []

    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    locked_names: List[str] = []

    for p in last_lineup:
        team = str(p.get("team", "")).upper()
        name = str(p.get("name", "")).strip()
        iso = kickoff_map.get(team)
        if not iso or not name:
            continue
        try:
            kdt = _parse_iso(iso)
            if kdt <= now:
                locked_names.append(name)
        except Exception as e:
            print(f"[kickoff_times] WARNING: bad kickoff for {team}: {iso} ({e})")

    return locked_names
