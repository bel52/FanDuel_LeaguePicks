import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Set

try:
    import pytz
except Exception:
    pytz = None

from .schedule_providers import fetch_kickoffs_from_oddsapi

DEFAULT_TZ = os.getenv("TIMEZONE", "America/Chicago")
LAST_LINEUP_PATH = "data/output/last_lineup.json"
KICKOFF_CACHE_PATH = "data/output/kickoffs.json"
CACHE_TTL_SECONDS = 2 * 60 * 60  # 2 hours

def _get_tz():
    if pytz is None:
        return None
    try:
        return pytz.timezone(DEFAULT_TZ)
    except Exception:
        return pytz.timezone("America/Chicago")

def _now_local():
    tz = _get_tz()
    return datetime.now(tz) if tz else datetime.now()

def _load_cache() -> Dict[str, str]:
    if not os.path.isfile(KICKOFF_CACHE_PATH):
        return {}
    try:
        with open(KICKOFF_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(ko_map: Dict[str, str]):
    os.makedirs(os.path.dirname(KICKOFF_CACHE_PATH), exist_ok=True)
    payload = {"saved_at": _now_local().isoformat(), "kickoffs": {}}
    # Serialize datetimes to iso
    for k, v in ko_map.items():
        payload["kickoffs"][k] = v.isoformat()
    with open(KICKOFF_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def build_kickoff_map(players_df) -> Dict[str, datetime]:
    """
    Build TEAM -> kickoff datetime map.
    Order:
      1) Cache if fresh (<2h)
      2) The Odds API (requires ODDS_API_KEY)
    """
    # 1) cache
    cache = _load_cache()
    kickoffs = cache.get("kickoffs", {})
    saved_at = cache.get("saved_at")
    if kickoffs and saved_at:
        try:
            saved_dt = datetime.fromisoformat(saved_at)
            age = (_now_local() - saved_dt.replace(tzinfo=None)).total_seconds()
            if age <= CACHE_TTL_SECONDS:
                # return cached map (parse iso to dt, local tz if necessary)
                out = {}
                for team, iso in kickoffs.items():
                    try:
                        dt = datetime.fromisoformat(iso)
                    except Exception:
                        continue
                    out[team] = dt
                if out:
                    logging.info(f"Using cached kickoff map ({int(age)}s old).")
                    return out
        except Exception:
            pass

    # 2) provider (Odds API)
    ko_map = fetch_kickoffs_from_oddsapi()
    if ko_map:
        _save_cache(ko_map)
        return ko_map

    logging.warning("No kickoff map available. Auto-locks disabled for this run.")
    return {}

def load_last_lineup() -> List[str]:
    if not os.path.isfile(LAST_LINEUP_PATH):
        return []
    try:
        with open(LAST_LINEUP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(x.get("name","")) for x in data.get("lineup", []) if x.get("name")]
    except Exception as e:
        logging.warning(f"Failed to read {LAST_LINEUP_PATH}: {e}")
        return []

def save_last_lineup(lineup_players: List[dict], meta: dict = None):
    os.makedirs(os.path.dirname(LAST_LINEUP_PATH), exist_ok=True)
    payload = {
        "saved_at": _now_local().isoformat(),
        "lineup": lineup_players,
        "meta": meta or {}
    }
    with open(LAST_LINEUP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def auto_lock_started_players(players_df, kickoff_map: Dict[str, datetime]) -> Tuple[List[int], List[str], List[str]]:
    """
    Returns:
      - lock_indices: df indices to lock (players whose team's kickoff <= now)
      - auto_locked_names: names locked
      - not_found_names: names from last lineup not found in df
    """
    if not kickoff_map:
        return [], [], []
    last_names = load_last_lineup()
    if not last_names:
        return [], [], []

    # Build lookup
    name_to_idx = {}
    for i, row in players_df.iterrows():
        nm = str(row.get("PLAYER NAME","")).strip()
        if nm:
            name_to_idx.setdefault(nm, []).append(i)

    now = _now_local()
    lock_idx, auto_locked, not_found = [], [], []
    for nm in last_names:
        candidates = name_to_idx.get(nm, [])
        if not candidates:
            not_found.append(nm)
            continue
        i0 = candidates[0]
        team = str(players_df.loc[i0, "TEAM"]).upper()
        ko = kickoff_map.get(team)
        if ko is None:
            continue
        # If kickoff is naive, compare using naive now
        ko_cmp = ko
        if hasattr(ko, 'tzinfo') and ko.tzinfo is not None and hasattr(now, 'tzinfo') and now.tzinfo is not None:
            pass
        else:
            ko_cmp = ko.replace(tzinfo=None)
            now = now.replace(tzinfo=None)
        if ko_cmp <= now:
            lock_idx.append(i0)
            auto_locked.append(nm)
    return lock_idx, auto_locked, not_found
