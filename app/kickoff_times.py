import os
import csv
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, List, Set

try:
    import pytz
except Exception:
    pytz = None

DEFAULT_TZ = os.getenv("TIMEZONE", "America/Chicago")
LAST_LINEUP_PATH = "data/output/last_lineup.json"
KICKOFF_CSV = "data/input/kickoffs.csv"

def _get_tz():
    if pytz is None:
        return None
    try:
        return pytz.timezone(DEFAULT_TZ)
    except Exception:
        return pytz.timezone("America/Chicago")

def load_kickoff_map_from_csv() -> Dict[str, datetime]:
    """
    Load team -> kickoff datetime from data/input/kickoffs.csv
    CSV columns: TEAM,KICKOFF (ISO, with timezone offset preferred)
    Example: CIN,2025-09-07T12:00:00-05:00
    """
    path = KICKOFF_CSV
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            team = str(row.get("TEAM","")).strip().upper()
            ts = str(row.get("KICKOFF","")).strip()
            if not team or not ts:
                continue
            try:
                # Try full ISO first
                dt = datetime.fromisoformat(ts)
            except Exception:
                # Fallback: assume local tz if no offset
                tz = _get_tz()
                try:
                    naive = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    dt = tz.localize(naive) if tz else naive
                except Exception:
                    logging.warning(f"Unable to parse kickoff '{ts}' for team {team}")
                    continue
            out[team] = dt
    if out:
        logging.info(f"Loaded {len(out)} team kickoffs from {path}")
    return out

def build_kickoff_map(players_df) -> Dict[str, datetime]:
    """
    Builds TEAM -> kickoff datetime map.
    Current implementation:
    1) Try CSV data/input/kickoffs.csv
    2) (Future) Attempt automated fetch (Playwright), else return {} and skip auto-locks.
    """
    kickoff_map = load_kickoff_map_from_csv()
    if kickoff_map:
        return kickoff_map

    # TODO: Add Playwright-based schedule scraping here if desired.
    logging.warning("No kickoff map found (no CSV). Auto-locks disabled for this run.")
    return {}

def teams_for_players(players_df, indices: List[int]) -> Set[str]:
    teams = set()
    for i in indices:
        try:
            t = str(players_df.loc[i, "TEAM"]).upper()
            if t:
                teams.add(t)
        except Exception:
            continue
    return teams

def load_last_lineup() -> List[str]:
    """
    Returns list of player names (as they appear in dataframe 'PLAYER NAME') from last lineup.
    """
    path = LAST_LINEUP_PATH
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = [str(x.get("name","")) for x in data.get("lineup", []) if x.get("name")]
        return names
    except Exception as e:
        logging.warning(f"Failed to read {path}: {e}")
        return []

def save_last_lineup(lineup_players: List[dict], meta: dict = None):
    os.makedirs(os.path.dirname(LAST_LINEUP_PATH), exist_ok=True)
    payload = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "lineup": lineup_players,
        "meta": meta or {}
    }
    with open(LAST_LINEUP_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def _now_local() -> datetime:
    tz = _get_tz()
    now = datetime.now(tz) if tz else datetime.now()
    return now

def auto_lock_started_players(players_df, kickoff_map: Dict[str, datetime]) -> Tuple[List[int], List[str], List[str]]:
    """
    Returns:
      - lock_indices: list of df indices to lock (already kicked off)
      - auto_locked_names: display names for UI
      - not_found_names: names present in last_lineup but not found in current df
    """
    if not kickoff_map:
        return [], [], []
    last_names = load_last_lineup()
    if not last_names:
        return [], [], []

    # Build a quick name -> index map for current df
    name_to_idx = {}
    for i, row in players_df.iterrows():
        nm = str(row.get("PLAYER NAME","")).strip()
        if nm:
            name_to_idx.setdefault(nm, []).append(i)

    now = _now_local()
    lock_indices, auto_locked_names, not_found = [], [], []
    for nm in last_names:
        idx_list = name_to_idx.get(nm, [])
        if not idx_list:
            not_found.append(nm)
            continue
        # Use TEAM to decide kickoff
        i0 = idx_list[0]
        team = str(players_df.loc[i0, "TEAM"]).upper()
        ko = kickoff_map.get(team)
        if ko is None:
            continue
        try:
            # Normalize ko to aware datetime in local tz if needed
            if ko.tzinfo is None and pytz is not None:
                ko = _get_tz().localize(ko)
        except Exception:
            pass
        if ko <= now:
            lock_indices.append(i0)
            auto_locked_names.append(nm)

    return lock_indices, auto_locked_names, not_found
