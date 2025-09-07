import datetime as dt
import json
import os
import csv
import ast
import re
from typing import Dict, List, Optional, Set, Tuple, Iterable

import requests

# ---------- Storage locations ----------
_SAVED_JSON = os.path.join("data", "output", "kickoffs.json")
_WEEKLY_BASE = os.path.join("data", "weekly")

# ESPN endpoints
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"

# Optional fallback: The Odds API (free tier key works)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_ENDPOINT = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"

# Basic team sanity pattern (e.g., KC, LAC, WSH)
_TEAM_RE = re.compile(r"^[A-Z]{2,4}$")


# Full team name -> abbreviation for The Odds API fallback
_ODDS_NAME_TO_ABBR: Dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WSH",
}


# ========== Week helpers ==========

def _current_week_id(today: Optional[dt.date | dt.datetime] = None) -> str:
    if today is None:
        today = dt.date.today()
    if isinstance(today, dt.datetime):
        today = today.date()
    year, week, _ = today.isocalendar()
    return f"{year}_w{week:02d}"


def _weekly_csv_path(week_id: Optional[str] = None) -> str:
    wid = week_id or _current_week_id()
    return os.path.join(_ WEEKLY_BASE, wid, "kickoffs.csv")


# ========== Parse helpers ==========

def _parse_iso(ts: str) -> Optional[dt.datetime]:
    try:
        x = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return x if x.tzinfo else x.replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _pairwise(items: List[str]) -> Iterable[Tuple[str, Optional[str]]]:
    it = iter(items)
    while True:
        try:
            a = next(it)
        except StopIteration:
            return
        b = next(it, None)
        yield a, b


def _games_from_team_map(team2iso: Dict[str, str]) -> List[Dict]:
    # group teams by identical kickoff ISO so we can pair them
    by_ts: Dict[str, List[str]] = {}
    for t, iso in team2iso.items():
        by_ts.setdefault(str(iso), []).append(t)
    out: List[Dict] = []
    for iso, teams in by_ts.items():
        teams = sorted(set(teams))
        for a, b in _pairwise(teams):
            out.append({"teams": [a] if not b else [a, b], "kickoff": iso})
    return out


def _load_json(path: str) -> Optional[object]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_team_map(raw: object) -> Dict[str, str]:
    """
    Normalize previously saved shapes:
      - list of games: [{"teams":[...], "kickoff": "..."}]
      - flat map: {"KC": "...", "PHI": "..."}
      - wrapped: {"SAVED_AT":"...", "KICKOFFS": {...}}  or KICKOFFS as a stringified dict
    """
    if raw is None:
        return {}
    if isinstance(raw, list):
        team2iso: Dict[str, str] = {}
        for g in raw:
            if not isinstance(g, dict):
                continue
            iso = g.get("kickoff")
            teams = g.get("teams", [])
            if not iso or not teams:
                continue
            for t in teams:
                t_up = str(t).upper()
                if _TEAM_RE.match(t_up):
                    team2iso[t_up] = str(iso)
        return team2iso
    if isinstance(raw, dict):
        # unwrap
        nested = raw.get("KICKOFFS")
        if isinstance(nested, str):
            try:
                nested = ast.literal_eval(nested)
            except Exception:
                nested = {}
        if isinstance(nested, dict):
            raw = nested
        out: Dict[str, str] = {}
        for k, v in raw.items():
            k_up = str(k).upper()
            if isinstance(v, str) and _TEAM_RE.match(k_up):
                out[k_up] = v
        return out
    return {}


# ========== CSV read/write ==========

def _read_weekly_csv(path: Optional[str] = None) -> List[Dict]:
    csv_path = path or _weekly_csv_path()
    if not os.path.exists(csv_path):
        return []
    games: List[Dict] = []
    try:
        with open(csv_path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                iso = row.get("kickoff") or row.get("kickoff_iso")
                ta = row.get("team_a") or row.get("team1")
                tb = row.get("team_b") or row.get("team2")
                teams = [t for t in [ta, tb] if t]
                if iso and teams:
                    games.append({"teams": [t.upper() for t in teams], "kickoff": str(iso)})
        return games
    except Exception as e:
        print(f"[kickoff_times] Failed reading CSV {csv_path}: {e}")
        return []


def _write_weekly_csv(games: List[Dict], path: Optional[str] = None) -> str:
    csv_path = path or _weekly_csv_path()
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["kickoff", "team_a", "team_b"])
            w.writeheader()
            for g in games:
                teams = [t.upper() for t in g.get("teams", [])]
                ta = teams[0] if len(teams) > 0 else ""
                tb = teams[1] if len(teams) > 1 else ""
                w.writerow({"kickoff": str(g.get("kickoff")), "team_a": ta, "team_b": tb})
    except Exception as e:
        print(f"[kickoff_times] Failed writing CSV {csv_path}: {e}")
    return csv_path


def _snapshot_weekly_csv(*, games: Optional[List[Dict]] = None, team_map: Optional[Dict[str, str]] = None,
                         week_id: Optional[str] = None) -> str:
    if games is None and team_map is not None:
        games = _games_from_team_map(team_map)
    return _write_weekly_csv(games or [], _weekly_csv_path(week_id))


# ========== Persistence of JSON ==========

def persist_kickoffs(team_map: Dict[str, dt.datetime | str], output_path: Optional[str] = None) -> None:
    path = output_path or _SAVED_JSON
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable: Dict[str, str] = {}
        for t, ts in team_map.items():
            if isinstance(ts, dt.datetime):
                serializable[t] = ts.isoformat()
            else:
                serializable[t] = str(ts)
        payload = {"SAVED_AT": dt.datetime.now(dt.timezone.utc).isoformat(), "KICKOFFS": serializable}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[kickoff_times] Persist error: {e}")


# ========== ESPN fetch ==========

def _fetch_espn_for_date(day: dt.date) -> List[Dict]:
    url = f"{ESPN_SCOREBOARD}?dates={day:%Y%m%d}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[kickoff_times] ESPN fetch failed {url}: {e}")
        return []

    events = data.get("events", []) if isinstance(data, dict) else []
    games: List[Dict] = []
    for ev in events:
        try:
            comps = (ev.get("competitions") or [])[0]
            iso = comps.get("date") or ev.get("date")
            teams: List[str] = []
            for c in comps.get("competitors", []):
                abbr = ((c.get("team") or {}).get("abbreviation") or "").upper()
                if abbr and _TEAM_RE.match(abbr):
                    teams.append(abbr)
            teams = sorted(set(teams))
            if iso and teams:
                # Only keep the first two teams if extra appear
                games.append({"teams": teams[:2], "kickoff": iso})
        except Exception:
            continue
    return games


def _fetch_espn_week_window(days_ahead: int = 7) -> List[Dict]:
    # Cover next 7–8 days so TNF/SNF/MNF are included regardless of current weekday
    start = dt.date.today()
    out: Dict[Tuple[str, str], Dict] = {}
    for i in range(days_ahead + 1):
        day = start + dt.timedelta(days=i)
        for g in _fetch_espn_for_date(day):
            teams = tuple(sorted([t.upper() for t in g.get("teams", [])])[:2])
            iso = str(g.get("kickoff"))
            if teams and iso:
                out[(teams[0], teams[1] if len(teams) > 1 else "")] = {"teams": list(teams) if teams[1] else [teams[0]], "kickoff": iso}
    return list(out.values())


# ========== The Odds API fallback (optional) ==========

def _fetch_oddsapi_upcoming() -> List[Dict]:
    if not ODDS_API_KEY:
        return []
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "dateFormat": "iso",
        "oddsFormat": "american",
    }
    try:
        r = requests.get(ODDS_ENDPOINT, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[kickoff_times] Odds API fetch failed: {e}")
        return []

    games: List[Dict] = []
    for ev in data if isinstance(data, list) else []:
        try:
            iso = ev.get("commence_time")
            home = _ODDS_NAME_TO_ABBR.get(ev.get("home_team", ""), "")
            away = _ODDS_NAME_TO_ABBR.get(ev.get("away_team", ""), "")
            teams = [t for t in [away, home] if t]
            if iso and teams:
                games.append({"teams": teams, "kickoff": iso})
        except Exception:
            continue
    return games


# ========== Public functions used by FastAPI app ==========

def build_kickoff_map(api_url: Optional[str] = None) -> Dict[str, dt.datetime]:
    """
    Returns {TEAM: kickoff_datetime_tzaware}. Source order:
      1) Provided API URL (if set) that returns either list-of-games or map.
      2) ESPN (free, default).
      3) Weekly CSV snapshot.
      4) Saved JSON cache.
      5) Odds API (only if key is set) as a final fallback.
    """
    # 1) Custom/internal API (if provided)
    if api_url:
        try:
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            team_map = _extract_team_map(data) if not isinstance(data, list) else _extract_team_map(data)
            if isinstance(data, list):
                # list-of-games -> team map
                team_map = {}
                for g in data:
                    iso = str(g.get("kickoff"))
                    for t in g.get("teams", []):
                        team_map[str(t).upper()] = iso
            if team_map:
                out: Dict[str, dt.datetime] = {}
                for t, iso in team_map.items():
                    ts = _parse_iso(iso)
                    if ts:
                        out[t] = ts
                return out
        except Exception as e:
            print(f"[kickoff_times] Custom API error {api_url}: {e}")

    # 2) ESPN default
    games = _fetch_espn_week_window()
    if games:
        tm: Dict[str, dt.datetime] = {}
        for g in games:
            ts = _parse_iso(g["kickoff"])
            if not ts:
                continue
            for t in g["teams"]:
                tm[t.upper()] = ts
        return tm

    # 3) Weekly CSV
    games = _read_weekly_csv()
    if games:
        tm: Dict[str, dt.datetime] = {}
        for g in games:
            ts = _parse_iso(g["kickoff"])
            if not ts:
                continue
            for t in g["teams"]:
                tm[t.upper()] = ts
        return tm

    # 4) Saved JSON
    raw = _load_json(_SAVED_JSON)
    team_map = _extract_team_map(raw)
    out: Dict[str, dt.datetime] = {}
    for t, iso in team_map.items():
        ts = _parse_iso(iso)
        if ts:
            out[t] = ts
    if out:
        return out

    # 5) Odds API fallback
    games = _fetch_oddsapi_upcoming()
    tm: Dict[str, dt.datetime] = {}
    for g in games:
        ts = _parse_iso(g["kickoff"])
        if not ts:
            continue
        for t in g["teams"]:
            tm[t.upper()] = ts
    return tm


def get_schedule(api_url: Optional[str] = None) -> List[Dict]:
    """
    Returns normalized list of games: [{"teams":[A,B], "kickoff":"ISO"}]
    Source order mirrors build_kickoff_map; ESPN is the default if nothing is configured.
    """
    # 1) Custom/internal API if provided
    if api_url:
        try:
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                # Assume it's already in our shape
                norm = []
                for g in data:
                    teams = [str(t).upper() for t in g.get("teams", [])]
                    norm.append({"teams": teams[:2], "kickoff": str(g.get("kickoff"))})
                if norm:
                    return norm
            # Or a map/wrapped form
            team_map = _extract_team_map(data)
            if team_map:
                return _games_from_team_map(team_map)
        except Exception as e:
            print(f"[kickoff_times] Custom API schedule error {api_url}: {e}")

    # 2) ESPN default
    games = _fetch_espn_week_window()
    if games:
        # Save snapshot(s) on the fly
        team_map: Dict[str, str] = {}
        for g in games:
            iso = str(g.get("kickoff"))
            for t in g.get("teams", []):
                team_map[str(t).upper()] = iso
        persist_kickoffs(team_map, output_path=_SAVED_JSON)
        if not os.path.exists(_weekly_csv_path()):
            _snapshot_weekly_csv(games=games)
        return games

    # 3) Weekly CSV
    games = _read_weekly_csv()
    if games:
        return games

    # 4) Saved JSON
    raw = _load_json(_SAVED_JSON)
    team_map = _extract_team_map(raw)
    if team_map:
        return _games_from_team_map(team_map)

    # 5) Odds API fallback
    games = _fetch_oddsapi_upcoming()
    if games:
        team_map: Dict[str, str] = {}
        for g in games:
            iso = str(g.get("kickoff"))
            for t in g.get("teams", []):
                team_map[str(t).upper()] = iso
        persist_kickoffs(team_map, output_path=_SAVED_JSON)
        if not os.path.exists(_weekly_csv_path()):
            _snapshot_weekly_csv(games=games)
    return games


def auto_lock_started_players() -> Set[str]:
    """
    Returns {team_abbr,...} whose games have already kicked off,
    using JSON/CSV (which will already be refreshed when you hit /schedule).
    """
    # Prefer saved JSON
    raw = _load_json(_SAVED_JSON)
    team_map = _extract_team_map(raw)

    # Fallback to weekly CSV if needed
    if not team_map:
        csv_games = _read_weekly_csv()
        for g in csv_games:
            for t in g.get("teams", []):
                team_map[t] = g.get("kickoff")

    if not team_map:
        return set()

    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    locked: Set[str] = set()
    for team, iso in team_map.items():
        ts = _parse_iso(iso)
        if ts and ts <= now:
            locked.add(team)
    return locked
