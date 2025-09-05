import os
import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict

try:
    import pytz
except Exception:
    pytz = None

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DEFAULT_TZ = os.getenv("TIMEZONE", "America/Chicago")

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAC","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
    "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}
ALT_NAMES = {
    "Washington Football Team":"WAS","Oakland Raiders":"LV","San Diego Chargers":"LAC","St. Louis Rams":"LAR",
    "Jacksonville":"JAC","New York Jets":"NYJ","NY Jets":"NYJ","New York Giants":"NYG","NY Giants":"NYG",
    "Tampa Bay":"TB","New England":"NE","San Francisco":"SF","Kansas City":"KC","Green Bay":"GB","Cleveland":"CLE",
    "Cincinnati":"CIN","Pittsburgh":"PIT","Baltimore":"BAL","Buffalo":"BUF","Miami":"MIA","Detroit":"DET",
    "Dallas":"DAL","Denver":"DEN","Chicago":"CHI","Carolina":"CAR","Atlanta":"ATL","Seattle":"SEA",
    "Indianapolis":"IND","Houston":"HOU","Tennessee":"TEN","Philadelphia":"PHI","Minnesota":"MIN","Arizona":"ARI",
    "Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Las Vegas":"LV","New Orleans":"NO","Jacksonville Jaguars":"JAC",
}

def _to_abbr(name: str) -> str|None:
    if not name: return None
    name = str(name).strip()
    if name in TEAM_NAME_TO_ABBR: return TEAM_NAME_TO_ABBR[name]
    if name in ALT_NAMES: return ALT_NAMES[name]
    if name.isupper() and 2 <= len(name) <= 3: return name
    return None

def _localize(dt_utc: datetime) -> datetime:
    if pytz is None: return dt_utc
    try:
        tz = pytz.timezone(DEFAULT_TZ)
        return dt_utc.astimezone(tz)
    except Exception:
        return dt_utc

def fetch_kickoffs_from_oddsapi_events() -> Dict[str, datetime]:
    """
    Odds API /events (no date filters). Returns free event list with commence_time.
    """
    if not ODDS_API_KEY:
        logging.info("ODDS_API_KEY not set; skipping Odds API /events.")
        return {}
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    try:
        r = requests.get(url, params={"apiKey": ODDS_API_KEY}, timeout=15)
        r.raise_for_status()
        events = r.json() or []
    except Exception as e:
        logging.error(f"Odds API /events request failed: {e}")
        return {}

    out: Dict[str, datetime] = {}
    for ev in events:
        ct = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not ct or not home or not away: continue
        try:
            dt_utc = datetime.fromisoformat(ct.replace("Z","+00:00"))
        except Exception:
            continue
        dt_local = _localize(dt_utc)
        for nm in (home, away):
            abbr = _to_abbr(nm)
            if abbr:
                out[abbr] = dt_local
    if out:
        logging.info(f"Odds API /events kickoff map size: {len(out)}")
    else:
        logging.warning("Odds API /events returned no parsable kickoffs.")
    return out

def fetch_kickoffs_from_oddsapi_odds() -> Dict[str, datetime]:
    """
    Odds API /odds (h2h) also includes commence_time. Costs credits; used only as last resort.
    """
    if not ODDS_API_KEY:
        logging.info("ODDS_API_KEY not set; skipping Odds API /odds.")
        return {}
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"regions": "us", "markets": "h2h", "oddsFormat": "american", "apiKey": ODDS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        events = r.json() or []
    except Exception as e:
        logging.error(f"Odds API /odds request failed: {e}")
        return {}

    out: Dict[str, datetime] = {}
    for ev in events:
        ct = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not ct or not home or not away: continue
        try:
            dt_utc = datetime.fromisoformat(ct.replace("Z","+00:00"))
        except Exception:
            continue
        dt_local = _localize(dt_utc)
        for nm in (home, away):
            abbr = _to_abbr(nm)
            if abbr:
                out[abbr] = dt_local
    if out:
        logging.info(f"Odds API /odds kickoff map size: {len(out)}")
    else:
        logging.warning("Odds API /odds returned no parsable kickoffs.")
    return out

def fetch_kickoffs_from_espn(days_ahead: int = 10) -> Dict[str, datetime]:
    """
    ESPN scoreboard fallback (free).
    """
    base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    today = datetime.utcnow().date()
    end = today + timedelta(days=days_ahead)
    params = {"limit":"1000","dates":f"{today:%Y%m%d}-{end:%Y%m%d}"}
    try:
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        events = data.get("events") or []
    except Exception as e:
        logging.error(f"ESPN scoreboard request failed: {e}")
        return {}

    out: Dict[str, datetime] = {}
    for ev in events:
        ct = ev.get("date")
        comps = (ev.get("competitions") or [])
        if not ct or not comps: continue
        try:
            dt_utc = datetime.fromisoformat(ct.replace("Z","+00:00"))
        except Exception:
            continue
        dt_local = _localize(dt_utc)
        comp = comps[0]
        for c in (comp.get("competitors") or []):
            team = (c.get("team") or {})
            abbr = team.get("abbreviation")
            name = team.get("displayName")
            key = abbr or _to_abbr(name)
            if key:
                out[key] = dt_local
    if out:
        logging.info(f"ESPN kickoff map size: {len(out)}")
    else:
        logging.warning("ESPN scoreboard returned no parsable kickoffs.")
    return out

def fetch_kickoffs(days_ahead: int = 10) -> Dict[str, datetime]:
    """
    Prefer FREE + reliable:
      1) ESPN scoreboard
      2) Odds API /events (free)
      3) Odds API /odds (paid credits) as last resort
    """
    ko = fetch_kickoffs_from_espn(days_ahead=days_ahead)
    if ko:
        return ko
    ko = fetch_kickoffs_from_oddsapi_events()
    if ko:
        return ko
    return fetch_kickoffs_from_oddsapi_odds()
