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
    "Tampa Bay":"TB","New England":"NE","San Francisco":"SF","Kansas City":"KC","Green Bay":"GB",
    "Cleveland":"CLE","Cincinnati":"CIN","Pittsburgh":"PIT","Baltimore":"BAL","Buffalo":"BUF","Miami":"MIA",
    "Detroit":"DET","Dallas":"DAL","Denver":"DEN","Chicago":"CHI","Carolina":"CAR","Atlanta":"ATL","Seattle":"SEA",
    "Indianapolis":"IND","Houston":"HOU","Tennessee":"TEN","Philadelphia":"PHI","Minnesota":"MIN",
    "Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Las Vegas":"LV","New Orleans":"NO","Arizona":"ARI",
    "Jacksonville Jaguars":"JAC",
}

def _to_abbr(name: str) -> str|None:
    if not name: return None
    name = str(name).strip()
    if name in TEAM_NAME_TO_ABBR: return TEAM_NAME_TO_ABBR[name]
    if name in ALT_NAMES: return ALT_NAMES[name]
    if name.isupper() and len(name) in (2,3): return name
    return None

def _localize(dt_utc: datetime) -> datetime:
    if pytz is None: return dt_utc
    try:
        tz = pytz.timezone(DEFAULT_TZ)
        return dt_utc.astimezone(tz)
    except Exception:
        return dt_utc

def fetch_kickoffs_from_oddsapi() -> Dict[str, datetime]:
    if not ODDS_API_KEY:
        logging.warning("ODDS_API_KEY not set; cannot fetch schedule from The Odds API.")
        return {}

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"regions": "us", "markets": "h2h", "oddsFormat": "american", "apiKey": ODDS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        events = r.json() or []
    except Exception as e:
        logging.error(f"Odds API request failed: {e}")
        return {}

    ko_map: Dict[str, datetime] = {}
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=10)
    for ev in events:
        ct = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        if not ct or not (home and away): continue
        try:
            dt_utc = datetime.fromisoformat(ct.replace("Z","+00:00"))
        except Exception:
            continue
        if not (now - timedelta(days=1) <= dt_utc <= horizon):
            continue
        for nm in (home, away):
            abbr = _to_abbr(nm)
            if not abbr:
                continue
            ko_map[abbr] = _localize(dt_utc)
    if not ko_map:
        logging.warning("No kickoff times parsed from Odds API.")
    else:
        logging.info(f"Built kickoff map for {len(ko_map)} teams from Odds API.")
    return ko_map
