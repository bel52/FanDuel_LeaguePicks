import os
import datetime as dt
from zoneinfo import ZoneInfo
import requests

def _to_local_iso(iso_utc: str, tz_name: str) -> str:
    # ESPN returns '2025-09-14T16:05Z' or with offset; normalize:
    iso = iso_utc.replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    target = ZoneInfo(tz_name)
    return t.astimezone(target).isoformat()

def fetch_kickoffs_from_espn(days_ahead: int = 10, tz_name: str = "America/New_York"):
    base = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    today = dt.date.today()
    end = today + dt.timedelta(days=days_ahead)
    params = {"limit":"1000","dates":f"{today:%Y%m%d}-{end:%Y%m%d}"}
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    events = data.get("events") or []
    m = {}
    for ev in events:
        comps = (ev.get("competitions") or [{}])[0]
        start = comps.get("date") or ev.get("date") or ev.get("startDate")
        if not start:
            continue
        for team in (comps.get("competitors") or []):
            abbr = ((team.get("team") or {}).get("abbreviation") or "").upper()
            if abbr:
                m[abbr] = _to_local_iso(start, tz_name)
    return m

def fetch_kickoffs_from_oddsapi_events(tz_name: str = "America/New_York"):
    key = os.getenv("ODDS_API_KEY")
    if not key:
        return {}
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    # Ask for a broad window (14d) to avoid 422 on missing params
    params = {"apiKey": key}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        events = r.json() or []
    except Exception:
        return {}
    # Try simple mapping by team abbreviations in 'home_team','away_team' if present
    m = {}
    for ev in events:
        start = ev.get("commence_time")
        if not start:
            continue
        start = start.replace("Z", "+00:00")
        t = dt.datetime.fromisoformat(start)
        target = ZoneInfo(tz_name)
        loc = t.astimezone(target).isoformat()
        # Odds API team names are long; we can't 100% map. Skip unless already an abbreviation-like 2-4 chars.
        for k in ("home_team","away_team"):
            name = ev.get(k) or ""
            abbr = name.strip().upper()
            if 2 <= len(abbr) <= 4:  # very naive; ESPN will be our authoritative source
                m[abbr] = loc
    return m
