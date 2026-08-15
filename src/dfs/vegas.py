"""The Odds API client → implied TEAM totals. 

Anti-double-count rule (locked): Vegas enters the system exactly once, here, as
implied team totals attached to the slate. blend.py consumes them; the optimizer
never sees game totals and applies no boosts.

implied_team_total = game_total/2 - spread/2  (team favored by 3 in a 47 game: 25.0)
Env: ODDS_API_KEY. Usage ~2 calls/week (well within 500/mo free tier).
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass

from .matching import norm_team as _canon
from typing import Optional

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# Odds API team names -> FanDuel abbreviations
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


class VegasError(Exception):
    pass


@dataclass
class TeamLine:
    team: str                  # FD abbreviation
    opponent: str
    game_total: float
    spread: float              # negative = favored
    implied_total: float
    kickoff_iso: str


class OddsClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            raise VegasError("ODDS_API_KEY not set")
        self.timeout = timeout
        self.last_quota: dict[str, str] = {}

    def _get(self, path: str, params: dict) -> list | dict:
        qs = "&".join(f"{k}={v}" for k, v in {**params, "apiKey": self.api_key}.items())
        url = f"{BASE}/{path}?{qs}"
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "dfs-v6/1.0"}),
                    timeout=self.timeout) as resp:
                self.last_quota = {
                    "remaining": resp.headers.get("x-requests-remaining", "?"),
                    "used": resp.headers.get("x-requests-used", "?"),
                }
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise VegasError("Odds API auth failed — rotate key (backlog P0 #5)") from e
            raise VegasError(f"Odds API HTTP {e.code}") from e

    def team_lines(self, slate_teams: set[str] | None = None) -> dict[str, TeamLine]:
        """Fetch spreads+totals; return implied totals keyed by FD team abbr.
        If slate_teams given, restrict to those teams (slate-scoped, not whole week)."""
        games = self._get(f"sports/{SPORT}/odds",
                          {"regions": "us", "markets": "spreads,totals", "oddsFormat": "american"})
        if not isinstance(games, list):
            raise VegasError(f"unexpected response type: {type(games)}")
        out: dict[str, TeamLine] = {}
        for g in games:
            home_full, away_full = g.get("home_team", ""), g.get("away_team", "")
            home = _canon(TEAM_ABBR.get(home_full, ""))
            away = _canon(TEAM_ABBR.get(away_full, ""))
            if not home or not away:
                continue
            total, spreads = None, {}
            for bk in g.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt["key"] == "totals" and total is None:
                        pts = [o.get("point") for o in mkt["outcomes"] if o.get("point")]
                        total = pts[0] if pts else None
                    if mkt["key"] == "spreads" and not spreads:
                        for o in mkt["outcomes"]:
                            abbr = _canon(TEAM_ABBR.get(o.get("name", ""), ""))
                            if abbr and o.get("point") is not None:
                                spreads[abbr] = float(o["point"])
                if total is not None and spreads:
                    break
            if total is None or home not in spreads or away not in spreads:
                continue
            for team, opp in ((home, away), (away, home)):
                out[team] = TeamLine(
                    team=team, opponent=opp, game_total=float(total),
                    spread=spreads[team],
                    implied_total=round(float(total) / 2 - spreads[team] / 2, 2),
                    kickoff_iso=g.get("commence_time", ""),
                )
        if slate_teams:
            want = {_canon(t) for t in slate_teams}
            out = {t: v for t, v in out.items() if t in want}
            missing = want - set(out)
            if missing:
                raise VegasError(f"no Vegas lines for slate teams: {sorted(missing)} — "
                                 "verify slate vs odds board before proceeding")
        if not out:
            raise VegasError("zero team lines parsed — API/schema problem")
        return out
