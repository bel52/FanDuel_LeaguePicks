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
from typing import Optional

from statistics import median

from .matching import norm_team

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


# The contest is on FanDuel, so FanDuel's own board is the relevant market; other
# books are fallbacks in order, and a median consensus is the last resort.
BOOK_PREFERENCE = ["fanduel", "draftkings", "betmgm", "caesars"]


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
    book: str = ""              # which book the total+spread pair came from
    last_update: str = ""       # that book's last_update, for staleness reporting


class OddsClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            raise VegasError("ODDS_API_KEY not set")
        self.timeout = timeout
        self.last_quota: dict[str, str] = {}
        self.missing_teams: list[str] = []
        self.books_used: dict[str, int] = {}
        self.newest_update: str = ""

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
        except VegasError:
            raise
        except Exception as e:
            # Timeouts, DNS failures, connection resets, malformed JSON — the caller
            # treats VegasError as "degrade to no-Vegas and keep building". A raw
            # URLError/JSONDecodeError instead CRASHED the build, defeating the
            # designed soft-skip (external review, 2026-08-30).
            raise VegasError(f"Odds API unreachable/unparseable: {type(e).__name__}: {e}") from e

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
            home, away = TEAM_ABBR.get(home_full), TEAM_ABBR.get(away_full)
            if not home or not away:
                continue
            # Parse each book INDEPENDENTLY, then choose. The previous version took
            # the first available total and the first available spreads separately,
            # which could pair a total from one book with a spread from another --
            # a combination no book ever offered, and the likely source of implied
            # totals that matched no real board (2026-09-07 build: BUF 26.75 when
            # every book had BUF@HOU at 44.5 / -1.5, i.e. 23.0).
            per_book: dict[str, tuple[float, dict[str, float], str]] = {}
            for bk in g.get("bookmakers", []):
                bkey = bk.get("key", "")
                b_total, b_spreads, b_upd = None, {}, bk.get("last_update", "")
                for mkt in bk.get("markets", []):
                    if mkt.get("key") == "totals":
                        pts = [o.get("point") for o in mkt.get("outcomes", [])
                               if o.get("point") is not None]
                        if pts:
                            b_total = float(pts[0])
                    elif mkt.get("key") == "spreads":
                        for o in mkt.get("outcomes", []):
                            abbr = TEAM_ABBR.get(o.get("name", ""))
                            if abbr and o.get("point") is not None:
                                b_spreads[abbr] = float(o["point"])
                    b_upd = max(b_upd or "", mkt.get("last_update", "") or "")
                if b_total is not None and home in b_spreads and away in b_spreads:
                    per_book[bkey] = (b_total, b_spreads, b_upd)
            if not per_book:
                continue
            chosen = next((b for b in BOOK_PREFERENCE if b in per_book), None)
            if chosen:
                total, spreads, upd = per_book[chosen]
                book = chosen
            else:
                # No preferred book on the board: consensus across whatever is there.
                total = median([v[0] for v in per_book.values()])
                spreads = {home: median([v[1][home] for v in per_book.values()]),
                           away: median([v[1][away] for v in per_book.values()])}
                upd = max((v[2] for v in per_book.values()), default="")
                book = f"consensus/{len(per_book)}"
            self.books_used[book] = self.books_used.get(book, 0) + 1
            if upd:
                self.newest_update = max(self.newest_update, upd)
            for team, opp in ((home, away), (away, home)):
                # Canonicalize through norm_team so a FanDuel "JAC" slate matches an
                # Odds-board "JAX" (and any future alias) — never key on raw abbrs.
                tn, on = norm_team(team), norm_team(opp)
                out[tn] = TeamLine(
                    team=tn, opponent=on, game_total=float(total),
                    spread=spreads[team],
                    implied_total=round(float(total) / 2 - spreads[team] / 2, 2),
                    kickoff_iso=g.get("commence_time", ""),
                    book=book,
                    last_update=upd,
                )
        if slate_teams:
            want = {norm_team(t) for t in slate_teams}
            out = {t: v for t, v in out.items() if t in want}
            # A slate team missing from the board is NORMAL on Sunday (TNF has kicked
            # off; lines vanish). Degrade to a warning the caller can print — a single
            # off-board team must never cost the build the entire Vegas layer.
            self.missing_teams = sorted(want - set(out))
        if not out:
            raise VegasError("zero team lines parsed — API/schema problem")
        return out
