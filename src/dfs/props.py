"""The Odds API player props -> market-implied FanDuel stat lines.

WHY THIS EXISTS
---------------
Under Total Points the league objective is (almost exactly) the sum of projections,
so projection accuracy is the only lever that matters. FantasyPros consensus moves
in hours or days; betting markets move in minutes and are priced by people risking
money. Player props are the sharpest public per-player signal available.

WHAT THIS MODULE PRODUCES
-------------------------
A market-implied *stat line* per player, not a points number. The stat line is then
scored by the SAME `scoring.score()` used for FantasyPros, so props and FP land in
identical FanDuel-points units with the same bonus model and the same auditable
breakdown. No second scoring path exists.

Markets consumed (1 credit per market per event):
  player_pass_yds        -> pass_yds
  player_pass_tds        -> pass_tds     (Poisson solve from the threshold)
  player_rush_yds        -> rush_yds
  player_reception_yds   -> rec_yds
  player_receptions      -> rec_rec
  player_anytime_td      -> non-passing TDs (split rush/rec is irrelevant: both 6 pts)

Categories the props board does not cover (interceptions, fumbles, return TDs,
two-point conversions) are taken from the player's FantasyPros stat line. Those are
small, negative-or-rare terms; dropping them would bias QBs up by ~0.7 points.

SCALE ANCHORING (the important design decision)
-----------------------------------------------
A prop line sits near the *median* outcome, but FanDuel points are linear in yards,
so what a projection needs is the *mean*. Yardage distributions are right-skewed, so
a median-anchored stat line under-projects by a few percent, and the size of that
skew is not identifiable from a single threshold.

Rather than invent a skew constant, props are used for RELATIVE signal only: after
scoring, each position's props points are multiplied by the median FP/props ratio for
that position on that slate. Markets decide how points are distributed between
players; FP consensus sets the overall level.

Two things fall out of this for free:
  * the median-vs-mean bias cancels, with no fabricated constant;
  * `distributions.json` (actual/projected ratio pools fit against FP-scaled
    projections) stays valid, because the blended projection remains on the FP scale.

The per-position scale factors are printed every build. A factor far from 1.0 means
the parser or the board schema broke, not that the market disagrees — so the factors
are gated (warn, then reject) rather than trusted.

Env: ODDS_API_KEY (same key as vegas.py). Credit cost: 6 per event, ~72 per full
slate pull. The /events listing is free. Responses are cached on disk so a rebuild
after the inactives sweep does not re-spend credits.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, median
from typing import Optional

from .matching import norm_name, norm_team, short_key, _first_names_compatible
from .scoring import score
from .vegas import BASE, SPORT, TEAM_ABBR, VegasError

MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]
CREDITS_PER_EVENT = len(MARKETS)

# Books to consult. Values are consensus (median) across whichever of these are on
# the board, so one book pulling a line does not swing the projection.
BOOKMAKERS = ["fanduel", "draftkings"]

# --- coarse priors, used ONLY where a single threshold cannot identify the value ---
# Outcome spread around a prop line, in the line's own units. These matter only when
# a book prices a market away from 50/50; at the typical -114/-114 the devigged
# probability is 0.500 and the mean lands exactly on the line regardless of spread.
LINE_SPREAD = {
    "pass_yds": 78.0,      # matches scoring.YD_SPREAD["pass"]
    "rush_yds": 34.0,      # matches scoring.YD_SPREAD["rush"]
    "rec_yds": 36.0,       # matches scoring.YD_SPREAD["rec"]
    "rec_rec": 2.2,
}
# Anytime-TD boards carry a large overround (books hold far more on TD scorer markets
# than on yardage). One-sided prices cannot be devigged pairwise, so a flat haircut is
# applied. [coarse prior — recalibrate from logged actuals once weeks accumulate]
ANYTIME_TD_OVERROUND = 1.14
# E[TDs] / P(>=1 TD). Poisson would imply ~1.39 at p=0.5, which overstates multi-TD
# games for skill players. [coarse prior — recalibrate from logged actuals]
MULTI_TD_FACTOR = 1.18

# Blend weight on (scale-anchored) props, by position, when the position's primary
# market is present. The remainder goes to FantasyPros consensus.
PROPS_WEIGHT = {"QB": 0.65, "RB": 0.60, "WR": 0.60, "TE": 0.55}
# Primary market a position must have priced for props to be used at all.
PRIMARY_MARKET = {"QB": "pass_yds", "RB": "rush_yds", "WR": "rec_yds", "TE": "rec_yds"}

# Scale-factor gates. Inside WARN the factor is applied silently; between WARN and
# REJECT it is applied with a warning; outside REJECT props are discarded for that
# position, because a factor that extreme is a schema/parser failure, not a market view.
SCALE_WARN = (0.80, 1.25)
SCALE_REJECT = (0.55, 1.80)
# A position needs this many players with both numbers before its own scale factor is
# trustworthy. Below that, the all-positions factor is used -- but only if IT has a
# real sample, because a cross-position factor imports whatever bias exists between
# positions. With too little of either, props are skipped for that position: no
# number is safer than a mis-scaled one.
MIN_SCALE_SAMPLE = 4
MIN_GLOBAL_SAMPLE = 8


class PropsError(Exception):
    """Props layer failed. Always degradable: the build continues on FP alone."""


def american_to_prob(price: float) -> float:
    """American odds -> raw implied probability (vig included)."""
    p = float(price)
    if p < 0:
        return (-p) / ((-p) + 100.0)
    return 100.0 / (p + 100.0)


def devig_two_way(over_price: float, under_price: float) -> float:
    """Two-sided market -> P(over), proportional (multiplicative) vig removal."""
    a, b = american_to_prob(over_price), american_to_prob(under_price)
    if a + b <= 0:
        raise PropsError("degenerate two-way prices")
    return a / (a + b)


def mean_from_line(line: float, p_over: float, spread: float) -> float:
    """Mean of a normal outcome consistent with P(X > line) = p_over.

    P(X > line) = p  =>  (line - mean)/sd = z_(1-p)  =>  mean = line + sd * z_p.
    At p = 0.5 this returns the line exactly.
    """
    p = min(max(p_over, 0.02), 0.98)
    return max(0.0, line + spread * NormalDist().inv_cdf(p))


def poisson_lambda_from_tail(threshold: float, p_tail: float,
                             lo: float = 0.01, hi: float = 8.0) -> float:
    """Solve Poisson lambda from P(X > threshold) = p_tail.

    Pass-TD boards price a single threshold (1.5 or 2.5). Pass TDs are close to
    Poisson, so the threshold identifies lambda. Verified on the live 2026 Week 1
    TB@CIN board: DraftKings' 1.5 line and FanDuel's 2.5 line for Joe Burrow both
    solve to lambda ~2.2, which is what makes this defensible rather than a guess.
    """
    k = int(math.floor(threshold)) + 1          # P(X >= k)

    def tail(lam: float) -> float:
        # 1 - CDF(k-1)
        c, term = 0.0, math.exp(-lam)
        for i in range(0, k):
            if i:
                term *= lam / i
            c += term
        return 1.0 - c

    p = min(max(p_tail, 0.01), 0.99)
    for _ in range(60):
        mid = (lo + hi) / 2
        if tail(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class PlayerProps:
    """Market-implied stat line for one player, plus provenance."""
    name: str
    event_teams: tuple[str, str]
    stats: dict = field(default_factory=dict)
    markets: set = field(default_factory=set)
    books: set = field(default_factory=set)
    last_update: str = ""

    def has(self, market: str) -> bool:
        return market in self.markets


# ---------------------------------------------------------------------------
# Board parsing
# ---------------------------------------------------------------------------

def _consensus(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def parse_event_board(board: dict) -> dict[str, PlayerProps]:
    """One /events/{id}/odds payload -> {norm_name: PlayerProps}.

    Values are consensus across books. Team is not on props outcomes, so the caller
    matches against the two teams named on the event.
    """
    home = TEAM_ABBR.get(board.get("home_team", ""), "")
    away = TEAM_ABBR.get(board.get("away_team", ""), "")
    if not home or not away:
        raise PropsError(f"unknown event teams: {board.get('away_team')} @ {board.get('home_team')}")
    teams = (norm_team(away), norm_team(home))

    # market key -> player -> list of per-book parsed values
    acc: dict[str, dict[str, list[float]]] = {}
    books: dict[str, set] = {}
    updates: dict[str, str] = {}

    for bk in board.get("bookmakers", []) or []:
        bkey = bk.get("key", "")
        for mkt in bk.get("markets", []) or []:
            key = mkt.get("key", "")
            upd = mkt.get("last_update", "")
            outs = mkt.get("outcomes", []) or []
            if key == "player_anytime_td":
                for o in outs:
                    if str(o.get("name", "")).lower() != "yes":
                        continue
                    who = o.get("description") or ""
                    if not who or o.get("price") is None:
                        continue
                    raw = american_to_prob(o["price"]) / ANYTIME_TD_OVERROUND
                    acc.setdefault("anytime_td_prob", {}).setdefault(
                        norm_name(who), []).append(min(raw, 0.95))
                    books.setdefault(norm_name(who), set()).add(bkey)
                    updates[norm_name(who)] = max(updates.get(norm_name(who), ""), upd)
                continue

            # two-sided markets: pair Over/Under per player
            pairs: dict[str, dict[str, tuple[float, float]]] = {}
            for o in outs:
                side = str(o.get("name", "")).lower()
                who = o.get("description") or ""
                if not who or o.get("price") is None or o.get("point") is None:
                    continue
                pairs.setdefault(norm_name(who), {})[side] = (float(o["point"]), float(o["price"]))
            for who, sides in pairs.items():
                if "over" not in sides or "under" not in sides:
                    continue
                line, over_price = sides["over"]
                _, under_price = sides["under"]
                try:
                    p_over = devig_two_way(over_price, under_price)
                except PropsError:
                    continue
                acc.setdefault(key, {}).setdefault(who, []).append(line)
                acc.setdefault(key + "__p", {}).setdefault(who, []).append(p_over)
                books.setdefault(who, set()).add(bkey)
                updates[who] = max(updates.get(who, ""), upd)

    out: dict[str, PlayerProps] = {}

    def rec(who: str) -> PlayerProps:
        if who not in out:
            out[who] = PlayerProps(name=who, event_teams=teams,
                                   books=books.get(who, set()),
                                   last_update=updates.get(who, ""))
        return out[who]

    def two_sided(market_key: str, stat: str, spread_key: str) -> None:
        for who, lines in acc.get(market_key, {}).items():
            line = _consensus(lines)
            p_over = _consensus(acc.get(market_key + "__p", {}).get(who, [0.5]))
            r = rec(who)
            r.stats[stat] = round(mean_from_line(line, p_over, LINE_SPREAD[spread_key]), 2)
            r.markets.add(stat)

    two_sided("player_pass_yds", "pass_yds", "pass_yds")
    two_sided("player_rush_yds", "rush_yds", "rush_yds")
    two_sided("player_reception_yds", "rec_yds", "rec_yds")
    two_sided("player_receptions", "rec_rec", "rec_rec")

    # pass TDs: Poisson solve per book, then consensus on lambda
    for who, lines in acc.get("player_pass_tds", {}).items():
        probs = acc.get("player_pass_tds__p", {}).get(who, [])
        lams = [poisson_lambda_from_tail(t, p) for t, p in zip(lines, probs)]
        if not lams:
            continue
        r = rec(who)
        r.stats["pass_tds"] = round(_consensus(lams), 3)
        r.markets.add("pass_tds")

    # anytime TD -> expected non-passing TDs
    for who, probs in acc.get("anytime_td_prob", {}).items():
        r = rec(who)
        r.stats["anytime_td_prob"] = round(_consensus(probs), 4)
        r.markets.add("anytime_td")

    return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PropsClient:
    """Fetches event ids and per-event prop boards, with an on-disk cache.

    The cache is what keeps this inside the free tier: a full slate is 72 credits,
    and a Saturday build followed by a post-inactives rebuild would otherwise cost
    144. Cached boards are reused until `max_age_hours` old.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 25,
                 cache_dir: str | Path | None = None, max_age_hours: float = 6.0):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            raise PropsError("ODDS_API_KEY not set")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_age_hours = max_age_hours
        self.credits_spent = 0
        self.cache_hits = 0
        self.stale_cache = 0
        self.last_quota: dict[str, str] = {}
        self.errors: list[str] = []

    def _get(self, path: str, params: dict) -> list | dict:
        qs = urllib.parse.urlencode({**params, "apiKey": self.api_key})
        url = f"{BASE}/{path}?{qs}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dfs-v6/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self.last_quota = {
                    "remaining": resp.headers.get("x-requests-remaining", "?"),
                    "used": resp.headers.get("x-requests-used", "?"),
                    "last": resp.headers.get("x-requests-last", "?"),
                }
                try:
                    self.credits_spent += int(self.last_quota["last"])
                except (TypeError, ValueError):
                    pass
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise PropsError("Odds API auth failed — rotate key") from e
            if e.code == 422:
                raise PropsError(f"Odds API 422 (event off-board or market unsupported)") from e
            raise PropsError(f"Odds API HTTP {e.code}") from e
        except PropsError:
            raise
        except Exception as e:
            raise PropsError(f"Odds API unreachable/unparseable: {type(e).__name__}: {e}") from e

    def events(self) -> list[dict]:
        """Upcoming events. Free — does not consume credits."""
        ev = self._get(f"sports/{SPORT}/events", {})
        if not isinstance(ev, list):
            raise PropsError(f"unexpected /events payload: {type(ev)}")
        return ev

    def _cache_path(self, event_id: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"props-{event_id}.json"

    def event_board(self, event_id: str) -> dict:
        """Cached prop board for one event.

        Freshness is judged on a `_fetched_at` stamp written INSIDE the file, not on
        the file's mtime. mtime is not a property of the data: `git clone` and
        `git pull` stamp every file they write with the current time, so a board
        cached last week would have looked brand new to an mtime check and been
        served as live market data. (The cache directory is gitignored as of
        2026-09-07, but a cache whose validity depends on the file never being
        copied is fragile regardless of who copies it.)
        """
        cp = self._cache_path(event_id)
        if cp and cp.exists():
            try:
                board = json.loads(cp.read_text())
                stamped = board.get("_fetched_at")
                if stamped:
                    fetched = datetime.fromisoformat(str(stamped).replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - fetched).total_seconds() / 3600.0
                    if 0 <= age_h <= self.max_age_hours:
                        self.cache_hits += 1
                        return board
                    self.stale_cache += 1
            except Exception:
                pass                      # corrupt / unstamped cache: refetch
        board = self._get(f"sports/{SPORT}/events/{event_id}/odds", {
            "regions": "us",
            "markets": ",".join(MARKETS),
            "oddsFormat": "american",
            "bookmakers": ",".join(BOOKMAKERS),
        })
        if not isinstance(board, dict):
            raise PropsError(f"unexpected board payload: {type(board)}")
        board["_fetched_at"] = datetime.now(timezone.utc).isoformat()
        if cp:
            cp.write_text(json.dumps(board))
        return board

    def slate_props(self, slate_games: set[str]) -> dict[str, PlayerProps]:
        """Fetch props for exactly the games on this slate.

        `slate_games` are FanDuel game strings ("TB@CIN"). Events not on the slate are
        never fetched — that is the difference between 72 credits and 96.
        A single event failing degrades that game to FP only; it does not stop the build.
        """
        want: dict[frozenset, str] = {}
        for g in slate_games:
            if "@" in g:
                a, h = g.split("@", 1)
                want[frozenset({norm_team(a), norm_team(h)})] = g

        merged: dict[str, PlayerProps] = {}
        matched_games: set[str] = set()
        for ev in self.events():
            home = norm_team(TEAM_ABBR.get(ev.get("home_team", ""), ""))
            away = norm_team(TEAM_ABBR.get(ev.get("away_team", ""), ""))
            keyset = frozenset({away, home})
            if keyset not in want:
                continue
            try:
                board = self.event_board(ev["id"])
                parsed = parse_event_board(board)
            except PropsError as e:
                self.errors.append(f"{want[keyset]}: {e}")
                continue
            merged.update(parsed)
            matched_games.add(want[keyset])

        for g in sorted(set(want.values()) - matched_games):
            if not any(g in e for e in self.errors):
                self.errors.append(f"{g}: no event on the props board")
        return merged


# ---------------------------------------------------------------------------
# Scoring + scale anchoring
# ---------------------------------------------------------------------------

def _merged_stats(pp: PlayerProps, fp_stats: dict, position: str) -> dict:
    """Props stat line, with props-uncovered categories taken from FantasyPros.

    Props never cover interceptions, fumbles, return TDs or two-point conversions.
    Leaving them at zero biases QBs up by roughly a point, so they come from FP.
    """
    s = dict(pp.stats)
    s.pop("anytime_td_prob", None)
    non_pass_td = MULTI_TD_FACTOR * float(pp.stats.get("anytime_td_prob", 0.0))
    if pp.has("anytime_td"):
        # Both rush and rec TDs score 6, so the split does not affect points. Put
        # them where the position expects them for a readable breakdown.
        if position in ("WR", "TE"):
            s["rec_tds"] = round(non_pass_td, 3)
        else:
            s["rush_tds"] = round(non_pass_td, 3)
    for k in ("pass_ints", "fumbles", "ret_tds", "2pt_tds"):
        if fp_stats.get(k) is not None:
            s[k] = fp_stats.get(k)
    return s


@dataclass
class PropsReport:
    covered: int = 0
    total_considered: int = 0
    scale: dict = field(default_factory=dict)        # position -> factor applied
    sample: dict = field(default_factory=dict)       # position -> what the factor was fit on
    rejected: list = field(default_factory=list)     # positions discarded by the gate
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    credits_spent: int = 0
    cache_hits: int = 0
    stale_cache: int = 0
    board_age_h: Optional[float] = None

    def summary(self) -> str:
        lines = [f"Props: {self.covered}/{self.total_considered} skill players priced "
                 f"({self.credits_spent} credits, {self.cache_hits} cached boards"
                 + (f", {self.stale_cache} refetched as stale" if self.stale_cache else "")
                 + ")"]
        if self.board_age_h is not None:
            lines.append(f"  board age: {self.board_age_h:.1f}h")
        if self.scale:
            lines.append("  scale to FP level: " +
                         ", ".join(f"{p} x{v:.3f} ({self.sample.get(p, '?')})"
                                   for p, v in sorted(self.scale.items())))
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        for r in self.rejected:
            lines.append(f"  REJECTED (props unused): {r}")
        for e in self.errors[:6]:
            lines.append(f"  board gap: {e}")
        return "\n".join(lines)


def match_props_to_slate(slate_players: list, props: dict[str, PlayerProps]) -> dict:
    """fd_id -> PlayerProps.

    Matching is scoped by the event's two teams, which makes it far safer than
    slate-wide name matching: 'B. Robinson' can only collide with players in the same
    game. Falls back to first-initial+surname only when first names are compatible —
    the same guard matching.py needed after Bijan/Brian Robinson.
    """
    by_name = {k: v for k, v in props.items()}
    by_short: dict[str, list[PlayerProps]] = {}
    for v in props.values():
        by_short.setdefault(short_key(v.name), []).append(v)

    out: dict[str, PlayerProps] = {}
    for sp in slate_players:
        if sp.position == "D":
            continue
        team = norm_team(sp.team)
        cand = by_name.get(norm_name(sp.name))
        if cand is None:
            pool = [c for c in by_short.get(short_key(sp.name), [])
                    if team in c.event_teams
                    and _first_names_compatible(sp.name, c.name)]
            cand = pool[0] if len(pool) == 1 else None
        if cand is None or team not in cand.event_teams:
            continue
        out[sp.fd_id] = cand
    return out


def props_points(slate_players: list, props: dict[str, PlayerProps],
                 fp_by_id: dict) -> tuple[dict, PropsReport]:
    """Score matched props into FanDuel points and anchor each position to the FP level.

    Returns (fd_id -> anchored props points, report). Positions whose scale factor
    fails the reject gate are dropped entirely: an extreme factor means the board or
    the parser changed shape, and a wrong number is more dangerous than no number.
    """
    rep = PropsReport(credits_spent=0)
    matched = match_props_to_slate(slate_players, props)
    raw: dict[str, float] = {}
    ratios: dict[str, list[float]] = {}
    considered = 0

    for sp in slate_players:
        if sp.position not in PRIMARY_MARKET:
            continue
        considered += 1
        pp = matched.get(sp.fd_id)
        if pp is None or not pp.has(PRIMARY_MARKET[sp.position]):
            continue
        fp = fp_by_id.get(sp.fd_id)
        fp_stats = getattr(fp, "stats", {}) or {}
        pts, _bd = score(_merged_stats(pp, fp_stats, sp.position), sp.position)
        if pts <= 0:
            continue
        raw[sp.fd_id] = round(pts, 2)
        fp_pts = float(getattr(fp, "points", 0.0) or 0.0)
        if fp_pts > 3.0 and pts > 3.0:
            ratios.setdefault(sp.position, []).append(fp_pts / pts)

    rep.total_considered = considered
    all_ratios = [r for v in ratios.values() for r in v]
    global_scale = float(median(all_ratios)) if all_ratios else 1.0

    # Only positions that actually produced a priced player get a scale factor. An
    # earlier version looped over every position in PRIMARY_MARKET and reported a
    # rejection for positions with nobody on the slate, which buried the one real
    # rejection in three phantom ones.
    pos_present = {sp.position for sp in slate_players if sp.fd_id in raw}
    scale: dict[str, float] = {}
    for pos in sorted(pos_present):
        vals = ratios.get(pos, [])
        if len(vals) >= MIN_SCALE_SAMPLE:
            f, basis = float(median(vals)), f"n={len(vals)}"
        elif len(all_ratios) >= MIN_GLOBAL_SAMPLE:
            f, basis = global_scale, f"all-positions n={len(all_ratios)}"
            rep.warnings.append(f"{pos} has only {len(vals)} priced players — scaled "
                                f"on the all-positions factor")
        else:
            rep.rejected.append(f"{pos} scale sample too small "
                                f"({len(vals)} in position, {len(all_ratios)} overall) "
                                "— props unused rather than mis-scaled")
            continue
        rep.sample[pos] = basis
        if not (SCALE_REJECT[0] <= f <= SCALE_REJECT[1]):
            rep.rejected.append(f"{pos} scale x{f:.2f} outside "
                                f"[{SCALE_REJECT[0]}, {SCALE_REJECT[1]}] — "
                                "treated as a board/parser failure")
            continue
        if not (SCALE_WARN[0] <= f <= SCALE_WARN[1]):
            rep.warnings.append(f"{pos} props sit {abs(1 - 1 / f) * 100:.0f}% "
                                f"{'below' if f > 1 else 'above'} FP consensus "
                                f"(scale x{f:.2f}) — applied, but verify the board")
        scale[pos] = f
    rep.scale = {p: round(v, 4) for p, v in scale.items()}

    pos_by_id = {sp.fd_id: sp.position for sp in slate_players}
    out = {}
    for fid, pts in raw.items():
        pos = pos_by_id.get(fid)
        if pos not in scale:
            continue
        out[fid] = round(pts * scale[pos], 2)
    rep.covered = len(out)

    ages = [p.last_update for p in props.values() if p.last_update]
    if ages:
        try:
            newest = max(ages)
            dt = datetime.fromisoformat(newest.replace("Z", "+00:00"))
            rep.board_age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
        except Exception:
            pass
    return out, rep
