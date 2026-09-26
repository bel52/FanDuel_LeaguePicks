"""Whole-pool outcomes from nflverse: FanDuel points and played/inactive for EVERY
projected player, not just the ones an opponent happened to roster.

Why this exists
---------------
Until 2026-09-26 the only actuals came from the contest results page, which shows
the ~50-60 players somebody in a 12-person league rostered. That sample is the worst
possible one for measuring a projection model: it is almost entirely high-projection
starters, and it is chosen by the same humans whose picks the model competes with.
Concrete failures it caused by Week 3:
  * the Week 1 shadow arm was permanently ungradable (Emeka Egbuka was never on a
    results page, so his actual did not exist);
  * component accuracy (FP vs market vs blend) sat at n=70 after two weeks;
  * two actuals were WRONG in results.db — cells transcribed by hand from an
    obscured screenshot. nflverse scored them correctly.

Validation (2026-09-26): nflverse stats scored through this module reproduced the
FanDuel contest page to the hundredth for 50 of 52 Week-2 players, including all six
defenses. The two misses were the two hand-inferred cells, where nflverse was right.

Rules that keep this from corrupting the dataset
------------------------------------------------
1. Only games that are FINAL and whose team stats are published are graded. A week
   in progress leaves unfinished games NULL; the next build or capture fills them.
2. Players are joined through nflverse's own id (gsis_id), never name-to-name
   across two tables: name variants map to a roster id, and the id carries the
   stats. A roster player with no stat row genuinely recorded nothing -> 0.0.
3. A pool player who cannot be matched to a roster is left NULL and reported. A
   false 0.0 is worse than a missing value.
4. `played` comes from snap counts when that game's snaps are published. It
   overrides the post-lock CSV flag, which cannot see a late scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from .matching import norm_name, norm_team, short_key

SKILL = ("QB", "RB", "WR", "TE")

# FanDuel DST points-allowed ladder (mirrors scoring.DST_PA_LADDER, applied to a
# realised score rather than a projected mean).
from .scoring import (DST_PA_LADDER, DST_SACK, DST_INT, DST_FUM_REC, DST_TD,
                      DST_SAFETY)


def _n(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f            # NaN -> 0


def skill_points(r: dict) -> float:
    """FanDuel points for a QB/RB/WR/TE nflverse weekly stat row."""
    pass_yds, rush_yds, rec_yds = (_n(r.get("passing_yards")), _n(r.get("rushing_yards")),
                                   _n(r.get("receiving_yards")))
    pts = (pass_yds * 0.04 + _n(r.get("passing_tds")) * 4.0
           - _n(r.get("passing_interceptions")) * 1.0
           + (3.0 if pass_yds >= 300 else 0.0)
           + rush_yds * 0.1 + _n(r.get("rushing_tds")) * 6.0
           + (3.0 if rush_yds >= 100 else 0.0)
           + _n(r.get("receptions")) * 0.5 + rec_yds * 0.1
           + _n(r.get("receiving_tds")) * 6.0 + (3.0 if rec_yds >= 100 else 0.0)
           + (_n(r.get("passing_2pt_conversions")) + _n(r.get("rushing_2pt_conversions"))
              + _n(r.get("receiving_2pt_conversions"))) * 2.0
           - (_n(r.get("sack_fumbles_lost")) + _n(r.get("rushing_fumbles_lost"))
              + _n(r.get("receiving_fumbles_lost"))) * 2.0
           # return TDs and an offensive player recovering a fumble for a TD
           + (_n(r.get("special_teams_tds")) + _n(r.get("fumble_recovery_tds"))) * 6.0)
    return round(pts, 2)


def _ladder(points_allowed: float) -> float:
    pa = int(round(points_allowed))
    for lo, hi, pts in DST_PA_LADDER:
        if lo <= pa <= hi:
            return pts
    return DST_PA_LADDER[-1][2]


def dst_points(team_row: dict, opp_row: dict, points_allowed: float) -> float:
    """FanDuel DST points. Sacks and interceptions come from the OPPONENT's offense:
    nflverse credits sacks to individual defenders and split sacks can sum short of
    the team total (Carolina W2 2026: defenders 2.0, FanDuel 3)."""
    pts = (_n(opp_row.get("sacks_suffered")) * DST_SACK
           + _n(opp_row.get("passing_interceptions")) * DST_INT
           + _n(team_row.get("fumble_recovery_opp")) * DST_FUM_REC
           + (_n(team_row.get("def_tds")) + _n(team_row.get("special_teams_tds"))) * DST_TD
           + _n(team_row.get("def_safeties")) * DST_SAFETY
           + _ladder(points_allowed))
    return round(pts, 2)


@dataclass
class WeekOutcomes:
    season: int
    week: int
    final_teams: set = field(default_factory=set)      # graded: final + stats loaded
    pending_teams: set = field(default_factory=set)    # on the schedule, not final
    points: dict = field(default_factory=dict)         # gsis_id -> FD points
    name_index: dict = field(default_factory=dict)     # (norm_name, team) -> {gsis}
    name_only: dict = field(default_factory=dict)      # norm_name -> {(gsis, team)}
    short_index: dict = field(default_factory=dict)    # (short_key, team, pos) -> {gsis}
    gsis_team: dict = field(default_factory=dict)      # gsis_id -> team
    dst: dict = field(default_factory=dict)            # team -> FD points
    played: dict = field(default_factory=dict)         # gsis_id -> bool
    snap_teams: set = field(default_factory=set)


class _NflverseLoader:
    """Thin seam over nflreadpy so tests can inject frames. Each loader returns a
    pandas DataFrame for the whole season; lru_cache keeps one download per run."""

    @staticmethod
    @lru_cache(maxsize=8)
    def _load(kind: str, season: int):
        import nflreadpy as nfl
        fn = {"player_stats": nfl.load_player_stats, "team_stats": nfl.load_team_stats,
              "schedules": nfl.load_schedules, "rosters": nfl.load_rosters_weekly,
              "snaps": nfl.load_snap_counts}[kind]
        return fn([season]).to_pandas()

    def player_stats(self, season):  return self._load("player_stats", season)
    def team_stats(self, season):    return self._load("team_stats", season)
    def schedules(self, season):     return self._load("schedules", season)
    def rosters(self, season):       return self._load("rosters", season)
    def snaps(self, season):         return self._load("snaps", season)


def _rows(df, week: int) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    sub = df[(df["week"] == week)]
    if "season_type" in sub.columns:
        sub = sub[sub["season_type"].isin(["REG", "POST"])]
    elif "game_type" in sub.columns:
        sub = sub[sub["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])]
    return sub.to_dict("records")


def load_week(season: int, week: int, loader=None) -> WeekOutcomes:
    ld = loader or _NflverseLoader()
    out = WeekOutcomes(season=season, week=week)

    # 1. which games are final
    score_for: dict[str, float] = {}
    for g in _rows(ld.schedules(season), week):
        home, away = norm_team(g.get("home_team")), norm_team(g.get("away_team"))
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None or hs != hs or as_ != as_:      # None / NaN
            out.pending_teams |= {home, away}
            continue
        score_for[home], score_for[away] = float(hs), float(as_)

    # 2. team stats published for those games -> graded teams + DST points
    team_rows = {norm_team(r.get("team")): r for r in _rows(ld.team_stats(season), week)}
    for team, r in team_rows.items():
        opp = norm_team(r.get("opponent_team"))
        if team not in score_for or opp not in team_rows:
            continue
        out.final_teams.add(team)
        out.dst[team] = dst_points(r, team_rows[opp], score_for[opp])

    # 3. roster = the id backbone (every player who could have appeared)
    pfr_to_gsis: dict[str, str] = {}
    for r in _rows(ld.rosters(season), week):
        gsis, team = r.get("gsis_id"), norm_team(r.get("team"))
        if not gsis or not isinstance(gsis, str):
            continue
        out.gsis_team[gsis] = team
        pos = str(r.get("position") or "")
        first = r.get("football_name") or r.get("first_name") or ""
        variants = {r.get("full_name"), f"{first} {r.get('last_name') or ''}"}
        for v in variants:
            if isinstance(v, str) and v.strip():
                k = norm_name(v)
                out.name_index.setdefault((k, team), set()).add(gsis)
                out.name_only.setdefault(k, set()).add((gsis, team))
                out.short_index.setdefault((short_key(v), team, pos), set()).add(gsis)
        if isinstance(r.get("pfr_id"), str) and r.get("pfr_id"):
            pfr_to_gsis[r["pfr_id"]] = gsis

    # 4. stat rows, keyed by the same id; display names widen the name index too
    for r in _rows(ld.player_stats(season), week):
        gsis, team = r.get("player_id"), norm_team(r.get("team"))
        if not isinstance(gsis, str) or team not in out.final_teams:
            continue
        if str(r.get("position") or "") in SKILL or r.get("position_group") in SKILL:
            out.points[gsis] = skill_points(r)
        out.gsis_team.setdefault(gsis, team)
        dn = r.get("player_display_name")
        if isinstance(dn, str) and dn.strip():
            out.name_index.setdefault((norm_name(dn), team), set()).add(gsis)
            out.name_only.setdefault(norm_name(dn), set()).add((gsis, team))

    # 5. snap counts -> played
    for r in _rows(ld.snaps(season), week):
        team = norm_team(r.get("team"))
        if team not in out.final_teams:
            continue
        out.snap_teams.add(team)
        gsis = pfr_to_gsis.get(r.get("pfr_player_id"))
        if gsis:
            snaps = (_n(r.get("offense_snaps")) + _n(r.get("defense_snaps"))
                     + _n(r.get("st_snaps")))
            out.played[gsis] = out.played.get(gsis, False) or snaps > 0
    return out


@dataclass
class Resolution:
    fd_id: str
    name: str
    position: str
    team: str
    salary: int
    points: float | None = None
    played: int | None = None
    how: str = ""                # stat | roster-zero | dst | team-moved | short-key


def resolve(pool_rows: list[dict], wo: WeekOutcomes) -> tuple[list[Resolution], list[dict]]:
    """Map the week's projected pool onto nflverse outcomes.

    Returns (resolved, unmatched). Players whose game is not final are neither —
    they are simply not graded yet."""
    resolved, unmatched = [], []
    for r in pool_rows:
        pos, team = str(r.get("position") or ""), norm_team(r.get("team") or "")
        res = Resolution(fd_id=r["fd_id"], name=r["name"], position=pos, team=team,
                         salary=int(r.get("salary") or 0))
        if pos in ("D", "DST", "DEF"):
            if team in wo.final_teams:
                res.points, res.how = wo.dst[team], "dst"
                resolved.append(res)
            continue
        if team not in wo.final_teams:
            continue
        k = norm_name(r["name"])
        ids = wo.name_index.get((k, team), set())
        how = ""
        if len(ids) == 1:
            how = "id"
        else:
            # traded / signed since the slate priced him: same name, one final team
            cands = {(g, t) for g, t in wo.name_only.get(k, set()) if t in wo.final_teams}
            if len({g for g, _ in cands}) == 1:
                ids, how = {next(iter(cands))[0]}, "team-moved"
            else:
                ids = wo.short_index.get((short_key(r["name"]), team, pos), set())
                how = "short-key" if len(ids) == 1 else ""
        if len(ids) != 1:
            unmatched.append({"name": r["name"], "team": team, "position": pos,
                              "salary": res.salary})
            continue
        gsis = next(iter(ids))
        if gsis in wo.points:
            res.points, res.how = wo.points[gsis], (how if how != "id" else "stat")
        else:
            res.points, res.how = 0.0, "roster-zero"
        g_team = wo.gsis_team.get(gsis, team)
        if g_team in wo.snap_teams:
            res.played = 1 if wo.played.get(gsis, False) else 0
        resolved.append(res)
    return resolved, unmatched


@dataclass
class IngestReport:
    week: int
    pool: int = 0
    written: int = 0
    zeros: int = 0
    played_set: int = 0
    pending_games: int = 0
    unmatched: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    shadows_graded: list = field(default_factory=list)
    error: str = ""

    def line(self) -> str:
        if self.error:
            return f"  week {self.week}: nflverse unavailable ({self.error}) — will retry next run"
        s = (f"  week {self.week}: {self.written}/{self.pool} pool players graded "
             f"({self.zeros} on a roster with no stats -> 0), played set on {self.played_set}")
        if self.pending_games:
            s += f"; {self.pending_games} team(s) not final yet"
        return s

    def details(self, report_salary: int = 5000) -> list[str]:
        out = []
        big = [u for u in self.unmatched if u["salary"] >= report_salary]
        if big:
            out.append(f"    unmatched >= ${report_salary} (left ungraded, never zeroed):")
            for u in sorted(big, key=lambda x: -x["salary"])[:10]:
                out.append(f"      ${u['salary']:5d} {u['position']:3s} {u['team']:4s} {u['name']}")
        if self.conflicts:
            out.append("    contest page vs nflverse disagree (nflverse used; a mismatch "
                       "usually means a capture transcription error):")
            for c in self.conflicts[:10]:
                out.append(f"      {c['name']:24s} page {c['fd']:6.2f}  nflverse {c['nfl']:6.2f}")
        for g in self.shadows_graded:
            out.append(f"    graded {g['contest']}: {g['score']:.2f}")
        return out


def ingest(rl, season: int, weeks: list[int], loader=None) -> list[IngestReport]:
    """Grade every projected player for each week, then grade any shadow entry whose
    nine players are all graded. Safe to re-run; never raises on a feed failure."""
    reports = []
    for wk in weeks:
        rep = IngestReport(week=wk)
        try:
            wo = load_week(season, wk, loader)
        except Exception as e:                       # network, schema, anything
            rep.error = f"{type(e).__name__}: {e}"[:160]
            reports.append(rep)
            continue
        with rl._c() as c:
            pool = [dict(r) for r in c.execute(
                """SELECT fd_id, name, position, team, salary FROM player_results
                   WHERE season=? AND week=?""", (season, wk)).fetchall()]
        rep.pool = len(pool)
        rep.pending_games = len(wo.pending_teams)
        resolved, rep.unmatched = resolve(pool, wo)
        rep.written, rep.played_set = rl.log_nflverse_outcomes(season, wk, resolved)
        rep.zeros = sum(1 for x in resolved if x.how == "roster-zero")
        rep.conflicts = rl.actual_conflicts(season, wk)
        rep.shadows_graded = rl.grade_shadow_entries(season, wk)
        reports.append(rep)
    return reports
