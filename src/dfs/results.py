"""Result logging and season standings.

This closes the loop, and under Total Scores it is not bookkeeping — it is an INPUT.
The objective weights depend on where the season stands (weeks_played, my_points,
leader_points), so the optimizer cannot weight Week 8 correctly unless Weeks 1-7 were
recorded. It also produces the dataset 2025 never had: entered lineup, projection at
lock time, and actual outcome, per player, per week.

Three things get logged:
  1. entered lineups   — what we submitted, with the projection we believed at lock
  2. actual results    — per-player actual FanDuel points (FantasyPros points endpoint)
  3. contest outcomes  — final rank, score, winnings, plus opponents from contest_parse

From those we get projection error tracking (is the model degrading mid-season?),
season standings (feeds SeasonContext), and eventually the opponent model.
"""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    contest TEXT NOT NULL,
    slate_id TEXT,
    submitted_ts TEXT NOT NULL,
    objective TEXT,
    exp_points REAL,
    p_win REAL,
    salary INTEGER,
    lineup_json TEXT NOT NULL,
    actual_score REAL,
    final_rank INTEGER,
    field_size INTEGER,
    winnings REAL,
    PRIMARY KEY (season, week, contest)
);
CREATE TABLE IF NOT EXISTS player_results (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    fd_id TEXT NOT NULL,
    name TEXT NOT NULL,
    position TEXT,
    team TEXT,
    salary INTEGER,
    projection REAL,
    actual REAL,
    in_lineup INTEGER DEFAULT 0,
    PRIMARY KEY (season, week, fd_id)
);
CREATE TABLE IF NOT EXISTS opponent_entries (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    contest TEXT NOT NULL,
    entrant TEXT NOT NULL,
    score REAL,
    rank INTEGER,
    lineup_json TEXT,
    PRIMARY KEY (season, week, contest, entrant)
);
CREATE TABLE IF NOT EXISTS ownership (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    contest TEXT NOT NULL,
    player TEXT NOT NULL,
    drafted_pct REAL NOT NULL,
    PRIMARY KEY (season, week, contest, player)
);
"""


@dataclass
class Standing:
    entrant: str
    total_points: float = 0.0
    weeks: int = 0
    wins: int = 0
    best: float = 0.0

    @property
    def avg(self) -> float:
        return self.total_points / self.weeks if self.weeks else 0.0


class ResultLog:
    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    def _c(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- writes ----

    def log_entry(self, season: int, week: int, contest: str, players: list,
                  slate_id: str = "", objective: str = "", exp_points: float = 0.0,
                  p_win: float = 0.0, mvp_id: str | None = None,
                  mvp_salary_mult: float = 1.5) -> None:
        """Record a submitted lineup WITH the projections believed at lock time.

        Showdown: the MVP is part of the lineup's identity (scores 1.5x, costs 1.5x),
        so it is stored per-player in lineup_json and the salary column records the
        CHARGED total (base + MVP premium), which is what FanDuel validated."""
        lineup = [{"fd_id": p.fd_id, "name": p.name, "pos": p.position, "team": p.team,
                   "salary": p.salary, "projection": p.projection,
                   "proj_source": p.proj_source, "implied_total": p.implied_team_total,
                   "mvp": p.fd_id == mvp_id,
                   # injury status AT LOG TIME — the card badges these so a Q player
                   # anchoring the entry is visible on the same screen as the lineup
                   # (measured 2026-08-30: a week of "Week 1 in doubt" reporting moved
                   # a Q player's consensus projection by 0.1 pts — consensus is slow,
                   # the human needs to see the flag)
                   "inj": (getattr(p, "injury_indicator", "") or ""),
                   "inj_detail": (getattr(p, "injury_details", "") or "")}
                  for p in players]
        charged = sum(p.salary for p in players)
        if mvp_id:
            charged += int(round((mvp_salary_mult - 1.0)
                                 * next((p.salary for p in players
                                         if p.fd_id == mvp_id), 0)))
        with self._c() as c:
            c.execute("""INSERT OR REPLACE INTO entries
                (season,week,contest,slate_id,submitted_ts,objective,exp_points,p_win,
                 salary,lineup_json,actual_score,final_rank,field_size,winnings)
                VALUES (?,?,?,?,?,?,?,?,?,?,
                    (SELECT actual_score FROM entries WHERE season=? AND week=? AND contest=?),
                    (SELECT final_rank   FROM entries WHERE season=? AND week=? AND contest=?),
                    (SELECT field_size   FROM entries WHERE season=? AND week=? AND contest=?),
                    (SELECT winnings     FROM entries WHERE season=? AND week=? AND contest=?))""",
                (season, week, contest, slate_id,
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), objective,
                 exp_points, p_win, charged, json.dumps(lineup),
                 season, week, contest, season, week, contest,
                 season, week, contest, season, week, contest))
            for p in players:
                c.execute("""INSERT OR REPLACE INTO player_results
                    (season,week,fd_id,name,position,team,salary,projection,actual,in_lineup)
                    VALUES (?,?,?,?,?,?,?,?,
                        (SELECT actual FROM player_results WHERE season=? AND week=? AND fd_id=?),1)""",
                    (season, week, p.fd_id, p.name, p.position, p.team, p.salary,
                     p.projection, season, week, p.fd_id))

    def log_outcome(self, season: int, week: int, contest: str, score: float,
                    rank: int, field_size: int, winnings: float = 0.0) -> None:
        with self._c() as c:
            c.execute("""UPDATE entries SET actual_score=?, final_rank=?, field_size=?, winnings=?
                         WHERE season=? AND week=? AND contest=?""",
                      (score, rank, field_size, winnings, season, week, contest))

    def log_capture(self, capture) -> None:
        """Ingest a parsed contest page: opponents, scores, and measured ownership."""
        with self._c() as c:
            for e in capture.leaderboard:
                lu = next((x for x in capture.entries_with_lineups
                           if x.entrant == e.entrant), None)
                c.execute("""INSERT OR REPLACE INTO opponent_entries
                    (season,week,contest,entrant,score,rank,lineup_json) VALUES (?,?,?,?,?,?,?)""",
                    (capture.season, capture.week, capture.contest, e.entrant, e.score, e.rank,
                     json.dumps([{"pos": p.position, "name": p.name,
                                  "own": p.drafted_pct, "pts": p.actual_points}
                                 for p in lu.players]) if lu else None))
            for player, pct in capture.ownership().items():
                c.execute("""INSERT OR REPLACE INTO ownership
                    (season,week,contest,player,drafted_pct) VALUES (?,?,?,?,?)""",
                    (capture.season, capture.week, capture.contest, player, pct))

    def log_player_actuals(self, season: int, week: int,
                           actuals: dict[str, float]) -> int:
        """actuals: fd_id -> actual FanDuel points."""
        n = 0
        with self._c() as c:
            for fd_id, pts in actuals.items():
                cur = c.execute("""UPDATE player_results SET actual=?
                                   WHERE season=? AND week=? AND fd_id=?""",
                                (pts, season, week, fd_id))
                n += cur.rowcount
        return n

    # ---- reads ----

    def standings(self, season: int, contest_like: str = "%") -> list[Standing]:
        """Season standings from logged opponent entries. Feeds SeasonContext."""
        acc: dict[str, Standing] = {}
        with self._c() as c:
            rows = c.execute("""SELECT entrant, score, rank FROM opponent_entries
                                WHERE season=? AND contest LIKE ? AND score IS NOT NULL""",
                             (season, contest_like)).fetchall()
        for r in rows:
            s = acc.setdefault(r["entrant"], Standing(entrant=r["entrant"]))
            s.total_points += r["score"]
            s.weeks += 1
            s.best = max(s.best, r["score"])
            if r["rank"] == 1:
                s.wins += 1
        return sorted(acc.values(), key=lambda s: -s.total_points)

    def season_context(self, season: int, me: str, weeks_total: int = 21,
                       contest_like: str = "%"):
        """Build a SeasonContext from logged results — drives objective weights."""
        from .objectives import SeasonContext, Leaderboard
        st = self.standings(season, contest_like)
        if not st:
            return SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES,
                                 weeks_total=weeks_total, weeks_played=0)
        mine = next((s for s in st if s.entrant == me), Standing(entrant=me))
        leader = st[0]
        return SeasonContext(
            leaderboard=Leaderboard.TOTAL_SCORES, weeks_total=weeks_total,
            weeks_played=mine.weeks, my_points=mine.total_points,
            leader_points=leader.total_points, my_wins=mine.wins,
            leader_wins=leader.wins, field_size=max(len(st), 2))

    def projection_accuracy(self, season: int, min_week: int = 1) -> dict:
        """Is the model holding up in-season? Compares locked projections to actuals."""
        with self._c() as c:
            rows = c.execute("""SELECT position, projection, actual FROM player_results
                                WHERE season=? AND week>=? AND actual IS NOT NULL
                                  AND projection IS NOT NULL""",
                             (season, min_week)).fetchall()
        if not rows:
            return {}
        import numpy as np
        out: dict = {}
        for pos in ("QB", "RB", "WR", "TE", "D", "ALL"):
            sel = [r for r in rows if pos == "ALL" or r["position"] == pos]
            if len(sel) < 5:
                continue
            p = np.array([r["projection"] for r in sel])
            a = np.array([r["actual"] for r in sel])
            out[pos] = {"n": len(sel), "mae": round(float(np.abs(a - p).mean()), 2),
                        "bias": round(float((a - p).mean()), 2),
                        "corr": round(float(np.corrcoef(p, a)[0, 1]), 3) if len(sel) > 2 else None}
        return out

    def measured_ownership(self, season: int, weeks: int = 0,
                           contest_like: str = "%") -> dict[str, float]:
        """Average measured ownership by player across logged weeks.

        contest_like scopes the model: league builds must learn ONLY from league
        contests — a public H2H's ownership says nothing about how the family drafts.
        """
        with self._c() as c:
            q = ("SELECT player, AVG(drafted_pct) v FROM ownership "
                 "WHERE season=? AND contest LIKE ?")
            args = [season, contest_like]
            if weeks:
                q += " AND week>=?"
                args.append(weeks)
            rows = c.execute(q + " GROUP BY player", args).fetchall()
        return {r["player"]: round(r["v"], 2) for r in rows}

    def ownership_week_count(self, season: int, contest_like: str = "%") -> int:
        with self._c() as c:
            return c.execute("SELECT COUNT(DISTINCT week) FROM ownership "
                             "WHERE season=? AND contest LIKE ?",
                             (season, contest_like)).fetchone()[0]
