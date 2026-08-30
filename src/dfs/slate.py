"""Canonical PlayerSlate. FanDuel player ID is the join key — never name-matching downstream.

Fail-loud rule: a slate with missing critical data raises SlateError; nothing silently
substitutes FPPG or salary-derived projections.
"""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .contest_spec import SlateType


class SlateError(Exception):
    """Raised when a slate is unusable. Build stops. No silent fallbacks."""


@dataclass
class SlatePlayer:
    fd_id: str                 # FanDuel Id column — canonical key
    name: str
    position: str              # QB/RB/WR/TE/D
    team: str
    opponent: str
    salary: int
    game: str                  # e.g. "PHI@DAL"
    injury_indicator: str = ""
    injury_details: str = ""
    injury_source: str = ""    # "fanduel_csv" until an injury sweep overwrites it
    injury_ts: str = ""        # when that source last updated the record
    roster_position: str = ""  # FD "Roster Position" col (showdown: MVP/UTIL etc.)
    fppg: float = 0.0          # kept ONLY as a sanity-check reference, never a projection
    # Filled by projection engine (blend.py). None = not yet projected.
    projection: Optional[float] = None
    proj_source: Optional[str] = None
    proj_ts: Optional[str] = None
    floor_p10: Optional[float] = None
    ceiling_p90: Optional[float] = None
    implied_team_total: Optional[float] = None


@dataclass
class PlayerSlate:
    slate_id: str              # e.g. "2026-w01-main"
    slate_type: SlateType
    season: int
    week: int
    created_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_csv: str = ""
    players: list[SlatePlayer] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return list of problems. Empty list = valid."""
        problems: list[str] = []
        if not self.players:
            problems.append("slate has zero players")
            return problems
        ids = [p.fd_id for p in self.players]
        if len(ids) != len(set(ids)):
            problems.append("duplicate FanDuel player IDs")
        teams = {p.team for p in self.players}
        if self.slate_type == SlateType.SINGLE_GAME and len(teams) != 2:
            problems.append(f"single-game slate must have exactly 2 teams, found {len(teams)}: {sorted(teams)}")
        if self.slate_type == SlateType.FULL and len(teams) < 8:
            problems.append(f"full slate has only {len(teams)} teams — wrong CSV?")
        pos_counts: dict[str, int] = {}
        for p in self.players:
            pos_counts[p.position] = pos_counts.get(p.position, 0) + 1
        if self.slate_type == SlateType.FULL:
            for pos, minimum in (("QB", 8), ("RB", 20), ("WR", 30), ("TE", 10), ("D", 8)):
                if pos_counts.get(pos, 0) < minimum:
                    problems.append(f"only {pos_counts.get(pos, 0)} {pos}s — expected >= {minimum} on a full slate")
        bad_sal = [p.name for p in self.players if p.salary < 3000 or p.salary > 20000]
        if bad_sal:
            problems.append(f"salaries out of range for: {bad_sal[:5]}")
        return problems

    def require_projections(self) -> None:
        missing = [p.name for p in self.players if p.projection is None]
        if missing:
            raise SlateError(
                f"{len(missing)} players lack projections (e.g. {missing[:5]}). "
                "Build stopped — no FPPG/salary fallback exists by design."
            )


# ---------------- persistence ----------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS slates (
    slate_id TEXT PRIMARY KEY,
    slate_type TEXT NOT NULL,
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    created_ts TEXT NOT NULL,
    source_csv TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    slate_id TEXT NOT NULL,
    fd_id TEXT NOT NULL,
    actual_points REAL,
    PRIMARY KEY (slate_id, fd_id)
);
CREATE TABLE IF NOT EXISTS entered_lineups (
    slate_id TEXT NOT NULL,
    contest_name TEXT NOT NULL,
    entrant TEXT NOT NULL,          -- 'brett' or opponent handle (opponent-model data)
    lineup_json TEXT NOT NULL,
    actual_score REAL,
    final_rank INTEGER,
    PRIMARY KEY (slate_id, contest_name, entrant)
);
"""


class SlateStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save(self, slate: PlayerSlate) -> None:
        payload = json.dumps({**asdict(slate), "slate_type": slate.slate_type.value})
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO slates VALUES (?,?,?,?,?,?,?)",
                (slate.slate_id, slate.slate_type.value, slate.season, slate.week,
                 slate.created_ts, slate.source_csv, payload),
            )

    def load(self, slate_id: str) -> PlayerSlate:
        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM slates WHERE slate_id=?", (slate_id,)).fetchone()
        if not row:
            raise SlateError(f"slate {slate_id!r} not found")
        d = json.loads(row[0])
        players = [SlatePlayer(**p) for p in d.pop("players")]
        d["slate_type"] = SlateType(d["slate_type"])
        return PlayerSlate(**d, players=players)
