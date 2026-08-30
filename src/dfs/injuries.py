"""Injury / inactives pipeline.

Why this is the highest-value operational feature: a player who is inactive scores 0.
Under Total Scores that zero is permanent damage to the season standing; under Most
Wins it ends the week outright. Nothing else in the system protects against a loss
that large for so little effort.

Three layers, increasing in freshness:

  1. FanDuel CSV "Injury Indicator"  — as of CSV download (Wed). Already handled at
     ingest: O/IR/NA/SUSP rows are dropped and reported.
  2. FantasyPros injuries endpoint   — practice participation and game status, updated
     through the week. Q/D/O with practice trend.
  3. Sunday inactives sweep          — ~90 min before kickoff, official actives lists.
     This is where the real money is; the FD CSV is 4 days stale by then.

Statuses are mapped to an ACTION, never to a silent projection haircut. v5 quietly
multiplied projections by fudge factors for injury status; that hid the decision. Here
a questionable player either stays (with the risk stated) or is flagged for the human.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .matching import norm_name, norm_team


class Status(str, Enum):
    ACTIVE = "active"
    PROBABLE = "probable"        # full practice, expected to play
    QUESTIONABLE = "questionable"
    DOUBTFUL = "doubtful"
    OUT = "out"
    IR = "ir"
    UNKNOWN = "unknown"


class Action(str, Enum):
    KEEP = "keep"
    FLAG = "flag"                # human decision required before lock
    REMOVE = "remove"            # cannot play; must not be in a lineup


# Status -> action. Deliberately conservative: DOUBTFUL is treated as REMOVE because
# historical doubtful-tag play rates are low and a zero is unrecoverable.
ACTION_FOR = {
    Status.ACTIVE: Action.KEEP,
    Status.PROBABLE: Action.KEEP,
    Status.QUESTIONABLE: Action.FLAG,
    Status.DOUBTFUL: Action.REMOVE,
    Status.OUT: Action.REMOVE,
    Status.IR: Action.REMOVE,
    Status.UNKNOWN: Action.KEEP,
}

_STATUS_PATTERNS = [
    (re.compile(r"\b(out|inactive|ruled out|will not play|did not travel)\b", re.I), Status.OUT),
    (re.compile(r"\b(injured reserve|\bIR\b|pup|nfi|suspend)", re.I), Status.IR),
    (re.compile(r"\bdoubtful\b|\bD\b$", re.I), Status.DOUBTFUL),
    (re.compile(r"\bquestionable\b|\bQ\b$", re.I), Status.QUESTIONABLE),
    (re.compile(r"\bprobable\b|full practice|expected to play", re.I), Status.PROBABLE),
]


def parse_status(raw: str) -> Status:
    if not raw:
        return Status.UNKNOWN
    s = raw.strip()
    if s.upper() in ("O", "OUT"):
        return Status.OUT
    if s.upper() in ("IR", "PUP", "NFI", "SUSP", "NA", "IR-R", "IR/R"):
        return Status.IR
    if s.upper() == "D":
        return Status.DOUBTFUL
    if s.upper() == "Q":
        return Status.QUESTIONABLE
    if s.upper() == "P":
        return Status.PROBABLE
    for pat, st in _STATUS_PATTERNS:
        if pat.search(s):
            return st
    return Status.UNKNOWN


@dataclass
class InjuryRecord:
    name: str
    team: str
    status: Status
    detail: str = ""
    source: str = ""
    fetched_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def action(self) -> Action:
        return ACTION_FOR[self.status]

    @property
    def key(self) -> str:
        return norm_name(self.name)


@dataclass
class SweepResult:
    removed: list = field(default_factory=list)     # (player, record)
    flagged: list = field(default_factory=list)
    unchanged: int = 0
    lineup_affected: bool = False
    source_ts: str = ""

    @property
    def ok(self) -> bool:
        return not self.removed and not self.flagged

    def summary(self) -> str:
        if self.ok:
            return f"Inactives sweep clean — {self.unchanged} players verified."
        out = [f"Inactives sweep: {len(self.removed)} REMOVE, {len(self.flagged)} FLAG, "
               f"{self.unchanged} clear"]
        for p, r in self.removed:
            out.append(f"  REMOVE  {p.position:3s} {p.name:24s} {r.status.value.upper():12s} "
                       f"{r.detail[:48]}")
        for p, r in self.flagged:
            out.append(f"  FLAG    {p.position:3s} {p.name:24s} {r.status.value.upper():12s} "
                       f"{r.detail[:48]}")
        if self.lineup_affected:
            out.append("  ** A player in the CURRENT lineup is affected — rebuild before lock. **")
        return "\n".join(out)

    def pushover(self) -> str:
        """Terse alert body for the Sunday 11:30 notification."""
        if self.ok:
            return "Lineup clear — no inactives affect your roster."
        parts = []
        for p, r in self.removed:
            parts.append(f"OUT: {p.name} ({p.team})")
        for p, r in self.flagged:
            parts.append(f"Q: {p.name} ({p.team})")
        return " | ".join(parts)


# FP practice codes: DNP = did not participate, LP = limited, FP = full.
# A Q tag with three DNPs is a very different player from a Q tag with a full Friday.
_PRACTICE_RANK = {"DNP": 0, "LP": 1, "FP": 2, "FULL": 2, "LIMITED": 1}


def _practice_trend(r: dict) -> tuple[str, int | None, int]:
    """Returns (display, last_rank, n_sessions). n_sessions matters: a lone Wednesday
    DNP with Thu/Fri still blank is routine veteran rest, not a signal — escalation
    on it falsely removed healthy players (external review, 2026-08-30)."""
    seq = [str(r.get(f"practice_{n}") or "").upper().strip() for n in (1, 2, 3)]
    seq = [s for s in seq if s]
    if not seq:
        return "", None, 0
    ranks = [_PRACTICE_RANK.get(s, None) for s in seq]
    ranks = [x for x in ranks if x is not None]
    return "/".join(seq), (ranks[-1] if ranks else None), len(ranks)


def records_from_fantasypros(rows: list[dict]) -> dict[str, InjuryRecord]:
    """Normalize the FP injuries payload (verified schema, 2026-08-16).

    Fields used: name, team_id, status / status_short, injury_type, comment,
    practice_1..3, probability_of_playing, injury_update_date.

    Practice participation escalates a Questionable to Doubtful when the player has
    not practiced: a Q tag with three DNPs historically plays far less often than a Q
    with a full session, and under Total Points a zero is unrecoverable.
    """
    out: dict[str, InjuryRecord] = {}
    for r in rows or []:
        name = r.get("name") or r.get("player_name") or ""
        if not name:
            continue
        raw = r.get("status") or r.get("status_short") or ""
        status = parse_status(str(raw))
        practice, last_rank, n_sessions = _practice_trend(r)
        prob = r.get("probability_of_playing")
        if status is Status.QUESTIONABLE and last_rank == 0 and n_sessions >= 2:
            status = Status.DOUBTFUL          # Q and repeatedly did not practice
        if isinstance(prob, (int, float)) and prob is not None:
            if prob <= 25 and status in (Status.QUESTIONABLE, Status.PROBABLE):
                status = Status.DOUBTFUL
            elif prob >= 75 and status is Status.QUESTIONABLE:
                status = Status.PROBABLE
        bits = [str(r.get("injury_type") or "").strip(),
                f"practice {practice}" if practice else "",
                f"{prob}% to play" if prob is not None else "",
                str(r.get("comment") or "").strip()[:60]]
        rec = InjuryRecord(name=name,
                           team=norm_team(r.get("team_id") or r.get("team") or ""),
                           status=status,
                           detail=" · ".join(b for b in bits if b),
                           source="fantasypros",
                           fetched_ts=str(r.get("injury_update_date") or "") or None
                           or datetime.now(timezone.utc).isoformat())
        if rec.status is not Status.UNKNOWN:
            out[rec.key] = rec
    return out


def records_from_slate(slate) -> dict[str, InjuryRecord]:
    """Injury indicators carried on the FanDuel CSV (stale by Sunday, but a floor)."""
    out: dict[str, InjuryRecord] = {}
    for p in slate.players:
        if not p.injury_indicator:
            continue
        rec = InjuryRecord(name=p.name, team=p.team,
                           status=parse_status(p.injury_indicator),
                           detail=p.injury_details, source="fanduel_csv")
        if rec.status is not Status.UNKNOWN:
            out[rec.key] = rec
    return out


def merge(*sources: dict[str, InjuryRecord]) -> dict[str, InjuryRecord]:
    """Later sources win, but a worse status never gets overwritten by a better one
    from a staler feed — the pessimistic read is the safe one before lock."""
    severity = {Status.ACTIVE: 0, Status.PROBABLE: 1, Status.UNKNOWN: 1,
                Status.QUESTIONABLE: 2, Status.DOUBTFUL: 3, Status.OUT: 4, Status.IR: 5}
    merged: dict[str, InjuryRecord] = {}
    for src in sources:
        for k, rec in src.items():
            cur = merged.get(k)
            if cur is None or severity[rec.status] >= severity[cur.status]:
                merged[k] = rec
    return merged


def sweep(slate, injuries: dict[str, InjuryRecord],
          lineup_ids: set[str] | None = None) -> SweepResult:
    """Apply injury records to a slate. REMOVE players are dropped from the pool.

    Returns what changed so the caller can decide whether to rebuild. Never silently
    haircuts a projection — a player is either in the pool or out of it.
    """
    res = SweepResult(source_ts=datetime.now(timezone.utc).isoformat())
    keep = []
    for p in slate.players:
        rec = injuries.get(norm_name(p.name))
        if rec is None or rec.action is Action.KEEP:
            keep.append(p)
            res.unchanged += 1
            continue
        if rec.action is Action.REMOVE:
            res.removed.append((p, rec))
            if lineup_ids and p.fd_id in lineup_ids:
                res.lineup_affected = True
            continue
        res.flagged.append((p, rec))
        keep.append(p)                      # flagged players stay; human decides
        if lineup_ids and p.fd_id in lineup_ids:
            res.lineup_affected = True
    slate.players = keep
    return res
