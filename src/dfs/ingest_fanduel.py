"""FanDuel salary CSV ingester. Hardened: validation report on every load; fail-loud on schema drift.

Known FD export columns (v5-era, re-verify against a live 2026 CSV in 1.2):
Id, Position, First Name, Nickname, Last Name, FPPG, Played, Salary, Game,
Team, Opponent, Injury Indicator, Injury Details, Tier, Roster Position
"""
from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path

from .contest_spec import SlateType
from .slate import PlayerSlate, SlatePlayer, SlateError

REQUIRED_COLS = {"Id", "Position", "Salary", "Team", "Opponent", "Game"}
NAME_COLS = {"First Name", "Last Name"}  # Nickname optional
POSITION_MAP = {"DST": "D", "DEF": "D", "D/ST": "D", "D": "D",
                "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
                "K": "K", "PK": "K"}  # kickers appear on showdown slates
EXCLUDE_INJURY = {"IR", "O", "OUT", "NA", "SUSP"}  # dropped at ingest, reported


@dataclass
class IngestReport:
    source: str
    total_rows: int = 0
    ingested: int = 0
    dropped_injury: list[str] = field(default_factory=list)
    dropped_bad_row: list[str] = field(default_factory=list)
    unknown_positions: list[str] = field(default_factory=list)
    detected_slate_type: SlateType = SlateType.FULL
    teams: list[str] = field(default_factory=list)
    validation_problems: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Ingest: {self.source}",
            f"  rows={self.total_rows} ingested={self.ingested} "
            f"dropped_injury={len(self.dropped_injury)} dropped_bad={len(self.dropped_bad_row)}",
            f"  slate_type={self.detected_slate_type.value} teams={len(self.teams)}",
        ]
        if self.dropped_injury:
            lines.append(f"  out/IR: {', '.join(self.dropped_injury[:12])}"
                         + (" …" if len(self.dropped_injury) > 12 else ""))
        if self.unknown_positions:
            lines.append(f"  UNKNOWN POSITIONS (dropped): {self.unknown_positions[:8]}")
        for p in self.validation_problems:
            lines.append(f"  PROBLEM: {p}")
        return "\n".join(lines)

    @property
    def ok(self) -> bool:
        return not self.validation_problems and self.ingested > 0


def ingest_csv(path: str | Path, slate_id: str, season: int, week: int,
               strict: bool = True) -> tuple[PlayerSlate, IngestReport]:
    path = Path(path)
    report = IngestReport(source=str(path))
    if not path.exists():
        raise SlateError(f"CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLS - cols
        if missing:
            raise SlateError(
                f"FanDuel CSV schema drift — missing columns {sorted(missing)}. "
                f"Found: {sorted(cols)}. Ingest refused (fail-loud)."
            )
        if not NAME_COLS.issubset(cols):
            raise SlateError(f"CSV missing name columns {sorted(NAME_COLS - cols)}")

        players: list[SlatePlayer] = []
        for row in reader:
            report.total_rows += 1
            try:
                fd_id = row["Id"].strip()
                raw_pos = row["Position"].strip().upper()
                pos = POSITION_MAP.get(raw_pos)
                if pos is None:
                    report.unknown_positions.append(f"{row.get('Last Name','?')}:{raw_pos}")
                    continue
                nickname = (row.get("Nickname") or "").strip()
                name = nickname or f"{row['First Name'].strip()} {row['Last Name'].strip()}".strip()
                salary = int(float(row["Salary"]))
                team = row["Team"].strip().upper()
                opp = row["Opponent"].strip().upper()
                if not fd_id or not name or not team:
                    report.dropped_bad_row.append(name or fd_id or "?")
                    continue
                inj = (row.get("Injury Indicator") or "").strip().upper()
                if inj in EXCLUDE_INJURY:
                    report.dropped_injury.append(f"{name}({inj})")
                    continue
                players.append(SlatePlayer(
                    fd_id=fd_id, name=name, position=pos, team=team, opponent=opp,
                    salary=salary, game=(row.get("Game") or "").strip(),
                    injury_indicator=inj,
                    injury_details=(row.get("Injury Details") or "").strip(),
                    roster_position=(row.get("Roster Position") or "").strip(),
                    fppg=float(row.get("FPPG") or 0.0),
                ))
            except (KeyError, ValueError) as e:
                report.dropped_bad_row.append(f"{row.get('Last Name','?')}: {e}")

    teams = sorted({p.team for p in players})
    report.teams = teams
    report.ingested = len(players)
    report.detected_slate_type = SlateType.SINGLE_GAME if len(teams) == 2 else SlateType.FULL

    slate = PlayerSlate(slate_id=slate_id, slate_type=report.detected_slate_type,
                        season=season, week=week, source_csv=str(path), players=players)
    report.validation_problems = slate.validate()
    if strict and not report.ok:
        raise SlateError("Slate validation failed:\n" + report.summary())
    return slate, report
