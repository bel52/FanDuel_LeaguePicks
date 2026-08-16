"""Lineup export — FanDuel upload CSV and a human-readable card.

FanDuel's authorized workflow: download the contest entries template, fill the player-ID
columns, upload. That keeps everything inside FanDuel's own tooling (no credentialed
automation, no ToS risk) while removing the manual click-through of nine players.

The template's exact column headers vary by slate type and can change between seasons.
Rather than hardcode them, `export_upload_csv` will mirror the headers of a real template
if you pass one (--template), and otherwise emit the documented default layout with a
warning. Same lesson as the salary CSV: verify against a real file before trusting it.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .contest_spec import SlateType

# Documented default column order for a FanDuel NFL classic entries template.
DEFAULT_CLASSIC_COLS = ["entry_id", "contest_id", "contest_name",
                        "QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"]
DEFAULT_SHOWDOWN_COLS = ["entry_id", "contest_id", "contest_name",
                         "MVP", "AnyFLEX", "AnyFLEX", "AnyFLEX", "AnyFLEX"]
# Roster order FanDuel expects for a classic lineup.
CLASSIC_FILL_ORDER = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "DEF"]


@dataclass
class ExportResult:
    path: Path
    template_used: str
    rows: int
    warnings: list

    def summary(self) -> str:
        s = [f"Wrote {self.rows} lineup row(s) -> {self.path}",
             f"  template: {self.template_used}"]
        s += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(s)


def _slot_players(players: list, slate_type: SlateType, mvp_id: str | None):
    """Assign players to FanDuel roster slots in the order the template expects."""
    if slate_type == SlateType.SINGLE_GAME:
        mvp = [p for p in players if p.fd_id == mvp_id]
        rest = [p for p in players if p.fd_id != mvp_id]
        return mvp + rest

    by_pos: dict[str, list] = {}
    for p in players:
        by_pos.setdefault(p.position, []).append(p)
    for v in by_pos.values():
        v.sort(key=lambda p: -p.salary)

    out, used = [], set()
    for slot in CLASSIC_FILL_ORDER:
        if slot == "FLEX":
            continue
        pos = "D" if slot == "DEF" else slot
        pick = next((p for p in by_pos.get(pos, []) if p.fd_id not in used), None)
        if pick is None:
            raise ValueError(f"no player available for slot {slot}")
        used.add(pick.fd_id)
        out.append((slot, pick))
    flex = next((p for p in players if p.fd_id not in used), None)
    if flex is None:
        raise ValueError("no FLEX player available")
    out.insert(CLASSIC_FILL_ORDER.index("FLEX"), ("FLEX", flex))
    return out


def export_upload_csv(players: list, out_path: str | Path,
                      slate_type: SlateType = SlateType.FULL,
                      mvp_id: str | None = None,
                      template: str | Path | None = None,
                      entry_id: str = "", contest_id: str = "",
                      contest_name: str = "") -> ExportResult:
    """Write a FanDuel-uploadable CSV for one lineup."""
    warnings: list[str] = []
    if template:
        with Path(template).open(newline="", encoding="utf-8-sig") as f:
            cols = next(csv.reader(f))
        tmpl_name = str(template)
    else:
        cols = (DEFAULT_SHOWDOWN_COLS if slate_type == SlateType.SINGLE_GAME
                else DEFAULT_CLASSIC_COLS)
        tmpl_name = "built-in default (UNVERIFIED)"
        warnings.append(
            "No FanDuel template supplied — column headers are the documented default "
            "and have NOT been verified against a live entries file. Download the "
            "contest's entries template and pass --template before trusting this upload.")

    slotted = _slot_players(players, slate_type, mvp_id)
    ids = ([p.fd_id for p in slotted] if slate_type == SlateType.SINGLE_GAME
           else [p.fd_id for _, p in slotted])

    row: list[str] = []
    id_i = 0
    for c in cols:
        cl = c.strip().lower()
        if cl in ("entry_id", "entry id"):
            row.append(entry_id)
        elif cl in ("contest_id", "contest id"):
            row.append(contest_id)
        elif cl in ("contest_name", "contest name"):
            row.append(contest_name)
        else:
            row.append(ids[id_i] if id_i < len(ids) else "")
            id_i += 1
    if id_i < len(ids):
        warnings.append(f"template had {id_i} player columns but lineup has {len(ids)} "
                        "players — extra players were dropped")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(row)
    return ExportResult(path=out, template_used=tmpl_name, rows=1, warnings=warnings)


def lineup_card(players: list, slate_type: SlateType = SlateType.FULL,
                mvp_id: str | None = None, title: str = "",
                metrics: str = "", notes: list | None = None) -> str:
    """Human-readable lineup, for terminal output and Pushover."""
    lines = []
    if title:
        lines.append(title)
    if metrics:
        lines.append(f"  {metrics}")
    slotted = (_slot_players(players, slate_type, mvp_id)
               if slate_type == SlateType.FULL
               else [("MVP" if p.fd_id == mvp_id else "FLEX", p)
                     for p in _slot_players(players, slate_type, mvp_id)])
    total_sal = sum(p.salary for _, p in slotted)
    total_proj = sum((p.projection or 0) for _, p in slotted)
    for slot, p in slotted:
        tot = f"{p.implied_team_total:.1f}" if p.implied_team_total else "  - "
        lines.append(f"  {slot:4s} {p.name:24s} {p.team:3s} ${p.salary:5d} "
                     f"proj {p.projection or 0:5.1f}  ITT {tot}")
    lines.append(f"  {'':4s} {'TOTAL':24s} {'':3s} ${total_sal:5d} proj {total_proj:5.1f}")
    for n in (notes or []):
        lines.append(f"  ! {n}")
    return "\n".join(lines)


def pushover_body(players: list, metrics: str, notes: list | None = None,
                  mvp_id: str | None = None) -> str:
    """Compact lineup for a phone notification."""
    parts = [metrics]
    for p in players:
        tag = "*" if p.fd_id == mvp_id else ""
        parts.append(f"{p.position} {p.name}{tag}")
    if notes:
        parts.append("! " + "; ".join(notes))
    return "\n".join(parts)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
