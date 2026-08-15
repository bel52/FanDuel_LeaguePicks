"""Parse a FanDuel contest-results page pasted as text (or TextEdit .rtf/.rtfd).

Why paste-based: FanDuel's authorized workflow has no results export, and credentialed
scraping is off the table (ToS/ban risk). Selecting the rendered page you are already
logged into and copying it is entirely within your rights and takes ~20 seconds.

What we harvest, and why it matters:
  * leaderboard  -> every entrant's score and rank (opponent skill distribution)
  * lineups      -> which players each entrant actually rostered (opponent tendencies)
  * DRAFTED %    -> MEASURED ownership in your exact 12-person field. n/12 exactly.
                    Public tools guess ownership for anonymous fields; you can observe
                    yours. This is the single most valuable input to the opponent model.

FanDuel's results view shows your entry vs one opponent at a time, so a full week needs
several pastes; each parses independently and merges by (week, entrant).
"""
from __future__ import annotations
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

POSITIONS = {"QB", "RB", "WR", "TE", "D", "DEF", "DST", "K", "FLEX"}
# FanDuel labels the flex slot "FLEX" and the defense "DEF"; the real position of a
# FLEX player is recovered from the projections/slate later, not from this page.
ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
MONEY = re.compile(r"^\$([\d,]+(?:\.\d{2})?)$")
PCT = re.compile(r"^(\d+(?:\.\d+)?)%$")
NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
GAME = re.compile(r"^([A-Z]{2,3})\s*\d+\s*@\s*([A-Z]{2,3})\s*\d+$")


# ---------- input normalisation ----------

def _rtf_to_text(raw: str) -> str:
    t = re.sub(r"\\'([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), raw)
    t = re.sub(r"\{\\\*?[^{}]*\}", "", t)
    t = t.replace("\\par", "\n").replace("\\line", "\n")
    t = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", t)
    return t.replace("{", "").replace("}", "")


def load_text(path: str | Path) -> str:
    """Accept .txt, .rtf, or a TextEdit .rtfd (directory or zipped)."""
    p = Path(path)
    if p.is_dir():
        inner = p / "TXT.rtf"
        return _rtf_to_text(inner.read_text(encoding="utf-8", errors="ignore"))
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next((n for n in z.namelist()
                         if n.endswith("TXT.rtf") and "__MACOSX" not in n), None)
            if not name:
                raise ValueError(f"no TXT.rtf inside {p}")
            return _rtf_to_text(z.read(name).decode("utf-8", "ignore"))
    raw = p.read_text(encoding="utf-8", errors="ignore")
    return _rtf_to_text(raw) if raw.lstrip().startswith("{\\rtf") else raw


def _lines(text: str) -> list[str]:
    out = []
    for ln in text.splitlines():
        s = ln.strip().rstrip("\\").strip()
        if not s or s in ("d",) or s.startswith(("deftab", "tightenfactor", "*HYPERLINK",
                                                 "\\*", "*\\", "cocoartf", "fonttbl")):
            continue
        out.append(s)
    return out


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9 ]", "", s.replace("-", " "))
    parts = [p for p in s.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


# ---------- data model ----------

@dataclass
class RosteredPlayer:
    position: str
    name: str
    drafted_pct: float | None = None     # measured field ownership (n/12)
    actual_points: float | None = None
    stat_line: str = ""
    game: str = ""

    @property
    def norm_name(self) -> str:
        return _norm(self.name)


@dataclass
class Entry:
    entrant: str
    rank: int | None = None
    score: float | None = None
    won: float = 0.0
    players: list[RosteredPlayer] = field(default_factory=list)


@dataclass
class ContestCapture:
    season: int
    week: int
    contest: str
    field_size: int = 0
    leaderboard: list[Entry] = field(default_factory=list)
    entries_with_lineups: list[Entry] = field(default_factory=list)

    def ownership(self) -> dict[str, float]:
        """Measured ownership by normalized player name (mean of observed DRAFTED%)."""
        acc: dict[str, list[float]] = {}
        for e in self.entries_with_lineups:
            for p in e.players:
                if p.drafted_pct is not None:
                    acc.setdefault(p.norm_name, []).append(p.drafted_pct)
        return {k: round(sum(v) / len(v), 2) for k, v in acc.items()}

    def summary(self) -> str:
        out = [f"{self.contest} — {self.season} week {self.week}",
               f"  leaderboard: {len(self.leaderboard)} entrants"]
        for e in self.leaderboard:
            out.append(f"    {str(e.rank) + '.':>4s} {e.entrant:16s} {e.score:7.2f}"
                       + (f"  ${e.won:.2f}" if e.won else ""))
        out.append(f"  lineups captured: {len(self.entries_with_lineups)}")
        for e in self.entries_with_lineups:
            out.append(f"    {e.entrant} ({len(e.players)} players):")
            for p in e.players:
                own = f"{p.drafted_pct:5.1f}%" if p.drafted_pct is not None else "    ?"
                pts = f"{p.actual_points:6.2f}" if p.actual_points is not None else "     ?"
                out.append(f"      {p.position:3s} {p.name:24s} own={own} pts={pts}")
        return "\n".join(out)


# ---------- parsing ----------

def _parse_leaderboard(L: list[str]) -> list[Entry]:
    try:
        start = next(i for i, s in enumerate(L) if s == "Leaderboard")
    except StopIteration:
        return []
    entries: list[Entry] = []
    i = start
    while i < len(L):
        m = ORDINAL.match(L[i])
        if not m:
            i += 1
            continue
        rank = int(m.group(1))
        # scan forward for: username, optional $won, then the score
        j, name, won, nums = i + 1, None, 0.0, []
        while j < len(L) and j < i + 10:
            s = L[j]
            if ORDINAL.match(s) or s.startswith("Jump to"):
                break
            mm = MONEY.match(s)
            if mm:
                won = float(mm.group(1).replace(",", ""))
            elif NUM.match(s):
                nums.append(float(s))
            elif name is None and not PCT.match(s) and s not in ("FINAL",):
                name = s
            j += 1
        # The row also carries "minutes remaining" (0) and can run into pagination
        # digits; the fantasy score is the largest number in the row.
        score = max(nums) if nums else None
        if name and score is not None:
            entries.append(Entry(entrant=name, rank=rank, score=score, won=won))
        i = j
    # dedupe by rank, keep first
    seen, out = set(), []
    for e in entries:
        if e.rank not in seen:
            seen.add(e.rank)
            out.append(e)
    return sorted(out, key=lambda e: e.rank or 999)


def _parse_lineups(L: list[str], known_users: set[str]) -> list[Entry]:
    """Player blocks look like:
         QB / <name> / <stat line> / <GAME> / <pct>% / DRAFTED / <points>
    Blocks are grouped under whichever username most recently appeared.
    """
    entries: dict[str, Entry] = {}
    current: str | None = None
    i = 0
    while i < len(L):
        s = L[i]
        if s in known_users:
            current = s
            entries.setdefault(s, Entry(entrant=s))
            i += 1
            continue
        if s in POSITIONS and i + 1 < len(L):
            pos = "D" if s in ("DEF", "DST") else s
            name = L[i + 1]
            if name in POSITIONS or ORDINAL.match(name):
                i += 1
                continue
            p = RosteredPlayer(position=pos, name=name)
            j = i + 2
            while j < len(L) and j < i + 12:
                tok = L[j]
                if tok in POSITIONS or tok in known_users:
                    break
                if PCT.match(tok):
                    p.drafted_pct = float(PCT.match(tok).group(1))
                elif GAME.match(tok):
                    p.game = tok
                elif NUM.match(tok) and p.drafted_pct is not None and p.actual_points is None:
                    p.actual_points = float(tok)
                elif tok not in ("DRAFTED", "FINAL") and not p.stat_line:
                    p.stat_line = tok
                j += 1
            key = current or "unknown"
            entries.setdefault(key, Entry(entrant=key)).players.append(p)
            i = j
            continue
        i += 1
    return [e for e in entries.values() if e.players]


def parse_contest(path: str | Path, season: int, week: int,
                  contest: str = "Leather League") -> ContestCapture:
    L = _lines(load_text(path))
    lb = _parse_leaderboard(L)
    users = {e.entrant for e in lb}
    lineups = _parse_lineups(L, users)
    # merge leaderboard rank/score onto lineup entries
    by_name = {e.entrant: e for e in lb}
    for e in lineups:
        src = by_name.get(e.entrant)
        if src:
            e.rank, e.score, e.won = src.rank, src.score, src.won
    return ContestCapture(season=season, week=week, contest=contest,
                          field_size=len(lb), leaderboard=lb,
                          entries_with_lineups=lineups)


def save(capture: ContestCapture, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(capture), indent=1))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="dfs.contest_parse")
    ap.add_argument("path")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--contest", default="Leather League")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cap = parse_contest(a.path, a.season, a.week, a.contest)
    print(cap.summary())
    own = cap.ownership()
    if own:
        print(f"\n  measured ownership ({len(own)} players):")
        for n, v in sorted(own.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {v:5.1f}%  {n}")
    out = a.out or f"data/contests/{a.season}-w{a.week:02d}.json"
    save(cap, out)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
