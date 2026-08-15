"""Player matching between FanDuel CSV and FantasyPros projections.

Team is NOT part of the primary key. Live evidence (2026-08-15): FanDuel and FantasyPros
disagree on team for many players (mid-season trades, practice-squad churn, stale rosters),
so name|team matching dropped ~48% of the slate including A.J. Brown. Name is the stable
identifier; team is a disambiguator for duplicate names and a warning signal otherwise.

Order of attempts:
  1. exact normalized full name
  2. normalized name + position (disambiguates same-name players)
  3. last name + first initial + position (handles "Marvin Mims" vs "Marvin Mims Jr.")
Ambiguity is resolved by team, then position, then highest projection. Unresolvable
ambiguity is reported, never silently guessed.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# FanDuel <-> FantasyPros abbreviation differences
TEAM_ALIASES = {"JAX": "JAC", "WSH": "WAS", "LA": "LAR", "SD": "LAC", "OAK": "LV",
                "STL": "LAR", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}


def norm_team(t: str) -> str:
    t = (t or "").strip().upper()
    return TEAM_ALIASES.get(t, t)


def norm_name(name: str) -> str:
    """Aggressive but reversible-in-spirit normalization: 'A.J. Brown' -> 'aj brown'."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("&", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)                  # drops . ' etc
    parts = [p for p in s.split() if p and p not in SUFFIXES]
    return " ".join(parts)


def short_key(name: str) -> str:
    """First initial + last name: 'aj brown' -> 'a|brown'."""
    p = norm_name(name).split()
    if not p:
        return ""
    return f"{p[0][0]}|{p[-1]}" if len(p) > 1 else p[0]


@dataclass
class MatchReport:
    matched: int = 0
    total: int = 0
    by_method: dict = field(default_factory=dict)
    unmatched: list = field(default_factory=list)          # (name, team, pos, salary)
    team_disagreements: list = field(default_factory=list)  # (name, fd_team, fp_team)
    ambiguous: list = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.matched / self.total if self.total else 0.0

    def summary(self, top_n: int = 12) -> str:
        lines = [f"Match: {self.matched}/{self.total} = {self.rate:.1%}",
                 f"  methods: {self.by_method}"]
        if self.team_disagreements:
            ex = ", ".join(f"{n} FD:{a}/FP:{b}" for n, a, b in self.team_disagreements[:6])
            lines.append(f"  team disagreements (matched anyway): {len(self.team_disagreements)} — {ex}")
        if self.ambiguous:
            lines.append(f"  AMBIGUOUS (unresolved): {self.ambiguous[:6]}")
        if self.unmatched:
            top = sorted(self.unmatched, key=lambda u: -u[3])[:top_n]
            lines.append("  unmatched (by salary):")
            lines += [f"    ${s:5d} {p:3s} {t:3s} {n}" for n, t, p, s in top]
        return "\n".join(lines)


class ProjectionIndex:
    """Index of FantasyPros projections supporting name-first lookup."""

    def __init__(self, projections: list):
        self.by_name: dict[str, list] = {}
        self.by_short: dict[str, list] = {}
        for p in projections:
            self.by_name.setdefault(norm_name(p.name), []).append(p)
            self.by_short.setdefault(short_key(p.name), []).append(p)

    @staticmethod
    def _pick(cands: list, team: str, position: str):
        """Resolve multiple candidates: team, then position, then highest projection."""
        if len(cands) == 1:
            return cands[0], False
        same_team = [c for c in cands if norm_team(c.team) == norm_team(team)]
        if len(same_team) == 1:
            return same_team[0], False
        pool = same_team or cands
        same_pos = [c for c in pool if c.position == position]
        if len(same_pos) == 1:
            return same_pos[0], False
        pool = same_pos or pool
        if not pool:
            return None, True
        best = max(pool, key=lambda c: c.points)
        return best, len(pool) > 1

    def lookup(self, name: str, team: str, position: str):
        """Return (projection|None, method, ambiguous)."""
        cands = self.by_name.get(norm_name(name))
        if cands:
            pick, amb = self._pick(cands, team, position)
            if pick is not None:
                return pick, "name", amb
        cands = [c for c in self.by_name.get(norm_name(name), []) if c.position == position]
        if cands:
            pick, amb = self._pick(cands, team, position)
            if pick is not None:
                return pick, "name+pos", amb
        cands = [c for c in self.by_short.get(short_key(name), []) if c.position == position]
        if cands:
            pick, amb = self._pick(cands, team, position)
            if pick is not None:
                return pick, "short+pos", amb
        return None, "none", False


def match_slate(slate_players: list, projections: list) -> tuple[dict, MatchReport]:
    """Map fd_id -> FPProjection. Returns (mapping, report)."""
    idx = ProjectionIndex(projections)
    rep = MatchReport(total=len(slate_players))
    mapping: dict = {}
    for sp in slate_players:
        proj, method, amb = idx.lookup(sp.name, sp.team, sp.position)
        if proj is None:
            rep.unmatched.append((sp.name, sp.team, sp.position, sp.salary))
            continue
        mapping[sp.fd_id] = proj
        rep.matched += 1
        rep.by_method[method] = rep.by_method.get(method, 0) + 1
        if norm_team(proj.team) != norm_team(sp.team):
            rep.team_disagreements.append((sp.name, sp.team, proj.team))
        if amb:
            rep.ambiguous.append(f"{sp.name} ({sp.team} {sp.position})")
    return mapping, rep
