"""Projection engine: FP consensus (half-PPR) + FanDuel bonus expectation + Vegas team-total tilt.

Every number source-traceable. Vegas applied ONCE here as a bounded tilt around the
consensus, scaled by how far the implied team total sits from league average.
No downstream boosts exist anywhere.
"""
from __future__ import annotations
from datetime import datetime, timezone

from .slate import PlayerSlate, SlateError
from .fantasypros import FPProjection
from .vegas import TeamLine
from .matching import norm_team
from .distributions import floor_ceiling
from .matching import match_slate, MatchReport

LEAGUE_AVG_TEAM_TOTAL = 22.0
# Bounded Vegas tilt: ±1 point of implied total ≈ ±1.5% projection for skill players.
# Rationale: consensus projections already price most matchup info; the tilt only
# nudges toward the sharpest same-morning market signal. Documented, capped at ±9%.
VEGAS_TILT_PER_POINT = 0.015
VEGAS_TILT_CAP = 0.09

# Bonus expectation now lives in scoring.py and is already baked into FPProjection.points.


def apply_projections(slate: PlayerSlate, fp_projections: list[FPProjection],
                      team_lines: dict[str, TeamLine], distributions: dict,
                      min_match_rate: float = 0.80,
                      critical_salary: int = 5500) -> MatchReport:
    """Attach FanDuel-scored projections to slate players.

    Fail-loud below min_match_rate, or if any high-salary player is unmatched.
    Low-salary unmatched players (deep backups FP does not project) are dropped from
    the optimizable pool rather than being given a fabricated number.
    """
    mapping, report = match_slate(slate.players, fp_projections)
    now = datetime.now(timezone.utc).isoformat()

    for sp in slate.players:
        fp = mapping.get(sp.fd_id)
        if fp is None or fp.points <= 0:
            continue
        base = fp.points                      # already true FanDuel points (scoring.py)
        line = team_lines.get(norm_team(sp.team))
        tilt = 0.0
        if line and sp.position != "D":
            tilt = max(-VEGAS_TILT_CAP,
                       min(VEGAS_TILT_CAP,
                           (line.implied_total - LEAGUE_AVG_TEAM_TOTAL) * VEGAS_TILT_PER_POINT))
            sp.implied_team_total = line.implied_total
        sp.projection = round(base * (1 + tilt), 2)
        sp.proj_source = f"fp_stats->fd_scoring{'+vegas' if tilt else ''}"
        sp.proj_ts = now
        sp.floor_p10, sp.ceiling_p90 = floor_ceiling(sp.projection, sp.position, distributions)

    # A matched player whose FP payload scores to <= 0 was silently skipped above
    # and then dropped from the slate — invisible to the match-rate gate AND the
    # critical-salary gate (both only see UNMATCHED). External review reproduced a
    # $9,100 player vanishing at a reported 100% match rate. Nonpositive matched
    # projections are a data/schema failure and gate exactly like unmatched.
    # Relevance is POSITION-RELATIVE. A flat dollar cutoff misjudges both ends: at
    # $7,000 it ignores a newly-promoted $5,600 WR starter (the exact player who
    # unlocks an optimal lineup — the entered lineup routinely rosters four players
    # under $6,500), while treating a $6,000 third-string QB as critical. Anything in
    # the top quartile of its own position's salary distribution is roster-relevant.
    from collections import defaultdict
    by_pos: dict[str, list[int]] = defaultdict(list)
    for sp in slate.players:
        by_pos[sp.position].append(sp.salary)
    pos_cut: dict[str, int] = {}
    for pos, sals in by_pos.items():
        s = sorted(sals, reverse=True)
        pos_cut[pos] = s[max(0, int(len(s) * 0.25) - 1)] if s else 0

    def _relevant(pos: str, salary: int) -> bool:
        return salary >= critical_salary or salary >= pos_cut.get(pos, critical_salary)

    nonpos = [(sp.name, sp.team, sp.position, sp.salary)
              for sp in slate.players
              if mapping.get(sp.fd_id) is not None
              and mapping[sp.fd_id].points <= 0]
    if nonpos:
        print(f"  WARNING: {len(nonpos)} matched player(s) with nonpositive FanDuel "
              "score from FP stats (schema problem or true zero):")
        for n, t, pos, sal in sorted(nonpos, key=lambda z: -z[3])[:8]:
            print(f"    ${sal:5d} {pos:3s} {t:4s} {n}")
    # report.unmatched rows are (name, team, position, salary)
    critical = ([u for u in report.unmatched if _relevant(u[2], u[3])]
                + [z for z in nonpos if _relevant(z[2], z[3])])
    if report.rate < min_match_rate or critical:
        cuts = ", ".join(f"{p} >= ${c}" for p, c in sorted(pos_cut.items()))
        raise SlateError(
            f"Projection match {report.rate:.0%} (min {min_match_rate:.0%}); "
            f"{len(critical)} roster-relevant player(s) unmatched or zero-projected.\n"
            f"  relevance gate (top quartile by position): {cuts}\n"
            + "".join(f"    ${s:5d} {p:3s} {t:4s} {n}\n"
                      for n, t, p, s in sorted(critical, key=lambda z: -z[3])[:10])
            + report.summary() +
            "\nBuild stopped — fix matching/schema, never fall back to FPPG."
        )
    slate.players = [p for p in slate.players if p.projection is not None]
    return report
