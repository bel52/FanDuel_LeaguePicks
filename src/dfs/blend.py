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
        line = team_lines.get(sp.team)
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

    critical = [u for u in report.unmatched if u[3] >= critical_salary]
    if report.rate < min_match_rate or critical:
        raise SlateError(
            f"Projection match {report.rate:.0%} (min {min_match_rate:.0%}); "
            f"{len(critical)} unmatched at >= ${critical_salary}.\n" + report.summary() +
            "\nBuild stopped — fix matching, never fall back to FPPG."
        )
    slate.players = [p for p in slate.players if p.projection is not None]
    return report
