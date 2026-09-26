"""Teammate effects: when a player is out (or may be), his production does not
vanish — part of it flows to his teammates. FantasyPros consensus is slow to move
teammates (measured 2026-08-30: a week of injury news moved a consensus projection
by 0.1 pts), so the FantasyPros component can be stale exactly when it matters.

Everything here is MEASURED, not assumed (nflverse 2023-2025 regular seasons,
2026-09-26):
  * a receiver with 5+ targets/game sits (271 games): teammates gained 61% of his
    usual receiving points (95% CI 45-77%)
  * a back with 10+ carries/game sits (103 games): teammates gained 58% of his
    usual rushing points (95% CI 40-76%)
  * the gain splits in proportion to each teammate's normal volume, with
    same-position teammates taking ~2.5x per unit of volume (2,096 recipient-games)

How it is applied, and why each choice is conservative:
  * Absorption uses the LOW end of each interval (45% / 40%): consensus has usually
    priced some of the change, and under-correcting is cheaper than double-counting.
  * Only the FantasyPros component moves. Betting markets reprice teammates within
    minutes of the news, so the props component already contains the effect; the
    blend then carries the boost at the FP weight only.
  * "Vacated" is the absent player's OWN FantasyPros projection x P(he sits). If
    FantasyPros has already zeroed him out, nothing is left to redistribute — the
    adjustment switches itself off exactly when consensus has caught up.
  * Per-player cap: +5 pts or +35% of his own projection, whichever is smaller.
  * Every adjustment is printed and logged (player_results.teammate_adj) so its
    accuracy can be measured against nflverse actuals, like every other layer.

Out of scope, deliberately: a QB out lowers his receivers, but that is a team-level
effect already carried by the Vegas implied total (and the opposing DST is scored
against that total too — Carolina W2 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .matching import norm_name, norm_team

ABSORB_REC = 0.45        # measured 0.61, CI [0.45, 0.77]
ABSORB_RUSH = 0.40       # measured 0.58, CI [0.40, 0.76]
SAME_POS_MULT = 2.5      # measured 2.48
MIN_MISS = 0.10          # ignore players 90%+ to play
MAX_BOOST_PTS = 5.0
MAX_BOOST_FRAC = 0.35
RECIPIENTS = ("WR", "TE", "RB")
DONORS = ("WR", "TE", "RB")


def _f(stats: dict, k: str) -> float:
    try:
        return float((stats or {}).get(k) or 0)
    except (TypeError, ValueError):
        return 0.0


def rec_points(stats: dict) -> float:
    return _f(stats, "rec_rec") * 0.5 + _f(stats, "rec_yds") * 0.1 + _f(stats, "rec_tds") * 6


def rush_points(stats: dict) -> float:
    return _f(stats, "rush_yds") * 0.1 + _f(stats, "rush_tds") * 6


@dataclass
class Adjustment:
    player: object
    boost: float = 0.0
    sources: list = field(default_factory=list)     # (donor name, P(sits), pts)
    capped: bool = False


def adjust_for_absences(slate, fp_projections, injuries: dict,
                        play_probability) -> list[Adjustment]:
    """Add vacated production to teammates' FantasyPros component. Returns every
    adjustment made (already applied to the slate). Must run AFTER the injury sweep
    (so p_active is known) and BEFORE the props blend (so only FP moves)."""
    fp_by_id = getattr(slate, "fp_by_id", {}) or {}
    # donors: anyone FantasyPros projects who may not play, including players the
    # sweep already removed from the pool (they are absent from the slate, not FP)
    donors = []
    for q in fp_projections or []:
        pos = str(getattr(q, "position", "") or "")
        if pos not in DONORS:
            continue
        rec = injuries.get(norm_name(q.name))
        if rec is None:
            continue
        miss = 1.0 - float(play_probability(rec))
        if miss < MIN_MISS:
            continue
        v_rec, v_rush = rec_points(q.stats) * miss, rush_points(q.stats) * miss
        if v_rec + v_rush <= 0.05:
            continue                     # consensus already has him near zero
        donors.append((q, norm_team(q.team), pos, miss, v_rec, v_rush))

    adj: dict[str, Adjustment] = {}
    for q, team, dpos, miss, v_rec, v_rush in donors:
        dkey = norm_name(q.name)
        mates = []
        for sp in slate.players:
            if (norm_team(sp.team) != team or sp.position not in RECIPIENTS
                    or sp.proj_fp is None or norm_name(sp.name) == dkey):
                continue
            fq = fp_by_id.get(sp.fd_id)
            if fq is None:
                continue
            mates.append((sp, fq.stats or {}))
        for vac, absorb, vol_key, eligible in (
                (v_rec, ABSORB_REC, "rec_rec", RECIPIENTS),
                (v_rush, ABSORB_RUSH, "rush_att", ("RB",))):
            if vac <= 0:
                continue
            w = {sp.fd_id: _f(st, vol_key) * (SAME_POS_MULT if sp.position == dpos else 1.0)
                 for sp, st in mates if sp.position in eligible}
            tot = sum(w.values())
            if tot <= 0:
                continue
            for sp, _ in mates:
                share = w.get(sp.fd_id, 0.0) / tot
                if share <= 0:
                    continue
                a = adj.setdefault(sp.fd_id, Adjustment(player=sp))
                pts = absorb * vac * share
                a.boost += pts
                a.sources.append((q.name, round(miss, 2), round(pts, 2)))

    out = []
    for a in adj.values():
        sp = a.player
        cap = min(MAX_BOOST_PTS, MAX_BOOST_FRAC * max(sp.proj_fp, 0.0))
        if a.boost > cap:
            a.boost, a.capped = cap, True
        a.boost = round(a.boost, 2)
        if a.boost < 0.05:
            continue
        sp.teammate_adj = a.boost
        old_fp = sp.proj_fp
        sp.proj_fp = round(old_fp + a.boost, 2)
        sp.projection = round((sp.projection or old_fp) + a.boost, 2)
        if sp.proj_blend is None or abs(sp.proj_blend - old_fp) < 1e-9:
            sp.proj_blend = sp.proj_fp          # not blended yet (the normal case)
        else:
            sp.proj_blend = round(sp.proj_blend + a.boost, 2)
        sp.proj_source = f"{sp.proj_source or 'fp'}+teammates"
        out.append(a)
    return sorted(out, key=lambda a: -a.boost)


def report(adjs: list[Adjustment], limit: int = 12) -> list[str]:
    if not adjs:
        return []
    lines = [f"\n  teammate effects applied to {len(adjs)} players (vacated production, "
             f"measured 2023-25; FantasyPros component only):"]
    for a in adjs[:limit]:
        p = a.player
        src = "; ".join(f"{n} {m:.0%} out" for n, m, _ in a.sources[:3])
        lines.append(f"    +{a.boost:4.1f}  {p.position:3s} {p.team:4s} {p.name:24s} "
                     f"<- {src}{'  (capped)' if a.capped else ''}")
    return lines
