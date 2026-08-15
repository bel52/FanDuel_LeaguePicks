"""Opponent field model + payout-EV lineup selection.

This is the module that makes the system different from a public optimizer: candidates
are ranked by P(rank=1) against a simulated field of the ACTUAL 11 league opponents,
not by projected points or a "ceiling score".

Field construction (v1, generic prior): opponents are simulated as lineups drawn from a
salary/projection-weighted popularity distribution, respecting roster rules. Once league
history is ingested (opponent_history.py), per-opponent tendencies replace the prior.

Ownership note: real per-player league ownership is observable in FanDuel contest
results as "% DRAFTED" (n/12). That is measured ownership for the exact field we play
against — vastly better than public GPP ownership projections.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .contest_spec import ContestSpec
from .optimize import Candidate
from .simulate import SlateSimulator
from .slate import PlayerSlate


@dataclass
class FieldModel:
    """Distribution over opponent lineups."""
    lineups: list[tuple[str, ...]]     # sampled opponent lineups
    source: str                        # 'prior' | 'history'


def build_prior_field(slate: PlayerSlate, spec: ContestSpec, n_opponents: int,
                      seed: int = 99, chalk_temp: float = 1.6) -> FieldModel:
    """Generic-field prior: opponents pick popular, high-value players.

    Popularity ~ softmax over projection-per-$1000 and raw projection. chalk_temp
    controls concentration: higher = more herd behaviour (friends leagues are chalky).

    Salary-cap handling uses REPAIR, not rejection: an over-cap lineup has its most
    expensive player swapped down until it fits. Pure rejection sampling silently
    under-produced the field (8 of 11 opponents), which quietly inflated every win
    probability — the field must always be exactly n_opponents.
    """
    rng = np.random.default_rng(seed)
    by_pos: dict[str, list] = {}
    for p in slate.players:
        by_pos.setdefault(p.position, []).append(p)
    for pos in ("QB", "RB", "WR", "TE", "D"):
        if len(by_pos.get(pos, [])) < 4:
            raise ValueError(f"field model needs >=4 {pos}, has {len(by_pos.get(pos, []))}")

    weights: dict[str, np.ndarray] = {}
    for pos, plist in by_pos.items():
        val = np.array([p.projection / max(p.salary / 1000, 0.1) for p in plist])
        proj = np.array([p.projection for p in plist])
        z = chalk_temp * ((val - val.mean()) / (val.std() + 1e-9)
                          + 0.8 * (proj - proj.mean()) / (proj.std() + 1e-9))
        w = np.exp(z - z.max())
        weights[pos] = w / w.sum()

    def _repair(picks: list) -> list | None:
        """Swap expensive players down until under cap. Preserves position counts."""
        for _ in range(40):
            total = sum(p.salary for p in picks)
            if total <= spec.salary_cap:
                return picks
            over = total - spec.salary_cap
            i = max(range(len(picks)), key=lambda k: picks[k].salary)
            cur = picks[i]
            chosen = {p.fd_id for p in picks}
            alts = [p for p in by_pos[cur.position]
                    if p.fd_id not in chosen and p.salary <= cur.salary - over]
            if not alts:
                alts = [p for p in by_pos[cur.position]
                        if p.fd_id not in chosen and p.salary < cur.salary]
            if not alts:
                return None
            # opponents downgrade to the best remaining value, not at random
            alts.sort(key=lambda p: -p.projection)
            picks[i] = alts[0]
        return None

    lineups: list[tuple[str, ...]] = []
    tries = 0
    while len(lineups) < n_opponents and tries < n_opponents * 200:
        tries += 1
        picks: list = []
        ok = True
        for pos, count in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("D", 1)):
            plist = by_pos.get(pos, [])
            if len(plist) < count:
                ok = False
                break
            idx = rng.choice(len(plist), size=count, replace=False, p=weights[pos])
            picks.extend(plist[i] for i in idx)
        if not ok:
            continue
        chosen = {p.fd_id for p in picks}
        flex_pool = [p for p in slate.players
                     if p.position in ("RB", "WR", "TE") and p.fd_id not in chosen]
        if not flex_pool:
            continue
        fv = np.array([p.projection / max(p.salary / 1000, 0.1) for p in flex_pool])
        fw = np.exp(chalk_temp * (fv - fv.mean()) / (fv.std() + 1e-9))
        picks.append(flex_pool[rng.choice(len(flex_pool), p=fw / fw.sum())])
        repaired = _repair(picks)
        if repaired is None:
            continue
        lineups.append(tuple(p.fd_id for p in repaired))

    if len(lineups) != n_opponents:
        raise ValueError(
            f"field model produced {len(lineups)}/{n_opponents} opponents after {tries} tries — "
            "win probabilities would be biased; refusing to proceed")
    return FieldModel(lineups=lineups, source="prior")


def baseline_lineups(slate: PlayerSlate, spec: ContestSpec, n: int = 200,
                     seed: int = 7) -> dict[str, list[tuple[str, ...]]]:
    """Reference lineups for honest comparison: random-valid and max-projection.

    Without these, a win probability is uninterpretable — 15% against 8 opponents is
    worse than random. Every build reports the edge over both baselines.
    """
    rng = np.random.default_rng(seed)
    by_pos: dict[str, list] = {}
    for p in slate.players:
        by_pos.setdefault(p.position, []).append(p)

    randoms: list[tuple[str, ...]] = []
    tries = 0
    while len(randoms) < n and tries < n * 200:
        tries += 1
        picks: list = []
        for pos, count in (("QB", 1), ("RB", 2), ("WR", 3), ("TE", 1), ("D", 1)):
            plist = by_pos.get(pos, [])
            if len(plist) < count:
                break
            picks.extend(plist[i] for i in rng.choice(len(plist), size=count, replace=False))
        if len(picks) != 8:
            continue
        chosen = {p.fd_id for p in picks}
        flex = [p for p in slate.players
                if p.position in ("RB", "WR", "TE") and p.fd_id not in chosen]
        if not flex:
            continue
        picks.append(flex[rng.integers(len(flex))])
        if sum(p.salary for p in picks) <= spec.salary_cap:
            randoms.append(tuple(p.fd_id for p in picks))
    return {"random": randoms}


@dataclass
class RankedLineup:
    candidate: Candidate
    p_win: float
    ev: float
    median: float
    p90: float
    mean_rank: float

    def summary(self) -> str:
        return (f"P(win)={self.p_win:.1%} EV=${self.ev:.2f} med={self.median:.1f} "
                f"p90={self.p90:.1f} avg_rank={self.mean_rank:.1f} "
                f"${self.candidate.salary} proj={self.candidate.proj_sum}")


def rank_candidates(candidates: list[Candidate], sim: SlateSimulator,
                    field: FieldModel, spec: ContestSpec) -> list[RankedLineup]:
    """Score every candidate against the simulated field. Returns sorted by EV desc."""
    if not field.lineups:
        raise ValueError("empty field model")
    field_totals = sim.score_many([(l, None) for l in field.lineups])   # (n_opp, n_sims)
    ranked: list[RankedLineup] = []
    for c in candidates:
        mine = sim.score(c.player_ids, c.mvp_id).totals                 # (n_sims,)
        beaten = (mine[None, :] > field_totals).sum(axis=0)             # opponents beaten per sim
        rank = field_totals.shape[0] + 1 - beaten                       # my rank, 1 = best
        p_win = float((rank == 1).mean())
        payouts = np.array([spec.payout_for_rank(int(r)) for r in range(1, field_totals.shape[0] + 2)])
        ev = float(payouts[rank - 1].mean() - spec.entry_fee)
        ranked.append(RankedLineup(
            candidate=c, p_win=p_win, ev=round(ev, 2),
            median=round(float(np.median(mine)), 2),
            p90=round(float(np.percentile(mine, 90)), 2),
            mean_rank=round(float(rank.mean()), 2),
        ))
    return sorted(ranked, key=lambda r: (r.ev, r.p_win), reverse=True)
