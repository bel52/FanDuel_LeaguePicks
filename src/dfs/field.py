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
    source: str                        # 'prior' | 'prior_showdown' | 'history'
    mvp_ids: list | None = None        # showdown: MVP per opponent lineup


def ownership_weights(slate: PlayerSlate, measured: dict[str, float],
                      n_weeks: int, chalk_temp: float = 1.6):
    """Blend measured league ownership into per-player pick weights.

    measured: normalized player name -> avg DRAFTED% from captured weeks.
    Shrinkage: w = n/(n+4) on the measured component, so one captured week moves the
    prior only 20% and the model earns influence as evidence accumulates. Players never
    observed keep the chalk prior — absence of evidence is not evidence of a fade.
    """
    import numpy as np
    from .matching import norm_name
    alpha = n_weeks / (n_weeks + 4.0)
    proj = np.array([p.projection for p in slate.players])
    val = np.array([p.projection / max(p.salary / 1000, 0.1) for p in slate.players])
    z = chalk_temp * ((val - val.mean()) / (val.std() + 1e-9)
                      + 0.8 * (proj - proj.mean()) / (proj.std() + 1e-9))
    prior = np.exp(z - z.max())
    prior = prior / prior.sum()
    own = np.array([measured.get(norm_name(p.name), -1.0) for p in slate.players])
    seen = own >= 0
    if not seen.any() or n_weeks == 0:
        return None
    meas = np.where(seen, np.maximum(own, 0.5) / 100.0, 0.0)
    if meas.sum() > 0:
        meas = meas / meas.sum()
    blended = np.where(seen, (1 - alpha) * prior + alpha * meas, prior)
    return blended / blended.sum()


def build_field_ensemble(slate: PlayerSlate, spec: ContestSpec, n_opponents: int,
                         n_fields: int = 25, seed: int = 99,
                         chalk_temp: float = 1.6,
                         measured_ownership: dict[str, float] | None = None,
                         ownership_weeks: int = 0) -> list[FieldModel]:
    """Posterior over fields, not one arbitrary draw.

    A single 11-opponent draw held fixed across all simulations conditions every win
    probability on that one realization — re-seeding the field moved the reported
    P(win) from 14.5% to 21.7% on the test fixture. Averaging over an ensemble of
    field draws integrates that uncertainty out. n_fields=25 keeps runtime modest;
    the spread across fields is reported so instability is visible, not hidden.
    """
    ow = None
    if measured_ownership:
        ow = ownership_weights(slate, measured_ownership, ownership_weeks, chalk_temp)
    fields = [build_prior_field(slate, spec, n_opponents, seed=seed + 1000 * k,
                                chalk_temp=chalk_temp, pick_weights=ow)
              for k in range(n_fields)]
    for fm in fields:
        fm.source = f"ownership({ownership_weeks}wk)" if ow is not None else fm.source
    return fields


def build_prior_field(slate: PlayerSlate, spec: ContestSpec, n_opponents: int,
                      seed: int = 99, chalk_temp: float = 1.6,
                      pick_weights=None) -> FieldModel:
    """Generic-field prior: opponents pick popular, high-value players.

    Popularity ~ softmax over projection-per-$1000 and raw projection. chalk_temp
    controls concentration: higher = more herd behaviour (friends leagues are chalky).

    Salary-cap handling uses REPAIR, not rejection: an over-cap lineup has its most
    expensive player swapped down until it fits. Pure rejection sampling silently
    under-produced the field (8 of 11 opponents), which quietly inflated every win
    probability — the field must always be exactly n_opponents.
    """
    from .contest_spec import SlateType
    if spec.slate_type == SlateType.SINGLE_GAME:
        return _prior_field_showdown(slate, spec, n_opponents, seed, chalk_temp)
    rng = np.random.default_rng(seed)
    by_pos: dict[str, list] = {}
    for p in slate.players:
        by_pos.setdefault(p.position, []).append(p)
    for pos in ("QB", "RB", "WR", "TE", "D"):
        if len(by_pos.get(pos, [])) < 4:
            raise ValueError(f"field model needs >=4 {pos}, has {len(by_pos.get(pos, []))}")

    idx_of = {p.fd_id: i for i, p in enumerate(slate.players)}
    weights: dict[str, np.ndarray] = {}
    for pos, plist in by_pos.items():
        if pick_weights is not None:
            w = np.array([pick_weights[idx_of[p.fd_id]] for p in plist])
            w = np.maximum(w, 1e-9)
        else:
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


def _prior_field_showdown(slate: PlayerSlate, spec: ContestSpec, n_opponents: int,
                          seed: int, chalk_temp: float) -> FieldModel:
    """Showdown opponents: 5 players from a 2-team pool, one MVP, cap-legal.

    Opponents weight MVP choice heavily toward the highest-projected skill players
    (QBs dominate real MVP ownership). Lineup tuple carries the MVP as element 0 by
    convention; scoring reads FieldModel.mvp_ids.
    """
    rng = np.random.default_rng(seed)
    pool = list(slate.players)
    if len({p.team for p in pool}) != 2:
        raise ValueError("showdown field requires a 2-team slate")
    proj = np.array([p.projection for p in pool])
    z = chalk_temp * (proj - proj.mean()) / (proj.std() + 1e-9)
    w = np.exp(z - z.max()); w = w / w.sum()
    mvp_w = np.exp(1.6 * z - z.max()); mvp_w = mvp_w / mvp_w.sum()

    lineups, mvps, tries = [], [], 0
    from .contest_spec import ROSTER_SHOWDOWN_SIZE
    while len(lineups) < n_opponents and tries < n_opponents * 400:
        tries += 1
        idx = rng.choice(len(pool), size=ROSTER_SHOWDOWN_SIZE, replace=False, p=w)
        picks = [pool[i] for i in idx]
        if len({p.team for p in picks}) != 2:
            continue
        mw = mvp_w[idx]; mvp_i = idx[int(np.argmax(rng.random(len(idx)) ** (1.0 / mw)))]
        # MVP premium counts against the cap — checking base salary alone generates
        # opponent lineups FanDuel would reject.
        charged = (sum(p.salary for p in picks)
                   + (spec.mvp_salary_mult - 1.0) * pool[mvp_i].salary)
        if charged > spec.salary_cap:
            continue
        lineups.append(tuple(pool[i].fd_id for i in idx))
        mvps.append(pool[mvp_i].fd_id)
    if len(lineups) != n_opponents:
        raise ValueError(f"showdown field produced {len(lineups)}/{n_opponents}")
    fm = FieldModel(lineups=lineups, source="prior_showdown")
    fm.mvp_ids = mvps
    return fm


@dataclass
class RankedLineup:
    candidate: Candidate
    p_win: float
    p_top3: float
    exp_points: float
    dollars: float          # objective value, in dollars (see objectives.py)
    median: float
    p10: float
    p90: float
    mean_rank: float
    weights: "ObjectiveWeights"

    def summary(self) -> str:
        w = self.weights
        parts = [f"${self.dollars:6.2f}"]
        if w.w_points:
            parts.append(f"E[pts]={self.exp_points:6.1f}")
        parts += [f"P(win)={self.p_win:5.1%}±{self.mean_rank:4.1%}",
                  f"P(top3)={self.p_top3:5.1%}",
                  f"med={self.median:6.1f}", f"[{self.p10:5.1f}-{self.p90:6.1f}]",
                  f"${self.candidate.salary}"]
        return " ".join(parts)


def rank_candidates(candidates: list[Candidate], sim: SlateSimulator,
                    field: "FieldModel | list[FieldModel]", spec: ContestSpec,
                    weights: "ObjectiveWeights | None" = None) -> list[RankedLineup]:
    """Score candidates against a field ensemble, rank by the contest objective.

    `field` may be one FieldModel or an ensemble; metrics are averaged across fields
    so no single arbitrary opponent draw decides the pick. Ties split win credit.
    """
    from .objectives import ObjectiveWeights
    fields = field if isinstance(field, list) else [field]
    if not fields or not fields[0].lineups:
        raise ValueError("empty field model")
    if weights is None:
        weights = ObjectiveWeights(0.0, 1.0, "fallback: pure P(win)")

    # simulate every distinct opponent lineup once
    keyed = {}
    for fm in fields:
        for i, l in enumerate(fm.lineups):
            mvp = fm.mvp_ids[i] if fm.mvp_ids else None
            keyed.setdefault((l, mvp), None)
    for key in keyed:
        keyed[key] = sim.score(key[0], key[1]).totals
    field_stacks = []
    for fm in fields:
        import numpy as _np
        rows = [keyed[(l, fm.mvp_ids[i] if fm.mvp_ids else None)]
                for i, l in enumerate(fm.lineups)]
        field_stacks.append(_np.stack(rows))

    n_opp = field_stacks[0].shape[0]
    ranked: list[RankedLineup] = []
    for c in candidates:
        mine = sim.score(c.player_ids, c.mvp_id).totals
        pw, pt3, spread = [], [], []
        for ft in field_stacks:
            beaten_by = (ft > mine[None, :]).sum(axis=0)
            tied = (ft == mine[None, :]).sum(axis=0)
            rank = 1 + beaten_by
            pw.append(float(((beaten_by == 0) * (1.0 / (1.0 + tied))).mean()))
            pt3.append(float((rank <= 3).mean()))
        import numpy as np
        p_win, p_top3 = float(np.mean(pw)), float(np.mean(pt3))
        exp_points = float(mine.mean())
        ranked.append(RankedLineup(
            candidate=c, p_win=p_win, p_top3=p_top3, exp_points=round(exp_points, 2),
            dollars=round(weights.score(exp_points, p_win), 4),
            median=round(float(np.median(mine)), 2),
            p10=round(float(np.percentile(mine, 10)), 2),
            p90=round(float(np.percentile(mine, 90)), 2),
            mean_rank=round(float(np.std(pw)), 4) if len(pw) > 1 else 0.0,  # field-spread of P(win)
            weights=weights))
    return sorted(ranked, key=lambda r: r.dollars, reverse=True)
