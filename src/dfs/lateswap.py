"""Late swap — rebuild a lineup around players whose games have started.

FanDuel's rule in late-swap-eligible contests: a player can be changed until his own
game kicks off. So after the 1:00 games start, the 4:05/4:25/SNF slots are still live.
The correct operation is a CONSTRAINED re-optimization:

  * every player whose game has started is locked into his slot (he cannot be removed,
    and his salary is spent)
  * every unlocked slot is re-optimized over players whose games have NOT started,
    using current information (fresh injuries, fresh Vegas)
  * the objective is unchanged — same dollar weights as the original build

A swap is only proposed when it actually improves the objective; a lineup that is
still the best available stays put, and the report says so explicitly.

Design rule carried over from the injury pipeline: the engine never silently changes
anything. It produces a proposal + reason; Brett approves and edits on FanDuel.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .contest_spec import ContestSpec, SlateType
from .kickoffs import KickoffSchedule
from .optimize import generate_pool, StackRule, Candidate
from .simulate import SlateSimulator
from .field import build_prior_field, rank_candidates, FieldModel
from .slate import PlayerSlate, SlateError


@dataclass
class SwapProposal:
    original_ids: tuple
    proposed_ids: tuple
    locked_ids: set
    swaps: list                       # (out_player, in_player)
    old_dollars: float
    new_dollars: float
    reason: str

    @property
    def improves(self) -> bool:
        return self.new_dollars > self.old_dollars + 1e-9

    def summary(self, byid: dict) -> str:
        if not self.swaps:
            return (f"NO SWAP — current lineup is still optimal for the unlocked slots "
                    f"(${self.old_dollars:.2f}). {self.reason}")
        lines = [f"SWAP PROPOSAL: ${self.old_dollars:.2f} -> ${self.new_dollars:.2f} "
                 f"({self.new_dollars - self.old_dollars:+.2f}/wk objective)"]
        for out_p, in_p in self.swaps:
            lines.append(f"  OUT {out_p.position:3s} {out_p.name:24s} "
                         f"(proj {out_p.projection or 0:5.1f}, ${out_p.salary})")
            lines.append(f"  IN  {in_p.position:3s} {in_p.name:24s} "
                         f"(proj {in_p.projection or 0:5.1f}, ${in_p.salary})")
        lines.append(f"  locked (game started): "
                     f"{', '.join(byid[i].name for i in sorted(self.locked_ids) if i in byid) or 'none'}")
        lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)


def propose_swap(slate: PlayerSlate, current_ids: tuple, spec: ContestSpec,
                 schedule: KickoffSchedule, sim: SlateSimulator, field_model: FieldModel,
                 weights, now: datetime | None = None,
                 stack: StackRule | None = None, pool_size: int = 60,
                 reason: str = "scheduled late-swap check") -> SwapProposal:
    """Re-optimize the unlocked portion of a lineup. Never touches locked players."""
    now = now or datetime.now(timezone.utc)
    locked_teams = schedule.locked_teams(now)
    byid = {p.fd_id: p for p in slate.players}

    missing = [i for i in current_ids if i not in byid]
    if missing:
        raise SlateError(f"current lineup has players not in slate: {missing} — "
                         "was a player removed by the inactives sweep? Rebuild instead.")

    locked_ids = {i for i in current_ids if byid[i].team in locked_teams}
    open_ids = [i for i in current_ids if i not in locked_ids]

    from .objectives import score_lineup
    field_totals = sim.score_many([(l, None) for l in field_model.lineups])
    cur = score_lineup(sim.score(current_ids).totals, field_totals, weights)

    if not open_ids:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur.dollars, cur.dollars,
                            "all players locked — nothing can be changed")

    # Candidate pool constrained to: locked players forced in, locked-team players
    # excluded from the open slots (their games already started).
    startable_exclusions = {p.fd_id for p in slate.players
                            if p.team in locked_teams and p.fd_id not in locked_ids}
    pool = generate_pool(slate, spec, n=pool_size, stack=stack or StackRule(),
                         min_unique=1, locked_ids=locked_ids,
                         excluded_ids=startable_exclusions)
    if not pool:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur.dollars, cur.dollars,
                            "no valid alternative lineups under lock constraints")

    ranked = rank_candidates(pool, sim, field_model, spec, weights)
    best = ranked[0]

    if set(best.candidate.player_ids) == set(current_ids) or best.dollars <= cur.dollars:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur.dollars, cur.dollars, reason)

    out_ids = set(current_ids) - set(best.candidate.player_ids)
    in_ids = set(best.candidate.player_ids) - set(current_ids)
    swaps = list(zip(sorted((byid[i] for i in out_ids), key=lambda p: p.position),
                     sorted((byid[i] for i in in_ids), key=lambda p: p.position)))
    return SwapProposal(current_ids, best.candidate.player_ids, locked_ids, swaps,
                        cur.dollars, best.dollars, reason)
