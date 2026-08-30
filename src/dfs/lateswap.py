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
    proposed_mvp: str | None = None   # showdown: MVP of the proposed lineup
    criterion: str = "$/wk objective" # what old/new numbers ARE (model $ vs proj pts)

    @property
    def improves(self) -> bool:
        return self.new_dollars > self.old_dollars + 1e-9

    def summary(self, byid: dict) -> str:
        if not self.swaps:
            return (f"NO SWAP — current lineup is still optimal for the unlocked slots "
                    f"({self.old_dollars:.2f} {self.criterion}). {self.reason}")
        lines = [f"SWAP PROPOSAL: {self.old_dollars:.2f} -> {self.new_dollars:.2f} "
                 f"({self.new_dollars - self.old_dollars:+.2f} {self.criterion})"]
        for out_p, in_p in self.swaps:
            lines.append(f"  OUT {out_p.position:3s} {out_p.name:24s} "
                         f"(proj {out_p.projection or 0:5.1f}, ${out_p.salary})")
            lines.append(f"  IN  {in_p.position:3s} {in_p.name:24s} "
                         f"(proj {in_p.projection or 0:5.1f}, ${in_p.salary})")
        lines.append(f"  locked (game started): "
                     f"{', '.join(byid[i].name for i in sorted(self.locked_ids) if i in byid) or 'none'}")
        lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)


def propose_swap_maxproj(slate: PlayerSlate, current_ids: tuple, spec: ContestSpec,
                         schedule: KickoffSchedule, now: datetime | None = None,
                         reason: str = "scheduled late-swap check") -> SwapProposal:
    """Late swap for a MAX-PROJECTION entry: re-solve max projection under lock
    constraints, and propose a change only if projection strictly improves (or a
    lineup player was removed). The simulator has NO say here — an entry chosen for
    projection must not be churned Sunday morning by the model objective it was
    chosen over. old/new numbers are PROJECTED POINTS, not model dollars."""
    from .optimize import _solve_full, _solve_showdown, LEGALITY_ONLY
    now = now or datetime.now(timezone.utc)
    locked_teams = schedule.locked_teams(now)
    byid = {p.fd_id: p for p in slate.players}

    swept = [i for i in current_ids if i not in byid]        # removed by inactives
    present = [i for i in current_ids if i in byid]
    locked_ids = {i for i in present if byid[i].team in locked_teams}
    cur_proj = round(sum(byid[i].projection for i in present), 2)

    open_slots = [i for i in present if i not in locked_ids] + swept
    if not open_slots:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur_proj, cur_proj,
                            "all players locked — nothing can be changed",
                            criterion="projected pts")

    # players whose games already started cannot fill an open slot
    startable = [p for p in slate.players
                 if p.team not in locked_teams or p.fd_id in locked_ids]
    solve = (_solve_showdown if spec.slate_type == SlateType.SINGLE_GAME
             else _solve_full)
    best = solve(startable, spec, LEGALITY_ONLY, locked_ids, [], 0)
    if best is None:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur_proj, cur_proj,
                            "no valid alternative under lock constraints",
                            criterion="projected pts")

    forced = bool(swept)
    if set(best.player_ids) == set(present) and not forced:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur_proj, cur_proj, reason, criterion="projected pts",
                            proposed_mvp=best.mvp_id)
    new_proj = round(sum(byid[i].projection for i in best.player_ids), 2)
    if not forced and new_proj <= cur_proj + 1e-9:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur_proj, cur_proj, reason, criterion="projected pts")

    out_ids = set(current_ids) - set(best.player_ids)
    in_ids = set(best.player_ids) - set(current_ids)
    from .slate import SlatePlayer
    def _ghost(fd_id):
        return byid.get(fd_id) or SlatePlayer(fd_id=fd_id, name=f"(ruled out) {fd_id}",
                                              position="?", team="?", opponent="?",
                                              salary=0, game="")
    swaps = list(zip(sorted((_ghost(i) for i in out_ids), key=lambda p: p.position),
                     sorted((byid[i] for i in in_ids), key=lambda p: p.position)))
    return SwapProposal(current_ids, best.player_ids, locked_ids, swaps,
                        cur_proj, new_proj, reason, criterion="projected pts",
                        proposed_mvp=best.mvp_id)


def propose_swap(slate: PlayerSlate, current_ids: tuple, spec: ContestSpec,
                 schedule: KickoffSchedule, sim: SlateSimulator, field_model: FieldModel,
                 weights, now: datetime | None = None,
                 stack: StackRule | None = None, pool_size: int = 60,
                 reason: str = "scheduled late-swap check",
                 mvp_id: str | None = None) -> SwapProposal:
    """Re-optimize the unlocked portion of a lineup. Never touches locked players.

    Showdown: the whole lineup is one game, so lock is all-or-nothing — once the game
    kicks off nothing can change; before kickoff everything (including MVP) can. The
    current lineup is scored WITH its MVP, and any proposal carries its own MVP."""
    now = now or datetime.now(timezone.utc)
    locked_teams = schedule.locked_teams(now)
    byid = {p.fd_id: p for p in slate.players}

    # A lineup player missing from the slate was removed by the inactives sweep
    # (ruled OUT after we entered). That is exactly when late swap matters most —
    # his slot is forcibly open and the objective already reflects his absence.
    swept = [i for i in current_ids if i not in byid]
    present = [i for i in current_ids if i in byid]
    locked_ids = {i for i in present if byid[i].team in locked_teams}
    open_ids = [i for i in present if i not in locked_ids] + swept
    if swept:
        reason = (f"{len(swept)} rostered player(s) ruled OUT after entry "
                  f"({', '.join(swept)}); their slots are open. " + reason)

    from .objectives import score_lineup
    fm = field_model[0] if isinstance(field_model, list) else field_model
    fm_mvps = getattr(fm, "mvp_ids", None)
    field_totals = sim.score_many(
        [(l, fm_mvps[i] if fm_mvps else None) for i, l in enumerate(fm.lineups)])
    # score current lineup with swept players contributing ZERO (they will not play);
    # a swept MVP scores zero either way, so drop the multiplier with the player
    cur_mvp = mvp_id if (mvp_id in byid and mvp_id not in swept) else None
    cur = score_lineup(sim.score(tuple(present), cur_mvp).totals, field_totals, weights)

    if not open_ids:
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur.dollars, cur.dollars,
                            "all players locked — nothing can be changed",
                            proposed_mvp=cur_mvp)
    if swept and all(byid.get(i) is None or byid[i].team in locked_teams
                     for i in current_ids if i not in swept):
        pass  # swept slots stay open even when everything else is locked

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
                            "no valid alternative lineups under lock constraints",
                            proposed_mvp=cur_mvp)

    ranked = rank_candidates(pool, sim, field_model, spec, weights)
    best = ranked[0]

    # any valid lineup beats one with a guaranteed zero, so a swept slot forces a swap
    same_lineup = (set(best.candidate.player_ids) == set(current_ids)
                   and best.candidate.mvp_id == cur_mvp)
    if same_lineup or (not swept and best.dollars <= cur.dollars):
        return SwapProposal(current_ids, current_ids, locked_ids, [],
                            cur.dollars, cur.dollars, reason, proposed_mvp=cur_mvp)

    out_ids = set(current_ids) - set(best.candidate.player_ids)
    in_ids = set(best.candidate.player_ids) - set(current_ids)
    from .slate import SlatePlayer
    def _ghost(fd_id):
        return byid.get(fd_id) or SlatePlayer(fd_id=fd_id, name=f"(ruled out) {fd_id}",
                                              position="?", team="?", opponent="?",
                                              salary=0, game="")
    swaps = list(zip(sorted((_ghost(i) for i in out_ids), key=lambda p: p.position),
                     sorted((byid[i] for i in in_ids), key=lambda p: p.position)))
    return SwapProposal(current_ids, best.candidate.player_ids, locked_ids, swaps,
                        cur.dollars, best.dollars, reason,
                        proposed_mvp=best.candidate.mvp_id)
