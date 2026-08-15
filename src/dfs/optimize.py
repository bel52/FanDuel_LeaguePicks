"""MIP candidate-pool generator.

Role change from v5 (locked): the optimizer does NOT pick "the best lineup". It emits a
diverse pool of valid, structurally-sensible candidates. Ranking happens in simulate.py
against the opponent field. No magic multipliers anywhere in this module — the objective
here is plain projected points; structure comes from CONSTRAINTS.

Diversity: each new lineup must differ from every previously generated lineup by at least
`min_unique` players (max-overlap constraint), which is a real portfolio control rather
than v5's randomize-projections-by-20% hack.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import pulp

from .contest_spec import (ContestSpec, SlateType, ROSTER_FULL, ROSTER_FULL_SIZE,
                           ROSTER_SHOWDOWN_SIZE, MVP_MULTIPLIER)
from .slate import PlayerSlate, SlatePlayer, SlateError


@dataclass
class StackRule:
    """Structural constraints. Enforced, never bonused."""
    require_qb_stack: int = 1          # min same-team pass-catchers with the QB
    require_bring_back: int = 0        # min opposing-team skill players from the QB's game
    max_per_team: int = 4
    max_per_game: int = 5


@dataclass
class Candidate:
    player_ids: tuple[str, ...]
    salary: int
    proj_sum: float
    mvp_id: Optional[str] = None       # showdown only
    stack_team: str = ""
    meta: dict = field(default_factory=dict)

    def key(self) -> frozenset:
        return frozenset(self.player_ids)


def _skill(p: SlatePlayer) -> bool:
    return p.position in ("RB", "WR", "TE")


def generate_pool(slate: PlayerSlate, spec: ContestSpec, n: int = 150,
                  stack: StackRule | None = None, min_unique: int = 3,
                  locked_ids: set[str] | None = None,
                  excluded_ids: set[str] | None = None) -> list[Candidate]:
    """Generate up to n diverse valid lineups."""
    slate.require_projections()
    stack = stack or StackRule()
    locked_ids = locked_ids or set()
    excluded_ids = excluded_ids or set()

    pool_players = [p for p in slate.players if p.fd_id not in excluded_ids]
    if len(pool_players) < 20 and spec.slate_type == SlateType.FULL:
        raise SlateError(f"player pool too small: {len(pool_players)}")

    candidates: list[Candidate] = []
    seen: set[frozenset] = set()

    for _ in range(n):
        cand = (_solve_showdown if spec.slate_type == SlateType.SINGLE_GAME else _solve_full)(
            pool_players, spec, stack, locked_ids, candidates, min_unique)
        if cand is None:
            break
        if cand.key() in seen:
            break
        seen.add(cand.key())
        candidates.append(cand)
    return candidates


def _diversity_constraints(prob, players, x, prior: list[Candidate], min_unique: int, size: int):
    """Each new lineup overlaps any prior lineup by at most size - min_unique players."""
    idx = {p.fd_id: i for i, p in enumerate(players)}
    for c in prior:
        overlap = [x[idx[pid]] for pid in c.player_ids if pid in idx]
        if overlap:
            prob += pulp.lpSum(overlap) <= size - min_unique


def _solve_full(players: list[SlatePlayer], spec: ContestSpec, stack: StackRule,
                locked: set[str], prior: list[Candidate], min_unique: int) -> Optional[Candidate]:
    prob = pulp.LpProblem("pool", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(players))}
    prob += pulp.lpSum(players[i].projection * x[i] for i in x)

    prob += pulp.lpSum(players[i].salary * x[i] for i in x) <= spec.salary_cap
    prob += pulp.lpSum(x.values()) == ROSTER_FULL_SIZE
    for pos, (lo, hi) in ROSTER_FULL.items():
        ix = [i for i, p in enumerate(players) if p.position == pos]
        if not ix:
            raise SlateError(f"no {pos} in pool")
        prob += pulp.lpSum(x[i] for i in ix) >= lo
        prob += pulp.lpSum(x[i] for i in ix) <= hi
    for i, p in enumerate(players):
        if p.fd_id in locked:
            prob += x[i] == 1

    teams: dict[str, list[int]] = {}
    games: dict[str, list[int]] = {}
    for i, p in enumerate(players):
        teams.setdefault(p.team, []).append(i)
        games.setdefault("|".join(sorted((p.team, p.opponent))), []).append(i)
    for ix in teams.values():
        prob += pulp.lpSum(x[i] for i in ix) <= stack.max_per_team
    for ix in games.values():
        prob += pulp.lpSum(x[i] for i in ix) <= stack.max_per_game

    # QB stack: if QB from team T is used, >= require_qb_stack pass-catchers from T.
    # Implemented as a linear implication (no bonus terms).
    if stack.require_qb_stack:
        for team, ix in teams.items():
            qbs = [i for i in ix if players[i].position == "QB"]
            catchers = [i for i in ix if players[i].position in ("WR", "TE")]
            if qbs:
                prob += (pulp.lpSum(x[i] for i in catchers)
                         >= stack.require_qb_stack * pulp.lpSum(x[i] for i in qbs))
    # Bring-back: QB from T implies >= k skill players from T's opponent
    if stack.require_bring_back:
        for team, ix in teams.items():
            qbs = [i for i in ix if players[i].position == "QB"]
            if not qbs:
                continue
            opp = players[ix[0]].opponent
            opp_skill = [i for i, p in enumerate(players) if p.team == opp and _skill(p)]
            if opp_skill:
                prob += (pulp.lpSum(x[i] for i in opp_skill)
                         >= stack.require_bring_back * pulp.lpSum(x[i] for i in qbs))

    _diversity_constraints(prob, players, x, prior, min_unique, ROSTER_FULL_SIZE)

    if prob.solve(pulp.PULP_CBC_CMD(msg=0)) != pulp.LpStatusOptimal:
        return None
    chosen = [players[i] for i in x if x[i].varValue and x[i].varValue > 0.5]
    if len(chosen) != ROSTER_FULL_SIZE:
        return None
    qb = next((p for p in chosen if p.position == "QB"), None)
    return Candidate(
        player_ids=tuple(p.fd_id for p in chosen),
        salary=sum(p.salary for p in chosen),
        proj_sum=round(sum(p.projection for p in chosen), 2),
        stack_team=qb.team if qb else "",
    )


def _solve_showdown(players: list[SlatePlayer], spec: ContestSpec, stack: StackRule,
                    locked: set[str], prior: list[Candidate], min_unique: int) -> Optional[Candidate]:
    """5 players (1 MVP at 1.5x + 4 utility), both teams represented."""
    prob = pulp.LpProblem("sd", pulp.LpMaximize)
    n = len(players)
    x = {i: pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n)}   # in lineup
    m = {i: pulp.LpVariable(f"m{i}", cat="Binary") for i in range(n)}   # is MVP

    prob += pulp.lpSum(players[i].projection * (x[i] + (MVP_MULTIPLIER - 1) * m[i]) for i in range(n))
    prob += pulp.lpSum(players[i].salary * x[i] for i in range(n)) <= spec.salary_cap
    prob += pulp.lpSum(x.values()) == ROSTER_SHOWDOWN_SIZE
    prob += pulp.lpSum(m.values()) == 1
    for i in range(n):
        prob += m[i] <= x[i]
        if players[i].fd_id in locked:
            prob += x[i] == 1

    teams: dict[str, list[int]] = {}
    for i, p in enumerate(players):
        teams.setdefault(p.team, []).append(i)
    if len(teams) != 2:
        raise SlateError(f"showdown slate must have 2 teams, has {len(teams)}")
    for ix in teams.values():
        prob += pulp.lpSum(x[i] for i in ix) >= 1      # both teams represented
        prob += pulp.lpSum(x[i] for i in ix) <= 4

    _diversity_constraints(prob, players, x, prior, min_unique, ROSTER_SHOWDOWN_SIZE)

    if prob.solve(pulp.PULP_CBC_CMD(msg=0)) != pulp.LpStatusOptimal:
        return None
    chosen = [i for i in range(n) if x[i].varValue and x[i].varValue > 0.5]
    mvp = next((i for i in range(n) if m[i].varValue and m[i].varValue > 0.5), None)
    return Candidate(
        player_ids=tuple(players[i].fd_id for i in chosen),
        salary=sum(players[i].salary for i in chosen),
        proj_sum=round(sum(players[i].projection * (MVP_MULTIPLIER if i == mvp else 1.0)
                           for i in chosen), 2),
        mvp_id=players[mvp].fd_id if mvp is not None else None,
    )
