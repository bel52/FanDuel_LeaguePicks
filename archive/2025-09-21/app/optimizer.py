from __future__ import annotations
from typing import List, Tuple
from dataclasses import dataclass

from app.models import Player, GameType, FLEX_ELIGIBLE
from app.config import SALARY_CAP

@dataclass
class RosterRule:
    qbs: int = 1
    rbs: int = 2
    wrs: int = 3
    tes: int = 1
    flex: int = 1
    defs: int = 1

def select_optimal(players: List[Player], game_type: GameType = "gpp") -> Tuple[List[Player], int, float, str]:
    rules = RosterRule()
    pool = list(players)
    by_pos = lambda pos: [p for p in pool if p.position == pos]
    sort_key = lambda p: (p.projection / max(1, p.salary), p.projection)

    chosen: List[Player] = []
    def take(best_from: List[Player], n: int):
        return sorted(best_from, key=sort_key, reverse=True)[:max(0, n)]

    chosen += take(by_pos("QB"), rules.qbs)
    chosen += take(by_pos("RB"), rules.rbs)
    chosen += take(by_pos("WR"), rules.wrs)
    chosen += take(by_pos("TE"), rules.tes)
    chosen += take(by_pos("DEF"), rules.defs)

    used_ids = {p.id for p in chosen}
    remaining_cap = SALARY_CAP - sum(p.salary for p in chosen)

    flex_cands = [p for p in pool if p.position in FLEX_ELIGIBLE and p.id not in used_ids]
    flex_cands = sorted(flex_cands, key=lambda p: p.projection, reverse=True)
    for p in flex_cands:
        if p.salary <= remaining_cap:
            chosen.append(p)
            used_ids.add(p.id)
            remaining_cap -= p.salary
            break

    def try_improve():
        nonlocal chosen, remaining_cap
        current_salary = sum(p.salary for p in chosen)
        current_proj = sum(p.projection for p in chosen)
        for i, p_out in enumerate(chosen):
            candidates = [x for x in pool if x.position == p_out.position and x.id not in used_ids]
            if p_out.position in FLEX_ELIGIBLE:
                candidates += [x for x in pool if x.position in FLEX_ELIGIBLE and x.id not in used_ids]
            for p_in in sorted(candidates, key=lambda x: x.projection, reverse=True)[:25]:
                new_salary = current_salary - p_out.salary + p_in.salary
                new_proj = current_proj - p_out.projection + p_in.projection
                if new_salary <= SALARY_CAP and new_proj > current_proj + 0.15:
                    used_ids.remove(p_out.id)
                    chosen[i] = p_in
                    used_ids.add(p_in.id)
                    remaining_cap = SALARY_CAP - new_salary
                    return True
        return False

    for _ in range(6):
        if not try_improve():
            break

    total_salary = sum(p.salary for p in chosen)
    total_proj = sum(p.projection for p in chosen)
    notes = f"Greedy+local optimization. Salary used {total_salary}/{SALARY_CAP}."
    return chosen, total_salary, total_proj, notes
