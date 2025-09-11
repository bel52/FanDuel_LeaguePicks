"""
Construct DFS lineups using a simple brute-force search with optional
adjustments for odds and weather data.

This module defines a ``build_lineup`` function that takes a list of
player dictionaries and returns the best projected lineup under a salary
cap. Players are grouped by position, filtered by a value threshold,
ranked by adjusted projection, and then brute-forced across a trimmed
pool to find the optimal combination. Odds and weather data, when
provided, are used to adjust player projections via ``lineup_rules``.

The search space is constrained by constants like ``QB_TOP`` or
``RB_TOP`` to keep runtime manageable. If no valid lineup can be built,
the function returns ``(None, -1.0)``.
"""

from typing import List, Dict, Any, Tuple, Optional
import itertools
from src import lineup_rules as rules

CAP = 60000

# Maximum number of players to consider at each position. Adjust these
# values to trade off runtime versus lineup quality.
QB_TOP = 4
RB_TOP = 7
WR_TOP = 10
TE_TOP = 4
DST_TOP = 4
FLEX_TOP = 15

Positions = ('QB', 'RB', 'WR', 'TE', 'DST')


def _by_pos(players: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    pos = {p: [] for p in Positions}
    for pl in players:
        if pl.get('Pos') in pos:
            pos[pl['Pos']].append(pl)
    return pos


def _adj(p: Dict[str, Any], game_info_map: Dict[str, Dict[str, Any]]) -> float:
    # Compute adjusted score; fallback to raw projection if no game info
    return float(rules.adjusted_score(p, game_info_map.get(p.get('Team'))))


def _val(p: Dict[str, Any]) -> float:
    from src.util import value_per_1k
    return value_per_1k(p.get('ProjFP', 0.0), p.get('Salary', 0))


def _top_pool(players: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    # Filter by value threshold then sort by value and adjusted projection
    val_ok = [p for p in players if rules.meets_value(p)]
    val_ok.sort(key=lambda p: (_val(p), p.get('ProjFP', 0.0)), reverse=True)
    if len(val_ok) >= top_n:
        return val_ok[:top_n]
    rem = [p for p in players if p not in val_ok]
    rem.sort(key=lambda p: (_val(p), p.get('ProjFP', 0.0)), reverse=True)
    return (val_ok + rem)[:top_n]


def build_lineup(players: List[Dict[str, Any]], odds_data: Optional[List[Dict[str, Any]]] = None,
                 weather_data: Optional[List[Dict[str, Any]]] = None,
                 injuries: Optional[Dict[str, str]] = None,
                 locked: Optional[set[str]] = None) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Build the optimal lineup from a list of players.

    Args:
        players: List of player dictionaries with keys ``Name``, ``Team``, ``Pos``, ``ProjFP``, ``Salary``.
        odds_data: Optional list of games with implied totals. Each dict should
            contain ``Home``, ``Away``, ``HomeImplied``, ``AwayImplied``.
        weather_data: Optional list of weather dicts. Each dict may include
            ``Team``, ``wind_mph``, ``precip_chance``.

    Returns:
        (lineup, score): The lineup is a dict mapping roster slots to
            player dicts; score is the sum of adjusted projections. If no
            lineup is possible, returns (None, -1.0).
    """
    if not players:
        return None, -1.0

    # Filter out injured or locked players
    filtered_players: List[Dict[str, Any]] = []
    injuries = injuries or {}
    locked = locked or set()
    for p in players:
        name = p.get('Name')
        # Exclude if player is injured (status OUT/IR/DOUBTFUL) or locked
        status = injuries.get(name, '').upper()
        if status in ('OUT', 'IR', 'DOUBTFUL'):
            continue
        if name in locked:
            continue
        filtered_players.append(p)
    players = filtered_players
    pools = _by_pos(players)
    # Create game info mapping for odds and weather
    game_info_map: Dict[str, Dict[str, Any]] = {}
    if odds_data:
        for g in odds_data:
            try:
                home = g.get('Home')
                away = g.get('Away')
                if home and g.get('HomeImplied'):
                    game_info_map.setdefault(home, {})['implied_total'] = float(g['HomeImplied'])
                if away and g.get('AwayImplied'):
                    game_info_map.setdefault(away, {})['implied_total'] = float(g['AwayImplied'])
            except Exception:
                continue
    if weather_data:
        for w in weather_data:
            try:
                team = w.get('Team')
                if not team:
                    continue
                info = game_info_map.setdefault(team, {})
                if 'wind_mph' in w and w['wind_mph'] != '':
                    info['wind_mph'] = int(w['wind_mph'])
                if 'precip_chance' in w and w['precip_chance'] != '':
                    info['precip_chance'] = int(w['precip_chance'])
            except Exception:
                continue

    # Determine candidate pools per position
    QBs = _top_pool(pools.get('QB', []), QB_TOP)
    RBs = _top_pool(pools.get('RB', []), RB_TOP)
    WRs = _top_pool(pools.get('WR', []), WR_TOP)
    TEs = _top_pool(pools.get('TE', []), TE_TOP)
    DSTs = _top_pool(pools.get('DST', []), DST_TOP)

    # Ensure minimum counts
    if not (QBs and len(RBs) >= 2 and len(WRs) >= 3 and TEs and DSTs):
        return None, -1.0

    # Pre-sort flex candidates once
    flex_sorted = sorted(RBs + WRs + TEs, key=lambda p: (_val(p), p.get('ProjFP', 0.0)), reverse=True)

    best_score = -1.0
    best: Optional[Dict[str, Any]] = None

    # Precompute adjusted projections for each player
    adj_scores = {id(p): _adj(p, game_info_map) for p in players}

    for qb in QBs:
        for rb2 in itertools.combinations(RBs, 2):
            for wr3 in itertools.combinations(WRs, 3):
                for te in TEs:
                    for dst in DSTs:
                        chosen = [qb, *rb2, *wr3, te, dst]
                        chosen_keys = {(c['Name'], c['Pos'], c['Team']) for c in chosen}
                        base_salary = sum(int(c.get('Salary', 0) or 0) for c in chosen)
                        if base_salary > CAP:
                            continue
                        # pick first FLEX that is not already in chosen and fits CAP
                        for fx in flex_sorted[:FLEX_TOP * 3]:
                            key = (fx['Name'], fx['Pos'], fx['Team'])
                            if key in chosen_keys:
                                continue
                            total_salary = base_salary + int(fx.get('Salary', 0) or 0)
                            if total_salary > CAP:
                                continue
                            lineup = {
                                'QB': qb, 'RB1': rb2[0], 'RB2': rb2[1],
                                'WR1': wr3[0], 'WR2': wr3[1], 'WR3': wr3[2],
                                'TE': te, 'FLEX': fx, 'DST': dst
                            }
                            total = sum(adj_scores[id(p)] for p in lineup.values())
                            if total > best_score:
                                best = lineup
                                best_score = round(total, 2)
                            break  # take best available flex for this core
    return best, best_score
