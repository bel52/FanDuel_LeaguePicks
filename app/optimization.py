import itertools
import logging
from typing import List, Optional, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def optimize_lineup(
    df: pd.DataFrame,
    salary_cap: int = 60000,
    enforce_stack: bool = True,
    min_stack_receivers: int = 1,
    lock_indices: Optional[List[int]] = None,
    ban_indices: Optional[List[int]] = None
) -> List[int]:
    """
    Advanced lineup optimization with stacking and constraints
    """
    
    lock_indices = set(lock_indices or [])
    ban_indices = set(ban_indices or [])
    
    # Remove banned players
    use_df = df[~df.index.isin(ban_indices)].copy()
    
    # Add value metrics
    use_df['value'] = use_df['PROJ PTS'] / (use_df['SALARY'] / 1000)
    use_df['adj_proj'] = use_df['PROJ PTS']
    
    # Adjust for ownership if available
    if 'OWN_PCT' in use_df.columns:
        use_df['leverage'] = use_df['PROJ PTS'] / (use_df['OWN_PCT'] + 1)
        use_df['adj_proj'] = use_df['PROJ PTS'] * 0.8 + use_df['leverage'] * 0.2
    
    # Try linear programming if PuLP is available
    try:
        return optimize_with_lp(
            use_df, salary_cap, enforce_stack, 
            min_stack_receivers, lock_indices
        )
    except ImportError:
        logger.info("PuLP not available, using heuristic optimization")
        return optimize_heuristic(
            use_df, salary_cap, enforce_stack,
            min_stack_receivers, lock_indices
        )

def optimize_with_lp(
    df: pd.DataFrame,
    salary_cap: int,
    enforce_stack: bool,
    min_stack_receivers: int,
    lock_indices: Set[int]
) -> List[int]:
    """Linear programming optimization using PuLP"""
    import pulp
    
    # Create problem
    prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
    
    # Decision variables
    player_vars = {}
    for idx in df.index:
        player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
    
    # Objective: maximize adjusted projections
    prob += pulp.lpSum([
        df.loc[idx, 'adj_proj'] * player_vars[idx] 
        for idx in df.index
    ])
    
    # Salary constraint
    prob += pulp.lpSum([
        df.loc[idx, 'SALARY'] * player_vars[idx] 
        for idx in df.index
    ]) <= salary_cap
    
    # Position constraints
    position_limits = {
        'QB': (1, 1),
        'RB': (2, 3),
        'WR': (3, 4),
        'TE': (1, 2),
        'DST': (1, 1)
    }
    
    for pos, (min_count, max_count) in position_limits.items():
        pos_players = df[df['POS'] == pos].index
        if pos != 'RB' and pos != 'WR' and pos != 'TE':
            # Exact constraint for QB and DST
            prob += pulp.lpSum([player_vars[idx] for idx in pos_players]) == min_count
        else:
            # Range constraint for flex positions
            prob += pulp.lpSum([player_vars[idx] for idx in pos_players]) >= min_count
            prob += pulp.lpSum([player_vars[idx] for idx in pos_players]) <= max_count
    
    # Total players constraint (9 players)
    prob += pulp.lpSum([player_vars[idx] for idx in df.index]) == 9
    
    # FLEX constraint (7 RB/WR/TE total, since we have minimums)
    flex_positions = df[df['POS'].isin(['RB', 'WR', 'TE'])].index
    prob += pulp.lpSum([player_vars[idx] for idx in flex_positions]) == 7
    
    # Lock constraints
    for idx in lock_indices:
        if idx in player_vars:
            prob += player_vars[idx] == 1
    
    # Stacking constraint
    if enforce_stack:
        # For each QB, ensure they have teammates
        for qb_idx in df[df['POS'] == 'QB'].index:
            qb_team = df.loc[qb_idx, 'TEAM']
            teammates = df[(df['TEAM'] == qb_team) & 
                          (df['POS'].isin(['WR', 'TE']))].index
            
            # If QB is selected, must have at least min_stack_receivers teammates
            prob += pulp.lpSum([player_vars[tm_idx] for tm_idx in teammates]) >= \
                   min_stack_receivers * player_vars[qb_idx]
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    if pulp.LpStatus[prob.status] == 'Optimal':
        lineup = []
        for idx in df.index:
            if player_vars[idx].varValue == 1:
                lineup.append(idx)
        return lineup
    else:
        logger.warning("LP optimization failed, falling back to heuristic")
        return optimize_heuristic(df, salary_cap, enforce_stack, min_stack_receivers, lock_indices)

def optimize_heuristic(
    df: pd.DataFrame,
    salary_cap: int,
    enforce_stack: bool,
    min_stack_receivers: int,
    lock_indices: Set[int]
) -> List[int]:
    """Heuristic optimization using greedy search"""
    
    # Sort by position and value
    def get_top_n(pos: str, n: int):
        return df[df['POS'] == pos].nlargest(n, 'adj_proj')
    
    qbs = get_top_n('QB', 5)
    rbs = get_top_n('RB', 10)
    wrs = get_top_n('WR', 12)
    tes = get_top_n('TE', 6)
    dsts = get_top_n('DST', 5)
    
    best_lineup = []
    best_score = -1
    
    # Try combinations
    for qb_idx in qbs.index:
        qb = df.loc[qb_idx]
        qb_team = qb['TEAM']
        
        # Get potential stack mates
        stack_mates = df[(df['TEAM'] == qb_team) & 
                        (df['POS'].isin(['WR', 'TE']))].nlargest(5, 'adj_proj')
        
        if enforce_stack and len(stack_mates) < min_stack_receivers:
            continue
        
        # Try different stack combinations
        for stack_indices in itertools.combinations(stack_mates.index, min(min_stack_receivers, len(stack_mates))):
            used_indices = {qb_idx} | set(stack_indices)
            remaining_salary = salary_cap - qb['SALARY'] - sum(df.loc[idx, 'SALARY'] for idx in stack_indices)
            
            # Fill remaining positions
            lineup_indices = list(used_indices)
            
            # Add RBs (need 2-3)
            available_rbs = rbs[~rbs.index.isin(used_indices)]
            for rb_count in [2, 3]:
                for rb_combo in itertools.combinations(available_rbs.index, rb_count):
                    rb_cost = sum(df.loc[idx, 'SALARY'] for idx in rb_combo)
                    if rb_cost > remaining_salary * 0.4:  # Don't spend too much on RBs
                        continue
                    
                    # Add WRs (need total of 3-4 WR/TE)
                    wr_te_have = sum(1 for idx in stack_indices if df.loc[idx, 'POS'] in ['WR', 'TE'])
                    wr_need = max(3 - wr_te_have, 0)
                    
                    available_wrs = wrs[~wrs.index.isin(used_indices | set(rb_combo))]
                    if len(available_wrs) < wr_need:
                        continue
                    
                    for wr_combo in itertools.combinations(available_wrs.index, wr_need):
                        # Add TE if needed
                        te_have = sum(1 for idx in stack_indices if df.loc[idx, 'POS'] == 'TE')
                        if te_have == 0:
                            available_tes = tes[~tes.index.isin(used_indices | set(rb_combo) | set(wr_combo))]
                            if len(available_tes) == 0:
                                continue
                            te_idx = available_tes.index[0]
                        else:
                            te_idx = None
                        
                        # Add DST
                        available_dsts = dsts[~dsts.index.isin(used_indices)]
                        if len(available_dsts) == 0:
                            continue
                        dst_idx = available_dsts.index[0]
                        
                        # Build complete lineup
                        test_lineup = list(used_indices) + list(rb_combo) + list(wr_combo)
                        if te_idx is not None:
                            test_lineup.append(te_idx)
                        test_lineup.append(dst_idx)
                        
                        # Add FLEX if needed (should have 9 total)
                        if len(test_lineup) < 9:
                            flex_candidates = df[df['POS'].isin(['RB', 'WR', 'TE']) & 
                                               ~df.index.isin(test_lineup)]
                            if len(flex_candidates) > 0:
                                flex_idx = flex_candidates.nlargest(1, 'adj_proj').index[0]
                                test_lineup.append(flex_idx)
                        
                        # Check constraints
                        if len(test_lineup) != 9:
                            continue
                        
                        total_salary = sum(df.loc[idx, 'SALARY'] for idx in test_lineup)
                        if total_salary > salary_cap:
                            continue
                        
                        # Check if all locked players are included
                        if not lock_indices.issubset(set(test_lineup)):
                            continue
                        
                        # Calculate score
                        score = sum(df.loc[idx, 'adj_proj'] for idx in test_lineup)
                        if score > best_score:
                            best_score = score
                            best_lineup = test_lineup
    
    return best_lineup
