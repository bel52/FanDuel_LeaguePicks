import itertools
import logging
from typing import List, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_lineup(
    df: pd.DataFrame,
    salary_cap: int = 60000,
    enforce_stack: bool = True,
    min_stack_receivers: int = 1,
    lock_indices: Optional[List[int]] = None,
    ban_indices: Optional[List[int]] = None,
    leverage_weight: float = 0.2
) -> List[int]:
    """
    Lineup optimization. Tries LP if PuLP is present; otherwise a heuristic.
    """
    lock_indices = set(lock_indices or [])
    ban_indices = set(ban_indices or [])

    # Keep original indices
    use_df = df[~df.index.isin(ban_indices)].copy()
    if use_df.empty:
        return []

    # Auto-disable stacking if teams look unreliable
    teams_nonunk = set([t for t in use_df["TEAM"].astype(str) if t and t != "UNK"])
    if (not teams_nonunk) or (use_df["TEAM"].nunique() <= 1):
        enforce_stack = False
        logger.info("Auto-disabled stacking due to unreliable team data")

    # Calculate value & adjusted projections
    use_df['value'] = use_df['PROJ PTS'] / (use_df['SALARY'] / 1000)
    use_df['adj_proj'] = use_df['PROJ PTS']

    if 'OWN_PCT' in use_df.columns:
        own = use_df['OWN_PCT'].fillna(0)
        use_df['leverage'] = use_df['PROJ PTS'] / (own + 1)
        use_df['adj_proj'] = use_df['PROJ PTS'] * (1 - leverage_weight) + use_df['leverage'] * leverage_weight

    # Try LP first if available
    try:
        import pulp
        logger.info("Using linear programming optimizer")
        return _opt_lp(use_df, salary_cap, enforce_stack, min_stack_receivers, lock_indices)
    except ImportError:
        logger.info("PuLP not available; using heuristic optimizer")
        lineup = _opt_heuristic(use_df, salary_cap, enforce_stack, min_stack_receivers, lock_indices)
        if lineup:
            return lineup
        
        # Fallback to cheapest valid lineup
        logger.info("Trying cheapest valid lineup fallback")
        cheap = _cheapest_valid_lineup(use_df)
        if not cheap:
            logger.error("Could not build any valid lineup")
            return []
        
        cheap_salary = sum(use_df.loc[cheap, 'SALARY'])
        if cheap_salary > salary_cap:
            logger.error(f"Even cheapest lineup costs ${cheap_salary}, exceeds cap ${salary_cap}")
            return []
        return cheap


def _opt_lp(df: pd.DataFrame, salary_cap: int, enforce_stack: bool, min_stack_receivers: int, lock_indices: Set[int]) -> List[int]:
    """Linear programming optimization"""
    import pulp
    
    prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
    
    # Decision variables for each player (using original indices)
    x = {i: pulp.LpVariable(f"p_{i}", cat="Binary") for i in df.index}
    
    # Objective: maximize adjusted projection
    prob += pulp.lpSum(df.loc[i, 'adj_proj'] * x[i] for i in df.index)
    
    # Salary constraint
    prob += pulp.lpSum(df.loc[i, 'SALARY'] * x[i] for i in df.index) <= salary_cap
    
    # Position constraints
    pos_indices = {}
    for pos in ['QB', 'RB', 'WR', 'TE', 'DST']:
        pos_indices[pos] = df[df['POS'] == pos].index.tolist()
    
    # Exact position requirements
    prob += pulp.lpSum(x[i] for i in pos_indices['QB']) == 1
    prob += pulp.lpSum(x[i] for i in pos_indices['DST']) == 1
    
    # Flexible position requirements
    prob += pulp.lpSum(x[i] for i in pos_indices['RB']) >= 2
    prob += pulp.lpSum(x[i] for i in pos_indices['RB']) <= 3
    prob += pulp.lpSum(x[i] for i in pos_indices['WR']) >= 3
    prob += pulp.lpSum(x[i] for i in pos_indices['WR']) <= 4
    prob += pulp.lpSum(x[i] for i in pos_indices['TE']) >= 1
    prob += pulp.lpSum(x[i] for i in pos_indices['TE']) <= 2
    
    # Total 9 players
    prob += pulp.lpSum(x[i] for i in df.index) == 9
    
    # Lock constraints
    for i in lock_indices:
        if i in x:
            prob += x[i] == 1
    
    # Stacking constraints
    if enforce_stack:
        for qb_i in pos_indices['QB']:
            qb_team = df.loc[qb_i, 'TEAM']
            if pd.notna(qb_team) and qb_team != "UNK":
                stack_mates = df[(df['TEAM'] == qb_team) & (df['POS'].isin(['WR', 'TE']))].index.tolist()
                if stack_mates:
                    prob += pulp.lpSum(x[j] for j in stack_mates) >= min_stack_receivers * x[qb_i]
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    if pulp.LpStatus[prob.status] == 'Optimal':
        return [i for i in df.index if x[i].value() == 1]
    else:
        logger.warning("LP solver did not find optimal solution")
        raise ImportError("LP optimization failed")


def _opt_heuristic(df: pd.DataFrame, salary_cap: int, enforce_stack: bool, min_stack_receivers: int, lock_indices: Set[int]) -> List[int]:
    """Heuristic optimization with improved efficiency"""
    
    # Create position pools
    qbs = df[df['POS'] == 'QB'].nlargest(10, 'adj_proj')
    rbs = df[df['POS'] == 'RB'].nlargest(25, 'adj_proj')
    wrs = df[df['POS'] == 'WR'].nlargest(30, 'adj_proj')
    tes = df[df['POS'] == 'TE'].nlargest(15, 'adj_proj')
    dsts = df[df['POS'] == 'DST'].nlargest(10, 'adj_proj')
    
    best_lineup = []
    best_score = -1
    attempts = 0
    max_attempts = 5000
    
    for qb_idx in qbs.index:
        if attempts >= max_attempts:
            break
            
        qb_team = df.loc[qb_idx, 'TEAM']
        qb_salary = df.loc[qb_idx, 'SALARY']
        
        # Get potential stack mates
        if enforce_stack and qb_team != "UNK":
            stack_pool = df[(df['TEAM'] == qb_team) & (df['POS'].isin(['WR', 'TE']))].nlargest(8, 'adj_proj')
        else:
            stack_pool = pd.DataFrame()  # Empty if no stacking
        
        # Try with different stack sizes
        stack_options = []
        if enforce_stack and len(stack_pool) >= min_stack_receivers:
            for size in range(min_stack_receivers, min(3, len(stack_pool) + 1)):
                for combo in itertools.combinations(stack_pool.index, size):
                    stack_options.append(combo)
        else:
            stack_options = [()]  # No stack
        
        for stack_indices in stack_options[:20]:  # Limit combinations
            attempts += 1
            if attempts >= max_attempts:
                break
            
            # Initialize lineup with QB and stack
            current_lineup = [qb_idx] + list(stack_indices)
            current_salary = qb_salary + sum(df.loc[list(stack_indices), 'SALARY']) if stack_indices else qb_salary
            
            if current_salary > salary_cap:
                continue
            
            # Count positions already filled
            positions_filled = {'QB': 1, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0}
            for idx in stack_indices:
                pos = df.loc[idx, 'POS']
                if pos in positions_filled:
                    positions_filled[pos] += 1
            
            # Fill remaining positions greedily
            remaining_budget = salary_cap - current_salary
            
            # Add RBs (need 2-3 total)
            rb_needed = 2 - positions_filled['RB']
            rb_candidates = rbs[~rbs.index.isin(current_lineup)]
            rb_candidates = rb_candidates[rb_candidates['SALARY'] <= remaining_budget * 0.4]  # Budget control
            
            if len(rb_candidates) >= rb_needed:
                selected_rbs = rb_candidates.head(rb_needed).index.tolist()
                current_lineup.extend(selected_rbs)
                current_salary += df.loc[selected_rbs, 'SALARY'].sum()
                positions_filled['RB'] += len(selected_rbs)
                remaining_budget = salary_cap - current_salary
            else:
                continue
            
            # Add WRs (need 3-4 total)
            wr_needed = max(0, 3 - positions_filled['WR'])
            wr_candidates = wrs[~wrs.index.isin(current_lineup)]
            wr_candidates = wr_candidates[wr_candidates['SALARY'] <= remaining_budget * 0.5]
            
            if len(wr_candidates) >= wr_needed:
                selected_wrs = wr_candidates.head(wr_needed).index.tolist()
                current_lineup.extend(selected_wrs)
                current_salary += df.loc[selected_wrs, 'SALARY'].sum()
                positions_filled['WR'] += len(selected_wrs)
                remaining_budget = salary_cap - current_salary
            else:
                continue
            
            # Add TE if needed (need 1-2 total)
            if positions_filled['TE'] == 0:
                te_candidates = tes[~tes.index.isin(current_lineup)]
                te_candidates = te_candidates[te_candidates['SALARY'] <= remaining_budget * 0.3]
                
                if not te_candidates.empty:
                    selected_te = te_candidates.index[0]
                    current_lineup.append(selected_te)
                    current_salary += df.loc[selected_te, 'SALARY']
                    positions_filled['TE'] += 1
                    remaining_budget = salary_cap - current_salary
                else:
                    continue
            
            # Add DST
            dst_candidates = dsts[~dsts.index.isin(current_lineup)]
            dst_candidates = dst_candidates[dst_candidates['SALARY'] <= remaining_budget * 0.2]
            
            if not dst_candidates.empty:
                selected_dst = dst_candidates.index[0]
                current_lineup.append(selected_dst)
                current_salary += df.loc[selected_dst, 'SALARY']
                positions_filled['DST'] += 1
                remaining_budget = salary_cap - current_salary
            else:
                continue
            
            # Add FLEX if needed (should have 9 total)
            if len(current_lineup) < 9:
                flex_pool = pd.concat([
                    rbs[~rbs.index.isin(current_lineup)],
                    wrs[~wrs.index.isin(current_lineup)],
                    tes[~tes.index.isin(current_lineup)]
                ])
                flex_pool = flex_pool[flex_pool['SALARY'] <= remaining_budget]
                
                if not flex_pool.empty:
                    flex_pool = flex_pool.nlargest(10, 'adj_proj')
                    selected_flex = flex_pool.index[0]
                    current_lineup.append(selected_flex)
                    current_salary += df.loc[selected_flex, 'SALARY']
                else:
                    continue
            
            # Validate lineup
            if len(current_lineup) != 9:
                continue
            
            # Check lock constraints
            if not lock_indices.issubset(set(current_lineup)):
                continue
            
            # Calculate score
            current_score = df.loc[current_lineup, 'adj_proj'].sum()
            
            if current_score > best_score:
                best_score = current_score
                best_lineup = current_lineup
    
    logger.info(f"Heuristic optimizer tried {attempts} combinations")
    return best_lineup


def _cheapest_valid_lineup(df: pd.DataFrame) -> List[int]:
    """Build the cheapest possible valid lineup"""
    lineup = []
    
    # Get cheapest at each position
    for pos, count in [('QB', 1), ('DST', 1), ('RB', 2), ('WR', 3), ('TE', 1)]:
        cheapest = df[df['POS'] == pos].nsmallest(count, 'SALARY')
        lineup.extend(cheapest.index.tolist())
    
    # Add cheapest FLEX
    used_indices = set(lineup)
    flex_pool = df[(df['POS'].isin(['RB', 'WR', 'TE'])) & (~df.index.isin(used_indices))]
    if not flex_pool.empty:
        cheapest_flex = flex_pool.nsmallest(1, 'SALARY')
        lineup.extend(cheapest_flex.index.tolist())
    
    return lineup if len(lineup) == 9 else []
