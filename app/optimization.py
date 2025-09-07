import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def optimize_lineup(
    player_df: pd.DataFrame,
    game_type: str = "league",
    salary_cap: int = 60000,
    enforce_stack: bool = True,
    min_stack_receivers: int = 1,
    lock_indices: Optional[List[int]] = None,
    ban_indices: Optional[List[int]] = None
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Optimize lineup using linear programming or heuristic approach
    
    Args:
        player_df: DataFrame with player data
        game_type: "league" or "h2h" 
        salary_cap: Maximum salary allowed
        enforce_stack: Whether to enforce QB stacking
        min_stack_receivers: Minimum receivers to stack with QB
        lock_indices: List of player indices to lock in lineup
        ban_indices: List of player indices to exclude
        
    Returns:
        Tuple of (lineup_indices, metadata)
    """
    
    if player_df.empty:
        return [], {"error": "No players available", "method": "none"}
    
    # Clean inputs
    lock_indices = lock_indices or []
    ban_indices = ban_indices or []
    
    # Remove banned players
    if ban_indices:
        player_df = player_df[~player_df.index.isin(ban_indices)].copy()
    
    # Validate required columns
    required_cols = ['PLAYER NAME', 'POS', 'TEAM', 'SALARY', 'PROJ PTS']
    missing_cols = [col for col in required_cols if col not in player_df.columns]
    if missing_cols:
        return [], {"error": f"Missing required columns: {missing_cols}", "method": "validation"}
    
    # Fill NaN values with defaults
    player_df['SALARY'] = pd.to_numeric(player_df['SALARY'], errors='coerce').fillna(4000)
    player_df['PROJ PTS'] = pd.to_numeric(player_df['PROJ PTS'], errors='coerce').fillna(0)
    
    # Remove players with zero salary or projection
    player_df = player_df[(player_df['SALARY'] > 0) & (player_df['PROJ PTS'] > 0)].copy()
    
    if len(player_df) < 9:
        return [], {"error": "Insufficient players after filtering", "method": "validation"}
    
    # Try linear programming optimization first
    try:
        return _optimize_with_pulp(
            player_df, game_type, salary_cap, enforce_stack,
            min_stack_receivers, lock_indices
        )
    except ImportError:
        logger.info("PuLP not available, using heuristic optimization")
        return _optimize_heuristic(
            player_df, game_type, salary_cap, enforce_stack,
            min_stack_receivers, lock_indices
        )
    except Exception as e:
        logger.warning(f"Linear programming failed: {e}, falling back to heuristic")
        return _optimize_heuristic(
            player_df, game_type, salary_cap, enforce_stack,
            min_stack_receivers, lock_indices
        )

def _optimize_with_pulp(
    player_df: pd.DataFrame,
    game_type: str,
    salary_cap: int,
    enforce_stack: bool,
    min_stack_receivers: int,
    lock_indices: List[int]
) -> Tuple[List[int], Dict[str, Any]]:
    """Optimize using PuLP linear programming"""
    
    import pulp
    
    # Create problem
    prob = pulp.LpProblem("FanDuel_DFS_Optimization", pulp.LpMaximize)
    
    # Decision variables
    player_vars = {}
    for idx in player_df.index:
        player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
    
    # Objective: maximize projected points (with game type adjustments)
    if game_type == "h2h":
        # For head-to-head, weight high projection players more heavily
        objective_weights = player_df['PROJ PTS'] * (1 + 0.1 * (player_df['PROJ PTS'] > player_df['PROJ PTS'].quantile(0.7)))
    else:
        # For league play, use straight projections
        objective_weights = player_df['PROJ PTS']
    
    prob += pulp.lpSum([
        objective_weights.loc[idx] * player_vars[idx] 
        for idx in player_df.index
    ])
    
    # Constraints
    
    # 1. Salary constraint
    prob += pulp.lpSum([
        player_df.loc[idx, 'SALARY'] * player_vars[idx] 
        for idx in player_df.index
    ]) <= salary_cap
    
    # 2. Total players = 9
    prob += pulp.lpSum([player_vars[idx] for idx in player_df.index]) == 9
    
    # 3. Position constraints
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'QB'
    ]) == 1
    
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'RB'
    ]) >= 2
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'RB'
    ]) <= 3
    
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'WR'
    ]) >= 3
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'WR'
    ]) <= 4
    
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'TE'
    ]) >= 1
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'TE'
    ]) <= 2
    
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] == 'DST'
    ]) == 1
    
    # 4. FLEX constraint (RB + WR + TE = 7)
    flex_positions = ['RB', 'WR', 'TE']
    prob += pulp.lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] in flex_positions
    ]) == 7
    
    # 5. Lock constraints
    for idx in lock_indices:
        if idx in player_vars:
            prob += player_vars[idx] == 1
    
    # 6. Stacking constraints (if enforced)
    if enforce_stack:
        qb_indices = player_df[player_df['POS'] == 'QB'].index
        for qb_idx in qb_indices:
            qb_team = player_df.loc[qb_idx, 'TEAM']
            
            # Find potential stack mates
            stack_mates = player_df[
                (player_df['TEAM'] == qb_team) & 
                (player_df['POS'].isin(['WR', 'TE'])) &
                (player_df.index != qb_idx)
            ].index
            
            if len(stack_mates) > 0:
                # If QB is selected, must have at least min_stack_receivers teammates
                prob += pulp.lpSum([
                    player_vars[mate_idx] for mate_idx in stack_mates
                ]) >= min_stack_receivers * player_vars[qb_idx]
    
    # 7. Team exposure limits (max 4 players from same team)
    for team in player_df['TEAM'].unique():
        team_players = player_df[player_df['TEAM'] == team].index
        if len(team_players) > 4:
            prob += pulp.lpSum([
                player_vars[idx] for idx in team_players
            ]) <= 4
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    if prob.status == 1:  # Optimal solution found
        lineup_indices = []
        for idx in player_df.index:
            if player_vars[idx].varValue == 1:
                lineup_indices.append(idx)
        
        total_proj = sum(player_df.loc[idx, 'PROJ PTS'] for idx in lineup_indices)
        total_salary = sum(player_df.loc[idx, 'SALARY'] for idx in lineup_indices)
        
        metadata = {
            "method": "linear_programming",
            "game_type": game_type,
            "total_projection": round(total_proj, 2),
            "total_salary": total_salary,
            "optimization_status": "optimal",
            "solver": "pulp"
        }
        
        return lineup_indices, metadata
    else:
        return [], {
            "error": f"No feasible solution found (status: {prob.status})", 
            "method": "linear_programming_failed"
        }

def _optimize_heuristic(
    player_df: pd.DataFrame,
    game_type: str,
    salary_cap: int,
    enforce_stack: bool,
    min_stack_receivers: int,
    lock_indices: List[int]
) -> Tuple[List[int], Dict[str, Any]]:
    """Optimize using heuristic greedy approach"""
    
    # Calculate value scores
    player_df = player_df.copy()
    player_df['value_score'] = player_df['PROJ PTS'] / (player_df['SALARY'] / 1000)
    
    # Adjust scores for game type
    if game_type == "h2h":
        # For H2H, boost high-projection players
        high_proj_bonus = (player_df['PROJ PTS'] > player_df['PROJ PTS'].quantile(0.8)).astype(float) * 0.1
        player_df['adj_score'] = player_df['PROJ PTS'] * (1 + high_proj_bonus)
    else:
        player_df['adj_score'] = player_df['value_score']
    
    # Get top players by position
    top_qbs = player_df[player_df['POS'] == 'QB'].nlargest(8, 'adj_score')
    top_rbs = player_df[player_df['POS'] == 'RB'].nlargest(15, 'adj_score')  
    top_wrs = player_df[player_df['POS'] == 'WR'].nlargest(20, 'adj_score')
    top_tes = player_df[player_df['POS'] == 'TE'].nlargest(12, 'adj_score')
    top_dsts = player_df[player_df['POS'] == 'DST'].nlargest(8, 'adj_score')
    
    best_lineup = []
    best_score = -1
    attempts = 0
    max_attempts = 500
    
    # Try different QB-stack combinations
    for _, qb in top_qbs.iterrows():
        if attempts >= max_attempts:
            break
            
        qb_idx = qb.name
        qb_team = qb['TEAM']
        
        # Find potential stack mates
        stack_candidates = player_df[
            (player_df['TEAM'] == qb_team) & 
            (player_df['POS'].isin(['WR', 'TE'])) &
            (player_df.index != qb_idx)
        ].nlargest(6, 'adj_score')
        
        if enforce_stack and len(stack_candidates) < min_stack_receivers:
            continue
        
        # Try different stack combinations
        if enforce_stack and len(stack_candidates) >= min_stack_receivers:
            import itertools
            for stack_combo in itertools.combinations(stack_candidates.index, min_stack_receivers):
                attempts += 1
                lineup = _build_heuristic_lineup(
                    qb_idx, list(stack_combo), player_df, top_rbs, top_wrs, 
                    top_tes, top_dsts, salary_cap, lock_indices
                )
                if lineup:
                    score = sum(player_df.loc[idx, 'PROJ PTS'] for idx in lineup)
                    if score > best_score:
                        best_score = score
                        best_lineup = lineup
        else:
            # No stacking - just build best lineup with this QB
            attempts += 1
            lineup = _build_heuristic_lineup(
                qb_idx, [], player_df, top_rbs, top_wrs, 
                top_tes, top_dsts, salary_cap, lock_indices
            )
            if lineup:
                score = sum(player_df.loc[idx, 'PROJ PTS'] for idx in lineup)
                if score > best_score:
                    best_score = score
                    best_lineup = lineup
    
    if best_lineup:
        total_proj = sum(player_df.loc[idx, 'PROJ PTS'] for idx in best_lineup)
        total_salary = sum(player_df.loc[idx, 'SALARY'] for idx in best_lineup)
        
        metadata = {
            "method": "heuristic",
            "game_type": game_type,
            "total_projection": round(total_proj, 2),
            "total_salary": total_salary,
            "attempts": attempts,
            "optimization_status": "completed"
        }
        
        return best_lineup, metadata
    else:
        return [], {
            "error": "No valid lineup found with heuristic approach",
            "method": "heuristic_failed",
            "attempts": attempts
        }

def _build_heuristic_lineup(
    qb_idx: int,
    stack_indices: List[int],
    player_df: pd.DataFrame,
    top_rbs: pd.DataFrame,
    top_wrs: pd.DataFrame, 
    top_tes: pd.DataFrame,
    top_dsts: pd.DataFrame,
    salary_cap: int,
    lock_indices: List[int]
) -> Optional[List[int]]:
    """Build a complete lineup using heuristic approach"""
    
    lineup = [qb_idx] + stack_indices + lock_indices
    used_salary = sum(player_df.loc[idx, 'SALARY'] for idx in lineup)
    
    if used_salary >= salary_cap:
        return None
    
    # Track what positions we still need
    positions_filled = {}
    for idx in lineup:
        pos = player_df.loc[idx, 'POS'] 
        positions_filled[pos] = positions_filled.get(pos, 0) + 1
    
    # Required positions: QB(1), RB(2-3), WR(3-4), TE(1-2), DST(1)
    needs = {
        'QB': max(0, 1 - positions_filled.get('QB', 0)),
        'RB': max(0, 2 - positions_filled.get('RB', 0)), 
        'WR': max(0, 3 - positions_filled.get('WR', 0)),
        'TE': max(0, 1 - positions_filled.get('TE', 0)),
        'DST': max(0, 1 - positions_filled.get('DST', 0))
    }
    
    # Fill required positions first
    remaining_budget = salary_cap - used_salary
    
    # RBs
    if needs['RB'] > 0:
        available_rbs = top_rbs[~top_rbs.index.isin(lineup)]
        for i, (_, rb) in enumerate(available_rbs.iterrows()):
            if needs['RB'] <= 0:
                break
            if rb['SALARY'] <= remaining_budget:
                lineup.append(rb.name)
                remaining_budget -= rb['SALARY']
                needs['RB'] -= 1
    
    # WRs  
    if needs['WR'] > 0:
        available_wrs = top_wrs[~top_wrs.index.isin(lineup)]
        for i, (_, wr) in enumerate(available_wrs.iterrows()):
            if needs['WR'] <= 0:
                break
            if wr['SALARY'] <= remaining_budget:
                lineup.append(wr.name)
                remaining_budget -= wr['SALARY']
                needs['WR'] -= 1
    
    # TE
    if needs['TE'] > 0:
        available_tes = top_tes[~top_tes.index.isin(lineup)]
        for i, (_, te) in enumerate(available_tes.iterrows()):
            if needs['TE'] <= 0:
                break
            if te['SALARY'] <= remaining_budget:
                lineup.append(te.name)
                remaining_budget -= te['SALARY']
                needs['TE'] -= 1
    
    # DST
    if needs['DST'] > 0:
        available_dsts = top_dsts[~top_dsts.index.isin(lineup)]
        # Avoid DST playing against QB's team
        qb_team = player_df.loc[qb_idx, 'TEAM']
        qb_opp = player_df.loc[qb_idx, 'OPP'] if 'OPP' in player_df.columns else ''
        
        for i, (_, dst) in enumerate(available_dsts.iterrows()):
            if needs['DST'] <= 0:
                break
            # Skip if DST is playing against our QB
            if dst['TEAM'] == qb_opp:
                continue
            if dst['SALARY'] <= remaining_budget:
                lineup.append(dst.name)
                remaining_budget -= dst['SALARY']
                needs['DST'] -= 1
    
    # Check if we have all required positions
    if any(needs[pos] > 0 for pos in ['QB', 'RB', 'WR', 'TE', 'DST']):
        return None
    
    # Fill FLEX spot if needed (need exactly 9 players)
    while len(lineup) < 9 and remaining_budget > 3000:
        # Get all remaining FLEX-eligible players (RB, WR, TE)
        flex_candidates = []
        
        for df, pos in [(top_rbs, 'RB'), (top_wrs, 'WR'), (top_tes, 'TE')]:
            available = df[~df.index.isin(lineup)]
            affordable = available[available['SALARY'] <= remaining_budget]
            flex_candidates.extend(affordable.index.tolist())
        
        if not flex_candidates:
            break
            
        # Sort by value score and take the best
        flex_scores = [(idx, player_df.loc[idx, 'adj_score']) for idx in flex_candidates]
        flex_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_flex_idx = flex_scores[0][0]
        lineup.append(best_flex_idx)
        remaining_budget -= player_df.loc[best_flex_idx, 'SALARY']
    
    # Final validation
    if len(lineup) != 9:
        return None
    
    # Check position constraints
    final_positions = {}
    for idx in lineup:
        pos = player_df.loc[idx, 'POS']
        final_positions[pos] = final_positions.get(pos, 0) + 1
    
    # Validate FanDuel constraints
    if (final_positions.get('QB', 0) != 1 or
        final_positions.get('RB', 0) < 2 or final_positions.get('RB', 0) > 3 or
        final_positions.get('WR', 0) < 3 or final_positions.get('WR', 0) > 4 or
        final_positions.get('TE', 0) < 1 or final_positions.get('TE', 0) > 2 or
        final_positions.get('DST', 0) != 1):
        return None
    
    # FLEX constraint (RB + WR + TE = 7)
    flex_total = final_positions.get('RB', 0) + final_positions.get('WR', 0) + final_positions.get('TE', 0)
    if flex_total != 7:
        return None
    
    return lineup
