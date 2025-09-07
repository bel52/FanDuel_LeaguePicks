import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

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
    Optimize lineup using linear programming
    """
    
    if player_df.empty:
        return [], {"error": "No players available"}
    
    # Remove banned players
    if ban_indices:
        player_df = player_df[~player_df.index.isin(ban_indices)]
    
    # Create LP problem
    prob = LpProblem(f"FanDuel_{game_type}_Optimization", LpMaximize)
    
    # Decision variables
    player_vars = {}
    for idx in player_df.index:
        player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')
    
    # Objective: maximize projected points
    prob += lpSum([
        player_df.loc[idx, 'PROJ PTS'] * player_vars[idx] 
        for idx in player_df.index
    ])
    
    # Constraints
    
    # 1. Salary constraint
    prob += lpSum([
        player_df.loc[idx, 'SALARY'] * player_vars[idx] 
        for idx in player_df.index
    ]) <= salary_cap
    
    # 2. Total players = 9
    prob += lpSum([player_vars[idx] for idx in player_df.index]) == 9
    
    # 3. Position constraints
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'QB']) == 1
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'RB']) >= 2
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'RB']) <= 3
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'WR']) >= 3
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'WR']) <= 4
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'TE']) >= 1
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'TE']) <= 2
    prob += lpSum([player_vars[idx] for idx in player_df.index if player_df.loc[idx, 'POS'] == 'DST']) == 1
    
    # 4. FLEX constraint (RB + WR + TE = 7)
    flex_positions = ['RB', 'WR', 'TE']
    prob += lpSum([
        player_vars[idx] for idx in player_df.index 
        if player_df.loc[idx, 'POS'] in flex_positions
    ]) == 7
    
    # 5. Lock constraints
    if lock_indices:
        for idx in lock_indices:
            if idx in player_vars:
                prob += player_vars[idx] == 1
    
    # 6. Stacking constraints (if enforced)
    if enforce_stack:
        for qb_idx in player_df[player_df['POS'] == 'QB'].index:
            qb_team = player_df.loc[qb_idx, 'TEAM']
            teammates = player_df[
                (player_df['TEAM'] == qb_team) & 
                (player_df['POS'].isin(['WR', 'TE']))
            ].index
            
            # If QB is selected, must have at least min_stack_receivers teammates
            if len(teammates) > 0:
                prob += lpSum([player_vars[tm_idx] for tm_idx in teammates]) >= \
                       min_stack_receivers * player_vars[qb_idx]
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))
    
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
            "optimization_status": "optimal"
        }
        
        return lineup_indices, metadata
    else:
        return [], {"error": "No feasible solution found", "status": prob.status}

def optimize_text(width: int = 100) -> str:
    """Generate optimized lineup as text"""
    from app.data_ingestion import load_data_from_input_dir
    from app.formatting import build_text_report
    
    df, warnings = load_data_from_input_dir()
    
    if df is None or df.empty:
        return "No player data available. Please upload CSV files to data/input/"
    
    lineup_indices, metadata = optimize_lineup(df, game_type="league")
    
    if not lineup_indices:
        return f"Optimization failed: {metadata.get('error', 'Unknown error')}"
    
    # Build result for formatting
    lineup_data = []
    total_salary = 0
    total_proj = 0
    
    for idx in lineup_indices:
        player = df.loc[idx]
        lineup_data.append(player.to_dict())
        total_salary += int(player['SALARY'])
        total_proj += float(player['PROJ PTS'])
    
    result = {
        "game_type": "league",
        "lineup": lineup_data,
        "total_projected_points": round(total_proj, 2),
        "cap_usage": {
            "total_salary": total_salary,
            "remaining": 60000 - total_salary
        }
    }
    
    return build_text_report(result, width=width)
