import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
from typing import List, Dict, Any, Tuple

def run_optimization(
    player_df: pd.DataFrame,
    game_mode: str,
    salary_cap: int = 60000,
    top_n: int = 1
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Runs the lineup optimization, ensuring the objective function is correctly set.
    """
    errors = []
    
    # --- Data Validation ---
    if player_df.empty:
        return [], ["Player data is empty."]
    if 'PROJ PTS' not in player_df.columns or player_df['PROJ PTS'].isnull().all():
        return [], ["'PROJ PTS' column is missing or contains all null values."]

    prob = LpProblem(f"FanDuel_{game_mode}_Optimization", LpMaximize)

    player_indices = player_df.index
    player_vars = LpVariable.dicts("player", player_indices, cat='Binary')

    # --- OBJECTIVE FUNCTION (CRITICAL FIX) ---
    # This line was the primary cause of the 0.0 projection. It is now corrected.
    prob += lpSum(player_df.loc[i, 'PROJ PTS'] * player_vars[i] for i in player_indices), "TotalProjectedPoints"

    # --- CONSTRAINTS ---
    # Salary Constraint
    prob += lpSum(player_df.loc[i, 'SALARY'] * player_vars[i] for i in player_indices) <= salary_cap, "TotalSalary"

    # Positional Constraints
    prob += lpSum(player_vars[i] for i in player_indices) == 9, "TotalPlayers"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'QB') == 1, "NumQB"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'RB') >= 2, "MinRB"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'RB') <= 3, "MaxRB"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'WR') >= 3, "MinWR"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'WR') <= 4, "MaxWR"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'TE') >= 1, "MinTE"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'TE') <= 2, "MaxTE"
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] == 'DST') == 1, "NumDST"
    
    # Flex Constraint
    flex_positions = ['RB', 'WR', 'TE']
    prob += lpSum(player_vars[i] for i in player_indices if player_df.loc[i, 'POS'] in flex_positions) == 7, "FlexCount"

    # --- SOLVE ---
    prob.solve(PULP_CBC_CMD(msg=0))

    # --- Extract Results ---
    selected_indices = [i for i in player_indices if player_vars[i].value() == 1]

    if len(selected_indices) != 9:
        errors.append("Optimization failed to find a valid 9-player lineup.")
        return [], errors

    lineup_info = {
        "players": selected_indices,
        "total_points": sum(player_df.loc[i, 'PROJ PTS'] for i in selected_indices),
        "total_salary": sum(player_df.loc[i, 'SALARY'] for i in selected_indices)
    }

    return [lineup_info], errors
