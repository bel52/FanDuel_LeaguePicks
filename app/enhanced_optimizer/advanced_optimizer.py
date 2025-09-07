# app/enhanced_optimizer/advanced_optimizer.py
from ortools.linear_solver import pywraplp
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AdvancedDFSOptimizer:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=8)
        
    async def optimize_multiple_lineups(
        self, 
        players_df: pd.DataFrame, 
        num_lineups: int = 150,
        uniqueness_threshold: float = 0.7
    ) -> List[Dict]:
        """Generate multiple unique optimal lineups"""
        
        loop = asyncio.get_running_loop()
        
        # Create tasks for parallel optimization
        tasks = []
        for i in range(num_lineups):
            task = loop.run_in_executor(
                self.executor,
                self._optimize_single_lineup,
                players_df,
                i,
                uniqueness_threshold
            )
            tasks.append(task)
        
        lineups = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed optimizations
        valid_lineups = [
            lineup for lineup in lineups 
            if not isinstance(lineup, Exception) and lineup is not None
        ]
        
        return valid_lineups[:num_lineups]
    
    def _optimize_single_lineup(
        self, 
        players_df: pd.DataFrame,
        iteration: int,
        uniqueness_threshold: float
    ) -> Optional[Dict]:
        """Optimize a single lineup using OR-Tools"""
        
        try:
            # Use OR-Tools for advanced constraint handling
            solver = pywraplp.Solver.CreateSolver('SCIP')
            if not solver:
                raise Exception("SCIP solver not available")
            
            # Decision variables (binary)
            x = {}
            for idx in players_df.index:
                x[idx] = solver.IntVar(0, 1, f'x_{idx}')
            
            # Objective: Maximize projected points
            objective = solver.Objective()
            for idx in players_df.index:
                objective.SetCoefficient(x[idx], players_df.loc[idx, 'projected_points'])
            objective.SetMaximization()
            
            # Salary cap constraint (FanDuel: $60,000)
            salary_constraint = solver.Constraint(0, 60000)
            for idx in players_df.index:
                salary_constraint.SetCoefficient(x[idx], players_df.loc[idx, 'salary'])
            
            # FanDuel position constraints
            self._add_fanduel_position_constraints(solver, x, players_df)
            
            # Team constraints (max 4 from same team)
            self._add_team_constraints(solver, x, players_df)
            
            # Solve optimization
            status = solver.Solve()
            
            if status == pywraplp.Solver.OPTIMAL:
                lineup = []
                total_salary = 0
                total_points = 0
                
                for idx in players_df.index:
                    if x[idx].solution_value() == 1:
                        player = players_df.loc[idx]
                        lineup.append({
                            'player_id': player['id'],
                            'name': f"{player['first_name']} {player['last_name']}",
                            'position': player['position'],
                            'salary': player['salary'],
                            'projected_points': player['projected_points'],
                            'team': player['team']
                        })
                        total_salary += player['salary']
                        total_points += player['projected_points']
                
                return {
                    'lineup': lineup,
                    'total_salary': total_salary,
                    'total_projected_points': total_points,
                    'salary_remaining': 60000 - total_salary
                }
            
            return None
            
        except Exception as e:
            print(f"Optimization failed for iteration {iteration}: {str(e)}")
            return None
    
    def _add_fanduel_position_constraints(self, solver, x, players_df):
        """Add FanDuel-specific position constraints"""
        positions = {
            'QB': (1, 1),    # Exactly 1 QB
            'RB': (2, 2),    # Exactly 2 RB  
            'WR': (3, 3),    # Exactly 3 WR
            'TE': (1, 1),    # Exactly 1 TE
            'K': (1, 1),     # Exactly 1 K
            'DST': (1, 1),   # Exactly 1 Defense
        }
        
        for position, (min_count, max_count) in positions.items():
            constraint = solver.Constraint(min_count, max_count)
            position_players = players_df[players_df['position'] == position]
            for idx in position_players.index:
                constraint.SetCoefficient(x[idx], 1)
        
        # FLEX constraint (1 additional RB/WR/TE)
        flex_constraint = solver.Constraint(1, 1)
        flex_positions = ['RB', 'WR', 'TE']
        flex_players = players_df[players_df['position'].isin(flex_positions)]
        for idx in flex_players.index:
            flex_constraint.SetCoefficient(x[idx], 1)
    
    def _add_team_constraints(self, solver, x, players_df):
        """Add team constraints (max 4 from same team)"""
        teams = players_df['team'].unique()
        
        for team in teams:
            team_constraint = solver.Constraint(0, 4)  # Max 4 from same team
            team_players = players_df[players_df['team'] == team]
            for idx in team_players.index:
                team_constraint.SetCoefficient(x[idx], 1)
