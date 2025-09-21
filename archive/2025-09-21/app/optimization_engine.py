# app/optimization_engine.py
import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus, value
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class OptimizationRequest:
    players_df: pd.DataFrame
    num_lineups: int = 1
    randomness: float = 0.25
    min_salary: int = 59200
    stack_qb_wr: bool = True
    team_limits: Dict[str, int] = None
    
@dataclass 
class LineupResult:
    lineups: List[pd.DataFrame]
    optimization_time: float
    status: str

class OptimizationEngine:
    def __init__(self):
        self.salary_cap = 60000
        self.roster_positions = {
            'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 
            'FLEX': 1, 'K': 1, 'DST': 1
        } 
        self.total_roster_size = 9
        self.min_teams = 3
        self.max_team_players = 4

    async def optimize_lineups(self, request: OptimizationRequest) -> LineupResult:
        """Generate optimized DFS lineups using linear programming"""
        start_time = datetime.now()
        
        try:
            # Validate input data
            self._validate_player_data(request.players_df)
            
            # Add randomness to projections if specified
            players_df = self._add_randomness(request.players_df, request.randomness)
            
            lineups = []
            used_lineups = set()
            
            for i in range(request.num_lineups):
                logger.info(f"Generating lineup {i + 1}/{request.num_lineups}")
                
                # Create optimization problem
                lineup = await self._solve_single_lineup(
                    players_df, used_lineups, request
                )
                
                if lineup is not None and not lineup.empty:
                    lineups.append(lineup)
                    # Add lineup signature to used set for uniqueness
                    lineup_signature = tuple(sorted(lineup['PlayerID'].tolist()))
                    used_lineups.add(lineup_signature)
                else:
                    logger.warning(f"Could not generate unique lineup {i + 1}")
                    break
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            return LineupResult(
                lineups=lineups,
                optimization_time=optimization_time,
                status=f"Generated {len(lineups)} lineups successfully"
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {str(e)}")

    async def _solve_single_lineup(
        self, 
        players_df: pd.DataFrame, 
        used_lineups: set,
        request: OptimizationRequest
    ) -> Optional[pd.DataFrame]:
        """Solve for a single optimal lineup using PuLP"""
        
        # Create linear programming problem
        prob = LpProblem("DFS_Lineup_Optimization", LpMaximize)
        
        # Decision variables - one binary variable per player
        player_vars = {}
        for idx, player in players_df.iterrows():
            player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')
        
        # Objective function - maximize projected points
        prob += lpSum([
            players_df.loc[idx, 'ProjectedPoints'] * player_vars[idx]
            for idx in players_df.index
        ])
        
        # Salary cap constraint
        prob += lpSum([
            players_df.loc[idx, 'Salary'] * player_vars[idx]
            for idx in players_df.index
        ]) <= self.salary_cap
        
        # Minimum salary constraint
        prob += lpSum([
            players_df.loc[idx, 'Salary'] * player_vars[idx]
            for idx in players_df.index
        ]) >= request.min_salary
        
        # Position constraints
        for position, count in self.roster_positions.items():
            if position == 'FLEX':
                # FLEX can be RB, WR, or TE
                flex_players = players_df[players_df['Position'].isin(['RB', 'WR', 'TE'])]
                prob += lpSum([
                    player_vars[idx] for idx in flex_players.index
                ]) >= count + sum([self.roster_positions.get(pos, 0) for pos in ['RB', 'WR', 'TE']])
            else:
                position_players = players_df[players_df['Position'] == position]
                prob += lpSum([
                    player_vars[idx] for idx in position_players.index
                ]) >= count
        
        # Total roster size constraint
        prob += lpSum([player_vars[idx] for idx in players_df.index]) == self.total_roster_size
        
        # Team diversity constraints
        teams = players_df['Team'].unique()
        for team in teams:
            team_players = players_df[players_df['Team'] == team]
            prob += lpSum([
                player_vars[idx] for idx in team_players.index
            ]) <= self.max_team_players
        
        # Custom team limits
        if request.team_limits:
            for team, limit in request.team_limits.items():
                team_players = players_df[players_df['Team'] == team]
                prob += lpSum([
                    player_vars[idx] for idx in team_players.index
                ]) <= limit
        
        # QB-WR stacking constraint
        if request.stack_qb_wr:
            self._add_stacking_constraints(prob, players_df, player_vars)
        
        # Uniqueness constraints (avoid duplicate lineups)
        for used_lineup in used_lineups:
            prob += lpSum([
                player_vars[idx] for idx in players_df.index 
                if players_df.loc[idx, 'PlayerID'] in used_lineup
            ]) <= len(used_lineup) - 1
        
        # Solve the problem
        prob.solve()
        
        if prob.status == 1:  # Optimal solution found
            # Extract selected players
            selected_players = []
            for idx in players_df.index:
                if player_vars[idx].value() == 1:
                    selected_players.append(idx)
            
            lineup_df = players_df.loc[selected_players].copy()
            return lineup_df
        else:
            logger.warning(f"No optimal solution found. Status: {LpStatus[prob.status]}")
            return None

    def _add_stacking_constraints(self, prob, players_df, player_vars):
        """Add QB-WR stacking constraints"""
        teams = players_df['Team'].unique()
        
        for team in teams:
            team_qbs = players_df[(players_df['Team'] == team) & (players_df['Position'] == 'QB')]
            team_wrs = players_df[(players_df['Team'] == team) & (players_df['Position'] == 'WR')]
            
            if not team_qbs.empty and not team_wrs.empty:
                # If a QB is selected, at least one WR from same team should be selected
                for qb_idx in team_qbs.index:
                    prob += lpSum([
                        player_vars[wr_idx] for wr_idx in team_wrs.index
                    ]) >= player_vars[qb_idx]

    def _validate_player_data(self, players_df: pd.DataFrame):
        """Validate that player data contains required fields"""
        required_fields = ['PlayerID', 'Name', 'Position', 'Team', 'Salary', 'ProjectedPoints']
        missing_fields = [field for field in required_fields if field not in players_df.columns]
        
        if missing_fields:
            raise OptimizationError(f"Missing required fields: {missing_fields}")
        
        if players_df.empty:
            raise OptimizationError("No player data provided")
        
        # Check for sufficient players by position
        position_counts = players_df['Position'].value_counts()
        for position, required_count in self.roster_positions.items():
            if position != 'FLEX':
                available_count = position_counts.get(position, 0)
                if available_count < required_count:
                    raise OptimizationError(
                        f"Insufficient {position} players. Need {required_count}, have {available_count}"
                    )

    def _add_randomness(self, players_df: pd.DataFrame, randomness: float) -> pd.DataFrame:
        """Add randomness to projected points based on normal distribution"""
        if randomness > 0:
            df_copy = players_df.copy()
            # Add random variance to projections (normal distribution)
            random_multiplier = np.random.normal(1.0, randomness, len(df_copy))
            df_copy['ProjectedPoints'] = df_copy['ProjectedPoints'] * random_multiplier
            # Ensure points stay positive
            df_copy['ProjectedPoints'] = np.maximum(df_copy['ProjectedPoints'], 0.1)
            return df_copy
        return players_df

# Custom exceptions
class OptimizationError(Exception):
    pass
