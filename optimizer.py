from pulp import *
import pulp
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from loguru import logger
import math
import csv
from pathlib import Path

# FanDuel constraints - CORRECTED for actual FanDuel format with FLEX
FANDUEL_SALARY_CAP = 60000
FANDUEL_POSITIONS = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'D': 1}  # Base positions
FLEX_POSITIONS = ['RB', 'WR', 'TE']  # FLEX can be any of these

@dataclass
class Player:
    """Player data structure"""
    id: str
    name: str
    position: str
    team: str
    salary: int
    projected_points: float
    ownership: Optional[float] = None
    ceiling: Optional[float] = None
    floor: Optional[float] = None

@dataclass
class Lineup:
    """Optimized lineup result"""
    players: List[Player]
    total_salary: int
    projected_points: float
    expected_ownership: float
    ceiling: float
    floor: float

class LineupOptimizer:
    """FanDuel lineup optimizer using PuLP"""
    
    def __init__(self, config: dict):
        self.config = config
        
    def optimize_lineup(self, players: List[Player], contest_type: str = 'GPP', 
                       used_combinations: Optional[List[Set[str]]] = None,
                       single_game_teams: Optional[List[str]] = None) -> Optional[Lineup]:
        """Generate optimized lineup"""
        
        if not players:
            logger.error("No players provided for optimization")
            return None
            
        # Create optimization problem
        prob = LpProblem("FanDuel_Lineup", LpMaximize)
        
        # Decision variables
        player_vars = {player.id: LpVariable(f"player_{player.id}", cat='Binary') 
                      for player in players}
        
        # Objective function - adjust based on contest type
        if contest_type == 'Cash':
            # Cash games: prioritize floor + projected
            prob += pulp.lpSum([
                player_vars[player.id] * (player.floor * 0.7 + player.projected_points * 0.3)
                for player in players
            ])
        else:
            # GPP/Tournament: prioritize ceiling + projected
            prob += pulp.lpSum([
                player_vars[player.id] * (player.ceiling * 0.6 + player.projected_points * 0.4)
                for player in players
            ])
        
        # Add constraints
        if single_game_teams:
            self._add_single_game_constraints(prob, players, player_vars)
        else:
            self._add_regular_constraints(prob, players, player_vars)
        
        # Avoid duplicate lineups
        if used_combinations:
            for combo in used_combinations:
                prob += pulp.lpSum([
                    player_vars.get(player_id, 0) for player_id in combo
                ]) <= len(combo) - 1
        
        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        return self._extract_result(prob, players, player_vars, contest_type)
    
    def _extract_result(self, prob, players: List[Player], player_vars: Dict, 
                       contest_type: str) -> Optional[Lineup]:
        """Extract lineup from solved optimization"""
        
        if prob.status != LpStatusOptimal:
            logger.error(f"Optimization failed with status: {LpStatus[prob.status]}")
            return None
            
        # Get selected players
        selected_players = []
        for player in players:
            if player_vars[player.id].value() == 1:
                selected_players.append(player)
        
        if not selected_players:
            logger.error("No players selected in optimization result")
            return None
            
        # Calculate metrics
        total_salary = sum(p.salary for p in selected_players)
        projected_points = sum(p.projected_points for p in selected_players)
        ownership_total = sum(p.ownership or 0 for p in selected_players)
        ceiling_score = sum(p.ceiling or p.projected_points * 1.5 for p in selected_players)
        floor_score = sum(p.floor or p.projected_points * 0.6 for p in selected_players)
        
        return Lineup(
            players=selected_players,
            total_salary=total_salary,
            projected_points=projected_points,
            expected_ownership=ownership_total,
            ceiling=ceiling_score,
            floor=floor_score
        )
    
    def _add_regular_constraints(self, prob, players: List[Player], player_vars: Dict):
        """Add standard FanDuel constraints with FLEX"""
        
        # Salary cap
        prob += pulp.lpSum([
            player_vars[player.id] * player.salary for player in players
        ]) <= FANDUEL_SALARY_CAP
        
        # Base position requirements
        for position, count in FANDUEL_POSITIONS.items():
            position_players = [p for p in players if p.position == position]
            prob += pulp.lpSum([
                player_vars[player.id] for player in position_players
            ]) == count
        
        # FLEX constraint - exactly 1 additional RB/WR/TE
        flex_players = [p for p in players if p.position in FLEX_POSITIONS]
        prob += pulp.lpSum([
            player_vars[player.id] for player in flex_players
        ]) == sum(FANDUEL_POSITIONS[pos] for pos in FLEX_POSITIONS) + 1  # Base + 1 FLEX
        
        # Total players (9 with FLEX)
        prob += pulp.lpSum([player_vars[player.id] for player in players]) == 9
    
    def _add_single_game_constraints(self, prob, players: List[Player], player_vars: Dict):
        """Add single-game/showdown constraints"""
        
        # Salary cap
        prob += pulp.lpSum([
            player_vars[player.id] * player.salary for player in players
        ]) <= FANDUEL_SALARY_CAP
        
        # Single game positions
        single_game_positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'D': 1}
        
        for position, count in single_game_positions.items():
            position_players = [p for p in players if p.position == position]
            if position_players:  # Only add constraint if players exist
                prob += pulp.lpSum([
                    player_vars[player.id] for player in position_players
                ]) <= count
        
        # Total 8 players for single game (no FLEX in showdown)
        prob += pulp.lpSum([player_vars[player.id] for player in players]) == 8
        
        # Must have at least 1 QB
        qb_players = [p for p in players if p.position == 'QB']
        if qb_players:
            prob += pulp.lpSum([
                player_vars[player.id] for player in qb_players
            ]) >= 1
