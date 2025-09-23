import pulp
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from loguru import logger
import math

# FanDuel constraints - CORRECTED for actual FanDuel format
FANDUEL_SALARY_CAP = 60000
FANDUEL_POSITIONS = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'D': 1}  # No K, D not DST

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
    vegas_implied: Optional[float] = None
    game_stack_id: Optional[str] = None
    bring_back_id: Optional[str] = None
    opponent: Optional[str] = None
    game_total: Optional[float] = None
    spread: Optional[float] = None
    weather_factor: Optional[float] = 1.0
    
    def __post_init__(self):
        if self.ceiling is None:
            self.ceiling = self.projected_points * 1.5
        if self.floor is None:
            self.floor = max(0, self.projected_points * 0.6)

@dataclass
class LineupResult:
    """Result structure for optimized lineup"""
    players: List[Player] = field(default_factory=list)
    total_salary: int = 0
    projected_points: float = 0.0
    expected_ownership: float = 0.0
    ceiling: float = 0.0
    floor: float = 0.0

class DFSOptimizer:
    """Enhanced DFS optimizer with late-swap support"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info("🚀 DFS Optimizer initialized")
    
    def optimize_dfs_lineups(self, players: List[Player], contest_type: str = 'gpp', 
                           num_lineups: int = 5, single_game_teams: List[str] = None,
                           locked_players: List[Player] = None) -> List[LineupResult]:
        """Main optimization entry point"""
        logger.info(f"🎯 Starting optimization: {contest_type} ({num_lineups} lineups)")
        
        if not players:
            logger.error("❌ No players provided")
            return []
            
        if single_game_teams:
            logger.info(f"🏈 Single game mode: {single_game_teams}")
            
        return self.generate_multiple_lineups(players, contest_type, num_lineups, single_game_teams)
    
    def prepare_players(self, player_data: Union[List[Player], List[Dict]], 
                       weather_data: Dict = None, locked_players: List[Player] = None) -> List[Player]:
        """Prepare and filter players for optimization"""
        players = []
        
        # Convert dict data to Player objects if needed
        for item in player_data:
            if isinstance(item, dict):
                player = Player(
                    id=str(item.get('id', item.get('name', ''))),
                    name=item.get('name', ''),
                    position=item.get('position', ''),
                    team=item.get('team', ''),
                    salary=int(item.get('salary', 0)),
                    projected_points=float(item.get('projected_points', 0)),
                    ownership=item.get('ownership'),
                    ceiling=item.get('ceiling'),
                    floor=item.get('floor'),
                    opponent=item.get('opponent'),
                    game_total=item.get('game_total'),
                    spread=item.get('spread'),
                    weather_factor=item.get('weather_factor', 1.0)
                )
                players.append(player)
            elif isinstance(item, Player):
                players.append(item)
        
        # Apply weather adjustments if provided
        if weather_data:
            for player in players:
                if player.team in weather_data:
                    player.weather_factor = weather_data[player.team].get('factor', 1.0)
                    
        return players
    
    def generate_multiple_lineups(self, players: List[Player], contest_type: str, 
                                num_lineups: int, single_game_teams: List[str] = None) -> List[LineupResult]:
        """Generate multiple optimized lineups"""
        lineups = []
        used_combinations = set()
        
        for i in range(num_lineups):
            # Add some randomization for diversity
            random_seed = random.randint(1, 10000) + i
            random.seed(random_seed)
            
            lineup = self.optimize_lineup(players, contest_type, single_game_teams, used_combinations)
            if lineup and lineup.players:
                lineups.append(lineup)
                # Track used combinations to avoid duplicates
                player_combo = tuple(sorted([p.id for p in lineup.players]))
                used_combinations.add(player_combo)
                
        logger.info(f"✅ Generated {len(lineups)} unique lineups")
        return lineups
    
    def optimize_lineup(self, players: List[Player], contest_type: str, 
                       single_game_teams: List[str] = None, used_combinations: set = None) -> LineupResult:
        """Optimize a single lineup using linear programming"""
        
        # Create the optimization problem
        prob = pulp.LpProblem("DFS_Lineup_Optimization", pulp.LpMaximize)
        
        # Create binary variables for each player
        player_vars = {}
        for player in players:
            player_vars[player.id] = pulp.LpVariable(f"player_{player.id}", cat='Binary')
        
        # Objective: maximize projected points
        if contest_type == 'cash':
            # Cash games: prioritize floor
            prob += pulp.lpSum([
                player_vars[player.id] * (player.floor * 0.7 + player.projected_points * 0.3)
                for player in players
            ])
        elif contest_type == 'contrarian':
            # Contrarian: fade ownership
            prob += pulp.lpSum([
                player_vars[player.id] * player.projected_points * (2.0 - (player.ownership or 0.1))
                for player in players
            ])
        else:  # GPP
            # Tournaments: prioritize upside
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
        return self._extract_result(prob, players, player_vars)
    
    def optimize_late_swap(self, current_lineup: List[Player], locked_players: List[Player],
                          available_players: List[Player], contest_type: str = 'gpp') -> LineupResult:
        """Optimize lineup with locked players from started games"""
        logger.info(f"🔄 Late swap optimization: {len(locked_players)} locked players")
        
        # Calculate locked salary and positions
        locked_salary = sum(p.salary for p in locked_players)
        locked_positions = {}
        for player in locked_players:
            locked_positions[player.position] = locked_positions.get(player.position, 0) + 1
        
        remaining_salary = FANDUEL_SALARY_CAP - locked_salary
        remaining_positions = {}
        for pos, count in FANDUEL_POSITIONS.items():
            remaining_positions[pos] = count - locked_positions.get(pos, 0)
        
        # Filter available players by remaining constraints
        valid_players = []
        for player in available_players:
            if (player.salary <= remaining_salary and 
                remaining_positions.get(player.position, 0) > 0):
                valid_players.append(player)
        
        if not valid_players:
            logger.warning("⚠️  No valid players for late swap")
            return LineupResult(players=locked_players)
        
        # Create optimization problem
        prob = pulp.LpProblem("Late_Swap_Optimization", pulp.LpMaximize)
        player_vars = {}
        
        for player in valid_players:
            player_vars[player.id] = pulp.LpVariable(f"player_{player.id}", cat='Binary')
        
        # Objective function based on contest type
        if contest_type == 'cash':
            prob += pulp.lpSum([
                player_vars[player.id] * player.floor for player in valid_players
            ])
        else:
            prob += pulp.lpSum([
                player_vars[player.id] * player.ceiling for player in valid_players
            ])
        
        # Salary constraint
        prob += pulp.lpSum([
            player_vars[player.id] * player.salary for player in valid_players
        ]) <= remaining_salary
        
        # Position constraints
        for position, needed in remaining_positions.items():
            if needed > 0:
                position_players = [p for p in valid_players if p.position == position]
                prob += pulp.lpSum([
                    player_vars[player.id] for player in position_players
                ]) == needed
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Combine locked + optimized players
        selected_players = locked_players.copy()
        if prob.status == pulp.LpStatusOptimal:
            for player in valid_players:
                if player_vars[player.id].varValue == 1:
                    selected_players.append(player)
        
        result = LineupResult(players=selected_players)
        result.total_salary = sum(p.salary for p in selected_players)
        result.projected_points = sum(p.projected_points for p in selected_players)
        
        logger.info(f"✅ Late swap complete: {len(selected_players)} players, ${result.total_salary}")
        return result
    
    def _extract_result(self, prob, players: List[Player], player_vars: Dict, 
                       locked_players: List[Player] = None) -> LineupResult:
        """Extract lineup result from solved optimization problem"""
        
        if prob.status != pulp.LpStatusOptimal:
            logger.warning(f"⚠️  Optimization not optimal: status {prob.status}")
            return LineupResult()
        
        selected_players = []
        
        # Add locked players first
        if locked_players:
            selected_players.extend(locked_players)
        
        # Add optimized players
        for player in players:
            if player_vars[player.id].varValue == 1:
                selected_players.append(player)
        
        # Calculate totals
        total_salary = sum(p.salary for p in selected_players)
        projected_points = sum(p.projected_points for p in selected_players)
        ceiling = sum(p.ceiling for p in selected_players)
        floor = sum(p.floor for p in selected_players)
        
        return LineupResult(
            players=selected_players,
            total_salary=total_salary,
            projected_points=projected_points,
            ceiling=ceiling,
            floor=floor
        )
    
    def _add_regular_constraints(self, prob, players: List[Player], player_vars: Dict):
        """Add standard FanDuel constraints"""
        
        # Salary cap
        prob += pulp.lpSum([
            player_vars[player.id] * player.salary for player in players
        ]) <= FANDUEL_SALARY_CAP
        
        # Position requirements - CORRECTED for actual FanDuel format
        for position, count in FANDUEL_POSITIONS.items():
            position_players = [p for p in players if p.position == position]
            prob += pulp.lpSum([
                player_vars[player.id] for player in position_players
            ]) == count
        
        # Total players (8 not 9 since no K)
        prob += pulp.lpSum([player_vars[player.id] for player in players]) == 8
    
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
        
        # Total players constraint for single game
        prob += pulp.lpSum([player_vars[player.id] for player in players]) == 8


# Main entry function for backward compatibility
def optimize_dfs_lineups(player_data: Union[List[Dict], List[Player]], weather_data: Dict = None,
                        num_lineups: int = 5, contest_type: str = 'gpp',
                        single_game_teams: List[str] = None, 
                        locked_players: List[str] = None) -> List[LineupResult]:
    """Main entry point for optimization with late-swap support"""
    logger.info(f"Starting DFS optimization: {contest_type}, {num_lineups} lineups")
    if single_game_teams:
        logger.info(f"Single game teams: {single_game_teams}")
    if locked_players:
        logger.info(f"Locked players: {locked_players}")
    
    optimizer = DFSOptimizer()
    players = optimizer.prepare_players(player_data, weather_data, locked_players)
    
    if not players:
        logger.error("No valid players for optimization")
        return []
    
    return optimizer.generate_multiple_lineups(players, contest_type, num_lineups, single_game_teams)

# Export the class with both names for compatibility
EnhancedDFSOptimizer = DFSOptimizer
