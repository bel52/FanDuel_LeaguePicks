"""
Fixed DFS optimizer with proper single game support and late-swap functionality
"""
import pulp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import random

from config import FANDUEL_POSITIONS, FANDUEL_SALARY_CAP

@dataclass
class Player:
    """Player data structure"""
    id: str
    name: str
    position: str
    team: str
    salary: int
    projection: float
    ownership: float = 10.0
    weather_factor: float = 1.0
    injury_risk: float = 0.0
    value: float = 0.0
    variance: float = 0.0
    is_locked: bool = False  # NEW: For late-swap functionality
    
    def __post_init__(self):
        self.value = self.projection / (self.salary / 1000) if self.salary > 0 else 0
        variance_multipliers = {'QB': 0.3, 'RB': 0.4, 'WR': 0.5, 'TE': 0.4, 'K': 0.6, 'DST': 0.5}
        self.variance = self.projection * variance_multipliers.get(self.position, 0.4)

@dataclass
class LineupResult:
    """Lineup result structure"""
    players: List[Player]
    total_salary: int
    projected_points: float
    total_value: float
    ownership_total: float
    correlation_score: float
    weather_impact: float
    contest_type: str
    ceiling_score: float = 0.0
    floor_score: float = 0.0
    locked_players: List[str] = None  # NEW: Track locked players for late swap

class EnhancedDFSOptimizer:
    """Enhanced optimizer with fixed single game support and late-swap functionality"""
    
    def prepare_players(self, player_data: List[Dict], weather_data: Dict = None, 
                       locked_players: List[str] = None) -> List[Player]:
        """Convert player data to Player objects with late-swap locking support"""
        players = []
        locked_set = set(locked_players or [])
        
        for data in player_data:
            try:
                player = Player(
                    id=str(data.get('player_id', data.get('name', ''))),
                    name=data.get('player_name', data.get('name', '')),
                    position=data.get('position', ''),
                    team=data.get('team', '').upper(),
                    salary=int(data.get('salary', 5000)),
                    projection=float(data.get('projection', data.get('fantasy_points_ppr', 0))),
                    is_locked=data.get('player_name', data.get('name', '')) in locked_set
                )
                
                player.value = player.projection / (player.salary / 1000) if player.salary > 0 else 0
                players.append(player)
                
                if player.is_locked:
                    logger.debug(f"🔒 Player locked for late swap: {player.name}")
                
            except Exception as e:
                logger.error(f"Error processing player {data}: {e}")
                continue
        
        logger.info(f"Prepared {len(players)} players for optimization ({len(locked_set)} locked)")
        return players

    # ============================================================================
    # LATE-SWAP FUNCTIONALITY (NEW)
    # ============================================================================
    
    def lock_started_players(self, lineups: List[LineupResult], started_games: List[Dict]) -> List[LineupResult]:
        """Lock players from started games in existing lineups"""
        if not started_games:
            logger.info("No started games to lock players from")
            return lineups
        
        started_teams = set()
        for game in started_games:
            started_teams.update(game.get('teams', []))
        
        logger.info(f"🔒 Locking players from started teams: {started_teams}")
        
        locked_lineups = []
        for lineup in lineups:
            locked_players = []
            for player in lineup.players:
                if player.team.upper() in started_teams:
                    player.is_locked = True
                    locked_players.append(player.name)
            
            # Update lineup with locked player info
            new_lineup = LineupResult(
                players=lineup.players,
                total_salary=lineup.total_salary,
                projected_points=lineup.projected_points,
                total_value=lineup.total_value,
                ownership_total=lineup.ownership_total,
                correlation_score=lineup.correlation_score,
                weather_impact=lineup.weather_impact,
                contest_type=lineup.contest_type,
                ceiling_score=lineup.ceiling_score,
                floor_score=lineup.floor_score,
                locked_players=locked_players
            )
            
            locked_lineups.append(new_lineup)
            logger.debug(f"Lineup locked {len(locked_players)} players: {locked_players[:3]}...")
        
        return locked_lineups
    
    def generate_late_swap_lineups(self, available_players: List[Player], 
                                  locked_lineup: LineupResult, 
                                  contest_type: str = 'gpp') -> List[LineupResult]:
        """Generate new lineups for late slate while keeping locked players"""
        logger.info(f"🔄 Generating late swap lineups for {contest_type}")
        
        if not locked_lineup.locked_players:
            logger.warning("No locked players found, treating as normal optimization")
            return self.generate_multiple_lineups(available_players, 5, contest_type)
        
        # Separate locked and available players
        locked_players = [p for p in locked_lineup.players if p.is_locked]
        unlocked_positions = []
        
        # Determine which positions need to be filled
        total_positions = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DST']
        used_positions = []
        
        for player in locked_players:
            if player.position in total_positions:
                used_positions.append(player.position)
                if used_positions.count(player.position) <= total_positions.count(player.position):
                    total_positions.remove(player.position)
        
        remaining_positions = total_positions
        logger.info(f"🎯 Need to fill positions: {remaining_positions}")
        
        if not remaining_positions:
            logger.warning("All positions already filled by locked players")
            return [locked_lineup]
        
        # Filter available players to late slate only
        late_slate_players = [p for p in available_players if not p.is_locked]
        
        if len(late_slate_players) < len(remaining_positions):
            logger.error(f"Not enough late slate players ({len(late_slate_players)}) for remaining positions ({len(remaining_positions)})")
            return [locked_lineup]
        
        # Generate optimized lineups with locked players as constraints
        late_swap_lineups = []
        
        for i in range(5):  # Generate 5 late swap options
            try:
                # Create optimization problem
                prob = pulp.LpProblem("Late_Swap_DFS", pulp.LpMaximize)
                
                player_vars = {}
                for j, player in enumerate(late_slate_players):
                    player_vars[j] = pulp.LpVariable(f"player_{j}", cat='Binary')
                
                # Objective function
                objective_terms = []
                for j, player in enumerate(late_slate_players):
                    value = self._calculate_contest_value(player, contest_type)
                    # Add randomization for diversity
                    random_factor = random.uniform(0.95, 1.08)
                    objective_terms.append(value * random_factor * player_vars[j])
                
                prob += pulp.lpSum(objective_terms)
                
                # Salary constraint (subtract locked players' salaries)
                locked_salary = sum(p.salary for p in locked_players)
                remaining_salary = FANDUEL_SALARY_CAP - locked_salary
                
                prob += pulp.lpSum([late_slate_players[j].salary * player_vars[j] 
                                  for j in range(len(late_slate_players))]) <= remaining_salary
                
                # Position constraints for remaining positions
                prob += pulp.lpSum([player_vars[j] for j in range(len(late_slate_players))]) == len(remaining_positions)
                
                # Solve
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                
                if prob.status == pulp.LpStatusOptimal:
                    selected_late_players = []
                    for j, player in enumerate(late_slate_players):
                        if player_vars[j].varValue == 1:
                            selected_late_players.append(player)
                    
                    # Combine locked and selected players
                    combined_players = locked_players + selected_late_players
                    
                    # Calculate new lineup metrics
                    total_salary = sum(p.salary for p in combined_players)
                    projected_points = sum(p.projection for p in combined_players)
                    ownership_total = sum(p.ownership for p in combined_players)
                    
                    late_swap_lineup = LineupResult(
                        players=combined_players,
                        total_salary=total_salary,
                        projected_points=projected_points,
                        total_value=sum(p.value for p in combined_players),
                        ownership_total=ownership_total,
                        correlation_score=0.5,  # Simplified for late swap
                        weather_impact=1.0,
                        contest_type=f"{contest_type}_late_swap",
                        ceiling_score=sum(p.projection + p.variance for p in combined_players),
                        floor_score=sum(max(0, p.projection - p.variance) for p in combined_players),
                        locked_players=[p.name for p in locked_players]
                    )
                    
                    late_swap_lineups.append(late_swap_lineup)
                    logger.info(f"✅ Late swap lineup {i+1}: {projected_points:.1f} pts, ${total_salary:,}")
                
            except Exception as e:
                logger.error(f"Error generating late swap lineup {i+1}: {e}")
                continue
        
        if not late_swap_lineups:
            logger.warning("No late swap lineups generated, returning original")
            return [locked_lineup]
        
        # Sort by appropriate metric
        if contest_type == 'cash':
            late_swap_lineups.sort(key=lambda x: x.floor_score, reverse=True)
        else:
            late_swap_lineups.sort(key=lambda x: x.ceiling_score, reverse=True)
        
        logger.info(f"🔄 Generated {len(late_swap_lineups)} late swap lineups")
        return late_swap_lineups
    
    def assess_late_swap_value(self, current_lineup: LineupResult, 
                              alternative_lineups: List[LineupResult]) -> Dict[str, Any]:
        """Assess whether late swap provides value over current lineup"""
        if not alternative_lineups:
            return {
                'recommendation': 'keep_current',
                'reason': 'No viable alternatives',
                'expected_value_change': 0.0
            }
        
        best_alternative = max(alternative_lineups, key=lambda x: x.projected_points)
        value_change = best_alternative.projected_points - current_lineup.projected_points
        
        # Consider ownership leverage
        ownership_change = best_alternative.ownership_total - current_lineup.ownership_total
        
        recommendation = 'keep_current'
        reason = 'Current lineup is optimal'
        
        if value_change > 1.0:  # More than 1 point improvement
            recommendation = 'swap_recommended'
            reason = f'Expected +{value_change:.1f} points improvement'
        elif value_change > 0.5 and ownership_change < -5:  # Small improvement but better leverage
            recommendation = 'swap_for_leverage'
            reason = f'Slight improvement (+{value_change:.1f} pts) with better ownership leverage'
        
        return {
            'recommendation': recommendation,
            'reason': reason,
            'expected_value_change': value_change,
            'ownership_change': ownership_change,
            'best_alternative': best_alternative,
            'current_lineup': current_lineup
        }

    # ============================================================================
    # EXISTING OPTIMIZATION METHODS (PRESERVED)
    # ============================================================================
    
    def optimize_lineup(self, players: List[Player], contest_type: str = 'gpp',
                       single_game_teams: List[str] = None) -> Optional[LineupResult]:
        """Main optimization function with single game support"""
        
        try:
            logger.info(f"Starting optimization for {contest_type}")
            
            # Handle single game filtering
            if single_game_teams and contest_type == 'single_game':
                players = self._filter_single_game_players(players, single_game_teams)
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)} (need 6)")
                    return None
            
            # Project ownership
            for player in players:
                player.ownership = self._predict_ownership(player, contest_type)
            
            # Create optimization problem
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
            
            player_vars = {}
            for i, player in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
            
            # Objective function
            objective_terms = []
            for i, player in enumerate(players):
                value = self._calculate_contest_value(player, contest_type)
                objective_terms.append(value * player_vars[i])
            
            prob += pulp.lpSum(objective_terms)
            
            # Add constraints
            if contest_type == 'single_game':
                self._add_single_game_constraints(prob, players, player_vars)
            else:
                self._add_regular_constraints(prob, players, player_vars)
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                return self._extract_result(prob, players, player_vars, contest_type)
            else:
                logger.warning(f"Optimization failed: {pulp.LpStatus[prob.status]}")
                return None
                
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _filter_single_game_players(self, players: List[Player], teams: List[str]) -> List[Player]:
        """Filter players for single game contest"""
        teams_upper = [t.upper() for t in teams]
        logger.info(f"Filtering for single game teams: {teams_upper}")
        
        filtered = []
        for player in players:
            if player.team.upper() in teams_upper:
                # Boost single game projections slightly
                sg_player = Player(
                    id=player.id, name=player.name, position=player.position,
                    team=player.team, salary=player.salary, 
                    projection=player.projection * 1.1,  # Single game boost
                    ownership=player.ownership, weather_factor=player.weather_factor,
                    injury_risk=player.injury_risk, value=player.value, variance=player.variance,
                    is_locked=player.is_locked
                )
                sg_player.value = sg_player.projection / (sg_player.salary / 1000)
                filtered.append(sg_player)
        
        # Log position breakdown
        pos_count = {}
        for p in filtered:
            pos_count[p.position] = pos_count.get(p.position, 0) + 1
        
        logger.info(f"Single game players: {len(filtered)}, positions: {pos_count}")
        return filtered
    
    def _predict_ownership(self, player: Player, contest_type: str) -> float:
        """Predict ownership percentage"""
        base = max(1.0, player.salary / 300)
        
        if contest_type == 'gpp':
            if player.value > 3.0:
                base *= 1.3
        elif contest_type == 'cash':
            base *= 0.9
        elif contest_type == 'contrarian':
            if player.value > 3.0:
                base *= 0.6
        elif contest_type == 'single_game':
            base *= 1.2  # Higher ownership in single game
        
        return min(50.0, base)
    
    def _calculate_contest_value(self, player: Player, contest_type: str) -> float:
        """Calculate contest-specific player value"""
        base = player.projection
        
        if contest_type == 'gpp':
            base += player.variance * 0.3
            if player.ownership > 25:
                base -= (player.ownership - 25) * 0.05
        elif contest_type == 'cash':
            base -= player.variance * 0.1
            if player.ownership < 5:
                base *= 0.9
        elif contest_type == 'contrarian':
            base += player.variance * 0.4
            if player.ownership > 15:
                base -= (player.ownership - 15) * 0.15
        elif contest_type == 'single_game':
            # Reward high-upside players in single game
            if player.position in ['QB', 'WR', 'TE']:
                base *= 1.2
            base += player.variance * 0.2
        
        return base
    
    def _add_single_game_constraints(self, prob, players: List[Player], player_vars: Dict):
        """Add single game constraints"""
        
        # Salary cap
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP
        
        # Exactly 6 players
        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 6
        
        # Position diversity - at least one from each major position if available
        position_groups = {}
        for i, player in enumerate(players):
            pos = player.position
            if pos not in position_groups:
                position_groups[pos] = []
            position_groups[pos].append(i)
        
        # Ensure we have at least 1 QB if available (for MVP)
        if 'QB' in position_groups and position_groups['QB']:
            prob += pulp.lpSum([player_vars[i] for i in position_groups['QB']]) >= 1
        
        logger.info(f"Single game constraints added for {len(position_groups)} position groups")
    
    def _add_regular_constraints(self, prob, players: List[Player], player_vars: Dict):
        """Add regular format constraints"""
        
        # Salary cap
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP
        
        # Position constraints
        for position, count in FANDUEL_POSITIONS.items():
            if position == 'FLEX':
                flex_players = [i for i, p in enumerate(players) if p.position in ['RB', 'WR', 'TE']]
                if flex_players:
                    prob += pulp.lpSum([player_vars[i] for i in flex_players]) >= count
            elif position == 'DST':
                dst_players = [i for i, p in enumerate(players) if p.position in ['DST', 'DEF']]
                if dst_players:
                    prob += pulp.lpSum([player_vars[i] for i in dst_players]) == count
            else:
                pos_players = [i for i, p in enumerate(players) if p.position == position]
                if pos_players:
                    prob += pulp.lpSum([player_vars[i] for i in pos_players]) == count
        
        # Total roster size
        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == sum(FANDUEL_POSITIONS.values())
    
    def _extract_result(self, prob, players: List[Player], player_vars: Dict, contest_type: str) -> LineupResult:
        """Extract optimization results"""
        selected_players = []
        total_salary = 0
        total_ownership = 0
        
        for i, player in enumerate(players):
            if player_vars[i].varValue == 1:
                selected_players.append(player)
                total_salary += player.salary
                total_ownership += player.ownership
        
        # Calculate projected points
        if contest_type == 'single_game' and len(selected_players) == 6:
            # Sort by projection for MVP selection
            selected_players.sort(key=lambda p: p.projection, reverse=True)
            mvp = selected_players[0]
            projected_points = mvp.projection * 1.5 + sum(p.projection for p in selected_players[1:])
            logger.info(f"Single game MVP: {mvp.name} ({mvp.position}) - {mvp.projection:.1f} pts")
        else:
            projected_points = sum(p.projection for p in selected_players)
        
        # Calculate ceiling/floor
        ceiling = sum(p.projection + p.variance for p in selected_players)
        floor = sum(max(0, p.projection - p.variance) for p in selected_players)
        
        if contest_type == 'single_game' and len(selected_players) >= 1:
            mvp = max(selected_players, key=lambda p: p.projection)
            ceiling = ceiling - mvp.projection + (mvp.projection * 1.5)
            floor = floor - mvp.projection + (mvp.projection * 1.5)
        
        return LineupResult(
            players=selected_players,
            total_salary=total_salary,
            projected_points=projected_points,
            total_value=sum(p.value for p in selected_players),
            ownership_total=total_ownership,
            correlation_score=0.5,
            weather_impact=1.0,
            contest_type=contest_type,
            ceiling_score=ceiling,
            floor_score=floor,
            locked_players=[p.name for p in selected_players if p.is_locked]
        )
    
    def generate_multiple_lineups(self, players: List[Player], num_lineups: int = 5,
                                 contest_type: str = 'gpp', single_game_teams: List[str] = None) -> List[LineupResult]:
        """Generate multiple lineups"""
        logger.info(f"Generating {num_lineups} {contest_type} lineups")
        
        lineups = []
        
        for i in range(num_lineups):
            # Add randomization for diversity
            randomized_players = []
            for player in players:
                new_player = Player(
                    id=player.id, name=player.name, position=player.position,
                    team=player.team, salary=player.salary, projection=player.projection,
                    ownership=player.ownership, weather_factor=player.weather_factor,
                    injury_risk=player.injury_risk, value=player.value, variance=player.variance,
                    is_locked=player.is_locked
                )
                
                # Add randomization for diversity
                if contest_type == 'gpp':
                    random_factor = random.uniform(0.95, 1.08)
                elif contest_type == 'cash':
                    random_factor = random.uniform(0.98, 1.02)
                elif contest_type == 'contrarian':
                    random_factor = random.uniform(0.90, 1.15)
                elif contest_type == 'single_game':
                    random_factor = random.uniform(0.95, 1.10)
                else:
                    random_factor = 1.0
                
                new_player.projection *= random_factor
                new_player.value = new_player.projection / (new_player.salary / 1000)
                randomized_players.append(new_player)
            
            lineup = self.optimize_lineup(randomized_players, contest_type, single_game_teams)
            if lineup:
                lineups.append(lineup)
                logger.info(f"Generated lineup {i+1}: {lineup.projected_points:.1f} pts, ${lineup.total_salary:,}")
        
        # Sort by appropriate metric
        if contest_type == 'cash':
            lineups.sort(key=lambda x: x.floor_score, reverse=True)
        else:
            lineups.sort(key=lambda x: x.ceiling_score, reverse=True)
        
        logger.info(f"Successfully generated {len(lineups)} {contest_type} lineups")
        return lineups
    
    def export_lineups_to_csv(self, lineups: List[LineupResult], filename: str = None):
        """Export lineups to CSV"""
        if not filename:
            filename = f"lineups_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        lineup_data = []
        for i, lineup in enumerate(lineups):
            lineup_row = {'Lineup': i + 1}
            
            for j, player in enumerate(lineup.players):
                lineup_row[f'Player_{j+1}'] = f"{player.name} ({player.position}) ${player.salary}"
            
            lineup_row.update({
                'Total_Salary': lineup.total_salary,
                'Projected_Points': round(lineup.projected_points, 2),
                'Contest_Type': lineup.contest_type,
                'Locked_Players': len(lineup.locked_players or [])
            })
            
            lineup_data.append(lineup_row)
        
        df = pd.DataFrame(lineup_data)
        df.to_csv(filename, index=False)
        return filename

def optimize_dfs_lineups(player_data: List[Dict], weather_data: Dict = None,
                        num_lineups: int = 5, contest_type: str = 'gpp',
                        single_game_teams: List[str] = None, 
                        locked_players: List[str] = None) -> List[LineupResult]:
    """Main entry point for optimization with late-swap support"""
    logger.info(f"Starting DFS optimization: {contest_type}, {num_lineups} lineups")
    if single_game_teams:
        logger.info(f"Single game teams: {single_game_teams}")
    if locked_players:
        logger.info(f"Locked players: {locked_players}")
    
    optimizer = EnhancedDFSOptimizer()
    players = optimizer.prepare_players(player_data, weather_data, locked_players)
    
    if not players:
        logger.error("No valid players for optimization")
        return []
    
    return optimizer.generate_multiple_lineups(players, num_lineups, contest_type, single_game_teams)

# Backward compatibility alias
DFSOptimizer = EnhancedDFSOptimizer
