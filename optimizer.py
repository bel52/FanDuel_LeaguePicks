"""
Enhanced DFS lineup optimization with proper FLEX position support
Fixed for tournament play with proper FanDuel format and CONSERVATIVE filtering
"""
import pulp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import random
import json
from datetime import datetime

from config import FANDUEL_POSITIONS, FANDUEL_SALARY_CAP, OPTIMIZATION_CONFIG, DATA_DIR

# AI Integration import (will gracefully fail if file doesn't exist)
try:
    from ai_analyzer import DFSAIAnalyzer
    AI_AVAILABLE = True
except ImportError:
    logger.warning("AI analyzer not available - continuing without AI analysis")
    AI_AVAILABLE = False

@dataclass
class Player:
    """Player data structure for optimization"""
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
    
    def __post_init__(self):
        self.value = self.projection / (self.salary / 1000) if self.salary > 0 else 0
        variance_multipliers = {'QB': 0.3, 'RB': 0.4, 'WR': 0.5, 'TE': 0.4, 'K': 0.6, 'DST': 0.5, 'D': 0.5}
        self.variance = self.projection * variance_multipliers.get(self.position, 0.4)

@dataclass
class LineupResult:
    """Optimization result structure"""
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

class EnhancedDFSOptimizer:
    """Enhanced DFS optimization for exact FanDuel format with CONSERVATIVE filtering"""
    
    def __init__(self):
        pass
        
    def prepare_players(self, player_data: List[Dict], weather_data: Dict = None) -> List[Player]:
        """Convert player data with MINIMAL filtering - data_collector already filtered conservatively"""
        players = []
        
        for data in player_data:
            try:
                player_name = data.get('player_name', data.get('name', ''))
                position = data.get('position', '')
                team = data.get('team', '')
                salary = int(data.get('salary', 5000))
                projection = float(data.get('projection', data.get('projected_points', 0)))
                
                # MINIMAL filtering - data_collector already did conservative filtering
                
                # Skip players with no name
                if not player_name or len(player_name.strip()) < 2:
                    continue
                
                # Skip players with negative projections
                if projection < 0:
                    continue
                
                # Normalize defense position
                if position in ['DST', 'DEF', 'D/ST']:
                    position = 'D'
                
                # Keep all players that made it through data_collector filtering
                player = Player(
                    id=str(data.get('player_id', data.get('id', player_name))),
                    name=player_name,
                    position=position,
                    team=team,
                    salary=salary,
                    projection=projection
                )
                
                # Apply weather adjustments
                if weather_data and team in weather_data:
                    weather_factor = weather_data[team].get('factor', 1.0)
                    player.weather_factor = weather_factor
                    player.projection *= weather_factor
                
                player.value = player.projection / (player.salary / 1000) if player.salary > 0 else 0
                players.append(player)
                
            except Exception as e:
                logger.error(f"Error processing player {data}: {e}")
                continue
        
        # Log position breakdown after filtering
        positions = {}
        for p in players:
            positions[p.position] = positions.get(p.position, 0) + 1
        logger.info(f"Final optimization player count by position: {positions}")
        
        return players

    def _format_lineup_for_fanduel(self, players: List[Player]) -> List[Player]:
        """Order players in exact FanDuel format: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF"""
        ordered = []
        
        # Sort players by position and salary
        by_position = {}
        for player in players:
            if player.position not in by_position:
                by_position[player.position] = []
            by_position[player.position].append(player)
        
        # Sort each position by salary (highest first)
        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.salary, reverse=True)
        
        # FanDuel order: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF
        if 'QB' in by_position:
            ordered.append(by_position['QB'][0])  # QB
        
        if 'RB' in by_position:
            ordered.extend(by_position['RB'][:2])  # RB, RB
        
        if 'WR' in by_position:
            ordered.extend(by_position['WR'][:3])  # WR, WR, WR
        
        if 'TE' in by_position:
            ordered.append(by_position['TE'][0])  # TE
        
        # FLEX - remaining highest salary RB/WR/TE
        flex_candidates = []
        if 'RB' in by_position and len(by_position['RB']) > 2:
            flex_candidates.extend(by_position['RB'][2:])
        if 'WR' in by_position and len(by_position['WR']) > 3:
            flex_candidates.extend(by_position['WR'][3:])
        if 'TE' in by_position and len(by_position['TE']) > 1:
            flex_candidates.extend(by_position['TE'][1:])
        
        if flex_candidates:
            flex_player = max(flex_candidates, key=lambda p: p.salary)
            ordered.append(flex_player)  # FLEX
        
        if 'D' in by_position:
            ordered.append(by_position['D'][0])  # DEF
        
        return ordered
    
    def optimize_lineup(self, players: List[Player], contest_type: str = 'gpp',
                       single_game_teams: List[str] = None) -> Optional[LineupResult]:
        """Optimize with EXACT FanDuel constraints"""
        
        try:
            # Filter for single game
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
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
                points_value = self._calculate_contest_value(player, contest_type)
                objective_terms.append(points_value * player_vars[i])
            
            prob += pulp.lpSum(objective_terms)
            
            # Add CORRECTED constraints
            self._add_fanduel_constraints(prob, players, player_vars, contest_type, single_game_teams)
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                return self._extract_result(prob, players, player_vars, contest_type)
            else:
                logger.warning(f"Optimization failed: {pulp.LpStatus[prob.status]}")
                return None
                
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return None
    
    def _add_fanduel_constraints(self, prob, players: List[Player], player_vars: Dict,
                                contest_type: str, single_game_teams: List[str]):
        """EXACT FanDuel tournament constraints: QB+2RB+3WR+1TE+1FLEX+1DEF=9"""
        
        # Salary cap
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP
        
        if single_game_teams:
            # Single game: 6 players total
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 6
            return
        
        # Get position indices
        qb_indices = [i for i, p in enumerate(players) if p.position == 'QB']
        rb_indices = [i for i, p in enumerate(players) if p.position == 'RB']
        wr_indices = [i for i, p in enumerate(players) if p.position == 'WR']
        te_indices = [i for i, p in enumerate(players) if p.position == 'TE']
        d_indices = [i for i, p in enumerate(players) if p.position == 'D']
        
        # EXACT FanDuel position requirements
        # 1 QB
        if qb_indices:
            prob += pulp.lpSum([player_vars[i] for i in qb_indices]) == 1
        else:
            logger.error("No QBs available!")
            return None
            
        # 1 DEF
        if d_indices:
            prob += pulp.lpSum([player_vars[i] for i in d_indices]) == 1
        else:
            logger.error("No defenses available!")
            return None
        
        # FLEX-eligible positions: RB, WR, TE
        flex_indices = rb_indices + wr_indices + te_indices
        
        if not flex_indices:
            logger.error("No flex-eligible players!")
            return None
        
        # Core requirements: 2 RB + 3 WR + 1 TE = 6 players minimum
        if rb_indices:
            prob += pulp.lpSum([player_vars[i] for i in rb_indices]) >= 2  # At least 2 RB
        
        if wr_indices:
            prob += pulp.lpSum([player_vars[i] for i in wr_indices]) >= 3  # At least 3 WR
        
        if te_indices:
            prob += pulp.lpSum([player_vars[i] for i in te_indices]) >= 1  # At least 1 TE
        
        # FLEX constraint: Total RB + WR + TE = 7 (2+3+1 core + 1 FLEX)
        prob += pulp.lpSum([player_vars[i] for i in flex_indices]) == 7
        
        # Position maximums to prevent 4+ of same position
        if rb_indices:
            prob += pulp.lpSum([player_vars[i] for i in rb_indices]) <= 3  # Max 3 RB (2 + 1 FLEX)
        if wr_indices:
            prob += pulp.lpSum([player_vars[i] for i in wr_indices]) <= 4  # Max 4 WR (3 + 1 FLEX)
        if te_indices:
            prob += pulp.lpSum([player_vars[i] for i in te_indices]) <= 2  # Max 2 TE (1 + 1 FLEX)
        
        # Total roster: QB + 7 flex + DEF = 9
        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 9
        
        # Team diversity constraints
        team_counts = {}
        for i, player in enumerate(players):
            if player.team not in team_counts:
                team_counts[player.team] = []
            team_counts[player.team].append(i)
        
        max_per_team = 3 if contest_type == 'cash' else 4
        for team, player_indices in team_counts.items():
            prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= max_per_team
        
        # Add stacking for tournaments
        if contest_type == 'gpp':
            self._add_stacking_incentive(prob, players, player_vars, qb_indices, wr_indices)
    
    def _add_stacking_incentive(self, prob, players: List[Player], player_vars: Dict, 
                               qb_indices: List[int], wr_indices: List[int]):
        """Add QB+WR stacking incentive for tournaments"""
        
        # Group players by team
        team_qbs = {}
        team_wrs = {}
        
        for i in qb_indices:
            team = players[i].team
            if team not in team_qbs:
                team_qbs[team] = []
            team_qbs[team].append(i)
        
        for i in wr_indices:
            team = players[i].team
            if team not in team_wrs:
                team_wrs[team] = []
            team_wrs[team].append(i)
        
        # Create stacking variables for each team
        for team in team_qbs:
            if team in team_wrs:
                # If we select QB from this team, encourage WR from same team
                qb_vars = [player_vars[i] for i in team_qbs[team]]
                wr_vars = [player_vars[i] for i in team_wrs[team]]
                
                if qb_vars and wr_vars:
                    # Soft constraint: if QB selected, prefer WR from same team
                    prob += pulp.lpSum(wr_vars) >= 0.5 * pulp.lpSum(qb_vars)
    
    def _predict_ownership(self, player: Player, contest_type: str) -> float:
        """Enhanced ownership prediction"""
        
        # Base ownership from salary
        base = (player.salary / 300) + (player.value * 2)
        
        # Contest type adjustments
        if contest_type == 'gpp':
            if player.value > 3.5:
                base *= 1.4  # High value players get more attention
            if player.position == 'QB' and player.salary > 8500:
                base *= 1.2  # Elite QBs are popular
        elif contest_type == 'cash':
            base *= 0.9  # Generally lower ownership in cash
        elif contest_type == 'contrarian':
            base *= 0.6  # Contrarian targets low ownership
        
        return max(2.0, min(50.0, base))
    
    def _calculate_contest_value(self, player: Player, contest_type: str) -> float:
        """Calculate contest-specific value - IMPROVED FOR TOURNAMENT PLAY"""
        
        base_value = player.projection
        
        # TOURNAMENT STRATEGY: Pure ceiling focus for winning lineups
        if contest_type == 'gpp':
            # For tournaments: ONLY care about ceiling, not ownership
            ceiling_bonus = player.variance * 1.0  # Strong ceiling bonus
            
            # MAJOR bonus for elite players (these win tournaments)
            if player.position == 'QB' and player.salary > 8500:
                ceiling_bonus *= 2.5  # Massive bonus for elite QBs
            elif player.position in ['RB', 'WR'] and player.salary > 8000:
                ceiling_bonus *= 2.0  # Big bonus for premium skill players
            
            # Strong penalty for low-projection players (avoid backups)
            if player.projection <= 5:
                return base_value * 0.05  # Make them essentially unusable
            
            # Reward high salary + high projection combinations
            if player.salary > 7500 and player.projection > 18:
                ceiling_bonus *= 1.5
            
            return base_value + ceiling_bonus
            
        elif contest_type == 'cash':
            # Cash: Balance floor and ceiling
            floor_bonus = -player.variance * 0.2  # Small variance penalty
            
            # Still reward good players in cash
            if player.projection > 15:
                floor_bonus += 1.5
            
            return base_value + floor_bonus
            
        elif contest_type == 'contrarian':
            # Contrarian: High upside players with lower ownership
            upside_bonus = player.variance * 0.8
            
            # Bonus for mid-tier players with high projections
            if 6000 <= player.salary <= 7500 and player.projection > 15:
                upside_bonus += 4.0
            
            return base_value + upside_bonus
            
        else:  # single_game
            return base_value * 1.15
    
    def _extract_result(self, prob, players: List[Player], player_vars: Dict, contest_type: str) -> LineupResult:
        """Extract lineup results with proper FanDuel ordering"""
        selected_players = []
        total_salary = 0
        total_ownership = 0
        
        for i, player in enumerate(players):
            if player_vars[i].varValue == 1:
                selected_players.append(player)
                total_salary += player.salary
                total_ownership += player.ownership
        
        # ORDER PLAYERS IN FANDUEL FORMAT
        ordered_players = self._format_lineup_for_fanduel(selected_players)
        
        # Calculate projected points
        if contest_type == 'single_game' and len(ordered_players) == 6:
            mvp = max(ordered_players, key=lambda p: p.projection)
            projected_points = mvp.projection * 1.5 + sum(p.projection for p in ordered_players if p != mvp)
        else:
            projected_points = sum(p.projection for p in ordered_players)
        
        # Calculate ceiling/floor
        ceiling = sum(p.projection + p.variance for p in ordered_players)
        floor = sum(max(0, p.projection - p.variance) for p in ordered_players)
        
        return LineupResult(
            players=ordered_players,  # Now properly ordered
            total_salary=total_salary,
            projected_points=projected_points,
            total_value=sum(p.value for p in ordered_players),
            ownership_total=total_ownership,
            correlation_score=self._calculate_correlation(ordered_players),
            weather_impact=np.mean([p.weather_factor for p in ordered_players]),
            contest_type=contest_type,
            ceiling_score=ceiling,
            floor_score=floor
        )
    
    def _calculate_correlation(self, players: List[Player]) -> float:
        """Calculate lineup correlation score"""
        correlation = 0.0
        
        # Find QB+WR same team stacks
        qb_teams = [p.team for p in players if p.position == 'QB']
        wr_teams = [p.team for p in players if p.position == 'WR']
        
        for qb_team in qb_teams:
            same_team_wrs = sum(1 for team in wr_teams if team == qb_team)
            if same_team_wrs > 0:
                correlation += 0.3 * same_team_wrs
        
        # Team stacking bonus
        team_counts = {}
        for player in players:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1
        
        for count in team_counts.values():
            if count >= 3:
                correlation += 0.4
            elif count >= 2:
                correlation += 0.2
        
        return min(1.0, correlation)
    
    def generate_multiple_lineups(self, players: List[Player], num_lineups: int = 10,
                                 contest_type: str = 'gpp', single_game_teams: List[str] = None) -> List[LineupResult]:
        """Generate multiple diverse lineups with BETTER randomization"""
        lineups = []
        used_combinations = set()
        max_attempts = num_lineups * 3  # Try 3x as many attempts
        
        for attempt in range(max_attempts):
            if len(lineups) >= num_lineups:
                break
                
            # More aggressive randomization for diversity
            randomized_players = []
            for player in players:
                new_player = Player(
                    id=player.id, name=player.name, position=player.position,
                    team=player.team, salary=player.salary, projection=player.projection,
                    ownership=player.ownership, weather_factor=player.weather_factor,
                    injury_risk=player.injury_risk, value=player.value, variance=player.variance
                )
                
                # Stronger randomization based on contest type
                if contest_type == 'gpp':
                    # Tournament: More variance for differentiation
                    random_factor = random.uniform(0.85, 1.15)
                elif contest_type == 'cash':
                    # Cash: Less variance for consistency
                    random_factor = random.uniform(0.95, 1.05)
                else:  # contrarian
                    # Contrarian: High variance to find unique builds
                    random_factor = random.uniform(0.8, 1.2)
                
                new_player.projection *= random_factor
                new_player.value = new_player.projection / (new_player.salary / 1000)
                randomized_players.append(new_player)
            
            lineup = self.optimize_lineup(randomized_players, contest_type, single_game_teams)
            if lineup:
                # More sophisticated uniqueness check
                core_players = tuple(sorted([p.id for p in lineup.players if p.salary > 7000]))
                if core_players not in used_combinations:
                    lineups.append(lineup)
                    used_combinations.add(core_players)
        
        # Sort by appropriate metric
        if contest_type == 'cash':
            lineups.sort(key=lambda x: x.floor_score, reverse=True)
        else:
            lineups.sort(key=lambda x: x.ceiling_score, reverse=True)
        
        logger.info(f"Generated {len(lineups)} unique {contest_type} lineups from {max_attempts} attempts")
        return lineups

    def export_lineups_to_csv(self, lineups: List[LineupResult], filename: str = None):
        """Export lineups to CSV"""
        if not filename:
            filename = f"data/lineups_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        lineup_data = []
        for i, lineup in enumerate(lineups):
            lineup_row = {'Lineup': i + 1}
            for j, player in enumerate(lineup.players):
                lineup_row[f'Player_{j+1}'] = f"{player.name} ({player.position}) ${player.salary}"
            lineup_row.update({
                'Total_Salary': lineup.total_salary,
                'Projected_Points': round(lineup.projected_points, 2),
                'Contest_Type': lineup.contest_type
            })
            lineup_data.append(lineup_row)
        
        df = pd.DataFrame(lineup_data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(lineups)} lineups to {filename}")
        return filename

# Main optimization function with AI integration
def optimize_dfs_lineups(player_data: List[Dict], weather_data: Dict = None,
                        num_lineups: int = 10, contest_type: str = 'gpp',
                        single_game_teams: List[str] = None) -> List[LineupResult]:
    """AI-Enhanced optimization entry point with CONSERVATIVE filtering"""
    
    logger.info(f"Starting AI-enhanced {contest_type} optimization...")
    
    # Step 1: Get AI strategic analysis (if available)
    if AI_AVAILABLE:
        try:
            analyzer = DFSAIAnalyzer()
            ai_analysis = analyzer.analyze_slate_for_optimization(
                player_data, weather_data or {}, {}, contest_type
            )
            
            if ai_analysis.get('ai_confidence', 0) > 0.5:
                logger.info(f"AI Strategy: {ai_analysis.get('strategy', 'No strategy')}")
                
                # Apply AI insights to player data
                for player in player_data:
                    player_name = player.get('name', '')
                    
                    # Apply AI ownership predictions
                    ownership_predictions = ai_analysis.get('ownership_predictions', {})
                    if player_name in ownership_predictions:
                        player['ai_ownership'] = ownership_predictions[player_name]
                    
                    # Boost leverage spots for tournaments
                    leverage_spots = ai_analysis.get('leverage_spots', [])
                    if contest_type == 'gpp' and any(player_name.lower() in spot.lower() for spot in leverage_spots):
                        player['projected_points'] = player.get('projected_points', 0) * 1.15
                        logger.info(f"AI Leverage boost: {player_name}")
                    
                    # Mark contrarian targets
                    contrarian_targets = ai_analysis.get('contrarian_targets', [])
                    if any(player_name.lower() in target.lower() for target in contrarian_targets):
                        player['contrarian_target'] = True
            
            # Log AI cost tracking
            cost_summary = analyzer.get_cost_summary()
            logger.info(f"AI Cost: ${cost_summary['weekly_spend']:.3f} of ${cost_summary['weekly_budget']:.2f} budget")
            
        except Exception as e:
            logger.warning(f"AI analysis failed, continuing without: {e}")
            ai_analysis = {'strategy': 'Fallback optimization', 'ai_confidence': 0}
    else:
        logger.info("AI analysis not available - using fallback optimization")
        ai_analysis = {'strategy': 'Fallback optimization', 'ai_confidence': 0}
    
    # Step 2: Run optimization with AI-enhanced data and CONSERVATIVE filtering
    optimizer = EnhancedDFSOptimizer()
    players = optimizer.prepare_players(player_data, weather_data)
    
    if not players:
        logger.error("No valid players for optimization")
        return []
    
    logger.info(f"Optimization: {num_lineups} {contest_type} lineups with {len(players)} active players")
    
    # Generate lineups
    lineups = optimizer.generate_multiple_lineups(players, num_lineups, contest_type, single_game_teams)
    
    return lineups
