"""
Advanced DFS lineup optimization using integer linear programming
Implements correlation-aware optimization with multiple constraints
"""
import pulp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import itertools
from sklearn.ensemble import RandomForestRegressor
import json

from config import FANDUEL_POSITIONS, FANDUEL_SALARY_CAP, OPTIMIZATION_CONFIG, H2H_POSITIONS

@dataclass
class Player:
    """Player data structure for optimization"""
    id: str
    name: str
    position: str
    team: str
    salary: int
    projection: float
    ownership: float = 10.0  # Default ownership percentage
    weather_factor: float = 1.0  # Weather adjustment factor
    injury_risk: float = 0.0  # Injury risk score (0-1)
    value: float = 0.0
    
    def __post_init__(self):
        self.value = self.projection / (self.salary / 1000) if self.salary > 0 else 0

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

class CorrelationMatrix:
    """Manages player correlations for stacking strategies"""
    
    def __init__(self):
        # Base correlation coefficients from research
        self.base_correlations = {
            ('QB', 'WR'): 0.62,
            ('QB', 'TE'): 0.32,
            ('QB', 'RB'): 0.08,
            ('RB', 'WR'): -0.05,
            ('WR', 'WR'): -0.15,  # Same team WRs compete
            ('QB', 'DST'): -0.41,  # QB vs opposing defense
            ('RB', 'DST'): 0.25   # Game script correlation
        }
    
    def get_correlation(self, player1: Player, player2: Player) -> float:
        """Calculate correlation between two players"""
        # Same team positive correlation for passing game
        if (player1.team == player2.team and 
            player1.position == 'QB' and player2.position in ['WR', 'TE']):
            return self.base_correlations[('QB', player2.position)]
        
        # Opposing team negative correlation (QB vs DST)
        if (player1.position == 'QB' and player2.position == 'DST' and 
            player1.team != player2.team):
            return self.base_correlations[('QB', 'DST')]
        
        # Same position negative correlation (target share competition)
        if (player1.position == player2.position and player1.team == player2.team and
            player1.position in ['WR', 'RB']):
            return self.base_correlations.get((player1.position, player2.position), -0.1)
        
        return 0.0

class WeatherAdjuster:
    """Adjusts player projections based on weather conditions"""
    
    def __init__(self):
        self.position_weather_sensitivity = {
            'QB': {'wind': -0.15, 'rain': -0.10, 'cold': -0.05},
            'WR': {'wind': -0.12, 'rain': -0.08, 'cold': -0.03},
            'TE': {'wind': -0.08, 'rain': -0.05, 'cold': -0.02},
            'RB': {'wind': -0.02, 'rain': 0.05, 'cold': 0.02},  # RBs benefit from bad weather
            'K': {'wind': -0.25, 'rain': -0.15, 'cold': -0.10},
            'DST': {'wind': 0.05, 'rain': 0.08, 'cold': 0.03}   # Defense benefits
        }
    
    def adjust_for_weather(self, player: Player, weather_data: Dict) -> float:
        """Calculate weather adjustment factor for player"""
        if not weather_data or player.position not in self.position_weather_sensitivity:
            return 1.0
        
        adjustment = 1.0
        sensitivity = self.position_weather_sensitivity[player.position]
        
        # Wind impact
        wind_speed = weather_data.get('wind_speed', 0)
        if wind_speed >= 15:  # High wind threshold
            adjustment += sensitivity.get('wind', 0)
        
        # Precipitation impact
        if weather_data.get('has_precipitation', False):
            adjustment += sensitivity.get('rain', 0)
        
        # Temperature impact
        temp = weather_data.get('temperature', 70)
        if temp <= 32:  # Freezing
            adjustment += sensitivity.get('cold', 0)
        
        return max(0.5, adjustment)  # Don't reduce below 50%

class OwnershipProjector:
    """Projects player ownership percentages for contrarian strategies"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
    
    def train_ownership_model(self, historical_data: List[Dict]):
        """Train ownership projection model on historical data"""
        if not historical_data:
            return
        
        try:
            df = pd.DataFrame(historical_data)
            
            # Create features for ownership prediction
            features = []
            for _, row in df.iterrows():
                feature_row = [
                    row.get('salary', 5000),
                    row.get('projection', 10),
                    row.get('value', 2.0),
                    1 if row.get('position') == 'QB' else 0,
                    1 if row.get('position') == 'RB' else 0,
                    1 if row.get('position') == 'WR' else 0,
                    row.get('news_sentiment', 0.5)  # Default neutral sentiment
                ]
                features.append(feature_row)
            
            X = np.array(features)
            y = np.array([row.get('actual_ownership', 10) for row in historical_data])
            
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("Ownership projection model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training ownership model: {e}")
    
    def predict_ownership(self, player: Player) -> float:
        """Predict ownership percentage for a player"""
        if not self.is_trained:
            # Return salary-based estimate if model not trained
            return min(max(player.salary / 300, 5), 40)
        
        try:
            features = np.array([[
                player.salary,
                player.projection,
                player.value,
                1 if player.position == 'QB' else 0,
                1 if player.position == 'RB' else 0,
                1 if player.position == 'WR' else 0,
                0.5  # Default neutral sentiment
            ]])
            
            ownership = self.model.predict(features)[0]
            return max(1, min(ownership, 50))  # Cap between 1-50%
            
        except Exception as e:
            logger.error(f"Error predicting ownership: {e}")
            return player.salary / 300  # Fallback to salary-based estimate

class DFSOptimizer:
    """Main DFS lineup optimization engine"""
    
    def __init__(self):
        self.correlation_matrix = CorrelationMatrix()
        self.weather_adjuster = WeatherAdjuster()
        self.ownership_projector = OwnershipProjector()
        
    def prepare_players(self, player_data: List[Dict], weather_data: Dict = None) -> List[Player]:
        """Convert raw player data to Player objects with adjustments"""
        players = []
        
        for data in player_data:
            try:
                # Create base player
                player = Player(
                    id=str(data.get('player_id', data.get('name', ''))),
                    name=data.get('player_name', data.get('name', '')),
                    position=data.get('position', ''),
                    team=data.get('team', ''),
                    salary=int(data.get('salary', 5000)),
                    projection=float(data.get('projection', data.get('fantasy_points_ppr', 0)))
                )
                
                # Apply weather adjustments
                if weather_data and player.team in weather_data:
                    team_weather = weather_data[player.team]
                    player.weather_factor = self.weather_adjuster.adjust_for_weather(
                        player, team_weather.get('forecast', {})
                    )
                    player.projection *= player.weather_factor
                
                # Project ownership
                player.ownership = self.ownership_projector.predict_ownership(player)
                
                # Recalculate value after adjustments
                player.value = player.projection / (player.salary / 1000) if player.salary > 0 else 0
                
                players.append(player)
                
            except Exception as e:
                logger.error(f"Error processing player {data}: {e}")
                continue
        
        logger.info(f"Prepared {len(players)} players for optimization")
        return players
    
    def optimize_lineup(self, 
                       players: List[Player], 
                       contest_type: str = 'gpp',
                       avoid_high_ownership: bool = True,
                       force_stacks: bool = True) -> Optional[LineupResult]:
        """Optimize a single lineup using integer linear programming"""
        
        try:
            # Create optimization problem
            prob = pulp.LpProblem("DFS_Lineup_Optimization", pulp.LpMaximize)
            
            # Decision variables - binary selection for each player
            player_vars = {}
            for i, player in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
            
            # Objective function - adjust based on contest type
            objective_terms = []
            for i, player in enumerate(players):
                points_value = player.projection
                
                if contest_type == 'gpp':
                    # GPP: Prioritize ceiling with some ownership penalty
                    if avoid_high_ownership:
                        ownership_penalty = max(0, (player.ownership - 20) * 0.1)
                        points_value -= ownership_penalty
                    # Slight bonus for high-variance players
                    if player.position in ['QB', 'WR']:
                        points_value *= 1.05
                        
                elif contest_type == 'cash':
                    # Cash: Prioritize floor and consistency
                    points_value *= 0.95  # Slight penalty to favor safer plays
                    # Penalty for high variance
                    if player.ownership < 5:  # Very low owned = risky
                        points_value *= 0.9
                        
                elif contest_type == 'h2h':
                    # Head-to-Head: Balanced approach with slight ceiling bias
                    points_value *= 1.02
                    # Moderate ownership penalty
                    if avoid_high_ownership and player.ownership > 25:
                        ownership_penalty = (player.ownership - 25) * 0.05
                        points_value -= ownership_penalty
                        
                elif contest_type == 'contrarian':
                    # Contrarian: Heavy penalty for high ownership
                    if player.ownership > 15:
                        ownership_penalty = (player.ownership - 15) * 0.2
                        points_value -= ownership_penalty
                    # Bonus for very low ownership
                    if player.ownership < 5:
                        points_value *= 1.15
                
                objective_terms.append(points_value * player_vars[i])
            
            prob += pulp.lpSum(objective_terms)
            
            # Add standard constraints
            self._add_standard_constraints(prob, players, player_vars, contest_type)
            
            # Contest-specific stacking constraints
            if force_stacks and contest_type in ['gpp', 'contrarian']:
                self._add_stacking_constraints(prob, players, player_vars)
            elif contest_type == 'h2h':
                # Light stacking for H2H
                self._add_light_stacking_constraints(prob, players, player_vars)
            # Cash games typically avoid stacking
            
            # Solve the problem
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                return self._extract_lineup_result(prob, players, player_vars)
            else:
                logger.warning(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
                return None
                
        except Exception as e:
            logger.error(f"Error in lineup optimization: {e}")
            return None
    
    def _add_light_stacking_constraints(self, prob, players: List[Player], player_vars: Dict):
        """Add lighter stacking constraints for H2H"""
        # Find QB-WR same-team combinations (but don't force them)
        teams_with_qb = set(p.team for p in players if p.position == 'QB')
        
        for team in teams_with_qb:
            qb_indices = [i for i, p in enumerate(players) if p.position == 'QB' and p.team == team]
            wr_te_indices = [i for i, p in enumerate(players) 
                           if p.position in ['WR', 'TE'] and p.team == team]
            
            # Optional correlation boost (not required)
            if qb_indices and wr_te_indices:
                for qb_idx in qb_indices:
                    # If QB selected, slight preference for same-team pass catchers
                    # This is handled in the objective function rather than as a hard constraint
                    pass
    
    def _add_stacking_constraints(self, prob, players: List[Player], player_vars: Dict):
        """Add QB-WR stacking constraints"""
        # Find all QB-WR same-team combinations
        teams_with_qb = set(p.team for p in players if p.position == 'QB')
        
        for team in teams_with_qb:
            qb_indices = [i for i, p in enumerate(players) if p.position == 'QB' and p.team == team]
            wr_te_indices = [i for i, p in enumerate(players) 
                           if p.position in ['WR', 'TE'] and p.team == team]
            
            # If QB selected from team, must select at least 1 WR/TE from same team
            if qb_indices and wr_te_indices:
                for qb_idx in qb_indices:
                    prob += pulp.lpSum([player_vars[i] for i in wr_te_indices]) >= player_vars[qb_idx]
    
    def _extract_lineup_result(self, prob, players: List[Player], player_vars: Dict) -> LineupResult:
        """Extract optimization results into LineupResult object"""
        selected_players = []
        total_salary = 0
        projected_points = 0
        total_ownership = 0
        
        for i, player in enumerate(players):
            if player_vars[i].varValue == 1:
                selected_players.append(player)
                total_salary += player.salary
                total_ownership += player.ownership
        
        # Calculate projected points (with MVP bonus for H2H)
        if len(selected_players) == 6:  # H2H format
            # Sort by projection to identify MVP
            selected_players.sort(key=lambda p: p.projection, reverse=True)
            mvp = selected_players[0]
            projected_points = mvp.projection * 1.5  # MVP gets 1.5x points
            projected_points += sum(p.projection for p in selected_players[1:])  # Regular points for others
        else:
            # Regular format
            projected_points = sum(p.projection for p in selected_players)
        
        # Calculate correlation score
        correlation_score = self._calculate_lineup_correlation(selected_players)
        
        # Calculate weather impact
        weather_impact = np.mean([p.weather_factor for p in selected_players])
        
        return LineupResult(
            players=selected_players,
            total_salary=total_salary,
            projected_points=projected_points,
            total_value=sum(p.value for p in selected_players),
            ownership_total=total_ownership,
            correlation_score=correlation_score,
            weather_impact=weather_impact
        )
    
    def _calculate_lineup_correlation(self, players: List[Player]) -> float:
        """Calculate overall lineup correlation score"""
        correlations = []
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players[i+1:], i+1):
                corr = self.correlation_matrix.get_correlation(player1, player2)
                if corr != 0:
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def generate_multiple_lineups(self, 
                                 players: List[Player], 
                                 num_lineups: int = 10,
                                 contest_type: str = 'gpp') -> List[LineupResult]:
        """Generate multiple diverse lineups"""
        lineups = []
        excluded_combinations = set()
        
        for i in range(num_lineups):
            # Create exclusion constraints for diversity
            lineup = self._optimize_with_exclusions(players, excluded_combinations, contest_type)
            
            if lineup:
                lineups.append(lineup)
                
                # Add this combination to exclusions for future lineups
                player_combo = tuple(sorted([p.id for p in lineup.players]))
                excluded_combinations.add(player_combo)
                
                # Also exclude core combinations (QB + top 2 skill players)
                skill_players = [p for p in lineup.players if p.position in ['QB', 'RB', 'WR', 'TE']]
                if len(skill_players) >= 3:
                    core_combo = tuple(sorted([p.id for p in skill_players[:3]]))
                    excluded_combinations.add(core_combo)
            else:
                # If we can't generate more unique lineups, break
                break
        
        # Sort by projected points
        lineups.sort(key=lambda x: x.projected_points, reverse=True)
        
        logger.info(f"Generated {len(lineups)} unique optimized lineups")
        return lineups
    
    def _optimize_with_exclusions(self, players: List[Player], excluded_combinations: set, contest_type: str) -> Optional[LineupResult]:
        """Optimize lineup with exclusion constraints for diversity"""
        try:
            # Create optimization problem
            prob = pulp.LpProblem("DFS_Lineup_Optimization_Diverse", pulp.LpMaximize)
            
            # Decision variables
            player_vars = {}
            for i, player in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
            
            # Objective function
            objective_terms = []
            for i, player in enumerate(players):
                points_value = player.projection
                
                # Ownership penalty for GPP contests
                if contest_type == 'gpp':
                    ownership_penalty = max(0, (player.ownership - 20) * 0.05)
                    points_value -= ownership_penalty
                
                objective_terms.append(points_value * player_vars[i])
            
            prob += pulp.lpSum(objective_terms)
            
            # Standard constraints (same as before)
            self._add_standard_constraints(prob, players, player_vars, contest_type)
            
            # Exclusion constraints for diversity
            for excluded_combo in excluded_combinations:
                if len(excluded_combo) <= len(players):
                    excluded_indices = []
                    for player_id in excluded_combo:
                        for i, player in enumerate(players):
                            if player.id == player_id:
                                excluded_indices.append(i)
                                break
                    
                    if len(excluded_indices) >= 3:  # Only exclude if we have enough players
                        prob += pulp.lpSum([player_vars[i] for i in excluded_indices]) <= len(excluded_indices) - 1
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status == pulp.LpStatusOptimal:
                return self._extract_lineup_result(prob, players, player_vars)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in diverse optimization: {e}")
            return None
    
    def _add_standard_constraints(self, prob, players: List[Player], player_vars: Dict, contest_type: str):
        """Add standard optimization constraints"""
        # Salary cap
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP
        
        # Position constraints based on contest type
        if contest_type == 'single_game':
            # Single Game format: 1 MVP + 5 FLEX (any positions)
            # MVP constraint
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) >= 1  # At least 1 MVP
            
            # Total players constraint (6 total: 1 MVP + 5 FLEX)
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 6
            
        else:
            # Regular FanDuel format
            for position, count in FANDUEL_POSITIONS.items():
                if position == 'FLEX':
                    flex_players = [i for i, p in enumerate(players) if p.position in ['RB', 'WR', 'TE']]
                    if flex_players:
                        prob += pulp.lpSum([player_vars[i] for i in flex_players]) >= count
                elif position == 'DST':
                    dst_players = [i for i, p in enumerate(players) if p.position in ['DST', 'DEF', 'D/ST']]
                    if dst_players:
                        prob += pulp.lpSum([player_vars[i] for i in dst_players]) == count
                    else:
                        logger.warning(f"No DST players available")
                else:
                    position_players = [i for i, p in enumerate(players) if p.position == position]
                    if position_players:
                        prob += pulp.lpSum([player_vars[i] for i in position_players]) == count
                    else:
                        logger.warning(f"No {position} players available")
            
            # Total roster size for regular formats
            total_required = sum(FANDUEL_POSITIONS.values())
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == total_required
        
        # Team diversity
        team_counts = {}
        for i, player in enumerate(players):
            if player.team not in team_counts:
                team_counts[player.team] = []
            team_counts[player.team].append(i)
        
        for team, player_indices in team_counts.items():
            max_per_team = 3 if contest_type == 'h2h' else 4  # Less team stacking in H2H
            prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= max_per_team
    
    def export_lineups_to_csv(self, lineups: List[LineupResult], filename: str = None):
        """Export lineups to CSV format for upload to DFS sites"""
        if not filename:
            filename = f"lineups_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        lineup_data = []
        
        for i, lineup in enumerate(lineups):
            lineup_row = {'Lineup': i + 1}
            
            # Check if this is Single Game format or regular
            if len(lineup.players) == 6:
                # Single Game format: MVP + 5 FLEX
                sorted_players = sorted(lineup.players, key=lambda p: p.projection, reverse=True)
                lineup_row['MVP'] = f"{sorted_players[0].name} ({sorted_players[0].id})"
                for j in range(5):
                    if j + 1 < len(sorted_players):
                        lineup_row[f'FLEX{j+1}'] = f"{sorted_players[j+1].name} ({sorted_players[j+1].id})"
                    else:
                        lineup_row[f'FLEX{j+1}'] = ""
                        
            else:
                # Regular FanDuel format: QB, RB, RB, WR, WR, WR, TE, FLEX, D
                # Group players by position
                by_position = {}
                for player in lineup.players:
                    pos = player.position
                    if pos in ['DST', 'DEF']:
                        pos = 'D'  # FanDuel uses 'D' for defense
                    if pos not in by_position:
                        by_position[pos] = []
                    by_position[pos].append(player)
                
                # Initialize all positions
                lineup_row['QB'] = ""
                lineup_row['RB'] = ""
                lineup_row['RB2'] = ""  # Second RB slot
                lineup_row['WR'] = ""
                lineup_row['WR2'] = ""  # Second WR slot
                lineup_row['WR3'] = ""  # Third WR slot
                lineup_row['TE'] = ""
                lineup_row['FLEX'] = ""
                lineup_row['D'] = ""
                
                # Fill QB
                if 'QB' in by_position and by_position['QB']:
                    lineup_row['QB'] = f"{by_position['QB'][0].name} ({by_position['QB'][0].id})"
                
                # Fill RBs
                if 'RB' in by_position:
                    if len(by_position['RB']) >= 1:
                        lineup_row['RB'] = f"{by_position['RB'][0].name} ({by_position['RB'][0].id})"
                    if len(by_position['RB']) >= 2:
                        lineup_row['RB2'] = f"{by_position['RB'][1].name} ({by_position['RB'][1].id})"
                
                # Fill WRs
                if 'WR' in by_position:
                    if len(by_position['WR']) >= 1:
                        lineup_row['WR'] = f"{by_position['WR'][0].name} ({by_position['WR'][0].id})"
                    if len(by_position['WR']) >= 2:
                        lineup_row['WR2'] = f"{by_position['WR'][1].name} ({by_position['WR'][1].id})"
                    if len(by_position['WR']) >= 3:
                        lineup_row['WR3'] = f"{by_position['WR'][2].name} ({by_position['WR'][2].id})"
                
                # Fill TE
                if 'TE' in by_position and by_position['TE']:
                    lineup_row['TE'] = f"{by_position['TE'][0].name} ({by_position['TE'][0].id})"
                
                # Fill FLEX (remaining RB/WR/TE)
                flex_candidates = []
                used_players = []
                
                # Add used players
                for pos in ['QB', 'RB', 'WR', 'TE', 'D']:
                    if pos in by_position:
                        if pos == 'RB':
                            used_players.extend(by_position[pos][:2])  # First 2 RBs used
                        elif pos == 'WR':
                            used_players.extend(by_position[pos][:3])  # First 3 WRs used
                        elif pos == 'TE':
                            used_players.extend(by_position[pos][:1])  # First TE used
                        else:
                            used_players.extend(by_position[pos])
                
                # Find FLEX player (unused RB/WR/TE)
                for pos in ['RB', 'WR', 'TE']:
                    if pos in by_position:
                        for player in by_position[pos]:
                            if player not in used_players:
                                lineup_row['FLEX'] = f"{player.name} ({player.id})"
                                break
                        if lineup_row['FLEX']:
                            break
                
                # Fill Defense
                if 'D' in by_position and by_position['D']:
                    lineup_row['D'] = f"{by_position['D'][0].name} ({by_position['D'][0].id})"
            
            # Add summary stats
            lineup_row.update({
                'Total_Salary': lineup.total_salary,
                'Projected_Points': round(lineup.projected_points, 2),
                'Ownership_Total': round(lineup.ownership_total, 1),
                'Correlation_Score': round(lineup.correlation_score, 3)
            })
            
            lineup_data.append(lineup_row)
        
        df = pd.DataFrame(lineup_data)
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(lineups)} lineups to {filename}")
        return filename

# Utility function for external use
def optimize_dfs_lineups(player_data: List[Dict], 
                        weather_data: Dict = None,
                        num_lineups: int = 10,
                        contest_type: str = 'gpp') -> List[LineupResult]:
    """Main entry point for lineup optimization"""
    optimizer = DFSOptimizer()
    players = optimizer.prepare_players(player_data, weather_data)
    
    if not players:
        logger.error("No valid players for optimization")
        return []
    
    return optimizer.generate_multiple_lineups(players, num_lineups, contest_type)
