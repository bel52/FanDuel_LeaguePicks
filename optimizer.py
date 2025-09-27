"""
Enhanced DFS lineup optimization with proper FLEX position support
Fixed for tournament play with proper FanDuel format and FRIENDS LEAGUE STRATEGY
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

# AI Integration import with better error handling
try:
    from ai_analyzer import DualAIDFSAnalyzer
    AI_AVAILABLE = True
    logger.info("AI analyzer imported successfully")
except ImportError as e:
    logger.warning(f"AI analyzer not available: {e}")
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
    """Enhanced DFS optimization for exact FanDuel format with FRIENDS LEAGUE STRATEGY"""

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
        """Optimize with EXACT FanDuel constraints + FRIENDS LEAGUE STRATEGY"""

        try:
            # Filter for single game
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
                    return None

            # Project ownership using FRIENDS LEAGUE psychology
            for player in players:
                player.ownership = self._predict_friends_league_ownership(player, contest_type)

            # Create optimization problem
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)

            player_vars = {}
            for i, player in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')

            # Objective function using FRIENDS LEAGUE strategy
            objective_terms = []
            for i, player in enumerate(players):
                points_value = self._calculate_contest_value(player, contest_type)
                objective_terms.append(points_value * player_vars[i])

            prob += pulp.lpSum(objective_terms)

            # Add CORRECTED constraints
            self._add_fanduel_constraints(prob, players, player_vars, contest_type, single_game_teams)

            # Add FRIENDS LEAGUE constraints
            if not single_game_teams:  # Only for regular contests, not single game
                self._add_friends_league_constraints(prob, players, player_vars, contest_type)

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

    def _add_friends_league_constraints(self, prob, players: List[Player], player_vars: Dict, contest_type: str):
        """Add constraints optimized for beating 11 friends, not perfect optimization"""

        # Friends league specific constraints
        if contest_type == 'gpp':
            # Force at least one "leverage" play
            expensive_players = [i for i, p in enumerate(players) if p.salary >= 9000]
            if expensive_players:
                prob += pulp.lpSum([player_vars[i] for i in expensive_players]) >= 1

            # Force some salary diversity to avoid obvious builds
            salary_buckets = {
                'cheap': [i for i, p in enumerate(players) if p.salary <= 5000],
                'mid': [i for i, p in enumerate(players) if 5000 < p.salary < 8000],
                'expensive': [i for i, p in enumerate(players) if p.salary >= 8000]
            }

            # Must have at least one from each bucket (forces creativity)
            for bucket_name, indices in salary_buckets.items():
                if indices:
                    prob += pulp.lpSum([player_vars[i] for i in indices]) >= 1

        elif contest_type == 'cash':
            # Force high-value plays (casuals miss obvious value)
            high_value_players = [i for i, p in enumerate(players) if p.value >= 3.5]
            if high_value_players:
                prob += pulp.lpSum([player_vars[i] for i in high_value_players]) >= 3

        elif contest_type == 'contrarian':
            # Force expensive studs (fade the "value" builds casuals love)
            premium_players = [i for i, p in enumerate(players) if p.salary >= 8500]
            if premium_players:
                prob += pulp.lpSum([player_vars[i] for i in premium_players]) >= 2

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

    def _predict_friends_league_ownership(self, player: Player, contest_type: str) -> float:
        """TARGETED FIX: Ultra-low ownership for 12-person league"""

        # Ultra-conservative baseline - start much lower
        ownership = 12.0  # 12% = ~1.4 people out of 12

        # Salary tiers (much more conservative)
        if player.salary >= 9500:
            ownership = 35.0  # Even expensive studs only ~4 people
        elif player.salary >= 8500:
            ownership = 25.0  # ~3 people
        elif player.salary >= 7500:
            ownership = 20.0  # ~2.4 people
        elif player.salary >= 6000:
            ownership = 15.0  # ~1.8 people
        elif player.salary <= 4500:
            ownership = 10.0  # ~1.2 people
        else:
            ownership = 12.0  # ~1.4 people

        # Much smaller position adjustments
        if player.position == 'QB':
            if player.salary >= 8500:
                ownership += 5  # Popular QBs
            elif player.salary <= 6500:
                ownership += 3  # Cheap QBs
        elif player.position == 'RB':
            ownership += 3  # RBs slightly more popular
        elif player.position == 'TE':
            ownership -= 5  # TEs less popular
        elif player.position == 'D':
            ownership -= 7  # Defense much less popular

        # Smaller value adjustments
        if player.value >= 4.0:
            ownership += 5  # Clear value
        elif player.value < 2.5:
            ownership -= 5  # Poor value

        # Very tight bounds for 12-person league
        return max(5.0, min(40.0, ownership))

    def _calculate_contest_value(self, player: Player, contest_type: str) -> float:
        """SIMPLE: Actually different strategies"""

        base_value = player.projection

        if contest_type == 'gpp':
            # GPP: Target 25-40% owned players (avoid super chalk)
            if 25 <= player.ownership <= 40:
                base_value += 2.0  # Sweet spot
            elif player.ownership >= 45:
                base_value -= 1.0  # Light penalty for chalk

            # Ceiling bonus
            return base_value + (player.variance * 1.2)

        elif contest_type == 'cash':
            # CASH: Just get the best value plays
            if player.value >= 3.5:
                base_value += 5.0  # Value is king

            # Floor bonus
            return base_value - (player.variance * 0.2)

        elif contest_type == 'contrarian':
            # CONTRARIAN: HARD fade anything >35% owned
            if player.ownership <= 20:
                base_value += 5.0  # Big boost for low owned
            elif player.ownership >= 35:
                base_value -= 8.0  # HEAVY penalty for chalk

            # Max ceiling
            return base_value + (player.variance * 2.0)

        else:  # bestball
            return base_value + (player.variance * 1.0)

    def _predict_friends_league_ownership(self, player: Player, contest_type: str) -> float:
        """SIMPLE: Realistic ownership for 12-person friends league"""

        # Start with realistic base
        ownership = 20.0  # 20% = ~2-3 people out of 12

        # Salary-based (friends are salary obsessed)
        if player.salary >= 9500:
            ownership = 45.0  # Jonathan Taylor type - 5-6 people pick him
        elif player.salary >= 8500:
            ownership = 35.0  # James Cook type - 4-5 people
        elif player.salary >= 7500:
            ownership = 28.0  # Solid players - 3-4 people
        elif player.salary >= 6000:
            ownership = 22.0  # Mid tier - 2-3 people
        elif player.salary <= 4500:
            ownership = 15.0  # Cheap plays - 1-2 people
        else:
            ownership = 18.0  # Boring mid tier - 2 people

        # Position adjustments (friends have clear biases)
        if player.position == 'QB':
            if player.salary >= 8500:
                ownership += 8  # Elite QBs very popular
            elif player.salary <= 6500:
                ownership += 5  # Cheap QB lottery tickets
        elif player.position == 'RB':
            ownership += 5  # Everyone loves RBs
        elif player.position == 'TE':
            ownership -= 8  # Friends hate TEs
        elif player.position == 'D':
            ownership -= 10  # Friends really hate defenses

        # Value boost (friends spot obvious value)
        if player.value >= 4.0:
            ownership += 10  # Clear value gets picked
        elif player.value < 2.5:
            ownership -= 8  # Poor value avoided

        # Team popularity
        if player.team in ['KC', 'BUF', 'DAL', 'BAL']:
            ownership += 5
        elif player.team in ['JAX', 'TEN', 'CAR']:
            ownership -= 5

        # STRICT bounds for 12-person league
        return max(8.0, min(55.0, ownership))

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
        """Generate diverse lineups optimized for 12-person friends league"""
        lineups = []
        used_combinations = set()
        max_attempts = num_lineups * 3  # Fewer attempts needed for friends league

        for attempt in range(max_attempts):
            if len(lineups) >= num_lineups:
                break

            # Moderate randomization for friends league diversity
            randomized_players = []
            for player in players:
                new_player = Player(
                    id=player.id, name=player.name, position=player.position,
                    team=player.team, salary=player.salary, projection=player.projection,
                    ownership=player.ownership, weather_factor=player.weather_factor,
                    injury_risk=player.injury_risk, value=player.value, variance=player.variance
                )

                # Moderate randomization for friends league
                if contest_type == 'gpp':
                    # Balanced variance for weekly wins
                    random_factor = random.uniform(0.85, 1.15)
                elif contest_type == 'cash':
                    # Low variance for cash games
                    random_factor = random.uniform(0.95, 1.05)
                else:  # contrarian
                    # Higher variance for contrarian builds
                    random_factor = random.uniform(0.75, 1.25)

                new_player.projection *= random_factor
                new_player.value = new_player.projection / (new_player.salary / 1000)
                randomized_players.append(new_player)

            lineup = self.optimize_lineup(randomized_players, contest_type, single_game_teams)
            if lineup:
                # FORCE diversity by blocking overused players
                if len(lineups) > 0:
                    # Count player usage
                    player_usage = {}
                    for existing_lineup in lineups:
                        for player in existing_lineup.players:
                            player_usage[player.id] = player_usage.get(player.id, 0) + 1

                    # Check if this lineup is too similar
                    overused_players = 0
                    max_usage = max(2, num_lineups // 4)  # Allow max 25% usage
                    for player in lineup.players:
                        usage_count = player_usage.get(player.id, 0)
                        if usage_count >= max_usage:
                            overused_players += 1

                    # Skip if too many overused players
                    if overused_players > 3:
                        continue

                # Original uniqueness check
                core_players = tuple(sorted([p.id for p in lineup.players if p.salary > 6500]))
                if core_players not in used_combinations:
                    lineups.append(lineup)
                    used_combinations.add(core_players)

        # Sort by appropriate metric for friends league
        if contest_type == 'cash':
            lineups.sort(key=lambda x: x.floor_score, reverse=True)
        else:
            # For friends league, balance ceiling and ownership
            lineups.sort(key=lambda x: (x.ceiling_score - (x.ownership_total * 0.5)), reverse=True)

        logger.info(f"Generated {len(lineups)} unique {contest_type} lineups for friends league")
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
    """AI-Enhanced optimization entry point with FRIENDS LEAGUE STRATEGY"""

    logger.info(f"Starting AI-enhanced {contest_type} optimization...")

    # Step 1: Get AI strategic analysis (if available)
    if AI_AVAILABLE:
        try:
            analyzer = DualAIDFSAnalyzer()
            ai_analysis = analyzer.analyze_slate_for_optimization(
                player_data, weather_data or {}, {}, contest_type
            )

            if ai_analysis.get('ai_enabled', False):
                logger.info(f"AI Strategy: {ai_analysis.get('ai_strategy', 'No strategy')[:100]}...")

                # Apply AI ownership adjustments to player data
                ownership_adjustments = ai_analysis.get('ownership_adjustments', {})
                if ownership_adjustments:
                    adjusted_count = 0
                    for player in player_data:
                        player_name = player.get('name', '')
                        if player_name in ownership_adjustments:
                            adjustment_factor = ownership_adjustments[player_name]
                            # Modify ownership to influence optimization
                            if 'ownership' not in player:
                                player['ownership'] = 15.0  # Default ownership
                            player['ownership'] *= adjustment_factor
                            adjusted_count += 1

                    logger.info(f"AI adjusted ownership for {adjusted_count} players")

            # Log AI cost tracking
            cost_summary = analyzer.get_cost_summary()
            logger.info(f"AI Cost: ${cost_summary['weekly_spend']:.3f} of ${cost_summary['weekly_budget']:.2f} budget")

        except Exception as e:
            logger.warning(f"AI analysis failed, continuing without: {e}")
            ai_analysis = {'ai_strategy': 'Fallback optimization', 'ai_enabled': False}
    else:
        logger.info("AI analysis not available - using fallback optimization")
        ai_analysis = {'ai_strategy': 'Fallback optimization', 'ai_enabled': False}

    # Step 2: Run optimization with AI-enhanced data and FRIENDS LEAGUE STRATEGY
    optimizer = EnhancedDFSOptimizer()
    players = optimizer.prepare_players(player_data, weather_data)

    if not players:
        logger.error("No valid players for optimization")
        return []

    logger.info(f"Optimization: {num_lineups} {contest_type} lineups with {len(players)} active players")

    # Generate lineups
    lineups = optimizer.generate_multiple_lineups(players, num_lineups, contest_type, single_game_teams)

    return lineups