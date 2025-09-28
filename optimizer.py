"""
Enhanced DFS lineup optimization with Monte Carlo variance analysis
Updated to use true ceiling/floor modeling for tournament wins
"""
import pulp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import random
import json
import asyncio
from datetime import datetime

from config import FANDUEL_POSITIONS, FANDUEL_SALARY_CAP, OPTIMIZATION_CONFIG, DATA_DIR

# Import the Monte Carlo engine
from monte_carlo_engine import (
    MonteCarloEngine,
    PlayerSimulation,
    convert_player_data_to_simulation,
    enhance_lineup_with_monte_carlo
)

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
    """Enhanced Player data structure with Monte Carlo variance"""
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
    # NEW Monte Carlo fields
    floor_10: float = 0.0
    ceiling_90: float = 0.0
    ceiling_95: float = 0.0
    boom_rate: float = 0.0
    bust_rate: float = 0.0
    monte_carlo_analyzed: bool = False

    def __post_init__(self):
        self.value = self.projection / (self.salary / 1000) if self.salary > 0 else 0
        variance_multipliers = {'QB': 0.28, 'RB': 0.35, 'WR': 0.45, 'TE': 0.38, 'D': 0.42}
        self.variance = self.projection * variance_multipliers.get(self.position, 0.35)

@dataclass
class LineupResult:
    """Enhanced lineup result with Monte Carlo analysis"""
    players: List[Player]
    total_salary: int
    projected_points: float
    total_value: float
    ownership_total: float
    correlation_score: float
    weather_impact: float
    contest_type: str
    # NEW Monte Carlo fields
    ceiling_90: float = 0.0
    ceiling_95: float = 0.0
    floor_10: float = 0.0
    floor_25: float = 0.0
    variance_score: float = 0.0
    sharpe_ratio: float = 0.0
    risk_level: str = "Unknown"
    boom_probability: float = 0.0
    bust_probability: float = 0.0
    monte_carlo_insights: Dict = None

class EnhancedDFSOptimizer:
    """Enhanced DFS optimization with Monte Carlo variance modeling"""

    def __init__(self, use_monte_carlo: bool = True, mc_simulations: int = 5000):
        self.use_monte_carlo = use_monte_carlo
        self.mc_simulations = mc_simulations
        self.monte_carlo_engine = MonteCarloEngine(num_simulations=mc_simulations) if use_monte_carlo else None

    async def prepare_players(self, player_data: List[Dict], weather_data: Dict = None,
                            vegas_data: Dict = None) -> List[Player]:
        """Convert player data with optional Monte Carlo enhancement"""
        players = []

        for data in player_data:
            try:
                player_name = data.get('player_name', data.get('name', ''))
                position = data.get('position', '')
                team = data.get('team', '')
                salary = int(data.get('salary', 5000))
                projection = float(data.get('projection', data.get('projected_points', 0)))

                # Basic filtering
                if not player_name or len(player_name.strip()) < 2 or projection < 0:
                    continue

                # Normalize defense position
                if position in ['DST', 'DEF', 'D/ST']:
                    position = 'D'

                # Create base player
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

                players.append(player)

            except Exception as e:
                logger.error(f"Error processing player {data}: {e}")
                continue

        # ENHANCE with Monte Carlo analysis
        if self.use_monte_carlo and players:
            logger.info(f"Running Monte Carlo analysis on {len(players)} players...")
            players = await self._enhance_players_with_monte_carlo(players, weather_data, vegas_data)

        return players

    async def _enhance_players_with_monte_carlo(self, players: List[Player],
                                             weather_data: Dict = None,
                                             vegas_data: Dict = None) -> List[Player]:
        """Enhance players with Monte Carlo variance analysis"""

        # Convert to simulation format
        sim_data = []
        for player in players:
            sim_data.append({
                'name': player.name,
                'position': player.position,
                'team': player.team,
                'salary': player.salary,
                'projected_points': player.projection
            })

        sim_players = convert_player_data_to_simulation(sim_data, weather_data, vegas_data)

        # Run Monte Carlo on each player
        enhanced_players = []

        # Process in batches for efficiency
        batch_size = 20
        for i in range(0, len(players), batch_size):
            batch_players = players[i:i + batch_size]
            batch_sims = sim_players[i:i + batch_size]

            # Run simulations for this batch
            sim_tasks = []
            for sim_player in batch_sims:
                task = self.monte_carlo_engine.simulate_player_performance(sim_player, num_sims=1000)
                sim_tasks.append(task)

            batch_results = await asyncio.gather(*sim_tasks)

            # Apply results to players
            for j, (player, sim_result) in enumerate(zip(batch_players, batch_results)):
                player.floor_10 = sim_result['floor_10']
                player.ceiling_90 = sim_result['ceiling_90']
                player.ceiling_95 = sim_result['ceiling_95']
                player.boom_rate = sim_result['boom_rate']
                player.bust_rate = sim_result['bust_rate']
                player.variance = sim_result['std']
                player.monte_carlo_analyzed = True

                enhanced_players.append(player)

        logger.info(f"Enhanced {len(enhanced_players)} players with Monte Carlo analysis")
        return enhanced_players

    def optimize_lineup(self, players: List[Player], contest_type: str = 'gpp',
                       single_game_teams: List[str] = None) -> Optional[LineupResult]:
        """Optimize with Monte Carlo-enhanced objective function"""

        try:
            # Filter for single game
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
                    return None

            # Project ownership using friends league psychology
            for player in players:
                player.ownership = self._predict_friends_league_ownership(player, contest_type)

            # Create optimization problem
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)

            player_vars = {}
            for i, player in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')

            # ENHANCED objective function using Monte Carlo data
            objective_terms = []
            for i, player in enumerate(players):
                if self.use_monte_carlo and player.monte_carlo_analyzed:
                    points_value = self._calculate_monte_carlo_value(player, contest_type)
                else:
                    points_value = self._calculate_contest_value(player, contest_type)
                objective_terms.append(points_value * player_vars[i])

            prob += pulp.lpSum(objective_terms)

            # Add constraints
            self._add_fanduel_constraints(prob, players, player_vars, contest_type, single_game_teams)

            if not single_game_teams:
                self._add_friends_league_constraints(prob, players, player_vars, contest_type)

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status == pulp.LpStatusOptimal:
                result = self._extract_result(prob, players, player_vars, contest_type)

                # Enhance result with Monte Carlo analysis
                if self.use_monte_carlo:
                    result = await self._enhance_lineup_result_with_monte_carlo(result)

                return result
            else:
                logger.warning(f"Optimization failed: {pulp.LpStatus[prob.status]}")
                return None

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return None

    def _calculate_monte_carlo_value(self, player: Player, contest_type: str) -> float:
        """Calculate player value using Monte Carlo variance data"""

        base_value = player.projection

        if contest_type == 'gpp':
            # GPP: Heavily weight ceiling potential and boom rate
            ceiling_bonus = (player.ceiling_90 - player.projection) * 2.0
            boom_bonus = player.boom_rate * 10.0  # Strong boom bonus
            bust_penalty = player.bust_rate * 5.0   # Light bust penalty

            # Ownership leverage
            if 20 <= player.ownership <= 35:
                ownership_bonus = 3.0
            elif player.ownership >= 40:
                ownership_bonus = -2.0
            else:
                ownership_bonus = 0.0

            return base_value + ceiling_bonus + boom_bonus - bust_penalty + ownership_bonus

        elif contest_type == 'cash':
            # CASH: Heavily weight floor and consistency
            floor_bonus = player.floor_10 * 2.0
            consistency_bonus = 5.0 if player.bust_rate < 0.15 else 0.0
            value_bonus = 3.0 if player.value >= 3.5 else 0.0

            # Penalize high variance heavily
            variance_penalty = player.variance * 0.5

            return base_value + floor_bonus + consistency_bonus + value_bonus - variance_penalty

        elif contest_type == 'contrarian':
            # CONTRARIAN: Max ceiling, low ownership
            ceiling_bonus = (player.ceiling_95 - player.projection) * 3.0

            # Heavy ownership leverage
            if player.ownership <= 15:
                ownership_bonus = 8.0
            elif player.ownership >= 35:
                ownership_bonus = -10.0
            else:
                ownership_bonus = 2.0

            boom_bonus = player.boom_rate * 15.0

            return base_value + ceiling_bonus + ownership_bonus + boom_bonus

        else:  # bestball
            return base_value + (player.ceiling_90 - player.projection) * 1.5

    async def _enhance_lineup_result_with_monte_carlo(self, lineup_result: LineupResult) -> LineupResult:
        """Enhance lineup result with full Monte Carlo analysis"""

        try:
            # Convert lineup to format for Monte Carlo
            lineup_data = []
            for player in lineup_result.players:
                lineup_data.append({
                    'name': player.name,
                    'position': player.position,
                    'team': player.team,
                    'salary': player.salary,
                    'projected_points': player.projection
                })

            # Run full lineup Monte Carlo simulation
            mc_results = await enhance_lineup_with_monte_carlo(
                lineup_data,
                num_simulations=self.mc_simulations
            )

            lineup_sim = mc_results['simulation_results']['lineup_simulation']
            insights = mc_results['insights']

            # Update lineup result with Monte Carlo data
            lineup_result.ceiling_90 = lineup_sim['ceiling_90']
            lineup_result.ceiling_95 = lineup_sim['ceiling_95']
            lineup_result.floor_10 = lineup_sim['floor_10']
            lineup_result.floor_25 = lineup_sim['floor_25']
            lineup_result.variance_score = lineup_sim['std']
            lineup_result.sharpe_ratio = lineup_sim['sharpe_ratio']
            lineup_result.risk_level = insights['risk_assessment']
            lineup_result.monte_carlo_insights = {
                'recommendations': insights['optimization_recommendations'],
                'player_analysis': insights['player_analysis'],
                'correlation_strength': insights['correlation_strength']
            }

            # Calculate boom/bust probabilities for lineup
            mean_score = lineup_sim['mean']
            lineup_result.boom_probability = self._calculate_boom_probability(lineup_sim, mean_score)
            lineup_result.bust_probability = self._calculate_bust_probability(lineup_sim, mean_score)

            logger.info(f"Enhanced lineup with Monte Carlo: {lineup_result.risk_level} risk, "
                       f"{lineup_result.boom_probability:.1%} boom rate")

        except Exception as e:
            logger.error(f"Error enhancing lineup with Monte Carlo: {e}")

        return lineup_result

    def _calculate_boom_probability(self, lineup_sim: Dict, mean_score: float) -> float:
        """Calculate probability of boom performance (90th+ percentile)"""
        boom_threshold = mean_score * 1.3  # 30% above projection
        ceiling_90 = lineup_sim.get('ceiling_90', mean_score)

        if ceiling_90 > boom_threshold:
            return 0.15  # Roughly 15% boom rate for good lineups
        else:
            return 0.05

    def _calculate_bust_probability(self, lineup_sim: Dict, mean_score: float) -> float:
        """Calculate probability of bust performance (bottom 25%)"""
        bust_threshold = mean_score * 0.75  # 25% below projection
        floor_25 = lineup_sim.get('floor_25', mean_score)

        if floor_25 < bust_threshold:
            return 0.3   # 30% bust rate indicates risk
        else:
            return 0.15  # 15% for safer lineups

    # Keep all existing methods (_add_fanduel_constraints, _predict_friends_league_ownership, etc.)
    # These remain the same from your current optimizer.py

    def _add_fanduel_constraints(self, prob, players: List[Player], player_vars: Dict,
                                contest_type: str, single_game_teams: List[str]):
        """EXACT FanDuel constraints (unchanged from existing code)"""
        # Salary cap
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP

        # Handle locked players
        locked_players = []
        for i, player in enumerate(players):
            if hasattr(player, 'locked') and player.locked:
                prob += player_vars[i] == 1
                locked_players.append(i)
                logger.info(f"🔒 LOCKED: {player.name} ({player.position})")

        if single_game_teams:
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 6
            return

        # Position requirements
        qb_indices = [i for i, p in enumerate(players) if p.position == 'QB']
        rb_indices = [i for i, p in enumerate(players) if p.position == 'RB']
        wr_indices = [i for i, p in enumerate(players) if p.position == 'WR']
        te_indices = [i for i, p in enumerate(players) if p.position == 'TE']
        d_indices = [i for i, p in enumerate(players) if p.position == 'D']

        # Exact FanDuel requirements
        if qb_indices:
            prob += pulp.lpSum([player_vars[i] for i in qb_indices]) == 1
        if d_indices:
            prob += pulp.lpSum([player_vars[i] for i in d_indices]) == 1

        flex_indices = rb_indices + wr_indices + te_indices

        if rb_indices:
            prob += pulp.lpSum([player_vars[i] for i in rb_indices]) >= 2
        if wr_indices:
            prob += pulp.lpSum([player_vars[i] for i in wr_indices]) >= 3
        if te_indices:
            prob += pulp.lpSum([player_vars[i] for i in te_indices]) >= 1

        prob += pulp.lpSum([player_vars[i] for i in flex_indices]) == 7

        # Position maximums
        if rb_indices:
            prob += pulp.lpSum([player_vars[i] for i in rb_indices]) <= 3
        if wr_indices:
            prob += pulp.lpSum([player_vars[i] for i in wr_indices]) <= 4
        if te_indices:
            prob += pulp.lpSum([player_vars[i] for i in te_indices]) <= 2

        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 9

        # Team diversity
        team_counts = {}
        for i, player in enumerate(players):
            if player.team not in team_counts:
                team_counts[player.team] = []
            team_counts[player.team].append(i)

        max_per_team = 3 if contest_type == 'cash' else 4
        for team, player_indices in team_counts.items():
            prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= max_per_team

        if contest_type == 'gpp':
            self._add_stacking_incentive(prob, players, player_vars, qb_indices, wr_indices)

    def _add_friends_league_constraints(self, prob, players: List[Player], player_vars: Dict, contest_type: str):
        """Friends league constraints (unchanged from existing code)"""
        if contest_type == 'gpp':
            expensive_players = [i for i, p in enumerate(players) if p.salary >= 9000]
            if expensive_players:
                prob += pulp.lpSum([player_vars[i] for i in expensive_players]) >= 1

        elif contest_type == 'cash':
            high_value_players = [i for i, p in enumerate(players) if p.value >= 3.5]
            if high_value_players:
                prob += pulp.lpSum([player_vars[i] for i in high_value_players]) >= 3

    def _add_stacking_incentive(self, prob, players: List[Player], player_vars: Dict,
                               qb_indices: List[int], wr_indices: List[int]):
        """QB+WR stacking (unchanged from existing code)"""
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

        for team in team_qbs:
            if team in team_wrs:
                qb_vars = [player_vars[i] for i in team_qbs[team]]
                wr_vars = [player_vars[i] for i in team_wrs[team]]

                if qb_vars and wr_vars:
                    prob += pulp.lpSum(wr_vars) >= 0.5 * pulp.lpSum(qb_vars)

    def _predict_friends_league_ownership(self, player: Player, contest_type: str) -> float:
        """Ultra-conservative ownership for 12-person league (unchanged)"""
        ownership = 12.0

        if player.salary >= 9500:
            ownership = 35.0
        elif player.salary >= 8500:
            ownership = 25.0
        elif player.salary >= 7500:
            ownership = 20.0
        elif player.salary >= 6000:
            ownership = 15.0
        elif player.salary <= 4500:
            ownership = 10.0
        else:
            ownership = 12.0

        if player.position == 'QB':
            if player.salary >= 8500:
                ownership += 5
            elif player.salary <= 6500:
                ownership += 3
        elif player.position == 'RB':
            ownership += 3
        elif player.position == 'TE':
            ownership -= 5
        elif player.position == 'D':
            ownership -= 7

        if player.value >= 4.0:
            ownership += 5
        elif player.value < 2.5:
            ownership -= 5

        return max(5.0, min(40.0, ownership))

    def _calculate_contest_value(self, player: Player, contest_type: str) -> float:
        """Fallback value calculation when Monte Carlo not available"""
        base_value = player.projection

        if contest_type == 'gpp':
            if 25 <= player.ownership <= 40:
                base_value += 2.0
            elif player.ownership >= 45:
                base_value -= 1.0
            return base_value + (player.variance * 1.2)

        elif contest_type == 'cash':
            if player.value >= 3.5:
                base_value += 5.0
            return base_value - (player.variance * 0.2)

        elif contest_type == 'contrarian':
            if player.ownership <= 20:
                base_value += 5.0
            elif player.ownership >= 35:
                base_value -= 8.0
            return base_value + (player.variance * 2.0)

        else:
            return base_value + (player.variance * 1.0)

    def _extract_result(self, prob, players: List[Player], player_vars: Dict, contest_type: str) -> LineupResult:
        """Extract lineup results with FanDuel ordering"""
        selected_players = []
        total_salary = 0
        total_ownership = 0

        for i, player in enumerate(players):
            if player_vars[i].varValue == 1:
                selected_players.append(player)
                total_salary += player.salary
                total_ownership += player.ownership

        ordered_players = self._format_lineup_for_fanduel(selected_players)

        if contest_type == 'single_game' and len(ordered_players) == 6:
            mvp = max(ordered_players, key=lambda p: p.projection)
            projected_points = mvp.projection * 1.5 + sum(p.projection for p in ordered_players if p != mvp)
        else:
            projected_points = sum(p.projection for p in ordered_players)

        return LineupResult(
            players=ordered_players,
            total_salary=total_salary,
            projected_points=projected_points,
            total_value=sum(p.value for p in ordered_players),
            ownership_total=total_ownership,
            correlation_score=self._calculate_correlation(ordered_players),
            weather_impact=np.mean([p.weather_factor for p in ordered_players]),
            contest_type=contest_type
        )

    def _format_lineup_for_fanduel(self, players: List[Player]) -> List[Player]:
        """Order players in FanDuel format"""
        ordered = []

        by_position = {}
        for player in players:
            if player.position not in by_position:
                by_position[player.position] = []
            by_position[player.position].append(player)

        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.salary, reverse=True)

        if 'QB' in by_position:
            ordered.append(by_position['QB'][0])

        if 'RB' in by_position:
            ordered.extend(by_position['RB'][:2])

        if 'WR' in by_position:
            ordered.extend(by_position['WR'][:3])

        if 'TE' in by_position:
            ordered.append(by_position['TE'][0])

        flex_candidates = []
        if 'RB' in by_position and len(by_position['RB']) > 2:
            flex_candidates.extend(by_position['RB'][2:])
        if 'WR' in by_position and len(by_position['WR']) > 3:
            flex_candidates.extend(by_position['WR'][3:])
        if 'TE' in by_position and len(by_position['TE']) > 1:
            flex_candidates.extend(by_position['TE'][1:])

        if flex_candidates:
            flex_player = max(flex_candidates, key=lambda p: p.salary)
            ordered.append(flex_player)

        if 'D' in by_position:
            ordered.append(by_position['D'][0])

        return ordered

    def _calculate_correlation(self, players: List[Player]) -> float:
        """Calculate lineup correlation"""
        correlation = 0.0

        qb_teams = [p.team for p in players if p.position == 'QB']
        wr_teams = [p.team for p in players if p.position == 'WR']

        for qb_team in qb_teams:
            same_team_wrs = sum(1 for team in wr_teams if team == qb_team)
            if same_team_wrs > 0:
                correlation += 0.3 * same_team_wrs

        team_counts = {}
        for player in players:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1

        for count in team_counts.values():
            if count >= 3:
                correlation += 0.4
            elif count >= 2:
                correlation += 0.2

        return min(1.0, correlation)

    async def generate_multiple_lineups(self, players: List[Player], num_lineups: int = 10,
                                      contest_type: str = 'gpp', single_game_teams: List[str] = None) -> List[LineupResult]:
        """Generate diverse lineups with Monte Carlo enhancement"""
        lineups = []
        used_combinations = set()
        max_attempts = num_lineups * 3

        for attempt in range(max_attempts):
            if len(lineups) >= num_lineups:
                break

            # Moderate randomization
            randomized_players = []
            for player in players:
                new_player = Player(
                    id=player.id, name=player.name, position=player.position,
                    team=player.team, salary=player.salary, projection=player.projection,
                    ownership=player.ownership, weather_factor=player.weather_factor,
                    injury_risk=player.injury_risk, value=player.value, variance=player.variance
                )

                # Copy Monte Carlo data
                if player.monte_carlo_analyzed:
                    new_player.floor_10 = player.floor_10
                    new_player.ceiling_90 = player.ceiling_90
                    new_player.ceiling_95 = player.ceiling_95
                    new_player.boom_rate = player.boom_rate
                    new_player.bust_rate = player.bust_rate
                    new_player.monte_carlo_analyzed = True

                # Apply randomization
                if contest_type == 'gpp':
                    random_factor = random.uniform(0.85, 1.15)
                elif contest_type == 'cash':
                    random_factor = random.uniform(0.95, 1.05)
                else:
                    random_factor = random.uniform(0.75, 1.25)

                new_player.projection *= random_factor
                new_player.value = new_player.projection / (new_player.salary / 1000)
                randomized_players.append(new_player)

            lineup = self.optimize_lineup(randomized_players, contest_type, single_game_teams)
            if lineup:
                # Diversity check
                if len(lineups) > 0:
                    player_usage = {}
                    for existing_lineup in lineups:
                        for player in existing_lineup.players:
                            player_usage[player.id] = player_usage.get(player.id, 0) + 1

                    overused_players = 0
                    max_usage = max(2, num_lineups // 4)
                    for player in lineup.players:
                        usage_count = player_usage.get(player.id, 0)
                        if usage_count >= max_usage:
                            overused_players += 1

                    if overused_players > 3:
                        continue

                core_players = tuple(sorted([p.id for p in lineup.players if p.salary > 6500]))
                if core_players not in used_combinations:
                    lineups.append(lineup)
                    used_combinations.add(core_players)

        # Sort by appropriate metric
        if contest_type == 'cash':
            if lineups and lineups[0].floor_25 > 0:  # Monte Carlo enhanced
                lineups.sort(key=lambda x: x.floor_25, reverse=True)
            else:
                lineups.sort(key=lambda x: x.projected_points - (x.variance_score * 0.5), reverse=True)
        else:
            if lineups and lineups[0].ceiling_90 > 0:  # Monte Carlo enhanced
                lineups.sort(key=lambda x: x.ceiling_90 - (x.ownership_total * 0.3), reverse=True)
            else:
                lineups.sort(key=lambda x: x.projected_points + (x.variance_score * 0.8), reverse=True)

        logger.info(f"Generated {len(lineups)} {contest_type} lineups with Monte Carlo enhancement")
        return lineups


# Enhanced main optimization function with Monte Carlo
async def optimize_dfs_lineups(player_data: List[Dict], weather_data: Dict = None, vegas_multipliers: Dict = None,
                              num_lineups: int = 10, contest_type: str = 'gpp',
                              single_game_teams: List[str] = None, use_monte_carlo: bool = True) -> List[LineupResult]:
    """AI-Enhanced optimization with Monte Carlo variance modeling"""

    logger.info(f"Starting Monte Carlo enhanced {contest_type} optimization...")

    # Step 1: Get AI strategic analysis (if available)
    if AI_AVAILABLE:
        try:
            analyzer = DualAIDFSAnalyzer()
            ai_analysis = analyzer.analyze_slate_for_optimization(
                player_data, weather_data or {}, {}, contest_type
            )

            if ai_analysis.get('ai_enabled', False):
                logger.info(f"AI Strategy: {ai_analysis.get('ai_strategy', 'No strategy')[:100]}...")

                # Apply AI ownership adjustments
                ownership_adjustments = ai_analysis.get('ownership_adjustments', {})
                if ownership_adjustments:
                    adjusted_count = 0
                    for player in player_data:
                        player_name = player.get('name', '')
                        if player_name in ownership_adjustments:
                            adjustment_factor = ownership_adjustments[player_name]
                            if 'ownership' not in player:
                                player['ownership'] = 15.0
                            player['ownership'] *= adjustment_factor
                            adjusted_count += 1

                    logger.info(f"AI adjusted ownership for {adjusted_count} players")

            cost_summary = analyzer.get_cost_summary()
            logger.info(f"AI Cost: ${cost_summary['weekly_spend']:.3f} of ${cost_summary['weekly_budget']:.2f} budget")

        except Exception as e:
            logger.warning(f"AI analysis failed, continuing without: {e}")
    else:
        logger.info("AI analysis not available - using Monte Carlo optimization")

    # Step 2: Run Monte Carlo enhanced optimization
    mc_simulations = 5000 if use_monte_carlo else 0
    optimizer = EnhancedDFSOptimizer(use_monte_carlo=use_monte_carlo, mc_simulations=mc_simulations)
    optimizer.vegas_multipliers = vegas_multipliers or {}

    players = await optimizer.prepare_players(player_data, weather_data, vegas_multipliers)

    if not players:
        logger.error("No valid players for optimization")
        return []

    if use_monte_carlo:
        monte_carlo_count = sum(1 for p in players if p.monte_carlo_analyzed)
        logger.info(f"Monte Carlo: {monte_carlo_count}/{len(players)} players analyzed with variance modeling")

    logger.info(f"Optimization: {num_lineups} {contest_type} lineups with {len(players)} active players")

    # Generate lineups
    lineups = await optimizer.generate_multiple_lineups(players, num_lineups, contest_type, single_game_teams)

    # Log Monte Carlo insights for generated lineups
    if use_monte_carlo and lineups:
        for i, lineup in enumerate(lineups[:3]):  # Show top 3
            if lineup.monte_carlo_insights:
                logger.info(f"Lineup {i+1} Monte Carlo: {lineup.risk_level} risk, "
                           f"Ceiling: {lineup.ceiling_90:.1f}, Floor: {lineup.floor_10:.1f}")

    return lineups