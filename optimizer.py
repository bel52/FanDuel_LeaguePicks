# optimizer.py
"""
WINNING DFS Optimizer for Friends League Domination
Fixes the 3 core problems:
1. Vegas data now DRIVES selection (not just mild multipliers)
2. AI identifies MUST PLAY/FADE players with projection boosts/penalties
3. Forces stacking from highest-total games
"""
import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pulp
from loguru import logger

from config import (
    DATA_DIR,
    FANDUEL_POSITIONS,
    FANDUEL_SALARY_CAP,
    OPTIMIZATION_CONFIG,
)

# Import the Monte Carlo engine
try:
    from monte_carlo_engine import (
        MonteCarloEngine,
        PlayerSimulation,
        convert_player_data_to_simulation,
        enhance_lineup_with_monte_carlo,
    )

    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    logger.warning("Monte Carlo engine not available")

# AI Integration import
try:
    from ai_analyzer import DualAIDFSAnalyzer

    AI_AVAILABLE = True
    logger.info("AI analyzer imported successfully")
except ImportError as e:
    logger.warning(f"AI analyzer not available: {e}")
    AI_AVAILABLE = False


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Player:
    """Enhanced Player data structure with Monte Carlo variance and game environment"""
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
    # Monte Carlo fields
    floor_10: float = 0.0
    ceiling_90: float = 0.0
    ceiling_95: float = 0.0
    boom_rate: float = 0.0
    bust_rate: float = 0.0
    monte_carlo_analyzed: bool = False
    # NEW: Game environment and AI fields
    game_total: float = 45.0
    game_environment_mult: float = 1.0
    ai_must_play: bool = False
    ai_must_fade: bool = False
    locked: bool = False

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
    # Monte Carlo fields
    ceiling_90: float = 0.0
    ceiling_95: float = 0.0
    floor_10: float = 0.0
    floor_25: float = 0.0
    variance_score: float = 0.0
    sharpe_ratio: float = 0.0
    risk_level: str = "Unknown"
    boom_probability: float = 0.0
    bust_probability: float = 0.0
    monte_carlo_insights: Dict = field(default_factory=dict)
    # NEW: Game environment info
    high_total_exposure: int = 0
    primary_stack_team: str = ""


# -----------------------------
# Optimizer
# -----------------------------
class EnhancedDFSOptimizer:
    """
    WINNING DFS Optimizer - Built for 12-person friends league domination

    Key changes from standard optimizer:
    1. Game environment (Vegas totals) drives 50%+ of player value
    2. Forces 3+ players from highest-total games
    3. AI must-play players get 25% boost, must-fade get 35% penalty
    4. Ownership is IGNORED (doesn't matter in 12-person league)
    """

    def __init__(self, use_monte_carlo: bool = True, mc_simulations: int = 5000):
        self.use_monte_carlo = use_monte_carlo and MONTE_CARLO_AVAILABLE
        self.mc_simulations = mc_simulations
        self.monte_carlo_engine = MonteCarloEngine(num_simulations=mc_simulations) if self.use_monte_carlo else None

        # NEW: Game environment tracking
        self.vegas_multipliers: Dict[str, float] = {}
        self.high_total_teams: List[str] = []
        self.game_totals: Dict[str, float] = {}  # team -> game total
        self.high_total_threshold = 47.0

        if use_monte_carlo and not MONTE_CARLO_AVAILABLE:
            logger.warning("Monte Carlo requested but not available - falling back to basic optimization")

    def set_vegas_data(self, vegas_multipliers: Dict[str, float], vegas_odds: Dict = None):
        """
        CRITICAL: Set Vegas data that drives player selection
        Call this BEFORE optimization
        """
        self.vegas_multipliers = vegas_multipliers or {}

        # Identify high-total teams
        self.high_total_teams = [
            team for team, mult in self.vegas_multipliers.items()
            if mult >= 1.25  # 47+ total games
        ]

        # Extract actual game totals if available
        if vegas_odds and 'games' in vegas_odds:
            for game_id, game_data in vegas_odds['games'].items():
                total = game_data.get('total_points', 45)
                home = game_data.get('home_team')
                away = game_data.get('away_team')
                if home:
                    self.game_totals[home] = total
                if away:
                    self.game_totals[away] = total

        logger.info(f"🎯 HIGH-TOTAL TEAMS (47+): {self.high_total_teams}")
        logger.info(f"📊 Vegas multipliers set for {len(self.vegas_multipliers)} teams")

    async def prepare_players(
            self,
            player_data: List[Dict],
            weather_data: Dict = None,
            vegas_multipliers: Dict = None,
    ) -> List[Player]:
        """Convert player data with game environment enhancement"""
        players: List[Player] = []

        # Store vegas multipliers
        if vegas_multipliers:
            self.vegas_multipliers = vegas_multipliers
            self.high_total_teams = [
                team for team, mult in vegas_multipliers.items()
                if mult >= 1.25
            ]

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

                # Get game environment data
                game_mult = self.vegas_multipliers.get(team, 1.0)
                game_total = self.game_totals.get(team, 45.0)

                # Create player with game environment
                player = Player(
                    id=str(data.get('player_id', data.get('id', player_name))),
                    name=player_name,
                    position=position,
                    team=team,
                    salary=salary,
                    projection=projection,
                    game_total=game_total,
                    game_environment_mult=game_mult,
                    ai_must_play=data.get('ai_must_play', False),
                    ai_must_fade=data.get('ai_must_fade', False),
                    locked=data.get('locked', False),
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

        # Enhance with Monte Carlo analysis
        if self.use_monte_carlo and players:
            logger.info(f"Running Monte Carlo analysis on {len(players)} players...")
            players = await self._enhance_players_with_monte_carlo(players, weather_data, vegas_multipliers)

        return players

    async def _enhance_players_with_monte_carlo(
            self,
            players: List[Player],
            weather_data: Dict = None,
            vegas_data: Dict = None,
    ) -> List[Player]:
        """Enhance players with Monte Carlo variance analysis"""
        sim_data: List[Dict[str, Any]] = []
        for player in players:
            sim_data.append({
                'name': player.name,
                'position': player.position,
                'team': player.team,
                'salary': player.salary,
                'projected_points': player.projection,
            })

        sim_players = convert_player_data_to_simulation(sim_data, weather_data, vegas_data)

        enhanced_players: List[Player] = []
        batch_size = 20

        for i in range(0, len(players), batch_size):
            batch_players = players[i:i + batch_size]
            batch_sims = sim_players[i:i + batch_size]

            sim_tasks = []
            for sim_player in batch_sims:
                task = self.monte_carlo_engine.simulate_player_performance(sim_player, num_sims=1000)
                sim_tasks.append(task)

            batch_results = await asyncio.gather(*sim_tasks)

            for player, sim_result in zip(batch_players, batch_results):
                player.floor_10 = sim_result.get('floor_10', 0.0)
                player.ceiling_90 = sim_result.get('ceiling_90', player.projection)
                player.ceiling_95 = sim_result.get('ceiling_95', player.ceiling_90)
                player.boom_rate = sim_result.get('boom_rate', 0.0)
                player.bust_rate = sim_result.get('bust_rate', 0.0)
                player.variance = sim_result.get('std', player.variance)
                player.monte_carlo_analyzed = True
                enhanced_players.append(player)

        logger.info(f"Enhanced {len(enhanced_players)} players with Monte Carlo analysis")
        return enhanced_players

    async def optimize_lineup(
            self,
            players: List[Player],
            contest_type: str = 'gpp',
            single_game_teams: List[str] = None,
    ) -> Optional[LineupResult]:
        """
        WINNING OPTIMIZATION - Forces high-total game exposure
        """
        try:
            # Filter for single game
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
                    return None

            # Predict ownership (for tracking only, not penalized)
            for player in players:
                player.ownership = self._predict_friends_league_ownership(player, contest_type)

            # Create optimization problem
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)

            player_vars: Dict[int, pulp.LpVariable] = {}
            for i, _ in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')

            # WINNING OBJECTIVE FUNCTION
            objective_terms = []
            for i, player in enumerate(players):
                # Use game-environment-weighted value
                points_value = self._calculate_winning_value(player, contest_type)
                objective_terms.append(points_value * player_vars[i])

            prob += pulp.lpSum(objective_terms)

            # Add constraints
            self._add_fanduel_constraints(prob, players, player_vars, contest_type, single_game_teams)

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status == pulp.LpStatusOptimal:
                result = self._extract_result(prob, players, player_vars, contest_type)

                # Enhance with Monte Carlo
                if self.use_monte_carlo:
                    result = await self._enhance_lineup_result_with_monte_carlo(result)

                return result
            else:
                logger.warning(f"Optimization failed: {pulp.LpStatus[prob.status]}")
                return None

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return None

    def _calculate_winning_value(self, player: Player, contest_type: str) -> float:
        """
        WINNING VALUE CALCULATION

        Key insight: Game environment is worth MORE than individual projection
        A mediocre player in a 50-point game beats an elite player in a 40-point game
        """
        base_value = player.projection
        game_mult = player.game_environment_mult

        # ===========================================
        # FIX #1: MASSIVE GAME ENVIRONMENT WEIGHTING
        # ===========================================
        if game_mult >= 1.35:  # 50+ total game
            game_boost = base_value * 0.50  # +50% for elite environment
        elif game_mult >= 1.25:  # 47+ total game
            game_boost = base_value * 0.35  # +35% for great environment
        elif game_mult >= 1.10:  # 44+ total game
            game_boost = base_value * 0.15  # +15% for good environment
        elif game_mult <= 0.90:  # Under 41 total
            game_boost = base_value * -0.30  # -30% PENALTY for bad environment
        else:
            game_boost = 0

        # Position-specific environment multipliers
        if player.position == 'QB' and game_mult >= 1.25:
            game_boost *= 1.5  # QBs in shootouts are AUTO-PLAYS
        elif player.position == 'WR' and game_mult >= 1.25:
            game_boost *= 1.3  # WRs in shootouts get extra boost
        elif player.position == 'TE' and game_mult >= 1.25:
            game_boost *= 1.2  # TEs benefit from high-scoring games

        # ===========================================
        # FIX #2: AI MUST-PLAY/MUST-FADE ENFORCEMENT
        # ===========================================
        ai_adjustment = 0
        if player.ai_must_play:
            ai_adjustment = base_value * 0.30  # +30% for AI must-plays
            logger.debug(f"🎯 AI BOOST: {player.name} +{ai_adjustment:.1f}")
        elif player.ai_must_fade:
            ai_adjustment = base_value * -0.40  # -40% for AI must-fades
            logger.debug(f"⛔ AI PENALTY: {player.name} {ai_adjustment:.1f}")

        # Contest-specific adjustments
        if contest_type in ['gpp', 'bestball']:
            # GPP: Maximize ceiling
            if player.monte_carlo_analyzed:
                ceiling_bonus = (player.ceiling_90 - player.projection) * 2.0
                boom_bonus = player.boom_rate * 12.0
            else:
                ceiling_bonus = player.variance * 1.5
                boom_bonus = 0

            # NO OWNERSHIP PENALTY - doesn't matter in 12-person league
            return base_value + game_boost + ai_adjustment + ceiling_bonus + boom_bonus

        elif contest_type == 'cash':
            # Cash: Floor + value
            if player.monte_carlo_analyzed:
                floor_bonus = player.floor_10 * 1.5
            else:
                floor_bonus = base_value * 0.3

            value_bonus = 5.0 if player.value >= 3.5 else 0.0

            return base_value + (game_boost * 0.5) + ai_adjustment + floor_bonus + value_bonus

        elif contest_type == 'contrarian':
            # Contrarian: Max ceiling from unexpected places
            if player.monte_carlo_analyzed:
                ceiling_bonus = (player.ceiling_95 - player.projection) * 3.0
            else:
                ceiling_bonus = player.variance * 2.5

            return base_value + game_boost + ai_adjustment + ceiling_bonus

        else:
            return base_value + game_boost + ai_adjustment

    def _add_fanduel_constraints(
            self,
            prob,
            players: List[Player],
            player_vars: Dict,
            contest_type: str,
            single_game_teams: List[str],
    ):
        """FanDuel constraints with FORCED high-total exposure"""

        # Salary cap
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP

        # Handle locked players
        for i, player in enumerate(players):
            if player.locked:
                prob += player_vars[i] == 1
                logger.info(f"🔒 LOCKED: {player.name} ({player.position})")

        if single_game_teams:
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 6
            return

        # Position indices
        qb_indices = [i for i, p in enumerate(players) if p.position == 'QB']
        rb_indices = [i for i, p in enumerate(players) if p.position == 'RB']
        wr_indices = [i for i, p in enumerate(players) if p.position == 'WR']
        te_indices = [i for i, p in enumerate(players) if p.position == 'TE']
        d_indices = [i for i, p in enumerate(players) if p.position == 'D']
        flex_indices = rb_indices + wr_indices + te_indices

        # Exact FanDuel requirements
        if qb_indices:
            prob += pulp.lpSum([player_vars[i] for i in qb_indices]) == 1
        if d_indices:
            prob += pulp.lpSum([player_vars[i] for i in d_indices]) == 1
        if rb_indices:
            prob += pulp.lpSum([player_vars[i] for i in rb_indices]) >= 2
            prob += pulp.lpSum([player_vars[i] for i in rb_indices]) <= 3
        if wr_indices:
            prob += pulp.lpSum([player_vars[i] for i in wr_indices]) >= 3
            prob += pulp.lpSum([player_vars[i] for i in wr_indices]) <= 4
        if te_indices:
            prob += pulp.lpSum([player_vars[i] for i in te_indices]) >= 1
            prob += pulp.lpSum([player_vars[i] for i in te_indices]) <= 2

        prob += pulp.lpSum([player_vars[i] for i in flex_indices]) == 7
        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 9

        # Team diversity (max 4 per team for stacking)
        team_counts: Dict[str, List[int]] = {}
        for i, player in enumerate(players):
            team_counts.setdefault(player.team, []).append(i)

        for team, player_indices in team_counts.items():
            prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= 4

        # ===========================================
        # FIX #3: FORCE HIGH-TOTAL GAME EXPOSURE
        # ===========================================
        if contest_type in ['gpp', 'bestball']:
            self._add_forced_high_total_stack(prob, players, player_vars)
            self._add_qb_wr_stack_requirement(prob, players, player_vars, qb_indices, wr_indices)

    def _add_forced_high_total_stack(self, prob, players: List[Player], player_vars: Dict):
        """
        CRITICAL: Force at least 3 players from highest-total games

        This is the #1 lever for friends league wins.
        70%+ of tournament winners have heavy exposure to the highest-total game.
        """
        if not self.vegas_multipliers:
            logger.warning("No Vegas multipliers - cannot force high-total exposure")
            return

        # Find teams in 47+ total games
        high_total_teams = [
            team for team, mult in self.vegas_multipliers.items()
            if mult >= 1.25
        ]

        if not high_total_teams:
            # Fallback: use teams with multiplier > 1.10
            high_total_teams = [
                team for team, mult in self.vegas_multipliers.items()
                if mult >= 1.10
            ]

        if not high_total_teams:
            logger.warning("No high-total teams found")
            return

        # Get player indices from high-total games
        high_total_indices = [
            i for i, p in enumerate(players)
            if p.team in high_total_teams
        ]

        if len(high_total_indices) >= 3:
            # FORCE at least 3 players from high-total games
            prob += pulp.lpSum([player_vars[i] for i in high_total_indices]) >= 3

            teams_in_constraint = set(players[i].team for i in high_total_indices)
            logger.info(f"🎯 FORCING 3+ players from high-total games: {teams_in_constraint}")
        else:
            logger.warning(f"Only {len(high_total_indices)} players in high-total games")

    def _add_qb_wr_stack_requirement(
            self,
            prob,
            players: List[Player],
            player_vars: Dict,
            qb_indices: List[int],
            wr_indices: List[int]
    ):
        """Force QB + at least 1 WR from same team (stacking)"""

        # Group by team
        team_qbs: Dict[str, List[int]] = {}
        team_wrs: Dict[str, List[int]] = {}

        for i in qb_indices:
            team_qbs.setdefault(players[i].team, []).append(i)

        for i in wr_indices:
            team_wrs.setdefault(players[i].team, []).append(i)

        # For each team with both QB and WR, if QB is selected, at least 1 WR must be too
        for team in team_qbs:
            if team in team_wrs:
                qb_vars = [player_vars[i] for i in team_qbs[team]]
                wr_vars = [player_vars[i] for i in team_wrs[team]]

                if qb_vars and wr_vars:
                    # If QB selected, at least 1 WR from same team
                    prob += pulp.lpSum(wr_vars) >= pulp.lpSum(qb_vars)

    def _predict_friends_league_ownership(self, player: Player, contest_type: str) -> float:
        """Predict ownership for tracking (not used in optimization)"""
        ownership = 15.0

        if player.salary >= 9500:
            ownership = 40.0
        elif player.salary >= 8500:
            ownership = 30.0
        elif player.salary >= 7500:
            ownership = 22.0
        elif player.salary >= 6000:
            ownership = 15.0
        else:
            ownership = 10.0

        if player.position == 'QB' and player.salary >= 8000:
            ownership += 5
        elif player.position == 'RB':
            ownership += 3

        return max(5.0, min(50.0, ownership))

    def _extract_result(self, prob, players: List[Player], player_vars: Dict, contest_type: str) -> LineupResult:
        """Extract lineup results with game environment tracking"""
        selected_players: List[Player] = []
        total_salary = 0
        total_ownership = 0
        high_total_count = 0

        for i, player in enumerate(players):
            if player_vars[i].varValue == 1:
                selected_players.append(player)
                total_salary += player.salary
                total_ownership += player.ownership
                if player.game_environment_mult >= 1.25:
                    high_total_count += 1

        ordered_players = self._format_lineup_for_fanduel(selected_players)
        projected_points = sum(p.projection for p in ordered_players)

        # Find primary stack team
        team_counts = {}
        for p in ordered_players:
            team_counts[p.team] = team_counts.get(p.team, 0) + 1
        primary_stack = max(team_counts, key=team_counts.get) if team_counts else ""

        result = LineupResult(
            players=ordered_players,
            total_salary=total_salary,
            projected_points=projected_points,
            total_value=sum(p.value for p in ordered_players),
            ownership_total=total_ownership,
            correlation_score=self._calculate_correlation(ordered_players),
            weather_impact=float(np.mean([p.weather_factor for p in ordered_players])) if ordered_players else 1.0,
            contest_type=contest_type,
            high_total_exposure=high_total_count,
            primary_stack_team=primary_stack,
        )

        logger.info(
            f"📊 Lineup: ${total_salary} | {projected_points:.1f}pts | {high_total_count} high-total players | Stack: {primary_stack}")

        return result

    async def _enhance_lineup_result_with_monte_carlo(self, lineup_result: LineupResult) -> LineupResult:
        """Enhance lineup result with Monte Carlo analysis"""
        try:
            lineup_data: List[Dict[str, Any]] = []
            for player in lineup_result.players:
                lineup_data.append({
                    'name': player.name,
                    'position': player.position,
                    'team': player.team,
                    'salary': player.salary,
                    'projected_points': player.projection,
                })

            mc_results = await enhance_lineup_with_monte_carlo(
                lineup_data,
                num_simulations=self.mc_simulations,
            )

            lineup_sim = mc_results['simulation_results']['lineup_simulation']
            insights = mc_results['insights']

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
                'correlation_strength': insights['correlation_strength'],
            }

            mean_score = lineup_sim['mean']
            lineup_result.boom_probability = 0.15 if lineup_sim.get('ceiling_90',
                                                                    mean_score) > mean_score * 1.3 else 0.05
            lineup_result.bust_probability = 0.30 if lineup_sim.get('floor_25',
                                                                    mean_score) < mean_score * 0.75 else 0.15

        except Exception as e:
            logger.error(f"Error enhancing lineup with Monte Carlo: {e}")

        return lineup_result

    def _format_lineup_for_fanduel(self, players: List[Player]) -> List[Player]:
        """Order players in FanDuel format: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF"""
        ordered: List[Player] = []

        by_position: Dict[str, List[Player]] = {}
        for player in players:
            by_position.setdefault(player.position, []).append(player)

        for pos in by_position:
            by_position[pos].sort(key=lambda p: p.salary, reverse=True)

        # QB
        if 'QB' in by_position:
            ordered.append(by_position['QB'][0])

        # RB x2
        if 'RB' in by_position:
            ordered.extend(by_position['RB'][:2])

        # WR x3
        if 'WR' in by_position:
            ordered.extend(by_position['WR'][:3])

        # TE
        if 'TE' in by_position:
            ordered.append(by_position['TE'][0])

        # FLEX
        flex_candidates: List[Player] = []
        if 'RB' in by_position and len(by_position['RB']) > 2:
            flex_candidates.extend(by_position['RB'][2:])
        if 'WR' in by_position and len(by_position['WR']) > 3:
            flex_candidates.extend(by_position['WR'][3:])
        if 'TE' in by_position and len(by_position['TE']) > 1:
            flex_candidates.extend(by_position['TE'][1:])

        if flex_candidates:
            flex_player = max(flex_candidates, key=lambda p: p.salary)
            ordered.append(flex_player)

        # DEF
        if 'D' in by_position:
            ordered.append(by_position['D'][0])

        return ordered

    def _calculate_correlation(self, players: List[Player]) -> float:
        """Calculate lineup correlation score"""
        correlation = 0.0

        qb_teams = [p.team for p in players if p.position == 'QB']
        wr_teams = [p.team for p in players if p.position == 'WR']
        te_teams = [p.team for p in players if p.position == 'TE']

        # QB-WR correlation
        for qb_team in qb_teams:
            same_team_wrs = sum(1 for team in wr_teams if team == qb_team)
            correlation += 0.3 * same_team_wrs

            # QB-TE correlation
            same_team_tes = sum(1 for team in te_teams if team == qb_team)
            correlation += 0.2 * same_team_tes

        # Team stacking bonus
        team_counts: Dict[str, int] = {}
        for player in players:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1

        for count in team_counts.values():
            if count >= 3:
                correlation += 0.4
            elif count >= 2:
                correlation += 0.2

        return min(1.0, correlation)

    async def generate_multiple_lineups(
            self,
            players: List[Player],
            num_lineups: int = 10,
            contest_type: str = 'gpp',
            single_game_teams: List[str] = None,
    ) -> List[LineupResult]:
        """Generate diverse lineups with forced high-total exposure"""
        lineups: List[LineupResult] = []
        used_combinations = set()
        max_attempts = num_lineups * 4

        for attempt in range(max_attempts):
            if len(lineups) >= num_lineups:
                break

            # Randomization for diversity
            randomized_players: List[Player] = []
            for player in players:
                new_player = Player(
                    id=player.id,
                    name=player.name,
                    position=player.position,
                    team=player.team,
                    salary=player.salary,
                    projection=player.projection,
                    ownership=player.ownership,
                    weather_factor=player.weather_factor,
                    injury_risk=player.injury_risk,
                    value=player.value,
                    variance=player.variance,
                    game_total=player.game_total,
                    game_environment_mult=player.game_environment_mult,
                    ai_must_play=player.ai_must_play,
                    ai_must_fade=player.ai_must_fade,
                    locked=player.locked,
                )

                # Copy Monte Carlo data
                if player.monte_carlo_analyzed:
                    new_player.floor_10 = player.floor_10
                    new_player.ceiling_90 = player.ceiling_90
                    new_player.ceiling_95 = player.ceiling_95
                    new_player.boom_rate = player.boom_rate
                    new_player.bust_rate = player.bust_rate
                    new_player.monte_carlo_analyzed = True

                # Apply randomization (less for cash games)
                if contest_type == 'gpp':
                    random_factor = random.uniform(0.88, 1.12)
                elif contest_type == 'cash':
                    random_factor = random.uniform(0.95, 1.05)
                else:
                    random_factor = random.uniform(0.80, 1.20)

                new_player.projection *= random_factor
                new_player.value = new_player.projection / (new_player.salary / 1000) if new_player.salary else 0.0
                randomized_players.append(new_player)

            lineup = await self.optimize_lineup(randomized_players, contest_type, single_game_teams)

            if lineup:
                # Check diversity
                core_players = tuple(sorted([p.id for p in lineup.players if p.salary > 6500]))
                if core_players not in used_combinations:
                    lineups.append(lineup)
                    used_combinations.add(core_players)

        # Sort by ceiling for GPP, floor for cash
        if contest_type == 'cash':
            if lineups and lineups[0].floor_10 > 0:
                lineups.sort(key=lambda x: x.floor_10, reverse=True)
            else:
                lineups.sort(key=lambda x: x.projected_points, reverse=True)
        else:
            if lineups and lineups[0].ceiling_90 > 0:
                lineups.sort(key=lambda x: (x.ceiling_90, x.high_total_exposure), reverse=True)
            else:
                lineups.sort(key=lambda x: (x.projected_points, x.high_total_exposure), reverse=True)

        logger.info(f"Generated {len(lineups)} {contest_type} lineups")

        # Log high-total exposure stats
        if lineups:
            avg_exposure = sum(l.high_total_exposure for l in lineups) / len(lineups)
            logger.info(f"📊 Average high-total exposure: {avg_exposure:.1f} players per lineup")

        return lineups


# -----------------------------
# Sync wrapper utilities
# -----------------------------
def _run_coro_sync(coro):
    """Run coroutine from sync context"""
    try:
        loop = asyncio.get_running_loop()
        is_running = loop.is_running()
    except RuntimeError:
        loop = None
        is_running = False

    if not is_running:
        return asyncio.run(coro)

    import threading
    result_box: Dict[str, Any] = {}
    exc_box: Dict[str, BaseException] = {}

    def runner():
        try:
            result_box["result"] = asyncio.run(coro)
        except BaseException as ex:
            exc_box["ex"] = ex

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()

    if exc_box:
        raise exc_box["ex"]

    return result_box.get("result")


# -----------------------------
# Main Public API
# -----------------------------
def optimize_dfs_lineups(
        player_data: List[Dict],
        weather_data: Dict = None,
        vegas_multipliers: Dict = None,
        num_lineups: int = 10,
        contest_type: str = 'gpp',
        single_game_teams: List[str] = None,
        use_monte_carlo: bool = True,
        mc_simulations: int = 5000,
        vegas_odds: Dict = None,
) -> List[LineupResult]:
    """
    WINNING DFS Optimization - Main entry point

    This function:
    1. Sets Vegas data to drive game environment weighting
    2. Applies AI must-play/must-fade recommendations
    3. Forces high-total game exposure
    4. Returns lineups optimized for CEILING, not median
    """
    logger.info(f"🏈 Starting {contest_type.upper()} optimization | Lineups: {num_lineups}")
    logger.info(f"   Monte Carlo: {'ON' if use_monte_carlo and MONTE_CARLO_AVAILABLE else 'OFF'}")

    # Build optimizer
    optimizer = EnhancedDFSOptimizer(use_monte_carlo=use_monte_carlo, mc_simulations=mc_simulations)

    # CRITICAL: Set Vegas data first
    optimizer.set_vegas_data(vegas_multipliers or {}, vegas_odds)

    # Prepare players
    players: List[Player] = _run_coro_sync(
        optimizer.prepare_players(player_data, weather_data or {}, vegas_multipliers or {})
    )

    if not players:
        logger.error("No valid players after preparation")
        return []

    # Log high-total team exposure available
    high_total_players = sum(1 for p in players if p.game_environment_mult >= 1.25)
    logger.info(f"📊 Players in high-total games: {high_total_players}/{len(players)}")

    # Generate lineups
    lineups: List[LineupResult] = _run_coro_sync(
        optimizer.generate_multiple_lineups(
            players=players,
            num_lineups=num_lineups,
            contest_type=contest_type,
            single_game_teams=single_game_teams,
        )
    )

    if not lineups:
        logger.error("No lineups generated")
        return []

    # Export lineups
    try:
        export_dir = Path(DATA_DIR) / "lineups"
        export_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / f"lineups_{contest_type}_{ts}.json"

        payload = []
        for lu in lineups:
            payload.append({
                "contest_type": lu.contest_type,
                "total_salary": lu.total_salary,
                "projected_points": round(lu.projected_points, 2),
                "ceiling_90": round(lu.ceiling_90, 2),
                "floor_10": round(lu.floor_10, 2),
                "high_total_exposure": lu.high_total_exposure,
                "primary_stack": lu.primary_stack_team,
                "players": [
                    {
                        "name": p.name,
                        "position": p.position,
                        "team": p.team,
                        "salary": p.salary,
                        "projection": round(p.projection, 2),
                        "game_mult": round(p.game_environment_mult, 2),
                        "ai_must_play": p.ai_must_play,
                    }
                    for p in lu.players
                ],
            })

        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"lineups": payload}, f, indent=2)
        logger.info(f"💾 Saved lineups to {out_path}")
    except Exception as e:
        logger.warning(f"Failed to export lineups: {e}")

    # Log summary
    if lineups:
        top = lineups[0]
        logger.info(f"🏆 TOP LINEUP: ${top.total_salary} | {top.projected_points:.1f}pts | "
                    f"Ceil90: {top.ceiling_90:.1f} | High-total: {top.high_total_exposure} | "
                    f"Stack: {top.primary_stack_team}")

    return lineups