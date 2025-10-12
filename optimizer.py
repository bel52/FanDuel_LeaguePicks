# optimizer.py
"""
Enhanced DFS lineup optimization with Monte Carlo variance analysis
Fixed async issues for tournament wins - FIXED LOCKED PLAYER HANDLING
PHASE 1: Added Smart Vegas Exposure constraints
"""
import asyncio
import json
import os
import random
from dataclasses import dataclass
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
    H2H_SALARY_CAP,      # ADD THIS
    H2H_ROSTER_SIZE,     # ADD THIS
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

# AI Integration import with better error handling
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
    locked: bool = False
    is_core: bool = False
    is_mvp: bool = False  # ADD THIS LINE
    mvp_candidate: bool = False  # ADD THIS LINE
    mvp_rank: int = 0  # ADD THIS LINE
    # Monte Carlo fields...
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


# -----------------------------
# Optimizer
# -----------------------------
def calculate_player_confidence(player: Player, vegas_data: Dict) -> float:
    """
    Score a player's "must play" confidence (0-100)

    High scores = exempt from exposure limits
    """
    confidence = 0.0

    # 1. VALUE (most important)
    if player.value >= 4.0:
        confidence += 30
    elif player.value >= 3.5:
        confidence += 20
    elif player.value >= 3.0:
        confidence += 10

    # 2. VEGAS ENVIRONMENT
    vegas_mult = vegas_data.get('vegas_multipliers', {}).get(player.team, 1.0)
    if vegas_mult >= 1.40:
        confidence += 25
    elif vegas_mult >= 1.25:
        confidence += 15

    # 3. MONTE CARLO CEILING
    if player.monte_carlo_analyzed:
        ceiling_ratio = player.ceiling_90 / player.projection if player.projection > 0 else 1
        if ceiling_ratio >= 1.35:
            confidence += 20
        elif ceiling_ratio >= 1.25:
            confidence += 10

    # 4. OWNERSHIP LEVERAGE
    if player.ownership <= 15 and player.projection >= 15:
        confidence += 15

    # 5. POSITION SCARCITY
    if player.position in ['RB', 'TE']:
        confidence += 10

    return min(100, confidence)


def identify_core_plays(players: List[Player], vegas_multipliers: Dict, num_lineups: int) -> List[Player]:
    """
    Mark players as 'core' if they exceed confidence threshold

    Core players get exemption from exposure limits
    """
    CORE_THRESHOLD = 60
    MAX_CORE_PLAYS = 3

    player_confidence = []
    for player in players:
        conf = calculate_player_confidence(player, vegas_multipliers)
        player_confidence.append((player, conf))

    player_confidence.sort(key=lambda x: x[1], reverse=True)

    core_count = 0
    for player, conf in player_confidence:
        if conf >= CORE_THRESHOLD and core_count < MAX_CORE_PLAYS:
            player.is_core = True
            core_count += 1
            logger.info(
                f"🔥 CORE PLAY: {player.name} ({player.position}) ${player.salary:,} - {conf:.0f} confidence (value={player.value:.2f}x)")
        else:
            player.is_core = False

    if core_count == 0:
        logger.info("📊 No core plays identified this week - normal exposure limits apply")
    else:
        logger.info(f"📊 Identified {core_count} core play(s) - will appear in most/all lineups")

    return players


def calculate_max_exposure(num_lineups: int, position: str) -> int:
    """
    Calculate max player appearances based on position and total lineups

    Philosophy: Let good players appear often in small sets.
    Diversity increases as lineup count grows.
    """

    if num_lineups <= 5:
        target_pct = {
            'QB': 0.80,
            'RB': 1.00,
            'WR': 0.60,
            'TE': 0.80,
            'D': 0.60,
        }.get(position, 0.70)

    elif num_lineups <= 15:
        target_pct = {
            'QB': 0.60,
            'RB': 0.65,
            'WR': 0.55,
            'TE': 0.60,
            'D': 0.50,
        }.get(position, 0.60)

    else:
        target_pct = {
            'QB': 0.50,
            'RB': 0.55,
            'WR': 0.50,
            'TE': 0.50,
            'D': 0.45,
        }.get(position, 0.50)

    max_uses = max(1, int(num_lineups * target_pct))

    return max_uses


class EnhancedDFSOptimizer:
    """Enhanced DFS optimization with Monte Carlo variance modeling"""

    def __init__(self, use_monte_carlo: bool = True, mc_simulations: int = 5000):
        self.use_monte_carlo = use_monte_carlo and MONTE_CARLO_AVAILABLE
        self.mc_simulations = mc_simulations
        self.monte_carlo_engine = MonteCarloEngine(num_simulations=mc_simulations) if self.use_monte_carlo else None

        if use_monte_carlo and not MONTE_CARLO_AVAILABLE:
            logger.warning("Monte Carlo requested but not available - falling back to basic optimization")

    async def prepare_players(
            self,
            player_data: List[Dict],
            weather_data: Dict = None,
            vegas_data: Dict = None,
    ) -> List[Player]:
        """Convert player data with optional Monte Carlo enhancement"""
        players: List[Player] = []

        for data in player_data:
            try:
                player_name = data.get('player_name', data.get('name', ''))
                position = data.get('position', '')
                team = data.get('team', '')
                salary = int(data.get('salary', 5000))
                projection = float(data.get('projection', data.get('projected_points', 0)))

                if not player_name or len(player_name.strip()) < 2 or projection < 0:
                    continue

                if position in ['DST', 'DEF', 'D/ST']:
                    position = 'D'

                is_locked = data.get('locked', False)

                player = Player(
                    id=str(data.get('player_id', data.get('id', player_name))),
                    name=player_name,
                    position=position,
                    team=team,
                    salary=salary,
                    projection=projection,
                    locked=is_locked,
                )

                # NEW: Check for injury opportunity boost
                if data.get('injury_opportunity', False):
                    opportunity_score = data.get('opportunity_score', 0)

                    if opportunity_score >= 0.7:
                        boost_factor = 1.0 + (opportunity_score * 0.25)
                        player.projection *= boost_factor

                        injured_starter = data.get('injured_starter', 'Unknown')
                        logger.info(f"🚑 INJURY OPPORTUNITY BOOST: {player.name} ({player.position}) "
                                    f"backing up {injured_starter} - "
                                    f"projection boosted {boost_factor:.2f}x")

                if weather_data and team in weather_data:
                    weather_factor = weather_data[team].get('factor', 1.0)
                    player.weather_factor = weather_factor
                    player.projection *= weather_factor

                players.append(player)

            except Exception as e:
                logger.error(f"Error processing player {data}: {e}")
                continue

        if self.use_monte_carlo and players:
            logger.info(f"Running Monte Carlo analysis on {len(players)} players...")
            players = await self._enhance_players_with_monte_carlo(players, weather_data, vegas_data)

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
        """Optimize with Monte Carlo-enhanced objective function"""
        try:
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
                    return None

            for player in players:
                player.ownership = self._predict_friends_league_ownership(player, contest_type)
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
                    return None

            for player in players:
                player.ownership = self._predict_friends_league_ownership(player, contest_type)

                # ========== ADD H2H MVP SELECTION HERE ==========
                # H2H: Select MVP and apply 1.5x multiplier
            if contest_type == 'h2h':
                # Find best MVP candidate (highest ceiling * value)
                mvp_candidates = []
                for player in players:
                    if player.monte_carlo_analyzed:
                        mvp_score = player.ceiling_90 * player.value * 0.5
                    else:
                        mvp_score = player.projection * player.value * 0.5
                    mvp_candidates.append((player, mvp_score))

                mvp_candidates.sort(key=lambda x: x[1], reverse=True)

                # Mark top 3 candidates for MVP consideration
                for i, (player, score) in enumerate(mvp_candidates[:3]):
                    player.mvp_candidate = True
                    player.mvp_rank = i + 1

                logger.info(f"🏆 H2H MVP Candidates:")
                for i, (player, score) in enumerate(mvp_candidates[:3]):
                    logger.info(f"   {i + 1}. {player.name} ({player.position}) ${player.salary} - Score: {score:.1f}")
                # ========== END H2H MVP SELECTION ==========

            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)

            player_vars: Dict[int, pulp.LpVariable] = {}
            for i, _ in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')

            objective_terms = []
            for i, player in enumerate(players):
                if self.use_monte_carlo and player.monte_carlo_analyzed:
                    points_value = self._calculate_monte_carlo_value(player, contest_type)
                else:
                    points_value = self._calculate_contest_value(player, contest_type)
                objective_terms.append(points_value * player_vars[i])

            prob += pulp.lpSum(objective_terms)

            self._add_fanduel_constraints(prob, players, player_vars, contest_type, single_game_teams)

            if not single_game_teams:
                self._add_friends_league_constraints(prob, players, player_vars, contest_type)

            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            logger.info(f"🔍 SOLVER STATUS: {pulp.LpStatus[prob.status]}")
            if hasattr(self, '_top_game_indices'):
                selected = sum(1 for i in self._top_game_indices if player_vars[i].varValue == 1)
                logger.info(f"🔍 VEGAS CHECK: Selected {selected} from top game (constraint: 3-4)")
            if selected > 0:
                selected_names = [players[i].name for i in self._top_game_indices if player_vars[i].varValue == 1]
                selected_teams = [players[i].team for i in self._top_game_indices if player_vars[i].varValue == 1]
                logger.info(f"🔍 VEGAS PLAYERS: {list(zip(selected_names, selected_teams))}")
            if prob.status != pulp.LpStatusOptimal:
                logger.error(f"❌ SOLVER FAILED: {pulp.LpStatus[prob.status]}")
                # Log constraint violations
                logger.error("Checking Vegas constraint...")
                if hasattr(self, '_top_game_indices'):
                    selected_from_game = sum(player_vars[i].varValue or 0 for i in self._top_game_indices)
                    logger.error(f"   Selected {selected_from_game} players from top game (need 3-4)")
                return None

            if prob.status == pulp.LpStatusOptimal:
                result = self._extract_result(prob, players, player_vars, contest_type)

                if self.use_monte_carlo:
                    result = await self._enhance_lineup_result_with_monte_carlo(result)

                return result
            else:
                logger.warning(f"Optimization failed: {pulp.LpStatus[prob.status]}")
                return None

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")  # CHANGE THIS LINE
            raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

    def _calculate_monte_carlo_value(self, player: Player, contest_type: str) -> float:
        """Enhanced value calculation with proper friends_league strategy"""
        base_value = player.projection

        if contest_type == 'friends_league':
            vegas_multipliers = getattr(self, 'vegas_multipliers', {})
            vegas_boost = vegas_multipliers.get(player.team, 1.0)

            if player.position == 'QB':
                # HARD FLOOR: Friends league needs reliable QB scoring
                if player.salary < 7000:
                    return 0.0  # Eliminate backup QBs entirely

                if vegas_boost >= 1.40:
                    base_value *= 2.80
                elif vegas_boost >= 1.25:
                    base_value *= 2.20
                elif vegas_boost >= 1.15:
                    base_value *= 1.60

                # TRIPLED ceiling bonuses for friends league
                ceiling_bonus = (player.ceiling_90 - player.projection) * 270.0  # Was 90
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 180.0  # Was 60
                boom_bonus = player.boom_rate * 450.0  # Was 150

                # Salary strategy: HEAVILY favor elite QBs
                if player.salary >= 9000:
                    salary_bonus = 80.0  # Was 35
                elif player.salary >= 8500:
                    salary_bonus = 60.0  # New tier
                elif player.salary >= 8000:
                    salary_bonus = 35.0
                elif player.salary >= 7500:
                    salary_bonus = 10.0  # Was 20
                elif player.salary >= 7000:
                    salary_bonus = -20.0  # Was -5
                else:
                    salary_bonus = -100.0  # Was -40


            elif player.position == 'RB':

                if vegas_boost >= 1.40:

                    base_value *= 1.80

                elif vegas_boost >= 1.25:

                    base_value *= 1.50

                elif vegas_boost >= 1.15:

                    base_value *= 1.25

                # DOUBLED ceiling bonuses
                ceiling_bonus = (player.ceiling_90 - player.projection) * 90.0  # Was 45
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 60.0  # Was 30
                boom_bonus = player.boom_rate * 300.0  # Was 150

                # Penalize cheap RBs harder
                if player.salary >= 9000:
                    salary_bonus = 40.0  # Was 25
                elif player.salary >= 7500:
                    salary_bonus = 20.0  # Was 12
                elif player.salary >= 6500:
                    salary_bonus = 5.0  # New
                elif player.salary <= 5000:
                    salary_bonus = -30.0  # Was -8 (HARSH penalty)
                else:
                    salary_bonus = -15.0


            elif player.position == 'WR':

                if vegas_boost >= 1.40:

                    base_value *= 1.65

                elif vegas_boost >= 1.25:

                    base_value *= 1.40

                elif vegas_boost >= 1.15:

                    base_value *= 1.20

                # DOUBLED ceiling bonuses
                ceiling_bonus = (player.ceiling_90 - player.projection) * 72.0  # Was 36
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 48.0  # Was 24
                boom_bonus = player.boom_rate * 288.0  # Was 144

                # Penalize cheap WRs harder
                if player.salary >= 8500:
                    salary_bonus = 30.0  # Was 18
                elif player.salary >= 7000:
                    salary_bonus = 15.0  # Was 10
                elif player.salary >= 6000:
                    salary_bonus = 5.0  # Was 12
                elif player.salary <= 5000:
                    salary_bonus = -40.0  # Was 0 (HARSH penalty)
                elif 5000 < player.salary <= 6000:
                    salary_bonus = -20.0  # Was -6
                else:
                    salary_bonus = -10.0


            elif player.position == 'TE':

                if vegas_boost >= 1.40:

                    base_value *= 1.45

                elif vegas_boost >= 1.25:

                    base_value *= 1.25

                # DOUBLED ceiling bonuses
                ceiling_bonus = (player.ceiling_90 - player.projection) * 60.0  # Was 30
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 36.0  # Was 18
                boom_bonus = player.boom_rate * 240.0  # Was 120

                # Penalize cheap TEs HARSHLY (position killers)
                if player.salary >= 6500:
                    salary_bonus = 35.0  # Was 20
                elif player.salary >= 5500:
                    salary_bonus = 15.0  # New
                elif player.salary <= 4500:
                    salary_bonus = -60.0  # Was 10 (MASSIVE penalty)
                else:
                    salary_bonus = -30.0  # Was -15

            else:
                # Defense - keep same, add small cheap penalty
                ceiling_bonus = (player.ceiling_90 - player.projection) * 6.0
                ceiling_95_bonus = 0.0
                boom_bonus = player.boom_rate * 25.0

                # Penalize ultra-cheap defenses
                if player.salary <= 3000:
                    salary_bonus = -20.0
                else:
                    salary_bonus = 0.0

            # FRIENDS LEAGUE: Ownership is irrelevant (12 people, 1 lineup each)
            # We want MAXIMUM POINTS, not differentiation
            # FRIENDS LEAGUE: Ownership is irrelevant (12 people, 1 lineup each)
            # We want MAXIMUM POINTS, not differentiation
            ownership_penalty = 0.0

            variance_bonus = player.variance * 2.5
            bust_penalty = player.bust_rate * 8.0

            # FRIENDS LEAGUE SCORING:
            # Pure points optimization - highest score wins the week
            # Ownership is meaningless, so we maximize: projection + ceiling + Vegas boost
            total_value = (base_value + ceiling_bonus + ceiling_95_bonus + boom_bonus +
                           salary_bonus + variance_bonus - bust_penalty)

            # Debug logging for high-salary QBs
            if player.position == 'QB' and player.salary >= 7000:
                logger.info(f"🎯 QB VALUE: {player.name} ${player.salary} = {total_value:.1f} "
                            f"(base={base_value:.1f}, vegas={vegas_boost:.2f}x, "
                            f"ceiling_bonus={ceiling_bonus:.1f}, salary_bonus={salary_bonus:.1f})")

            return total_value
        elif contest_type == 'h2h':
            # H2H uses friends_league style but MORE aggressive for single game
            vegas_multipliers = getattr(self, 'vegas_multipliers', {})
            vegas_boost = vegas_multipliers.get(player.team, 1.0)

            # H2H is all about CEILING - you need the highest score to win
            if player.position == 'QB':
                base_value *= 3.20  # QBs dominate single game
                ceiling_bonus = (player.ceiling_90 - player.projection) * 45.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 25.0
                boom_bonus = player.boom_rate * 80.0

            elif player.position == 'RB':
                base_value *= 2.00
                ceiling_bonus = (player.ceiling_90 - player.projection) * 22.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 15.0
                boom_bonus = player.boom_rate * 65.0

            elif player.position == 'WR':
                base_value *= 1.85
                ceiling_bonus = (player.ceiling_90 - player.projection) * 18.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 12.0
                boom_bonus = player.boom_rate * 60.0

            elif player.position == 'TE':
                base_value *= 1.65
                ceiling_bonus = (player.ceiling_90 - player.projection) * 15.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 10.0
                boom_bonus = player.boom_rate * 50.0

            else:  # DEF
                ceiling_bonus = (player.ceiling_90 - player.projection) * 10.0
                ceiling_95_bonus = 0.0
                boom_bonus = player.boom_rate * 30.0

            # Vegas boost for game environment
            if vegas_boost >= 1.40:
                base_value *= 1.50
            elif vegas_boost >= 1.25:
                base_value *= 1.25

            # Salary efficiency matters in H2H
            if player.value >= 3.5:
                salary_bonus = 25.0
            elif player.value <= 2.0:
                salary_bonus = -20.0
            else:
                salary_bonus = 0.0

            # Low ownership leverage
            if player.ownership <= 20:
                ownership_penalty = 8.0
            else:
                ownership_penalty = 0.0

            variance_bonus = player.variance * 3.5  # High variance good for H2H
            bust_penalty = player.bust_rate * 5.0

            return (base_value + ceiling_bonus + ceiling_95_bonus + boom_bonus +
                    salary_bonus + ownership_penalty + variance_bonus - bust_penalty)
        elif contest_type == 'gpp':
            ceiling_bonus = (player.ceiling_90 - player.projection) * 8.0
            ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 5.0
            boom_bonus = player.boom_rate * 40.0

            if player.ownership >= 40:
                ownership_penalty = -15.0
            elif player.ownership >= 30:
                ownership_penalty = -8.0
            elif 15 <= player.ownership <= 25:
                ownership_penalty = 5.0
            else:
                ownership_penalty = 0.0

            bust_penalty = player.bust_rate * 2.0

            return base_value + ceiling_bonus + ceiling_95_bonus + boom_bonus + ownership_penalty - bust_penalty

        elif contest_type == 'cash':
            floor_bonus = player.floor_10 * 2.0
            consistency_bonus = 5.0 if player.bust_rate < 0.15 else 0.0
            variance_penalty = player.variance * 0.5
            return base_value + floor_bonus + consistency_bonus - variance_penalty

        elif contest_type == 'contrarian':
            ceiling_bonus = (player.ceiling_95 - player.projection) * 10.0

            if player.ownership <= 10:
                ownership_bonus = 20.0
            elif player.ownership <= 15:
                ownership_bonus = 10.0
            elif player.ownership >= 35:
                ownership_bonus = -20.0
            else:
                ownership_bonus = 0.0

            boom_bonus = player.boom_rate * 20.0
            return base_value + ceiling_bonus + ownership_bonus + boom_bonus

        else:
            return base_value + (player.variance * 1.0)

    async def _enhance_lineup_result_with_monte_carlo(self, lineup_result: LineupResult) -> LineupResult:
        """Enhanced MC with ULTRA AGGRESSIVE GPP metrics"""
        try:
            lineup_data = []
            for player in lineup_result.players:
                lineup_data.append({
                    'name': player.name,
                    'position': player.position,
                    'team': player.team,
                    'salary': player.salary,
                    'projected_points': player.projection,
                })

            mc_results = await enhance_lineup_with_monte_carlo(lineup_data, num_simulations=self.mc_simulations)
            lineup_sim = mc_results['simulation_results']['lineup_simulation']
            insights = mc_results['insights']

            lineup_result.ceiling_90 = lineup_sim['ceiling_90']
            lineup_result.ceiling_95 = lineup_sim['ceiling_95']
            lineup_result.floor_10 = lineup_sim['floor_10']
            lineup_result.floor_25 = lineup_sim['floor_25']
            lineup_result.variance_score = lineup_sim['std']
            lineup_result.sharpe_ratio = lineup_sim['sharpe_ratio']

            mean_score = lineup_sim['mean']
            ceiling_distance = (lineup_result.ceiling_90 - mean_score) / mean_score if mean_score > 0 else 0

            if lineup_result.contest_type == 'gpp':
                if ceiling_distance > 0.30:
                    lineup_result.risk_level = "Tournament Winner"
                elif ceiling_distance > 0.25:
                    lineup_result.risk_level = "High Ceiling"
                elif ceiling_distance > 0.20:
                    lineup_result.risk_level = "Medium Ceiling"
                else:
                    lineup_result.risk_level = "Too Safe"
            else:
                lineup_result.risk_level = insights['risk_assessment']

            lineup_result.monte_carlo_insights = insights
            lineup_result.boom_probability = self._calculate_boom_probability(lineup_sim, mean_score)
            lineup_result.bust_probability = self._calculate_bust_probability(lineup_sim, mean_score)

            logger.info(
                f"Enhanced lineup: {lineup_result.risk_level}, {lineup_result.boom_probability:.1%} boom, Ceil90={lineup_result.ceiling_90:.1f}")

        except Exception as e:
            logger.error(f"MC enhancement error: {e}")

        return lineup_result

    def _calculate_boom_probability(self, lineup_sim: Dict, mean_score: float) -> float:
        """ULTRA AGGRESSIVE boom calculation for GPP"""
        ceiling_90 = lineup_sim.get('ceiling_90', mean_score)
        ceiling_95 = lineup_sim.get('ceiling_95', ceiling_90)

        ceiling_distance_90 = (ceiling_90 - mean_score) / mean_score if mean_score > 0 else 0
        ceiling_distance_95 = (ceiling_95 - mean_score) / mean_score if mean_score > 0 else 0

        if ceiling_distance_95 > 0.35:
            return 0.25
        elif ceiling_distance_90 > 0.30:
            return 0.20
        elif ceiling_distance_90 > 0.25:
            return 0.15
        elif ceiling_distance_90 > 0.20:
            return 0.12
        else:
            return 0.08

    def _calculate_bust_probability(self, lineup_sim: Dict, mean_score: float) -> float:
        """Calculate probability of bust performance (bottom 25%)"""
        bust_threshold = mean_score * 0.75
        floor_25 = lineup_sim.get('floor_25', mean_score)

        if floor_25 < bust_threshold:
            return 0.3
        else:
            return 0.15

    def _add_fanduel_constraints(
            self,
            prob,
            players: List[Player],
            player_vars: Dict,
            contest_type: str,
            single_game_teams: List[str],
    ):
        """EXACT FanDuel constraints - handles both main slate and H2H single game"""

        # Handle H2H Single Game Format
        if contest_type == 'h2h':
            logger.info(f"🎯 Applying H2H single-game constraints for teams: {single_game_teams}")

            # Filter to only players from the selected game
            if single_game_teams:
                game_player_indices = [
                    i for i, p in enumerate(players)
                    if p.team in single_game_teams
                ]

                if len(game_player_indices) < 6:
                    raise ValueError(
                        f"Not enough players from {single_game_teams} ({len(game_player_indices)} available, need 6)")

                logger.info(f"✅ {len(game_player_indices)} players available from {single_game_teams}")

            # MVP constraints
            mvp_var = pulp.LpVariable("mvp_selected", cat='Binary')

            # Salary constraint with MVP multiplier
            salary_expr = []
            for i in range(len(players)):
                # MVP costs 1.5x salary, regular FLEX costs normal salary
                # We'll handle MVP selection through a separate variable
                base_salary = players[i].salary * player_vars[i]
                salary_expr.append(base_salary)

            # Add extra 0.5x salary cost for whichever player is MVP
            for i in range(len(players)):
                mvp_bonus_cost = players[i].salary * 0.5 * pulp.LpVariable(f"is_mvp_{i}", cat='Binary')
                salary_expr.append(mvp_bonus_cost)

                # Link mvp_bonus to actual MVP selection (we'll set this in objective)
                # For now, just ensure salary cap

            prob += pulp.lpSum(salary_expr) <= H2H_SALARY_CAP

            # Roster size: exactly 6 players (1 MVP + 5 FLEX)
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == H2H_ROSTER_SIZE

            # If locked players exist, enforce them
            locked_count = sum(1 for p in players if p.locked)
            if locked_count > H2H_ROSTER_SIZE:
                raise ValueError(f"Too many locked players ({locked_count}) for H2H format (max {H2H_ROSTER_SIZE})")

            for i, player in enumerate(players):
                if player.locked:
                    prob += player_vars[i] == 1
                    logger.info(f"🔒 H2H LOCKED: {player.name} ({player.position}) ${player.salary}")

            # Team restriction: all players from selected game teams
            if single_game_teams:
                for i, player in enumerate(players):
                    if player.team not in single_game_teams:
                        prob += player_vars[i] == 0

            logger.info(f"✅ H2H constraints applied: 6 players, ${H2H_SALARY_CAP} cap, MVP 1.5x")
            return

        # EXISTING MAIN SLATE CODE CONTINUES BELOW...
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP

        locked_players_indices = []
        locked_salary = 0
        locked_positions = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'D': 0}

        for i, player in enumerate(players):
            if player.locked:
                prob += player_vars[i] == 1
                locked_players_indices.append(i)
                locked_salary += player.salary
                locked_positions[player.position] += 1
                logger.info(f"🔒 LOCKED: {player.name} ({player.position}) ${player.salary}")

        if locked_salary > FANDUEL_SALARY_CAP:
            raise ValueError(f"Locked players exceed salary cap: ${locked_salary:,}")

        if locked_positions['QB'] > 1 or locked_positions['RB'] > 3 or locked_positions['WR'] > 4 or locked_positions[
            'TE'] > 2 or locked_positions['D'] > 1:
            raise ValueError(f"Locked players violate position limits: {locked_positions}")

        logger.info(f"✅ Locked validation passed: ${locked_salary:,} salary, {locked_positions}")

        if single_game_teams:
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 6
            return

        qb_indices = [i for i, p in enumerate(players) if p.position == 'QB']
        rb_indices = [i for i, p in enumerate(players) if p.position == 'RB']
        wr_indices = [i for i, p in enumerate(players) if p.position == 'WR']
        te_indices = [i for i, p in enumerate(players) if p.position == 'TE']
        d_indices = [i for i, p in enumerate(players) if p.position == 'D']

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

        if rb_indices:
            prob += pulp.lpSum([player_vars[i] for i in rb_indices]) <= 3
        if wr_indices:
            prob += pulp.lpSum([player_vars[i] for i in wr_indices]) <= 4
        if te_indices:
            prob += pulp.lpSum([player_vars[i] for i in te_indices]) <= 2

        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 9

        team_counts: Dict[str, List[int]] = {}
        for i, player in enumerate(players):
            team_counts.setdefault(player.team, []).append(i)

        max_per_team = 3 if contest_type == 'cash' else 4
        for team, player_indices in team_counts.items():
            prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= max_per_team

        if contest_type == 'gpp':
            self._add_stacking_incentive(prob, players, player_vars, qb_indices, wr_indices)

    def _add_friends_league_constraints(self, prob, players: List[Player], player_vars: Dict, contest_type: str):
        """Friends league constraints with SMART Vegas exposure"""

        vegas_data = getattr(self, 'vegas_data', {})

        if contest_type == 'friends_league':
            top_game_indices = self._add_smart_vegas_exposure(prob, players, player_vars, vegas_data)
            if top_game_indices:
                self._top_game_indices = top_game_indices  # Store for this optimization only

        if contest_type == 'gpp':
            expensive_players = [i for i, p in enumerate(players) if p.salary >= 9000]
            if expensive_players:
                prob += pulp.lpSum([player_vars[i] for i in expensive_players]) >= 1
                # NEW: Force at least 3 TRUE boom candidates (top 25% ceiling)
                if contest_type == 'friends_league':
                    boom_candidates = []
                    for i, player in enumerate(players):
                        is_boom = False

                        if player.monte_carlo_analyzed:
                            ceiling_ratio = player.ceiling_90 / player.projection if player.projection > 0 else 1
                            # FIXED: Realistic threshold - 15%+ ceiling OR high boom rate
                            is_boom = (
                                    (ceiling_ratio >= 1.15 and player.salary >= 7000) or  # Any 15%+ ceiling + $7K+
                                    (player.boom_rate >= 0.20 and player.salary >= 7500) or  # High boom rate
                                    (player.salary >= 9000 and ceiling_ratio >= 1.10)  # Elite salary + decent ceiling
                            )
                        else:
                            # Fallback: expensive studs only
                            is_boom = (player.salary >= 8500 and player.projection >= 18)

                        if is_boom:
                            boom_candidates.append(i)

                    # Constraint: At least 3 boom candidates
                    if boom_candidates and len(boom_candidates) >= 3:
                        prob += pulp.lpSum([player_vars[i] for i in boom_candidates]) >= 3
                        logger.info(f"✅ Boom constraint: {len(boom_candidates)} candidates, forcing 3+")
                    else:
                        logger.warning(f"⚠️ Only {len(boom_candidates)} boom candidates - constraint may be too strict")
    def _add_stacking_incentive(
            self,
            prob,
            players: List[Player],
            player_vars: Dict,
            qb_indices: List[int],
            wr_indices: List[int],
    ):
        """QB+WR stacking"""
        team_qbs: Dict[str, List[int]] = {}
        team_wrs: Dict[str, List[int]] = {}

        for i in qb_indices:
            team_qbs.setdefault(players[i].team, []).append(i)

        for i in wr_indices:
            team_wrs.setdefault(players[i].team, []).append(i)

        for team in team_qbs:
            if team in team_wrs:
                qb_vars = [player_vars[i] for i in team_qbs[team]]
                wr_vars = [player_vars[i] for i in team_wrs[team]]

                if qb_vars and wr_vars:
                    prob += pulp.lpSum(wr_vars) >= 0.5 * pulp.lpSum(qb_vars)

    def _add_smart_vegas_exposure(
            self,
            prob,
            players: List[Player],
            player_vars: Dict,
            vegas_data: Dict
    ):
        """SMART Vegas exposure: Force 3-4 from #1 game, enforce QB+WR stacks, cap at 4/game"""

        high_total_games = vegas_data.get('high_total_games', [])

        if not high_total_games:
            logger.warning("No high-total games for smart exposure")
            return

        # Get #1 highest-total game
        top_game = high_total_games[0]
        top_game_teams = top_game.get('teams', [])
        top_game_total = top_game.get('total', 0)

        if not top_game_teams:
            logger.warning("Top game has no teams")
            return

        logger.info(f"🎯 SMART VEGAS: Targeting {top_game['game_id']} ({top_game_total} total)")

        # Find all players from top game
        top_game_indices = [
            i for i, p in enumerate(players)
            if p.team in top_game_teams and p.salary >= 6000  # Only consider $6K+ players
        ]

        if not top_game_indices:
            logger.warning(f"No players found from {top_game_teams}")
            return

        # ADD THIS DEBUG BLOCK
        logger.info(f"🔍 DEBUG: Found {len(top_game_indices)} players from {top_game_teams}")
        for idx in top_game_indices[:10]:  # Show first 10
            p = players[idx]
            logger.info(f"   Player {idx}: {p.name} ({p.position}) ${p.salary} - {p.projection:.1f}pts")

        # Check if any are locked
        locked_in_game = sum(1 for i in top_game_indices if players[i].locked)
        logger.info(f"🔍 DEBUG: {locked_in_game} locked players from top game")

        # CONSTRAINT 1: Force 3-4 players from top game
        prob += pulp.lpSum([player_vars[i] for i in top_game_indices]) >= 3
        prob += pulp.lpSum([player_vars[i] for i in top_game_indices]) <= 4

        logger.info(f"✅ CONSTRAINT: 3-4 players from {top_game_teams}")
        return top_game_indices  # Return for immediate checking
        # CONSTRAINT 2: If QB from top game, must roster 1+ WR from same team
        for team in top_game_teams:
            qb_indices = [i for i in top_game_indices if players[i].position == 'QB' and players[i].team == team]
            wr_indices = [i for i in top_game_indices if players[i].position == 'WR' and players[i].team == team]

            if qb_indices and wr_indices:
                for qb_idx in qb_indices:
                    prob += pulp.lpSum([player_vars[i] for i in wr_indices]) >= player_vars[qb_idx]

                logger.info(f"✅ CONSTRAINT: {team} QB → must roster {team} WR")

        # CONSTRAINT 3: Cap any single game at 4 players max
        game_groups = {}
        for i, player in enumerate(players):
            player_game = None
            for game in high_total_games:
                if player.team in game.get('teams', []):
                    player_game = game['game_id']
                    break

            if player_game:
                game_groups.setdefault(player_game, []).append(i)

        for game_id, player_indices in game_groups.items():
            prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= 4
            logger.info(f"✅ CONSTRAINT: Max 4 players from {game_id}")

    def _predict_friends_league_ownership(self, player: Player, contest_type: str) -> float:
        """
        FRIENDS LEAGUE: Ownership is IRRELEVANT (12 people, 1 lineup each)
        We're maximizing points to beat 11 opponents, not differentiating from field
        """
        if contest_type == 'friends_league':
            return 0.0  # Ownership doesn't matter - just score the most points

        # For other contest types (GPP/Cash), use simplified ownership estimate
        if player.salary >= 9500:
            return 35.0
        elif player.salary >= 7500:
            return 20.0
        elif player.salary <= 5000:
            return 10.0
        else:
            return 15.0

    def _calculate_contest_value(self, player: Player, contest_type: str) -> float:
        """Fallback value calculation when Monte Carlo not available"""
        base_value = player.projection

        if contest_type == 'friends_league':
            vegas_multipliers = getattr(self, 'vegas_multipliers', {})
            vegas_boost = vegas_multipliers.get(player.team, 1.0)

            if player.position == 'QB':
                if vegas_boost >= 1.40:
                    base_value *= 2.80
                elif vegas_boost >= 1.25:
                    base_value *= 2.20
                elif vegas_boost >= 1.15:
                    base_value *= 1.60

                ceiling_bonus = (player.ceiling_90 - player.projection) * 180.0  # Was 60.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 120.0  # Was 40.0
                boom_bonus = player.boom_rate * 300.0  # Was 100.0

                if player.salary >= 8000:
                    salary_bonus = 35.0
                elif player.salary >= 7000:
                    salary_bonus = 20.0
                elif player.salary >= 6500:
                    salary_bonus = -5.0
                else:
                    salary_bonus = -40.0

            elif player.position == 'RB':
                if vegas_boost >= 1.40:
                    base_value *= 1.80
                elif vegas_boost >= 1.25:
                    base_value *= 1.50
                elif vegas_boost >= 1.15:
                    base_value *= 1.25

                ceiling_bonus = (player.ceiling_90 - player.projection) * 90.0  # Was 30.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 60.0  # Was 20.0
                boom_bonus = player.boom_rate * 300.0  # Was 100.0

                if player.salary >= 9000:
                    salary_bonus = 25.0
                elif player.salary >= 7500:
                    salary_bonus = 12.0
                elif player.salary <= 5500 and player.value >= 2.8:
                    salary_bonus = 15.0
                else:
                    salary_bonus = -8.0

            elif player.position == 'WR':
                if vegas_boost >= 1.40:
                    base_value *= 1.65
                elif vegas_boost >= 1.25:
                    base_value *= 1.40
                elif vegas_boost >= 1.15:
                    base_value *= 1.20

                ceiling_bonus = (player.ceiling_90 - player.projection) * 72.0  # Was 24.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 48.0  # Was 16.0
                boom_bonus = player.boom_rate * 288.0  # Was 96.0

                if player.salary >= 8500:
                    salary_bonus = 18.0
                elif player.salary >= 7000:
                    salary_bonus = 10.0
                elif player.salary <= 6000 and player.value >= 2.5:
                    salary_bonus = 12.0
                elif 6000 <= player.salary <= 7000:
                    salary_bonus = -6.0
                else:
                    salary_bonus = 0.0

            elif player.position == 'TE':
                if vegas_boost >= 1.40:
                    base_value *= 1.45
                elif vegas_boost >= 1.25:
                    base_value *= 1.25

                ceiling_bonus = (player.ceiling_90 - player.projection) * 20.0
                ceiling_95_bonus = (player.ceiling_95 - player.ceiling_90) * 12.0
                boom_bonus = player.boom_rate * 80.0

                if player.salary >= 6500:
                    salary_bonus = 20.0
                elif player.salary <= 4800:
                    salary_bonus = 10.0
                else:
                    salary_bonus = -15.0

            else:
                ceiling_bonus = (player.ceiling_90 - player.projection) * 6.0
                ceiling_95_bonus = 0.0
                boom_bonus = player.boom_rate * 25.0
                salary_bonus = 0.0

            ownership_penalty = 0.0
            variance_bonus = player.variance * 2.5
            bust_penalty = player.bust_rate * 8.0

            total_value = (base_value + ceiling_bonus + ceiling_95_bonus + boom_bonus +
                           salary_bonus + variance_bonus - bust_penalty)

            if player.position == 'QB' and player.salary >= 7000:
                logger.info(f"🎯 QB VALUE: {player.name} ${player.salary} = {total_value:.1f} "
                            f"(base={base_value:.1f}, vegas={vegas_boost:.2f}x, "
                            f"ceiling_bonus={ceiling_bonus:.1f}, salary_bonus={salary_bonus:.1f})")

            return total_value
        elif contest_type == 'gpp':
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
        """Extract lineup results with FanDuel ordering - handles H2H MVP selection"""
        selected_players: List[Player] = []
        total_salary = 0
        total_ownership = 0

        for i, player in enumerate(players):
            if player_vars[i].varValue == 1:
                selected_players.append(player)
                total_salary += player.salary
                total_ownership += player.ownership

        # H2H: Select MVP from the 6 players
        if contest_type == 'h2h':
            if len(selected_players) != 6:
                logger.error(f"H2H lineup has {len(selected_players)} players, expected 6")
                return None

            # Find best MVP (highest ceiling + value combo)
            mvp = None
            best_mvp_score = 0

            for player in selected_players:
                if player.monte_carlo_analyzed:
                    mvp_score = player.ceiling_90 * player.value
                else:
                    mvp_score = player.projection * player.value * 1.5

                if mvp_score > best_mvp_score:
                    best_mvp_score = mvp_score
                    mvp = player

            if not mvp:
                mvp = max(selected_players, key=lambda p: p.projection)

            # Mark MVP
            mvp.is_mvp = True

            # Calculate total salary with MVP 1.5x cost
            mvp_salary_cost = int(mvp.salary * 1.5)
            other_salary = sum(p.salary for p in selected_players if p != mvp)
            total_salary = mvp_salary_cost + other_salary

            # Calculate projected points with MVP 1.5x
            projected_points = (mvp.projection * 1.5) + sum(p.projection for p in selected_players if p != mvp)

            # Order: MVP first, then FLEX by salary
            flex_players = [p for p in selected_players if p != mvp]
            flex_players.sort(key=lambda p: p.salary, reverse=True)
            ordered_players = [mvp] + flex_players

            logger.info(
                f"🏆 H2H MVP: {mvp.name} ({mvp.position}) ${mvp_salary_cost:,} (1.5x) - {mvp.projection:.1f} → {mvp.projection * 1.5:.1f} pts")

            return LineupResult(
                players=ordered_players,
                total_salary=total_salary,
                projected_points=projected_points,
                total_value=sum(p.value for p in ordered_players),
                ownership_total=total_ownership,
                correlation_score=1.0,  # All same game
                weather_impact=float(np.mean([p.weather_factor for p in ordered_players])) if ordered_players else 1.0,
                contest_type=contest_type,
            )

        # EXISTING CODE for non-H2H contests...
        ordered_players = self._format_lineup_for_fanduel(selected_players)

        # CRITICAL FIX: Use original projections for display, not randomized ones
        if contest_type == 'single_game' and len(ordered_players) == 6:
            mvp = max(ordered_players, key=lambda p: getattr(p, '_original_projection', p.projection))
            projected_points = mvp._original_projection * 1.5 + sum(
                getattr(p, '_original_projection', p.projection) for p in ordered_players if p != mvp)
        else:
            projected_points = sum(getattr(p, '_original_projection', p.projection) for p in ordered_players)

        return LineupResult(
            players=ordered_players,
            total_salary=total_salary,
            projected_points=projected_points,
            total_value=sum(p.value for p in ordered_players),
            ownership_total=total_ownership,
            correlation_score=self._calculate_correlation(ordered_players),
            weather_impact=float(np.mean([p.weather_factor for p in ordered_players])) if ordered_players else 1.0,
            contest_type=contest_type,
        )

    def _format_lineup_for_fanduel(self, players: List[Player]) -> List[Player]:
        """Order players in FanDuel format"""
        ordered: List[Player] = []

        by_position: Dict[str, List[Player]] = {}
        for player in players:
            by_position.setdefault(player.position, []).append(player)

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
        """Generate diverse lineups with AGGRESSIVE diversity enforcement"""

        vegas_multipliers = getattr(self, 'vegas_multipliers', {})
        players = identify_core_plays(players, vegas_multipliers, num_lineups)

        lineups: List[LineupResult] = []
        used_combinations = set()

        max_appearances = {
            pos: calculate_max_exposure(num_lineups, pos)
            for pos in ['QB', 'RB', 'WR', 'TE', 'D']
        }

        if single_game_teams:
            for pos in max_appearances:
                max_appearances[pos] = max(1, int(max_appearances[pos] * 0.90))

        logger.info(f"Diversity limits: {max_appearances}")
        exposure_pcts = {pos: f"{(max_appearances[pos] / num_lineups) * 100:.0f}%" for pos in max_appearances}
        logger.info(f"Exposure rates: {exposure_pcts}")

        player_usage_tracker = {}

        max_attempts = num_lineups * 8

        for attempt in range(max_attempts):
            if len(lineups) >= num_lineups:
                break

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
                    locked=player.locked,
                )

                if player.monte_carlo_analyzed:
                    new_player.floor_10 = player.floor_10
                    new_player.ceiling_90 = player.ceiling_90
                    new_player.ceiling_95 = player.ceiling_95
                    new_player.boom_rate = player.boom_rate
                    new_player.bust_rate = player.bust_rate
                    new_player.monte_carlo_analyzed = True

                random_factor = 1.0
                if not player.locked:
                    if contest_type == 'friends_league':
                        is_boom_candidate = False

                        # STRICT: Only protect elite players ($8500+ with real boom potential)
                        if player.salary >= 8500 and player.monte_carlo_analyzed:
                            ceiling_ratio = player.ceiling_90 / player.projection if player.projection > 0 else 1
                            is_boom_candidate = (ceiling_ratio >= 1.50 and player.boom_rate >= 0.25)
                        elif player.salary >= 9000:
                            # Very expensive players protected by default
                            is_boom_candidate = True

                        if is_boom_candidate:
                            random_factor = 1.0
                            logger.info(f"🛡️ BOOM PROTECTED: {player.name} (${player.salary}) - no randomization")
                        else:
                            random_factor = random.uniform(0.85, 1.15)
                    elif contest_type == 'gpp':
                        random_factor = random.uniform(0.70, 1.30)
                    elif contest_type == 'cash':
                        random_factor = random.uniform(0.92, 1.08)
                    else:
                        random_factor = random.uniform(0.60, 1.40)

                new_player.projection *= random_factor
                new_player._original_projection = player.projection  # SAVE ORIGINAL
                new_player.value = (new_player.projection / (new_player.salary / 1000.0)
                                    if new_player.salary > 0 else 0.0)
                variance_multipliers = {'QB': 0.28, 'RB': 0.35, 'WR': 0.45, 'TE': 0.38, 'D': 0.42}
                new_player.variance = new_player.projection * variance_multipliers.get(new_player.position, 0.35)

                randomized_players.append(new_player)

            lineup = await self.optimize_lineup(randomized_players, contest_type, single_game_teams)
            if not lineup:
                continue

            passes_diversity = True
            overused_players = []

            for player in lineup.players:
                position = player.position
                player_key = f"{player.id}_{position}"

                current_usage = player_usage_tracker.get(player_key, 0)
                max_allowed = max_appearances.get(position, 5)

                if player.is_core or player.locked:
                    continue

                if current_usage >= max_allowed:
                    overused_players.append(f"{player.name}({current_usage}/{max_allowed})")
                    passes_diversity = False

            if not passes_diversity:
                logger.debug(f"Rejected lineup - overused: {overused_players}")
                continue

            expensive_core = tuple(sorted([
                p.id for p in lineup.players
                if p.salary > 7000
            ]))

            if expensive_core in used_combinations:
                logger.debug(f"Rejected lineup - duplicate expensive core")
                continue

            for player in lineup.players:
                player_key = f"{player.id}_{player.position}"
                player_usage_tracker[player_key] = player_usage_tracker.get(player_key, 0) + 1

            lineups.append(lineup)
            used_combinations.add(expensive_core)

            logger.info(f"✅ Lineup {len(lineups)}/{num_lineups} generated (attempt {attempt + 1})")

        logger.info("=" * 60)
        logger.info("FINAL PLAYER USAGE ACROSS LINEUPS:")
        for player_key, count in sorted(player_usage_tracker.items(), key=lambda x: x[1], reverse=True)[:15]:
            player_id, position = player_key.rsplit('_', 1)
            player_name = next((p.name for p in players if p.id == player_id), player_id)
            logger.info(f"  {player_name} ({position}): {count}/{num_lineups} lineups")
        logger.info("=" * 60)

        # Sort lineups based on contest type
        if contest_type == 'friends_league':
            # FRIENDS LEAGUE: Pure ceiling/points focus (ownership irrelevant)
            if lineups and lineups[0].ceiling_90 > 0:
                lineups.sort(key=lambda x: x.ceiling_90, reverse=True)
            else:
                lineups.sort(key=lambda x: x.projected_points + (x.variance_score * 1.5), reverse=True)
        elif contest_type == 'cash':
            if lineups and lineups[0].floor_25 > 0:
                lineups.sort(key=lambda x: x.floor_25, reverse=True)
            else:
                lineups.sort(key=lambda x: x.projected_points - (x.variance_score * 0.5), reverse=True)
        else:
            # GPP/Contrarian: Use ownership leverage
            if lineups and lineups[0].ceiling_90 > 0:
                lineups.sort(key=lambda x: x.ceiling_90 - (x.ownership_total * 0.2), reverse=True)
            else:
                lineups.sort(key=lambda x: x.projected_points + (x.variance_score * 1.0), reverse=True)

        if len(lineups) < num_lineups:
            logger.warning(f"⚠️ Only generated {len(lineups)}/{num_lineups} lineups")
        else:
            logger.info(f"✅ Generated {len(lineups)} diverse {contest_type} lineups")

        return lineups


# -----------------------------
# Public API (sync)
# -----------------------------
def _run_coro_sync(coro):
    """
    Run coroutine from both sync and async contexts without raising "loop is running".
    Spawns a dedicated thread with its own loop when already inside an event loop.
    """
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


def optimize_dfs_lineups(
        player_data: List[Dict],
        weather_data: Dict = None,
        vegas_multipliers: Dict = None,
        vegas_data: Dict = None,
        num_lineups: int = 10,
        contest_type: str = 'gpp',
        single_game_teams: List[str] = None,
        use_monte_carlo: bool = True,
        mc_simulations: int = 5000,
) -> List[LineupResult]:
    """
    AI-Enhanced optimization with Monte Carlo variance modeling (synchronous wrapper).
    Safe to call from CLI, scripts, and FastAPI handlers.
    """
    logger.info(f"Starting {contest_type.upper()} optimization | "
                f"Lineups: {num_lineups} | Monte Carlo: {'ON' if use_monte_carlo and MONTE_CARLO_AVAILABLE else 'OFF'}")

    ai_enabled = os.getenv('AI_ENABLED', 'true').lower() == 'true'

    if AI_AVAILABLE and ai_enabled:
        try:
            analyzer = DualAIDFSAnalyzer()
            ai_analysis = analyzer.analyze_slate_for_optimization(
                player_data, weather_data or {}, vegas_multipliers or {}, contest_type
            )

            if ai_analysis and isinstance(ai_analysis, dict):
                ownership_adj = ai_analysis.get('ownership_adjustments') or {}
                if ownership_adj:
                    adjusted = 0
                    for rec in player_data:
                        name = rec.get('player_name', rec.get('name', ''))
                        if name in ownership_adj:
                            if 'ownership' in rec and isinstance(rec['ownership'], (int, float)):
                                rec['ownership'] = max(0.0, float(rec['ownership']) * float(ownership_adj[name]))
                            else:
                                rec['ownership_hint'] = float(ownership_adj[name])
                            adjusted += 1
                    logger.info(f"AI adjusted ownership hints for {adjusted} players")

                cost_summary = getattr(analyzer, "get_cost_summary", lambda: {})()
                if cost_summary:
                    logger.info(
                        f"AI cost: used ${cost_summary.get('weekly_spend', 0):.2f} / "
                        f"${cost_summary.get('weekly_budget', 0):.2f}"
                    )
        except Exception as e:
            logger.warning(f"AI analysis failed, continuing without it: {e}")
        # NEW: Apply AI edge case boosts to player projections
        if AI_AVAILABLE and ai_enabled:
            try:
                from ai_analyzer import DualAIDFSAnalyzer

                edge_analyzer = DualAIDFSAnalyzer()
                edge_analysis = edge_analyzer.analyze_edge_case_players(player_data)
                edge_recommendations = edge_analysis.get('edge_case_recommendations', [])

                boost_count = 0
                for rec in edge_recommendations:
                    player_name = rec.get('player_name', '')
                    confidence = rec.get('confidence', 0)
                    recommendation = rec.get('recommendation', '')

                    if recommendation == 'START' and confidence >= 7:
                        for player in player_data:
                            if player.get('name', '') == player_name:
                                original_proj = player.get('projected_points', 0)
                                boost_factor = 1.0 + ((confidence / 10.0) * 0.25)
                                player['projected_points'] = original_proj * boost_factor
                                player['projection'] = player['projected_points']

                                boost_count += 1
                                logger.info(f"🚀 AI EDGE BOOST: {player_name} "
                                            f"{original_proj:.1f} → {player['projected_points']:.1f} pts "
                                            f"(confidence {confidence}/10)")
                                break

                if boost_count > 0:
                    logger.info(f"✅ Applied {boost_count} AI edge case boosts")

            except Exception as e:
                logger.warning(f"AI edge case analysis failed: {e}")
    optimizer = EnhancedDFSOptimizer(use_monte_carlo=use_monte_carlo, mc_simulations=mc_simulations)

    setattr(optimizer, "vegas_multipliers", vegas_multipliers or {})
    setattr(optimizer, "vegas_data", vegas_data or {})

    # DIAGNOSTIC: Verify vegas multipliers reached optimizer
    logger.info(f"🎯 OPTIMIZER RECEIVED vegas_multipliers: {vegas_multipliers}")
    if vegas_multipliers:
        logger.info(f"   Sample teams: {list(vegas_multipliers.items())[:5]}")

    players: List[Player] = _run_coro_sync(
        optimizer.prepare_players(player_data, weather_data or {}, vegas_multipliers or {})
    )

    if not players:
        logger.error("No valid players after preparation")
        return []

    locked_count = sum(1 for p in players if p.locked)
    if locked_count > 0:
        logger.info(f"🔒 Found {locked_count} locked players in optimizer")

    if use_monte_carlo and MONTE_CARLO_AVAILABLE:
        mc_count = sum(1 for p in players if p.monte_carlo_analyzed)
        logger.info(f"Monte Carlo enriched players: {mc_count}/{len(players)}")

    logger.info(f"Optimization dataset size: {len(players)} active players")

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
                "total_value": round(lu.total_value, 3),
                "ownership_total": round(lu.ownership_total, 2),
                "correlation_score": round(lu.correlation_score, 3),
                "weather_impact": round(lu.weather_impact, 3),
                "ceiling_90": round(lu.ceiling_90, 2),
                "ceiling_95": round(lu.ceiling_95, 2),
                "floor_10": round(lu.floor_10, 2),
                "floor_25": round(lu.floor_25, 2),
                "variance_score": round(lu.variance_score, 3),
                "sharpe_ratio": round(lu.sharpe_ratio, 3),
                "risk_level": lu.risk_level,
                "boom_probability": round(lu.boom_probability, 4),
                "bust_probability": round(lu.bust_probability, 4),
                "players": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "position": p.position,
                        "team": p.team,
                        "salary": p.salary,
                        "projection": round(p.projection, 2),
                        "ownership": round(p.ownership, 2),
                        "value": round(p.value, 3),
                        "locked": p.locked,
                        "floor_10": round(getattr(p, "floor_10", 0.0), 2),
                        "ceiling_90": round(getattr(p, "ceiling_90", 0.0), 2),
                        "ceiling_95": round(getattr(p, "ceiling_95", 0.0), 2),
                        "boom_rate": round(getattr(p, "boom_rate", 0.0), 4),
                        "bust_rate": round(getattr(p, "bust_rate", 0.0), 4),
                    }
                    for p in lu.players
                ],
                "insights": lu.monte_carlo_insights or {},
            })

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": datetime.now().isoformat(),
                    "contest_type": contest_type,
                    "num_lineups": len(lineups),
                    "use_monte_carlo": bool(use_monte_carlo and MONTE_CARLO_AVAILABLE),
                    "lineups": payload,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved lineups to {out_path}")
    except Exception as e:
        logger.warning(f"Failed to export lineups: {e}")

    if use_monte_carlo and MONTE_CARLO_AVAILABLE:
        top = lineups[0]
        if top.ceiling_90 or top.floor_10:
            logger.info(
                f"Top lineup MC: risk={top.risk_level} | "
                f"Ceil90={top.ceiling_90:.1f} | Floor10={top.floor_10:.1f} | "
                f"Boom={top.boom_probability:.1%} | Bust={top.bust_probability:.1%}"
            )

    return lineups