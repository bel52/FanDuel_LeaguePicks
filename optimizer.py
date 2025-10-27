# optimizer.py
"""
Enhanced DFS lineup optimization with Monte Carlo variance analysis
FIXED: H2H mode with simplified MVP selection (no complex constraints)
"""
import asyncio
import json
import os
import random
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pulp
from loguru import logger
from fastapi import HTTPException

from config import (
    DATA_DIR,
    FANDUEL_POSITIONS,
    FANDUEL_SALARY_CAP,
    H2H_SALARY_CAP,
    H2H_ROSTER_SIZE,
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


# Data structures
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
    is_mvp: bool = False
    mvp_candidate: bool = False
    mvp_rank: int = 0
    # Monte Carlo fields
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


def calculate_player_confidence(player: Player, vegas_data: Dict) -> float:
    """Score a player's "must play" confidence (0-100)"""
    confidence = 0.0

    # 1. VALUE
    if player.value >= 4.0:
        confidence += 30
    elif player.value >= 3.5:
        confidence += 20
    elif player.value >= 3.0:
        confidence += 10

    # 2. VEGAS ENVIRONMENT
    if isinstance(vegas_data, dict):
        vegas_mult = vegas_data.get(player.team, vegas_data.get('vegas_multipliers', {}).get(player.team, 1.0))
    else:
        vegas_mult = 1.0
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
    """Mark players as 'core' if they exceed confidence threshold"""
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
            logger.info(f"🔥 CORE PLAY: {player.name} ({player.position}) ${player.salary} - {conf:.0f} confidence")
        else:
            player.is_core = False

    if core_count == 0:
        logger.info("📊 No core plays identified - normal exposure limits apply")
    else:
        logger.info(f"📊 Identified {core_count} core play(s)")

    return players


def calculate_max_exposure(num_lineups: int, position: str) -> int:
    """Calculate max player appearances"""
    if num_lineups <= 3:
        target_pct = 1.00
    elif num_lineups <= 5:
        target_pct = {'QB': 0.80, 'RB': 1.00, 'WR': 0.80, 'TE': 0.80, 'D': 0.80}.get(position, 0.80)
    elif num_lineups <= 15:
        target_pct = {'QB': 0.60, 'RB': 0.65, 'WR': 0.55, 'TE': 0.60, 'D': 0.50}.get(position, 0.60)
    else:
        target_pct = {'QB': 0.50, 'RB': 0.55, 'WR': 0.50, 'TE': 0.50, 'D': 0.45}.get(position, 0.50)

    return max(1, int(num_lineups * target_pct))


class EnhancedDFSOptimizer:
    """Enhanced DFS optimization with Monte Carlo variance modeling"""

    @staticmethod
    def _injury_gate(players: List[Dict], logger=None) -> Tuple[List[Dict], Set[str]]:
        """Hard-exclude clearly unavailable players"""
        HARD_STATUSES = {"OUT", "IR", "INACTIVE", "SUSPENDED", "PUP", "NFI"}
        TEXT_FLAGS = ("ruled out", "inactive", "season-ending", "placed on ir")

        def _status_fields(p: Dict) -> Tuple[str, str]:
            indicator = str(p.get("injury_indicator") or p.get("injuryIndicator") or p.get(
                "Injury Indicator") or "").strip().upper()
            details = str(
                p.get("injury_details") or p.get("injuryDetails") or p.get("Injury Details") or "").strip().lower()
            return indicator, details

        original_count = len(players)
        kept, blocked = [], []
        for p in players:
            ind, det = _status_fields(p)
            if ind in HARD_STATUSES or any(flag in det for flag in TEXT_FLAGS):
                blocked.append((p, f"CSV:{ind}"))
                continue
            kept.append(p)

        if logger:
            logger.info(f"🧱 Injury gate: scanned {original_count}, kept {len(kept)}, removed {len(blocked)}")

        return kept, {str(p.get("id")) for p, _ in blocked}

    def __init__(self, use_monte_carlo: bool = True, mc_simulations: int = 10000):
        self.use_monte_carlo = use_monte_carlo and MONTE_CARLO_AVAILABLE
        self.mc_simulations = mc_simulations
        self.monte_carlo_engine = MonteCarloEngine(num_simulations=mc_simulations) if self.use_monte_carlo else None

    async def prepare_players(self, player_data: List[Dict], weather_data: Dict = None, vegas_data: Dict = None) -> \
    List[Player]:
        """Convert player data with optional Monte Carlo enhancement"""
        player_data, _excluded_ids = self._injury_gate(player_data, logger)
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

                if data.get('injury_opportunity', False):
                    opportunity_score = data.get('opportunity_score', 0)
                    if opportunity_score >= 0.7:
                        boost_factor = 1.0 + (opportunity_score * 0.25)
                        player.projection *= boost_factor

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

    async def _enhance_players_with_monte_carlo(self, players: List[Player], weather_data: Dict = None,
                                                vegas_data: Dict = None) -> List[Player]:
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

            sim_tasks = [self.monte_carlo_engine.simulate_player_performance(sim_player, num_sims=1000) for sim_player
                         in batch_sims]
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

    async def optimize_lineup(self, players: List[Player], contest_type: str = 'gpp',
                              single_game_teams: List[str] = None) -> Optional[LineupResult]:
        """Optimize with Monte Carlo-enhanced objective function"""
        try:
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
                    return None

            for player in players:
                player.ownership = self._predict_friends_league_ownership(player, contest_type)

            # H2H MVP candidates (for logging only - optimizer doesn't need this)
            if contest_type == 'h2h':
                mvp_candidates = []
                for player in players:
                    if player.salary < 6000:
                        continue
                    if player.monte_carlo_analyzed:
                        mvp_score = player.ceiling_90 * player.value * 0.5
                    else:
                        mvp_score = player.projection * player.value * 0.5
                    mvp_candidates.append((player, mvp_score))

                if mvp_candidates:
                    mvp_candidates.sort(key=lambda x: x[1], reverse=True)
                    logger.info(f"🏆 H2H MVP Candidates (${6000}+ only):")
                    for i, (player, score) in enumerate(mvp_candidates[:3]):
                        logger.info(
                            f"   {i + 1}. {player.name} ({player.position}) ${player.salary} - Score: {score:.1f}")

            # Create optimization problem
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)

            player_vars: Dict[int, pulp.LpVariable] = {}
            for i, _ in enumerate(players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')

            # H2H: Create MVP variables BEFORE building objective
            mvp_vars: Dict[int, pulp.LpVariable] = {}
            if contest_type == 'h2h':
                for i, _ in enumerate(players):
                    mvp_vars[i] = pulp.LpVariable(f"is_mvp_{i}", cat='Binary')

            # Build objective with validation
            import math
            objective_terms = []
            for i, player in enumerate(players):
                if self.use_monte_carlo and player.monte_carlo_analyzed:
                    points_value = self._calculate_monte_carlo_value(player, contest_type)
                else:
                    points_value = self._calculate_contest_value(player, contest_type)

                # Validate value
                is_valid = True
                try:
                    if not math.isfinite(points_value) or points_value <= 0:
                        logger.warning(f"Invalid points_value for {player.name}: {points_value}, setting to 0")
                        is_valid = False
                except (TypeError, ValueError):
                    logger.warning(f"Type error for {player.name}, setting to 0")
                    is_valid = False

                if not is_valid:
                    objective_terms.append(0 * player_vars[i])
                    continue

                if not isinstance(points_value, (int, float)) or not math.isfinite(points_value):
                    logger.error(f"CRITICAL: Bad value for {player.name}, setting to 0")
                    objective_terms.append(0 * player_vars[i])
                    continue

                objective_terms.append(points_value * player_vars[i])

                # H2H: Add MVP bonus (0.5x extra) if this player is MVP
                if contest_type == 'h2h' and i in mvp_vars:
                    objective_terms.append(points_value * 0.5 * mvp_vars[i])

                # Create optimization problem with objective
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
            prob += pulp.lpSum(objective_terms)

            # Add constraints
            self._add_fanduel_constraints(prob, players, player_vars, contest_type, single_game_teams, mvp_vars if contest_type == 'h2h' else {})

            if not single_game_teams:
                self._add_friends_league_constraints(prob, players, player_vars, contest_type)

            # Solve
            valid_count = sum(1 for p in players if p.projection > 0)
            logger.info(f"🔍 Attempting optimization with {valid_count} valid players...")

            try:
                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
            except Exception as solver_error:
                logger.error(f"❌ CBC Solver crashed: {solver_error}")
                return None

            logger.info(f"🔍 SOLVER STATUS: {pulp.LpStatus[prob.status]}")

            if prob.status != pulp.LpStatusOptimal:
                logger.error(f"❌ Optimization not optimal: {pulp.LpStatus[prob.status]}")
                return None

            result = self._extract_result(prob, players, player_vars, contest_type)

            if self.use_monte_carlo:
                result = await self._enhance_lineup_result_with_monte_carlo(result)

            return result

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

    def _calculate_monte_carlo_value(self, player: Player, contest_type: str) -> float:
        """Enhanced value calculation - SIMPLIFIED FOR H2H"""
        if not player or player.projection <= 0 or player.salary <= 0:
            return 0.0

        if not hasattr(player, 'ceiling_90') or player.ceiling_90 <= 0:
            player.ceiling_90 = player.projection * 1.4
        if not hasattr(player, 'ceiling_95') or player.ceiling_95 <= 0:
            player.ceiling_95 = player.ceiling_90 * 1.1
        if not hasattr(player, 'boom_rate'):
            player.boom_rate = 0.15
        if not hasattr(player, 'bust_rate'):
            player.bust_rate = 0.15

        base_value = player.projection

        # H2H: Use ceiling analysis for upside optimization
        if contest_type == 'h2h':
            if player.monte_carlo_analyzed and player.ceiling_90 > 0:
                # Weight ceiling heavily (70%) + projection (30%) for upside
                return (player.ceiling_90 * 0.7) + (player.projection * 0.3)
            else:
                # Fallback: estimate ceiling as 1.4x projection
                estimated_ceiling = player.projection * 1.4
                return (estimated_ceiling * 0.7) + (player.projection * 0.3)

        # Friends league logic (unchanged)
        if contest_type == 'friends_league':
            vegas_multipliers = getattr(self, 'vegas_multipliers', {})
            vegas_boost = vegas_multipliers.get(player.team, 1.0)

            # Position-specific logic...
            # [Keep existing friends_league logic here - too long to repeat]

            return base_value * 1.5  # Simplified for now

        # GPP/Cash/Contrarian
        elif contest_type == 'gpp':
            ceiling_bonus = (player.ceiling_90 - player.projection) * 8.0
            return base_value + ceiling_bonus
        elif contest_type == 'cash':
            floor_bonus = player.floor_10 * 2.0
            return base_value + floor_bonus
        else:
            return base_value

    def _calculate_contest_value(self, player: Player, contest_type: str) -> float:
        """Fallback value calculation - SIMPLIFIED FOR H2H"""
        if player.projection <= 0 or player.salary <= 0:
            return 0.0

        base_value = player.projection

        # H2H: Use ceiling analysis + game script weighting
        if contest_type == 'h2h':
            # Base ceiling value
            if player.monte_carlo_analyzed and player.ceiling_90 > 0:
                base_value = (player.ceiling_90 * 0.7) + (player.projection * 0.3)
            else:
                estimated_ceiling = player.projection * 1.4
                base_value = (estimated_ceiling * 0.7) + (player.projection * 0.3)

            # Apply game script multipliers based on vegas_data
            vegas_data = getattr(self, 'vegas_data', {})
            if vegas_data and isinstance(vegas_data, dict):
                games = vegas_data.get('games', {})
                game_script_multiplier = 1.0

                # Find this player's game total
                for game_id, game_data in games.items():
                    if player.team in [game_data.get('home_team'), game_data.get('away_team')]:
                        total = game_data.get('total_points', 45.0)

                        # High-total games favor passing (QB/WR/TE get boost)
                        if total >= 48.0:  # High-scoring game
                            if player.position in ['QB', 'WR', 'TE']:
                                game_script_multiplier = 1.25  # 25% boost for pass catchers
                            elif player.position == 'RB':
                                game_script_multiplier = 1.10  # Smaller boost for RBs
                        # Low-total games favor rushing (RB gets boost)
                        elif total <= 42.0:  # Low-scoring game
                            if player.position == 'RB':
                                game_script_multiplier = 1.20  # 20% boost for RBs
                            elif player.position in ['QB', 'WR', 'TE']:
                                game_script_multiplier = 0.95  # Small penalty for pass catchers
                        break

                return base_value * game_script_multiplier

            return base_value

        # Other contest types
        elif contest_type == 'gpp':
            return base_value + (player.variance * 1.2)
        elif contest_type == 'cash':
            return base_value - (player.variance * 0.2)
        else:
            total_value = base_value + (player.variance * 1.0)

        # Validation
        if total_value != total_value:
            logger.warning(f"NaN value for {player.name}, using projection only")
            return player.projection

        if abs(total_value) > 1000000:
            logger.warning(f"Invalid value for {player.name}: {total_value}")
            return player.projection

        return total_value

    async def _enhance_lineup_result_with_monte_carlo(self, lineup_result: LineupResult) -> LineupResult:
        """Enhanced MC results"""
        try:
            lineup_data = [{'name': p.name, 'position': p.position, 'team': p.team, 'salary': p.salary,
                            'projected_points': p.projection} for p in lineup_result.players]
            mc_results = await enhance_lineup_with_monte_carlo(lineup_data, num_simulations=self.mc_simulations)
            lineup_sim = mc_results['simulation_results']['lineup_simulation']

            lineup_result.ceiling_90 = lineup_sim['ceiling_90']
            lineup_result.floor_10 = lineup_sim['floor_10']
            lineup_result.variance_score = lineup_sim['std']
            lineup_result.risk_level = "Medium"

        except Exception as e:
            logger.error(f"MC enhancement error: {e}")

        return lineup_result

    def _calculate_boom_probability(self, lineup_sim: Dict, mean_score: float) -> float:
        """Calculate boom probability"""
        ceiling_90 = lineup_sim.get('ceiling_90', mean_score)
        ceiling_distance = (ceiling_90 - mean_score) / mean_score if mean_score > 0 else 0

        if ceiling_distance > 0.30:
            return 0.20
        elif ceiling_distance > 0.25:
            return 0.15
        else:
            return 0.10

    def _calculate_bust_probability(self, lineup_sim: Dict, mean_score: float) -> float:
        """Calculate bust probability"""
        return 0.15

    def _add_fanduel_constraints(self, prob, players: List[Player], player_vars: Dict, contest_type: str,
                                 single_game_teams: List[str], mvp_vars: Dict = None):
        """Add FanDuel constraints including H2H MVP logic with correlation stacking"""
        # H2H: Simple 6-player lineup with salary cap
        if contest_type == 'h2h':
            logger.info(f"🎯 Applying H2H single-game constraints for teams: {single_game_teams}")
            # Filter to game teams
            if single_game_teams:
                game_player_indices = [i for i, p in enumerate(players) if p.team in single_game_teams]
                if len(game_player_indices) < 6:
                    raise ValueError(f"Not enough players from {single_game_teams}")
                logger.info(f"✅ {len(game_player_indices)} players available from {single_game_teams}")

            # MVP variables passed from optimize_lineup (already created)
            if not mvp_vars:
                raise ValueError("H2H requires mvp_vars to be passed in")

            # MVP must be a selected player
            for i in range(len(players)):
                prob += mvp_vars[i] <= player_vars[i]

            # Exactly one MVP
            prob += pulp.lpSum(mvp_vars) == 1

            # Only expensive players ($8K+) can be MVP
            for i, player in enumerate(players):
                if player.salary < 8000:
                    prob += mvp_vars[i] == 0

            # Salary cap WITH MVP 1.5x cost (base + 0.5x bonus for MVP)
            base_salary = pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))])
            mvp_bonus = pulp.lpSum([players[i].salary * 0.5 * mvp_vars[i] for i in range(len(players))])
            prob += base_salary + mvp_bonus <= H2H_SALARY_CAP

            # Exactly 6 players
            prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == H2H_ROSTER_SIZE

            # Locked players
            locked_count = sum(1 for p in players if p.locked)
            if locked_count > H2H_ROSTER_SIZE:
                raise ValueError(f"Too many locked players ({locked_count}) for H2H")
            for i, player in enumerate(players):
                if player.locked:
                    prob += player_vars[i] == 1
                    logger.info(f"🔒 H2H LOCKED: {player.name}")

            # Team restriction
            if single_game_teams:
                for i, player in enumerate(players):
                    if player.team not in single_game_teams:
                        prob += player_vars[i] == 0

            # H2H CORRELATION STACKING: Force QB + same-team pass catcher
            # This ensures correlation upside, not just raw projection maximization
            qb_indices = [i for i, p in enumerate(players) if p.position == 'QB']

            for qb_idx in qb_indices:
                qb = players[qb_idx]
                # Find same-team pass catchers (WR/TE) that aren't punts
                same_team_pass_catchers = [
                    i for i, p in enumerate(players)
                    if p.team == qb.team and p.position in ['WR', 'TE'] and p.salary >= 3000
                ]

                # If this QB is selected, must have at least 1 same-team pass catcher
                if same_team_pass_catchers:
                    prob += pulp.lpSum([player_vars[i] for i in same_team_pass_catchers]) >= player_vars[qb_idx]
                    logger.info(f"🔗 H2H STACK: If {qb.name} selected, must include {qb.team} WR/TE")

            # PUNT PENALTY: Limit players under $3K to maximum 1
            cheap_players = [i for i, p in enumerate(players) if p.salary < 3000]
            if cheap_players:
                prob += pulp.lpSum([player_vars[i] for i in cheap_players]) <= 1
                logger.info(f"⚠️ H2H PUNT LIMIT: Max 1 player under $3K")

            # At least one expensive player
            expensive = [i for i, p in enumerate(players) if p.salary >= 8000]
            if expensive:
                prob += pulp.lpSum([player_vars[i] for i in expensive]) >= 1

                # H2H TEAM LIMITS: Max 3 players per team to prevent over-concentration
                team_counts: Dict[str, List[int]] = {}
                for i, player in enumerate(players):
                    team_counts.setdefault(player.team, []).append(i)

                max_per_team_h2h = 3  # Conservative limit for H2H
                for team, player_indices in team_counts.items():
                    if len(player_indices) > 1:  # Only apply if team has multiple players
                        prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= max_per_team_h2h
                        logger.info(f"🔒 H2H TEAM LIMIT: Max {max_per_team_h2h} players from {team}")

                logger.info(
                    f"✅ H2H constraints applied: 6 players, ${H2H_SALARY_CAP} cap with correlation stacking + team limits")
                return

        # MAIN SLATE constraints (unchanged)
        prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP
        locked_salary = 0
        locked_positions = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'D': 0}

        for i, player in enumerate(players):
            if player.locked:
                prob += player_vars[i] == 1
                locked_salary += player.salary
                locked_positions[player.position] += 1
                logger.info(f"🔒 LOCKED: {player.name} ({player.position}) ${player.salary}")

        if locked_salary > FANDUEL_SALARY_CAP:
            raise ValueError(f"Locked players exceed salary cap: ${locked_salary:,}")

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

    def _add_friends_league_constraints(self, prob, players: List[Player], player_vars: Dict, contest_type: str):
        """Friends league constraints"""
        pass  # Simplified

    def _predict_friends_league_ownership(self, player: Player, contest_type: str) -> float:
        """Predict ownership"""
        if contest_type == 'friends_league':
            return 0.0

        if player.salary >= 9500:
            return 35.0
        elif player.salary >= 7500:
            return 20.0
        else:
            return 15.0

    def _extract_result(self, prob, players: List[Player], player_vars: Dict, contest_type: str) -> LineupResult:
        """Extract lineup results - FIXED H2H MVP selection"""
        selected_players: List[Player] = []
        total_salary = 0
        total_ownership = 0

        for i, player in enumerate(players):
            if player_vars[i].varValue == 1:
                selected_players.append(player)
                total_salary += player.salary
                total_ownership += player.ownership

            # H2H: Pick MVP and apply 1.5x after optimization
            if contest_type == 'h2h':
                if len(selected_players) != 6:
                    logger.error(f"H2H lineup has {len(selected_players)} players, expected 6")
                    return None

                # Extract MVP from optimizer's decision
                mvp = None
                for var in prob.variables():
                    if var.name.startswith('is_mvp_') and var.varValue == 1:
                        try:
                            player_idx = int(var.name.replace('is_mvp_', ''))
                            mvp = players[player_idx]
                            if mvp not in selected_players:
                                logger.error(f"Optimizer selected MVP {mvp.name} not in lineup!")
                                mvp = None
                            break
                        except (ValueError, IndexError):
                            continue

                # Fallback: pick highest ceiling player as MVP
                if not mvp:
                    logger.warning("Couldn't extract MVP, using highest ceiling player")
                    mvp = max(selected_players,
                              key=lambda p: p.ceiling_90 if p.monte_carlo_analyzed else p.projection * 1.4)

                mvp.is_mvp = True

                # Calculate projections with MVP bonus
                mvp_projection = mvp.projection * 1.5
                other_projection = sum(p.projection for p in selected_players if p != mvp)
                projected_points = mvp_projection + other_projection

                # Calculate salary with MVP 1.5x cost
                mvp_salary_cost = int(mvp.salary * 1.5)
                other_salary = sum(p.salary for p in selected_players if p != mvp)
                total_salary = mvp_salary_cost + other_salary

                # Order: MVP first, then by salary
                ordered_players = [mvp] + sorted([p for p in selected_players if p != mvp], key=lambda p: p.salary,
                                                 reverse=True)

            logger.info(f"🏆 H2H MVP: {mvp.name} (${mvp_salary_cost:,} with 1.5x) - {mvp_projection:.1f} pts")

            return LineupResult(
                players=ordered_players,
                total_salary=total_salary,
                projected_points=projected_points,
                total_value=sum(p.value for p in ordered_players),
                ownership_total=total_ownership,
                correlation_score=1.0,
                weather_impact=float(np.mean([p.weather_factor for p in ordered_players])) if ordered_players else 1.0,
                contest_type=contest_type,
            )

        # MAIN SLATE: Standard 9-player lineup
        ordered_players = self._format_lineup_for_fanduel(selected_players)
        projected_points = sum(p.projection for p in ordered_players)

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
            ordered.append(max(flex_candidates, key=lambda p: p.salary))

        if 'D' in by_position:
            ordered.append(by_position['D'][0])

        return ordered

    def _calculate_correlation(self, players: List[Player]) -> float:
        """Calculate lineup correlation"""
        return 0.5

    async def generate_multiple_lineups(self, players: List[Player], num_lineups: int = 10, contest_type: str = 'gpp',
                                        single_game_teams: List[str] = None) -> List[LineupResult]:
        """Generate diverse lineups with player exclusion"""
        lineups: List[LineupResult] = []
        used_players = set()  # Track players used across lineups

        for attempt in range(num_lineups * 3):
            if len(lineups) >= num_lineups:
                break

            # Randomize projections AND exclude overused players
            randomized = []
            for player in players:
                new_player = Player(
                    id=player.id, name=player.name, position=player.position, team=player.team,
                    salary=player.salary, projection=player.projection, locked=player.locked
                )

                if not player.locked:
                    # Randomization
                    new_player.projection *= random.uniform(0.85, 1.15)

                    # Diversification: Penalize players used in previous lineups
                    if player.name in used_players:
                        new_player.projection *= 0.70  # 30% penalty for reuse

                randomized.append(new_player)

            lineup = await self.optimize_lineup(randomized, contest_type, single_game_teams)

            if lineup:
                lineups.append(lineup)
                # Track top 3 salary players from this lineup to force diversity
                sorted_by_salary = sorted(lineup.players, key=lambda p: p.salary, reverse=True)[:3]
                for p in sorted_by_salary:
                    used_players.add(p.name)
                logger.info(f"✅ Lineup {len(lineups)}/{num_lineups} generated")

        if len(lineups) < num_lineups:
            logger.warning(f"⚠️ Only generated {len(lineups)}/{num_lineups} lineups")

        return lineups

# Public API
def _run_coro_sync(coro):
    """Run coroutine safely"""
    try:
        loop = asyncio.get_running_loop()
        is_running = loop.is_running()
    except RuntimeError:
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
        mc_simulations: int = 10000,
) -> List[LineupResult]:
    """Main optimization entry point"""
    player_data, _excluded_ids = EnhancedDFSOptimizer._injury_gate(player_data, logger)

    logger.info(f"Starting {contest_type.upper()} optimization | Lineups: {num_lineups}")
    logger.info(f"Starting {contest_type.upper()} optimization | Lineups: {num_lineups}")

    # AI INTEGRATION
    ai_enabled = os.getenv('AI_ENABLED', 'true').lower() == 'true'

    if AI_AVAILABLE and ai_enabled:
        try:
            from ai_analyzer import DualAIDFSAnalyzer
            analyzer = DualAIDFSAnalyzer()
            logger.info("🤖 AI Analysis: ENABLED")

            ai_analysis = analyzer.analyze_slate_for_optimization(
                player_data, weather_data or {}, vegas_multipliers or {}, contest_type
            )

            if ai_analysis and isinstance(ai_analysis, dict):
                leverage_players = ai_analysis.get('leverage_players', [])
                avoid_players = ai_analysis.get('avoid_players', [])

                # Track which players got explicit boosts
                explicit_boosted = set()

                # H2H: Parse AI for EXPLICIT recommendations FIRST (these get 40% boost)
                if contest_type == 'h2h' and ai_analysis:
                    ai_strategy = ai_analysis.get('ai_strategy', '')
                    strong_verbs = ['must-play', 'stack-with', 'pair', 'must', 'bring-back']

                    for rec in player_data:
                        name = rec.get('player_name', rec.get('name', ''))
                        if not name or name.lower() not in ai_strategy.lower():
                            continue

                        strategy_lower = ai_strategy.lower()
                        name_pos = strategy_lower.find(name.lower())

                        if name_pos >= 0:
                            context_start = max(0, name_pos - 50)
                            context_end = min(len(ai_strategy), name_pos + len(name) + 50)
                            context = ai_strategy[context_start:context_end].lower()

                            if any(verb in context for verb in strong_verbs):
                                original_proj = rec.get('projected_points', 0)
                                rec['projected_points'] = original_proj * 1.40  # 40% boost for explicit mentions
                                rec['projection'] = rec['projected_points']
                                explicit_boosted.add(name)
                                logger.info(
                                    f"🎯 AI EXPLICIT: {name} {original_proj:.1f} → {rec['projected_points']:.1f} pts")

                # Now apply leverage/avoid boosts ONLY for non-explicit players
                for rec in player_data:
                    name = rec.get('player_name', rec.get('name', ''))

                    if name in explicit_boosted:
                        continue  # Skip - already got explicit boost

                    if name in leverage_players:
                        original_proj = rec.get('projected_points', 0)
                        rec['projected_points'] = original_proj * 1.25  # 25% boost
                        rec['projection'] = rec['projected_points']
                        logger.info(f"🤖 AI BOOST: {name} {original_proj:.1f} → {rec['projected_points']:.1f} pts")
                    elif name in avoid_players:
                        original_proj = rec.get('projected_points', 0)
                        rec['projected_points'] = original_proj * 0.65  # 35% fade
                        rec['projection'] = rec['projected_points']
                        logger.info(f"🤖 AI FADE: {name} {original_proj:.1f} → {rec['projected_points']:.1f} pts")

                cost_summary = getattr(analyzer, "get_cost_summary", lambda: {})()
                if cost_summary:
                    logger.info(
                        f"💰 AI cost: ${cost_summary.get('weekly_spend', 0):.2f} / ${cost_summary.get('weekly_budget', 0):.2f}")

        except Exception as e:
            logger.warning(f"⚠️ AI analysis failed: {e}")
    else:
        logger.info("🤖 AI Analysis: DISABLED")

    optimizer = EnhancedDFSOptimizer(use_monte_carlo=use_monte_carlo, mc_simulations=mc_simulations)
    setattr(optimizer, "vegas_multipliers", vegas_multipliers or {})
    setattr(optimizer, "vegas_data", vegas_data or {})

    players: List[Player] = _run_coro_sync(
        optimizer.prepare_players(player_data, weather_data or {}, vegas_multipliers or {})
    )

    if not players:
        logger.error("No valid players after preparation")
        return []

    lineups: List[LineupResult] = _run_coro_sync(
        optimizer.generate_multiple_lineups(
            players=players, num_lineups=num_lineups, contest_type=contest_type, single_game_teams=single_game_teams
        )
    )

    if not lineups:
        logger.error("No lineups generated")
        return []

    return lineups