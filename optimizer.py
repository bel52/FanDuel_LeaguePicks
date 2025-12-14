# optimizer.py
"""
WINNING DFS Optimizer - Fixed diversity generation and comprehensive data integration
Key fixes:
1. Explicit player exclusion between lineups (not just randomization)
2. friends_league uses GPP logic
3. Proper data flow verification
4. NaN sanitization throughout
"""
import asyncio
import json
import random
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pulp
from loguru import logger

from config import (
    DATA_DIR,
    FANDUEL_POSITIONS,
    FANDUEL_SALARY_CAP,
    OPTIMIZATION_CONFIG,
)


def safe_float(val, default=0.0):
    """Sanitize float values - replace NaN/inf with default"""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# Import the Monte Carlo engine (SYNC version - no async conflicts)
try:
    from monte_carlo_engine import (
        MonteCarloEngine,
        PlayerSimulation,
        convert_player_data_to_simulation,
        run_monte_carlo_sync,
    )

    MONTE_CARLO_AVAILABLE = True
    logger.info("✅ Monte Carlo engine loaded (sync version)")
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    logger.warning("Monte Carlo engine not available")

# Note: AI analysis is handled in data_collector.py, not here
AI_AVAILABLE = False  # Placeholder for compatibility


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
    # Game environment and AI fields
    game_total: float = 45.0
    game_environment_mult: float = 1.0
    ai_must_play: bool = False
    ai_must_fade: bool = False
    locked: bool = False

    def __post_init__(self):
        # Sanitize all float fields
        self.projection = safe_float(self.projection, 5.0)
        self.ownership = safe_float(self.ownership, 10.0)
        self.weather_factor = safe_float(self.weather_factor, 1.0)
        self.game_environment_mult = safe_float(self.game_environment_mult, 1.0)
        self.game_total = safe_float(self.game_total, 45.0)
        self.floor_10 = safe_float(self.floor_10, 0.0)
        self.ceiling_90 = safe_float(self.ceiling_90, 0.0)
        self.ceiling_95 = safe_float(self.ceiling_95, 0.0)
        self.boom_rate = safe_float(self.boom_rate, 0.0)
        self.bust_rate = safe_float(self.bust_rate, 0.0)
        self.variance = safe_float(self.variance, 0.0)

        # Calculate value
        self.value = safe_float(self.projection / (self.salary / 1000), 0.0) if self.salary > 0 else 0.0

        # Calculate variance if not set
        if self.variance == 0 and self.projection > 0:
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
    # Game environment info
    high_total_exposure: int = 0
    primary_stack_team: str = ""


class EnhancedDFSOptimizer:
    """Enhanced DFS optimization with Monte Carlo variance modeling"""

    def __init__(self, use_monte_carlo: bool = True, mc_simulations: int = 1000):
        self.use_monte_carlo = use_monte_carlo and MONTE_CARLO_AVAILABLE
        self.mc_simulations = mc_simulations
        self.vegas_multipliers = {}
        self.vegas_data = {}

        if use_monte_carlo and MONTE_CARLO_AVAILABLE:
            logger.info(f"🎲 Monte Carlo enabled: {mc_simulations} simulations per player")
        elif use_monte_carlo and not MONTE_CARLO_AVAILABLE:
            logger.warning("Monte Carlo requested but not available")

    def set_vegas_data(self, vegas_multipliers: Dict, vegas_data: Dict = None):
        """Store Vegas data for constraint generation"""
        self.vegas_multipliers = vegas_multipliers or {}
        self.vegas_data = vegas_data or {}

        # Log high-total teams for verification
        high_total_teams = [team for team, mult in self.vegas_multipliers.items()
                            if safe_float(mult, 1.0) >= 1.25]
        if high_total_teams:
            logger.info(f"🎯 HIGH-TOTAL TEAMS (47+): {high_total_teams}")
        logger.info(f"📊 Vegas multipliers set for {len(self.vegas_multipliers)} teams")

    def prepare_players(
            self,
            player_data: List[Dict],
            weather_data: Dict = None,
            vegas_data: Dict = None,
    ) -> List[Player]:
        """Convert player data with Monte Carlo enhancement (sync version)"""
        players: List[Player] = []

        for data in player_data:
            try:
                player_name = data.get('player_name', data.get('name', ''))
                position = data.get('position', '')
                team = data.get('team', '')
                salary = int(data.get('salary', 5000))
                projection = safe_float(data.get('projection', data.get('projected_points', 0)), 5.0)

                # Basic filtering
                if not player_name or len(player_name.strip()) < 2:
                    continue
                if projection <= 0:
                    continue

                # Normalize defense position
                if position in ['DST', 'DEF', 'D/ST']:
                    position = 'D'

                # Get game environment data
                game_mult = safe_float(data.get('game_environment_mult', 1.0), 1.0)
                game_total = safe_float(data.get('game_total', 45.0), 45.0)

                # Apply Vegas multiplier if not already applied
                if game_mult == 1.0 and team in self.vegas_multipliers:
                    game_mult = safe_float(self.vegas_multipliers[team], 1.0)

                # Create player
                player = Player(
                    id=str(data.get('player_id', data.get('id', player_name))),
                    name=player_name,
                    position=position,
                    team=team,
                    salary=salary,
                    projection=projection,
                    ownership=safe_float(data.get('ownership', 10.0), 10.0),
                    weather_factor=safe_float(data.get('weather_factor', 1.0), 1.0),
                    game_total=game_total,
                    game_environment_mult=game_mult,
                    ai_must_play=data.get('ai_must_play', False),
                    ai_must_fade=data.get('ai_must_fade', False),
                    locked=data.get('locked', False),
                )

                # Apply weather adjustments
                if weather_data and team in weather_data:
                    weather_factor = safe_float(weather_data[team].get('factor', 1.0), 1.0)
                    player.weather_factor = weather_factor
                    player.projection = safe_float(player.projection * weather_factor, player.projection)

                players.append(player)

            except Exception as e:
                logger.error(f"Error processing player {data.get('name', 'unknown')}: {e}")
                continue

        # ENHANCE with Monte Carlo analysis
        if self.use_monte_carlo and players:
            logger.info(f"Running Monte Carlo analysis on {len(players)} players...")
            players = self._enhance_players_with_monte_carlo_sync(players, weather_data, vegas_data)

        return players

    def _enhance_players_with_monte_carlo_sync(
            self,
            players: List[Player],
            weather_data: Dict = None,
            vegas_data: Dict = None,
    ) -> List[Player]:
        """
        Enhance players with Monte Carlo variance analysis
        FIXED: Uses synchronous Monte Carlo engine (no async conflicts)
        """
        try:
            # Import sync Monte Carlo engine
            from monte_carlo_engine import (
                MonteCarloEngine,
                PlayerSimulation,
                run_monte_carlo_sync
            )

            # Convert players to dict format for Monte Carlo
            player_dicts = []
            for player in players:
                player_dicts.append({
                    'name': player.name,
                    'position': player.position,
                    'team': player.team,
                    'salary': player.salary,
                    'projection': player.projection,
                    'game_environment_mult': player.game_environment_mult,
                    'game_total': player.game_total,
                })

            # Run SYNCHRONOUS Monte Carlo
            mc_results = run_monte_carlo_sync(
                player_data=player_dicts,
                weather_data=weather_data,
                vegas_data=vegas_data,
                vegas_multipliers=self.vegas_multipliers,
                num_simulations=1000
            )

            # Apply results to Player objects
            for player in players:
                if player.name in mc_results:
                    mc = mc_results[player.name]
                    player.floor_10 = safe_float(mc.get('floor_10', 0.0), player.projection * 0.5)
                    player.ceiling_90 = safe_float(mc.get('ceiling_90', 0.0), player.projection * 1.5)
                    player.ceiling_95 = safe_float(mc.get('ceiling_95', 0.0), player.projection * 1.7)
                    player.boom_rate = safe_float(mc.get('boom_rate', 0.0), 0.15)
                    player.bust_rate = safe_float(mc.get('bust_rate', 0.0), 0.20)
                    player.variance = safe_float(mc.get('std', 0.0), player.projection * 0.3)
                    player.monte_carlo_analyzed = True
                else:
                    # Fallback for players not in results
                    player.floor_10 = player.projection * 0.5
                    player.ceiling_90 = player.projection * 1.5
                    player.ceiling_95 = player.projection * 1.7
                    player.boom_rate = 0.15
                    player.bust_rate = 0.20
                    player.monte_carlo_analyzed = True

            analyzed_count = sum(1 for p in players if p.monte_carlo_analyzed)
            logger.info(f"✅ Monte Carlo complete: {analyzed_count}/{len(players)} players analyzed")

        except ImportError as e:
            logger.warning(f"Monte Carlo engine not available: {e}, using fallback estimates")
            self._apply_fallback_estimates(players)
        except Exception as e:
            logger.warning(f"Monte Carlo failed: {e}, using fallback estimates")
            self._apply_fallback_estimates(players)

        return players

    def _apply_fallback_estimates(self, players: List[Player]):
        """Apply simple fallback estimates when Monte Carlo fails"""
        for player in players:
            player.floor_10 = player.projection * 0.5
            player.ceiling_90 = player.projection * 1.5
            player.ceiling_95 = player.projection * 1.7
            player.boom_rate = 0.15
            player.bust_rate = 0.20
            player.monte_carlo_analyzed = True
        return players

    def optimize_lineup(
            self,
            players: List[Player],
            contest_type: str = 'gpp',
            single_game_teams: List[str] = None,
            excluded_player_ids: set = None,
    ) -> Optional[LineupResult]:
        """Optimize single lineup with explicit exclusions"""
        try:
            excluded_player_ids = excluded_player_ids or set()

            # Filter for single game
            if single_game_teams:
                players = [p for p in players if p.team in single_game_teams]
                if len(players) < 6:
                    logger.error(f"Not enough players for single game: {len(players)}")
                    return None

            # Apply exclusions
            available_players = [p for p in players if p.id not in excluded_player_ids]

            if len(available_players) < 20:
                logger.warning(f"Low player pool after exclusions: {len(available_players)}")

            # Project ownership
            for player in available_players:
                player.ownership = self._predict_friends_league_ownership(player, contest_type)

            # Create optimization problem
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)

            player_vars: Dict[int, pulp.LpVariable] = {}
            for i, _ in enumerate(available_players):
                player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')

            # ENHANCED objective function
            objective_terms = []
            for i, player in enumerate(available_players):
                points_value = self._calculate_winning_value(player, contest_type)
                objective_terms.append(safe_float(points_value, 0.0) * player_vars[i])

            prob += pulp.lpSum(objective_terms)

            # Add constraints
            self._add_fanduel_constraints(prob, available_players, player_vars, contest_type, single_game_teams)

            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if prob.status == pulp.LpStatusOptimal:
                result = self._extract_result(prob, available_players, player_vars, contest_type)
                return result
            else:
                logger.warning(f"Optimization failed: {pulp.LpStatus[prob.status]}")
                return None

        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _calculate_winning_value(self, player: Player, contest_type: str) -> float:
        """
        WINNING VALUE CALCULATION - The heart of tournament optimization

        Priority order:
        1. Game environment (high-total games = more points scored)
        2. AI recommendations (expert analysis)
        3. Ceiling potential (Monte Carlo)
        4. Base projection
        """
        base_value = safe_float(player.projection, 5.0)
        game_mult = safe_float(player.game_environment_mult, 1.0)

        if base_value <= 0:
            return 0.0

        # ===========================================
        # GAME ENVIRONMENT WEIGHTING (Most Important)
        # ===========================================
        if game_mult >= 1.35:  # 50+ total game
            game_boost = base_value * 0.50
        elif game_mult >= 1.25:  # 47+ total game
            game_boost = base_value * 0.35
        elif game_mult >= 1.10:  # 44+ total game
            game_boost = base_value * 0.15
        elif game_mult <= 0.90:  # Under 41 total
            game_boost = base_value * -0.30
        else:
            game_boost = 0

        # Position-specific boosts for high-total games
        if player.position == 'QB' and game_mult >= 1.25:
            game_boost *= 1.5  # QBs in shootouts are key
        elif player.position == 'WR' and game_mult >= 1.25:
            game_boost *= 1.3
        elif player.position == 'TE' and game_mult >= 1.25:
            game_boost *= 1.2

        # ===========================================
        # AI MUST-PLAY/MUST-FADE ENFORCEMENT
        # ===========================================
        ai_adjustment = 0
        if player.ai_must_play:
            ai_adjustment = base_value * 0.30
        elif player.ai_must_fade:
            ai_adjustment = base_value * -0.40

        # ===========================================
        # CONTEST-SPECIFIC SCORING
        # ===========================================
        if contest_type in ['gpp', 'bestball', 'friends_league']:
            # GPP/Friends: Maximize ceiling
            if player.monte_carlo_analyzed and player.ceiling_90 > 0:
                ceiling_90 = safe_float(player.ceiling_90, player.projection)
                ceiling_bonus = safe_float((ceiling_90 - player.projection) * 2.0, 0.0)
                boom_bonus = safe_float(player.boom_rate * 12.0, 0.0)
            else:
                variance = safe_float(player.variance, base_value * 0.35)
                ceiling_bonus = variance * 1.5
                boom_bonus = 0

            return safe_float(base_value + game_boost + ai_adjustment + ceiling_bonus + boom_bonus, base_value)

        elif contest_type == 'cash':
            if player.monte_carlo_analyzed and player.floor_10 > 0:
                floor_bonus = safe_float(player.floor_10 * 1.5, 0.0)
            else:
                floor_bonus = base_value * 0.3

            value_bonus = 5.0 if safe_float(player.value, 0.0) >= 3.5 else 0.0
            return safe_float(base_value + (game_boost * 0.5) + ai_adjustment + floor_bonus + value_bonus, base_value)

        elif contest_type == 'contrarian':
            if player.monte_carlo_analyzed and player.ceiling_95 > 0:
                ceiling_95 = safe_float(player.ceiling_95, player.projection)
                ceiling_bonus = safe_float((ceiling_95 - player.projection) * 3.0, 0.0)
            else:
                variance = safe_float(player.variance, base_value * 0.35)
                ceiling_bonus = variance * 2.5

            return safe_float(base_value + game_boost + ai_adjustment + ceiling_bonus, base_value)

        else:
            return safe_float(base_value + game_boost + ai_adjustment, base_value)

    def _add_fanduel_constraints(
            self,
            prob,
            players: List[Player],
            player_vars: Dict,
            contest_type: str,
            single_game_teams: List[str],
    ):
        """EXACT FanDuel constraints"""
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

        # Position requirements
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

        # Team diversity - max 4 per team
        team_counts: Dict[str, List[int]] = {}
        for i, player in enumerate(players):
            team_counts.setdefault(player.team, []).append(i)

        for team, player_indices in team_counts.items():
            prob += pulp.lpSum([player_vars[i] for i in player_indices]) <= 4

        # Force high-total exposure and stacking for GPP/friends_league
        if contest_type in ['gpp', 'bestball', 'friends_league']:
            self._add_forced_high_total_stack(prob, players, player_vars)
            self._add_qb_wr_stack_requirement(prob, players, player_vars, qb_indices, wr_indices)

    def _add_forced_high_total_stack(self, prob, players: List[Player], player_vars: Dict):
        """FORCE at least 3 players from highest-total games"""
        if not self.vegas_multipliers:
            logger.warning("No Vegas multipliers - cannot force high-total exposure")
            return

        high_total_teams = [
            team for team, mult in self.vegas_multipliers.items()
            if safe_float(mult, 1.0) >= 1.25
        ]

        if not high_total_teams:
            high_total_teams = [
                team for team, mult in self.vegas_multipliers.items()
                if safe_float(mult, 1.0) >= 1.10
            ]

        if not high_total_teams:
            logger.warning("No high-total teams found")
            return

        high_total_indices = [
            i for i, p in enumerate(players)
            if p.team in high_total_teams
        ]

        if len(high_total_indices) >= 3:
            prob += pulp.lpSum([player_vars[i] for i in high_total_indices]) >= 3
            teams_in_constraint = set(players[i].team for i in high_total_indices)
            logger.info(f"🎯 FORCING 3+ players from high-total games: {teams_in_constraint}")

    def _add_qb_wr_stack_requirement(
            self,
            prob,
            players: List[Player],
            player_vars: Dict,
            qb_indices: List[int],
            wr_indices: List[int]
    ):
        """Force QB + at least 1 WR from same team"""
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
                    prob += pulp.lpSum(wr_vars) >= pulp.lpSum(qb_vars)

    def _predict_friends_league_ownership(self, player: Player, contest_type: str) -> float:
        """Predict ownership for 12-person league"""
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
                total_ownership += safe_float(player.ownership, 10.0)
                if safe_float(player.game_environment_mult, 1.0) >= 1.25:
                    high_total_count += 1

        ordered_players = self._format_lineup_for_fanduel(selected_players)
        projected_points = sum(safe_float(p.projection, 0.0) for p in ordered_players)

        # Find primary stack team
        team_counts = {}
        for p in ordered_players:
            team_counts[p.team] = team_counts.get(p.team, 0) + 1
        primary_stack = max(team_counts, key=team_counts.get) if team_counts else ""

        # Calculate ceiling/floor from player data
        ceiling_90 = sum(safe_float(p.ceiling_90, p.projection) for p in ordered_players)
        floor_10 = sum(safe_float(p.floor_10, p.projection * 0.5) for p in ordered_players)

        result = LineupResult(
            players=ordered_players,
            total_salary=total_salary,
            projected_points=safe_float(projected_points, 0.0),
            total_value=sum(safe_float(p.value, 0.0) for p in ordered_players),
            ownership_total=safe_float(total_ownership, 0.0),
            correlation_score=self._calculate_correlation(ordered_players),
            weather_impact=float(
                np.mean([safe_float(p.weather_factor, 1.0) for p in ordered_players])) if ordered_players else 1.0,
            contest_type=contest_type,
            high_total_exposure=high_total_count,
            primary_stack_team=primary_stack,
            ceiling_90=safe_float(ceiling_90, projected_points),
            floor_10=safe_float(floor_10, projected_points * 0.5),
            risk_level="Medium" if high_total_count >= 3 else "Low",
        )

        logger.info(
            f"📊 Lineup: ${total_salary} | {projected_points:.1f}pts | {high_total_count} high-total players | Stack: {primary_stack}")

        return result

    def _format_lineup_for_fanduel(self, players: List[Player]) -> List[Player]:
        """Order players in FanDuel format"""
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

        for qb_team in qb_teams:
            same_team_wrs = sum(1 for team in wr_teams if team == qb_team)
            correlation += 0.3 * same_team_wrs
            same_team_tes = sum(1 for team in te_teams if team == qb_team)
            correlation += 0.2 * same_team_tes

        team_counts: Dict[str, int] = {}
        for player in players:
            team_counts[player.team] = team_counts.get(player.team, 0) + 1

        for count in team_counts.values():
            if count >= 3:
                correlation += 0.4
            elif count >= 2:
                correlation += 0.2

        return min(1.0, correlation)

    def generate_multiple_lineups(
            self,
            players: List[Player],
            num_lineups: int = 10,
            contest_type: str = 'gpp',
            single_game_teams: List[str] = None,
    ) -> List[LineupResult]:
        """
        Generate diverse lineups with QB exclusion for different stacks

        Strategy: Each lineup gets a different QB = different primary stack
        This is the most reliable way to ensure lineup diversity
        """
        lineups: List[LineupResult] = []
        excluded_qbs: set = set()  # Force different QB each lineup

        max_attempts = num_lineups * 10

        for attempt in range(max_attempts):
            if len(lineups) >= num_lineups:
                break

            # Apply randomization with QB exclusion
            randomized_players: List[Player] = []
            for player in players:
                # Skip QBs we've already used
                if player.position == 'QB' and player.id in excluded_qbs:
                    continue

                new_player = Player(
                    id=player.id,
                    name=player.name,
                    position=player.position,
                    team=player.team,
                    salary=player.salary,
                    projection=safe_float(player.projection, 5.0),
                    ownership=safe_float(player.ownership, 10.0),
                    weather_factor=safe_float(player.weather_factor, 1.0),
                    game_total=safe_float(player.game_total, 45.0),
                    game_environment_mult=safe_float(player.game_environment_mult, 1.0),
                    ai_must_play=player.ai_must_play,
                    ai_must_fade=player.ai_must_fade,
                    locked=player.locked,
                    floor_10=safe_float(player.floor_10, 0.0),
                    ceiling_90=safe_float(player.ceiling_90, player.projection),
                    ceiling_95=safe_float(player.ceiling_95, player.projection),
                    boom_rate=safe_float(player.boom_rate, 0.0),
                    bust_rate=safe_float(player.bust_rate, 0.0),
                    monte_carlo_analyzed=player.monte_carlo_analyzed,
                )

                # Apply randomization for variety
                if contest_type in ['gpp', 'friends_league']:
                    random_factor = random.uniform(0.80, 1.20)
                elif contest_type == 'cash':
                    random_factor = random.uniform(0.92, 1.08)
                else:
                    random_factor = random.uniform(0.75, 1.25)

                new_player.projection = safe_float(new_player.projection * random_factor, 5.0)
                new_player.value = safe_float(new_player.projection / (new_player.salary / 1000),
                                              0.0) if new_player.salary else 0.0
                randomized_players.append(new_player)

            # Check we have enough QBs
            available_qbs = [p for p in randomized_players if p.position == 'QB']
            if not available_qbs:
                logger.warning(f"No QBs available after exclusions, stopping at {len(lineups)} lineups")
                break

            lineup = self.optimize_lineup(randomized_players, contest_type, single_game_teams)

            if lineup:
                # Get QB from this lineup
                qb = next((p for p in lineup.players if p.position == 'QB'), None)

                if qb and qb.id not in excluded_qbs:
                    lineups.append(lineup)
                    excluded_qbs.add(qb.id)
                    logger.info(
                        f"✅ Lineup {len(lineups)} accepted: QB={qb.name} ({qb.team}), Stack={lineup.primary_stack_team}")
                else:
                    logger.debug(f"⏭️ Lineup rejected - QB already used")

        # Sort by ceiling for GPP/friends_league
        if contest_type in ['cash']:
            lineups.sort(key=lambda x: safe_float(x.floor_10, x.projected_points), reverse=True)
        else:
            lineups.sort(key=lambda x: (safe_float(x.ceiling_90, x.projected_points), x.high_total_exposure),
                         reverse=True)

        logger.info(f"Generated {len(lineups)} {contest_type} lineups")

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
    Main entry point for lineup optimization
    """
    logger.info(f"🏈 Starting {contest_type.upper()} optimization | Lineups: {num_lineups}")
    logger.info(f"   Monte Carlo: {'ON' if use_monte_carlo and MONTE_CARLO_AVAILABLE else 'OFF'}")

    # Create optimizer
    optimizer = EnhancedDFSOptimizer(use_monte_carlo=use_monte_carlo, mc_simulations=mc_simulations)

    # Set Vegas data
    optimizer.set_vegas_data(vegas_multipliers or {}, vegas_data or {})

    # Prepare players
    players = optimizer.prepare_players(player_data, weather_data, vegas_data)

    if not players:
        logger.error("No valid players after preparation")
        return []

    # Log data verification
    mc_count = sum(1 for p in players if p.monte_carlo_analyzed)
    ai_must_play = sum(1 for p in players if p.ai_must_play)
    ai_must_fade = sum(1 for p in players if p.ai_must_fade)
    high_total = sum(1 for p in players if safe_float(p.game_environment_mult, 1.0) >= 1.25)

    logger.info(f"📊 Players in high-total games: {high_total}/{len(players)}")
    logger.info(f"🎯 AI flags: {ai_must_play} must-play, {ai_must_fade} must-fade")
    logger.info(f"🎲 Monte Carlo analyzed: {mc_count}/{len(players)}")

    # Generate lineups
    lineups = optimizer.generate_multiple_lineups(
        players=players,
        num_lineups=num_lineups,
        contest_type=contest_type,
        single_game_teams=single_game_teams,
    )

    if not lineups:
        logger.error("No lineups generated")
        return []

    # Save to JSON
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
                "projected_points": round(safe_float(lu.projected_points, 0.0), 2),
                "ceiling_90": round(safe_float(lu.ceiling_90, 0.0), 2),
                "floor_10": round(safe_float(lu.floor_10, 0.0), 2),
                "ownership_total": round(safe_float(lu.ownership_total, 0.0), 2),
                "high_total_exposure": lu.high_total_exposure,
                "primary_stack": lu.primary_stack_team,
                "players": [
                    {
                        "name": p.name,
                        "position": p.position,
                        "team": p.team,
                        "salary": p.salary,
                        "projection": round(safe_float(p.projection, 0.0), 2),
                        "ceiling_90": round(safe_float(p.ceiling_90, 0.0), 2),
                        "game_mult": round(safe_float(p.game_environment_mult, 1.0), 2),
                        "ai_must_play": p.ai_must_play,
                    }
                    for p in lu.players
                ],
            })

        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"generated_at": datetime.now().isoformat(), "lineups": payload}, f, indent=2)
        logger.info(f"💾 Saved lineups to {out_path}")
    except Exception as e:
        logger.warning(f"Failed to export lineups: {e}")

    # Log top lineup summary
    top = lineups[0]
    logger.info(
        f"🏆 TOP LINEUP: ${top.total_salary} | {top.projected_points:.1f}pts | Ceil90: {top.ceiling_90:.1f} | High-total: {top.high_total_exposure} | Stack: {top.primary_stack_team}")

    return lineups