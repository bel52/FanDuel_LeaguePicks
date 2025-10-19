"""
Monte Carlo Variance Engine for Tournament-Winning DFS Optimization
Simulates 10,000+ player performance scenarios to find true ceiling/floor/variance
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
from pathlib import Path


@dataclass
class PlayerSimulation:
    """Player simulation parameters with position-specific variance modeling"""
    name: str
    position: str
    team: str
    salary: int
    base_projection: float
    variance_factor: float
    floor_multiplier: float
    ceiling_multiplier: float
    game_script_correlation: float
    weather_impact: float
    injury_risk: float

    def __post_init__(self):
        # Position-specific variance based on real DFS data
        variance_by_position = {
            'QB': 0.28,  # QBs most consistent
            'RB': 0.35,  # RBs moderate variance
            'WR': 0.45,  # WRs highest variance
            'TE': 0.38,  # TEs moderate-high variance
            'D': 0.42  # Defenses high variance
        }

        if self.variance_factor == 0:
            self.variance_factor = variance_by_position.get(self.position, 0.35)


class MonteCarloEngine:
    """Advanced Monte Carlo simulation for DFS variance analysis"""

    def __init__(self, num_simulations: int = 10000, num_threads: int = 4):
        self.num_simulations = num_simulations
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

        # Cache for performance
        self.simulation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Advanced correlation matrices
        self.position_correlations = {
            ('QB', 'WR_same_team'): 0.65,
            ('QB', 'TE_same_team'): 0.45,
            ('RB', 'WR_same_team'): -0.15,  # Negative correlation
            ('QB', 'RB_same_team'): -0.25,  # QBs vs RBs
            ('WR', 'WR_same_team'): 0.20,  # Multiple WRs
            ('D', 'QB_opponent'): -0.35  # Defense vs opposing QB
        }

        # Game script correlations
        self.game_script_factors = {
            'blowout_positive': 1.2,  # Winning team gets more touches
            'blowout_negative': 0.8,  # Losing team in garbage time
            'close_game': 1.1,  # Close games = more action
            'weather_negative': 0.85  # Bad weather hurts all
        }

    async def simulate_player_performance(self, player: PlayerSimulation,
                                          num_sims: int = None) -> Dict[str, float]:
        """Simulate individual player performance with friends league calibration"""

        if num_sims is None:
            num_sims = min(1000, self.num_simulations // 10)

        # Check cache first
        cache_key = f"{player.name}_{player.base_projection}_{num_sims}"
        if cache_key in self.simulation_cache:
            self.cache_hits += 1
            return self.simulation_cache[cache_key]

        self.cache_misses += 1

        # Run simulation in thread pool for CPU-intensive work
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._simulate_player_scores_friends_league,  # CHANGED: Use friends league version
            player,
            num_sims
        )

        # Cache result
        self.simulation_cache[cache_key] = result
        return result

    def _simulate_player_scores_friends_league(self, player: PlayerSimulation, num_sims: int) -> Dict[str, float]:
        """FRIENDS LEAGUE: Reduced variance simulation for 12-person format"""

        scores = []
        base_proj = player.base_projection

        # REDUCED variance for friends league (not tournament variance)
        variance_multipliers = {
            'QB': 0.20,  # Reduced from 0.28
            'RB': 0.25,  # Reduced from 0.35
            'WR': 0.30,  # Reduced from 0.45
            'TE': 0.25,  # Reduced from 0.38
            'D': 0.30  # Reduced from 0.42
        }
        variance = base_proj * variance_multipliers.get(player.position, 0.25)

        for _ in range(num_sims):
            # Use normal distribution for friends league (not log-normal extremes)
            if player.position == 'D':
                score = np.random.normal(base_proj, variance)
            else:
                # Reduced variance normal distribution
                score = np.random.normal(base_proj, variance)
                score = max(0, score)  # No negative scores

            # REDUCED game script impact for friends league
            script_factor = self._sample_game_script_factor_friends(player)
            score *= script_factor

            # Apply weather impact (unchanged)
            if player.weather_impact < 1.0:
                weather_factor = np.random.uniform(player.weather_impact, 1.0)
                score *= weather_factor

            # REDUCED injury risk for friends league
            if np.random.random() < (player.injury_risk * 0.5):  # Half the injury risk
                score = 0

            # Position-specific floors (unchanged)
            if player.position == 'QB' and score < 8:
                score = max(score, np.random.uniform(8, 12))
            elif player.position in ['RB', 'WR', 'TE'] and score < 2:
                score = max(score, np.random.uniform(2, 5))

            scores.append(max(0, score))

        scores = np.array(scores)

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'floor_10': float(np.percentile(scores, 10)),
            'floor_25': float(np.percentile(scores, 25)),
            'median': float(np.percentile(scores, 50)),
            'ceiling_75': float(np.percentile(scores, 75)),  # Use 75th instead of 90th
            'ceiling_90': float(np.percentile(scores, 90)),  # Keep but de-emphasize
            'ceiling_95': float(np.percentile(scores, 95)),
            'max': float(np.max(scores)),
            'min': float(np.min(scores)),
            'bust_rate': float(np.sum(scores < base_proj * 0.7) / len(scores)),  # 70% threshold
            'boom_rate': float(np.sum(scores > base_proj * 1.25) / len(scores)),  # 1.25x threshold
            'zero_rate': float(np.sum(scores == 0) / len(scores))
        }

    def _sample_game_script_factor_friends(self, player: PlayerSimulation) -> float:
        """FRIENDS LEAGUE: Reduced game script impact"""

        correlation = player.game_script_correlation

        if correlation > 0.3:
            # Reduced impact for positive script
            factors = [1.0, 1.05, 1.10]  # Max 10% boost instead of 20%
            weights = [0.5, 0.3, 0.2]
        elif correlation < -0.3:
            # Reduced impact for negative script
            factors = [0.90, 0.95, 1.0]  # Max 10% penalty instead of 20%
            weights = [0.2, 0.3, 0.5]
        else:
            # Neutral (most common)
            factors = [0.95, 1.0, 1.05]
            weights = [0.25, 0.5, 0.25]

        return np.random.choice(factors, p=weights)

    async def simulate_lineup_performance(self, players: List[PlayerSimulation],
                                          include_correlations: bool = True) -> Dict[str, Any]:
        """Simulate complete lineup performance with player correlations"""

        # Get individual player simulations
        player_sims = {}
        tasks = []

        for player in players:
            task = self.simulate_player_performance(player)
            tasks.append(task)

        player_results = await asyncio.gather(*tasks)

        for i, player in enumerate(players):
            player_sims[player.name] = player_results[i]

        # Run correlated lineup simulation
        lineup_scores = await self._simulate_correlated_lineup(players, player_sims, include_correlations)

        return {
            'player_details': player_sims,
            'lineup_simulation': lineup_scores,
            'total_simulations': self.num_simulations,
            'cache_efficiency': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            }
        }

    async def _simulate_correlated_lineup(self, players: List[PlayerSimulation],
                                          player_sims: Dict, include_correlations: bool) -> Dict[str, float]:
        """Simulate lineup scores with player correlations"""

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._run_correlated_simulation,
            players,
            player_sims,
            include_correlations
        )

        return result

    def _run_correlated_simulation(self, players: List[PlayerSimulation],
                                   player_sims: Dict, include_correlations: bool) -> Dict[str, float]:
        """Core correlated lineup simulation"""

        lineup_scores = []
        num_players = len(players)

        for sim in range(self.num_simulations):
            lineup_score = 0
            player_performances = {}

            # Generate base performances for each player
            for player in players:
                sim_stats = player_sims[player.name]

                # Sample from the distribution we calculated
                mean = sim_stats['mean']
                std = sim_stats['std']

                # Use truncated normal to respect floor/ceiling
                score = np.random.normal(mean, std)
                score = max(sim_stats['min'], min(sim_stats['max'], score))

                player_performances[player.name] = score

            # Apply correlations if enabled
            if include_correlations:
                player_performances = self._apply_correlations(players, player_performances)

            # Sum lineup score
            lineup_score = sum(player_performances.values())
            lineup_scores.append(lineup_score)

        lineup_scores = np.array(lineup_scores)

        return {
            'mean': float(np.mean(lineup_scores)),
            'std': float(np.std(lineup_scores)),
            'floor_10': float(np.percentile(lineup_scores, 10)),
            'floor_25': float(np.percentile(lineup_scores, 25)),
            'median': float(np.percentile(lineup_scores, 50)),
            'ceiling_75': float(np.percentile(lineup_scores, 75)),
            'ceiling_90': float(np.percentile(lineup_scores, 90)),
            'ceiling_95': float(np.percentile(lineup_scores, 95)),
            'max': float(np.max(lineup_scores)),
            'min': float(np.min(lineup_scores)),
            'sharpe_ratio': float(np.mean(lineup_scores) / np.std(lineup_scores)) if np.std(lineup_scores) > 0 else 0,
            'downside_deviation': float(np.std(lineup_scores[lineup_scores < np.mean(lineup_scores)])),
            'upside_potential': float(np.percentile(lineup_scores, 90) - np.percentile(lineup_scores, 50))
        }

    def _apply_correlations(self, players: List[PlayerSimulation],
                            performances: Dict[str, float]) -> Dict[str, float]:
        """Apply realistic player correlations"""

        # Group players by team for correlation analysis
        team_players = {}
        for player in players:
            if player.team not in team_players:
                team_players[player.team] = []
            team_players[player.team].append(player)

        adjusted_performances = performances.copy()

        # Apply same-team correlations
        for team, team_roster in team_players.items():
            if len(team_roster) < 2:
                continue

            # Find QB-WR correlations (strongest)
            qb_players = [p for p in team_roster if p.position == 'QB']
            wr_players = [p for p in team_roster if p.position == 'WR']
            te_players = [p for p in team_roster if p.position == 'TE']

            for qb in qb_players:
                qb_performance = performances[qb.name]
                qb_multiplier = qb_performance / qb.base_projection

                # Correlate WRs with QB performance
                for wr in wr_players:
                    correlation = self.position_correlations.get(('QB', 'WR_same_team'), 0.65)
                    wr_base = performances[wr.name]
                    correlated_boost = (qb_multiplier - 1.0) * correlation
                    adjusted_performances[wr.name] = wr_base * (1.0 + correlated_boost)

                # Correlate TEs with QB performance
                for te in te_players:
                    correlation = self.position_correlations.get(('QB', 'TE_same_team'), 0.45)
                    te_base = performances[te.name]
                    correlated_boost = (qb_multiplier - 1.0) * correlation
                    adjusted_performances[te.name] = te_base * (1.0 + correlated_boost)

        return adjusted_performances

    async def tournament_simulation(self, lineups: List[List[PlayerSimulation]],
                                    field_size: int = 1000, entry_fee: float = 5.0,
                                    payout_structure: Dict = None) -> Dict[str, Any]:
        """Simulate tournament performance against field"""

        if not payout_structure:
            payout_structure = self._get_default_payout_structure(field_size)

        # Generate field performance distribution
        field_mean = 145  # Typical NFL DFS tournament average
        field_std = 25  # Standard deviation

        tournament_results = []

        for sim_round in range(min(100, self.num_simulations // 100)):  # Sample tournaments
            # Generate field scores
            field_scores = np.random.normal(field_mean, field_std, field_size)
            field_scores = np.maximum(field_scores, 0)  # No negative scores

            # Simulate your lineups
            your_scores = []

            for lineup in lineups:
                lineup_sim = await self.simulate_lineup_performance(lineup, include_correlations=True)
                # Sample one score from the distribution
                mean_score = lineup_sim['lineup_simulation']['mean']
                std_score = lineup_sim['lineup_simulation']['std']
                sampled_score = max(0, np.random.normal(mean_score, std_score))
                your_scores.append(sampled_score)

            # Combine and rank all scores
            all_scores = list(field_scores) + your_scores
            all_scores.sort(reverse=True)

            # Calculate payouts for your lineups
            round_payout = 0
            round_stats = {
                'min_cash': 0,
                'top_10': 0,
                'top_1': 0,
                'total_entries': len(your_scores)
            }

            for score in your_scores:
                rank = all_scores.index(score) + 1
                payout = self._calculate_payout(rank, field_size, payout_structure)
                round_payout += payout

                # Track placement stats
                if rank <= field_size * 0.20:  # Top 20% typically cash
                    round_stats['min_cash'] += 1
                if rank <= 10:
                    round_stats['top_10'] += 1
                if rank == 1:
                    round_stats['top_1'] += 1

            round_roi = (round_payout - (len(your_scores) * entry_fee)) / (len(your_scores) * entry_fee)

            tournament_results.append({
                'payout': round_payout,
                'roi': round_roi,
                'stats': round_stats,
                'your_scores': your_scores,
                'field_avg': np.mean(field_scores)
            })

        # Calculate aggregated results
        total_payouts = [r['payout'] for r in tournament_results]
        total_rois = [r['roi'] for r in tournament_results]

        # Calculate placement rates
        total_entries = sum(r['stats']['total_entries'] for r in tournament_results)
        total_cashes = sum(r['stats']['min_cash'] for r in tournament_results)
        total_top10 = sum(r['stats']['top_10'] for r in tournament_results)
        total_wins = sum(r['stats']['top_1'] for r in tournament_results)

        return {
            'expected_roi': float(np.mean(total_rois)),
            'roi_std': float(np.std(total_rois)),
            'min_cash_rate': float(total_cashes / total_entries) if total_entries > 0 else 0,
            'top_10_rate': float(total_top10 / total_entries) if total_entries > 0 else 0,
            'win_rate': float(total_wins / total_entries) if total_entries > 0 else 0,
            'profit_probability': float(np.sum(np.array(total_rois) > 0) / len(total_rois)),
            'break_even_probability': float(np.sum(np.array(total_rois) >= -0.05) / len(total_rois)),
            'tournament_simulations': len(tournament_results),
            'avg_payout': float(np.mean(total_payouts)),
            'max_payout': float(np.max(total_payouts)),
            'kelly_criterion': self._calculate_kelly_criterion(total_rois, entry_fee)
        }

    def _get_default_payout_structure(self, field_size: int) -> Dict[int, float]:
        """Generate realistic DFS payout structure"""

        # Typical DFS tournament payout (20% of field cashes)
        cash_line = int(field_size * 0.20)
        total_pool = field_size * 5.0  # $5 entry
        rake = 0.10  # 10% rake
        prize_pool = total_pool * (1 - rake)

        payouts = {}

        # Winner gets ~20% of prize pool
        payouts[1] = prize_pool * 0.20

        # Top 10 progressive payouts
        remaining_pool = prize_pool * 0.80

        # Simple payout distribution
        for rank in range(2, cash_line + 1):
            if rank <= 10:
                payouts[rank] = remaining_pool * 0.40 * (0.8 ** (rank - 2)) / 10
            elif rank <= cash_line // 4:
                payouts[rank] = remaining_pool * 0.35 / (cash_line // 4 - 10)
            else:
                payouts[rank] = remaining_pool * 0.25 / (cash_line - cash_line // 4)

        return payouts

    def _calculate_payout(self, rank: int, field_size: int, payout_structure: Dict) -> float:
        """Calculate payout for given rank"""
        return payout_structure.get(rank, 0.0)

    def _calculate_kelly_criterion(self, rois: List[float], entry_fee: float) -> float:
        """Calculate Kelly Criterion bet sizing"""

        wins = [roi for roi in rois if roi > 0]
        losses = [roi for roi in rois if roi <= 0]

        if not wins or not losses:
            return 0.0

        win_prob = len(wins) / len(rois)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return 0.0

        kelly_f = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_loss

        return max(0.0, min(0.25, kelly_f))  # Cap at 25% of bankroll

    def export_simulation_results(self, results: Dict, filename: str = None) -> str:
        """Export simulation results to JSON"""

        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monte_carlo_results_{timestamp}.json"

        filepath = Path("data/simulations") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_results = json.loads(json.dumps(results, default=convert_numpy))

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Exported simulation results to {filepath}")
        return str(filepath)

    def get_optimization_insights(self, simulation_results: Dict) -> Dict[str, Any]:
        """Extract optimization insights from simulation results"""

        lineup_sim = simulation_results.get('lineup_simulation', {})
        player_details = simulation_results.get('player_details', {})

        insights = {
            'risk_assessment': self._assess_lineup_risk(lineup_sim),
            'player_analysis': self._analyze_player_contributions(player_details),
            'correlation_strength': self._measure_correlation_impact(simulation_results),
            'optimization_recommendations': []
        }

        # Generate recommendations
        if lineup_sim.get('downside_deviation', 0) > lineup_sim.get('std', 0) * 0.7:
            insights['optimization_recommendations'].append("High downside risk - consider more floor plays")

        if lineup_sim.get('upside_potential', 0) < lineup_sim.get('mean', 0) * 0.3:
            insights['optimization_recommendations'].append("Low ceiling - consider more boom-bust players")

        if insights['correlation_strength'] < 0.2:
            insights['optimization_recommendations'].append("Low correlation - consider stacking opportunities")

        return insights

    def _assess_lineup_risk(self, lineup_sim: Dict) -> str:
        """Assess overall lineup risk level"""

        sharpe = lineup_sim.get('sharpe_ratio', 0)
        downside_dev = lineup_sim.get('downside_deviation', 0)
        std = lineup_sim.get('std', 1)

        downside_ratio = downside_dev / std if std > 0 else 1

        if sharpe > 0.8 and downside_ratio < 0.6:
            return "Low Risk"
        elif sharpe > 0.5 and downside_ratio < 0.8:
            return "Medium Risk"
        else:
            return "High Risk"

    def _analyze_player_contributions(self, player_details: Dict) -> Dict[str, Any]:
        """Analyze individual player contributions to lineup variance"""

        analysis = {
            'highest_variance': None,
            'most_consistent': None,
            'boom_bust_players': [],
            'safe_plays': []
        }

        for player_name, stats in player_details.items():
            variance_coef = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
            boom_rate = stats.get('boom_rate', 0)
            bust_rate = stats.get('bust_rate', 0)

            if boom_rate > 0.2 and bust_rate > 0.2:
                analysis['boom_bust_players'].append(player_name)
            elif variance_coef < 0.25:
                analysis['safe_plays'].append(player_name)

            # Track extremes
            if not analysis['highest_variance'] or variance_coef > player_details[analysis['highest_variance']]['std'] / \
                    player_details[analysis['highest_variance']]['mean']:
                analysis['highest_variance'] = player_name

            if not analysis['most_consistent'] or variance_coef < player_details[analysis['most_consistent']]['std'] / \
                    player_details[analysis['most_consistent']]['mean']:
                analysis['most_consistent'] = player_name

        return analysis

    def _measure_correlation_impact(self, simulation_results: Dict) -> float:
        """Measure the impact of correlations on lineup performance"""

        # This would require running simulation with and without correlations
        # For now, return a placeholder based on lineup composition
        return 0.3  # Default moderate correlation


# Integration functions for existing optimizer

def convert_player_data_to_simulation(player_data: List[Dict], weather_data: Dict = None,
                                      vegas_data: Dict = None) -> List[PlayerSimulation]:
    """Convert standard player data to PlayerSimulation objects"""

    simulations = []

    for player in player_data:
        # Extract weather impact
        weather_impact = 1.0
        if weather_data and player.get('team') in weather_data:
            weather_impact = weather_data[player['team']].get('factor', 1.0)

        # Estimate game script correlation based on Vegas data
        game_script_correlation = 0.0
        if vegas_data:
            # Simple heuristic based on game total and spread
            # More sophisticated logic could be added here
            pass

        # Create simulation object
        sim_player = PlayerSimulation(
            name=player.get('name', ''),
            position=player.get('position', ''),
            team=player.get('team', ''),
            salary=player.get('salary', 5000),
            base_projection=player.get('projected_points', 0),
            variance_factor=0,  # Will be set in __post_init__
            floor_multiplier=0.6,
            ceiling_multiplier=1.5,
            game_script_correlation=game_script_correlation,
            weather_impact=weather_impact,
            injury_risk=0.02  # 2% base injury risk
        )

        simulations.append(sim_player)

    return simulations


async def enhance_lineup_with_monte_carlo(lineup_players: List[Dict],
                                          weather_data: Dict = None,
                                          vegas_data: Dict = None,
                                          num_simulations: int = 5000) -> Dict[str, Any]:
    """Enhance lineup analysis with Monte Carlo simulation"""

    # Convert to simulation format
    sim_players = convert_player_data_to_simulation(lineup_players, weather_data, vegas_data)

    # Run simulation
    monte_carlo = MonteCarloEngine(num_simulations=num_simulations)

    simulation_results = await monte_carlo.simulate_lineup_performance(
        sim_players, include_correlations=True
    )

    # Get optimization insights
    insights = monte_carlo.get_optimization_insights(simulation_results)

    return {
        'simulation_results': simulation_results,
        'insights': insights,
        'risk_level': insights['risk_assessment'],
        'recommendations': insights['optimization_recommendations']
    }