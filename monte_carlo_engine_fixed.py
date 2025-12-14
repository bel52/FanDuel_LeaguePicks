"""
Monte Carlo Variance Engine for Tournament-Winning DFS Optimization
FIXED: Synchronous execution to avoid FastAPI event loop conflicts
"""
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from loguru import logger
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
    variance_factor: float = 0.0
    floor_multiplier: float = 0.6
    ceiling_multiplier: float = 1.5
    game_script_correlation: float = 0.0
    weather_impact: float = 1.0
    injury_risk: float = 0.02
    game_total: float = 45.0  # Vegas game total
    game_environment_mult: float = 1.0  # Vegas multiplier

    def __post_init__(self):
        # Position-specific variance based on real DFS data
        variance_by_position = {
            'QB': 0.28,
            'RB': 0.35,
            'WR': 0.45,
            'TE': 0.38,
            'D': 0.42,
            'DEF': 0.42
        }
        if self.variance_factor == 0:
            self.variance_factor = variance_by_position.get(self.position, 0.35)


class MonteCarloEngine:
    """
    Monte Carlo simulation for DFS variance analysis
    FIXED: All methods are synchronous to work within FastAPI
    """

    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
        self.simulation_cache = {}

    def simulate_player(self, player: PlayerSimulation, num_sims: int = None) -> Dict[str, float]:
        """
        SYNCHRONOUS player simulation - no async conflicts
        Returns floor, ceiling, boom/bust rates for a single player
        """
        if num_sims is None:
            num_sims = self.num_simulations

        # Check cache
        cache_key = f"{player.name}_{player.base_projection}_{num_sims}"
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]

        scores = []
        base_proj = max(1.0, player.base_projection)  # Avoid zero projections
        
        # Position-specific variance (friends league = reduced variance)
        variance_mult = {
            'QB': 0.22, 'RB': 0.28, 'WR': 0.35, 'TE': 0.28, 'D': 0.35, 'DEF': 0.35
        }
        variance = base_proj * variance_mult.get(player.position, 0.30)

        # Game environment affects variance - high total games have more upside
        if player.game_environment_mult >= 1.25:
            variance *= 1.15  # More variance in shootouts
            
        for _ in range(num_sims):
            # Normal distribution centered on projection
            score = np.random.normal(base_proj, variance)
            score = max(0, score)

            # Game script factor (simplified)
            if player.game_environment_mult >= 1.25:
                # High-total game: slightly boost ceiling scenarios
                script_factor = np.random.choice([1.0, 1.05, 1.10], p=[0.5, 0.3, 0.2])
            else:
                script_factor = np.random.choice([0.95, 1.0, 1.05], p=[0.25, 0.5, 0.25])
            score *= script_factor

            # Weather impact
            if player.weather_impact < 1.0:
                weather_factor = np.random.uniform(player.weather_impact, 1.0)
                score *= weather_factor

            # Injury/zero game risk
            if np.random.random() < player.injury_risk:
                score = 0

            # Position floors (QBs rarely score under 8)
            if player.position == 'QB' and 0 < score < 8:
                score = max(score, np.random.uniform(8, 12))
            elif player.position in ['RB', 'WR', 'TE'] and 0 < score < 2:
                score = max(score, np.random.uniform(2, 4))

            scores.append(score)

        scores = np.array(scores)

        result = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'floor_10': float(np.percentile(scores, 10)),
            'floor_25': float(np.percentile(scores, 25)),
            'median': float(np.percentile(scores, 50)),
            'ceiling_75': float(np.percentile(scores, 75)),
            'ceiling_90': float(np.percentile(scores, 90)),
            'ceiling_95': float(np.percentile(scores, 95)),
            'max': float(np.max(scores)),
            'bust_rate': float(np.sum(scores < base_proj * 0.7) / len(scores)),
            'boom_rate': float(np.sum(scores > base_proj * 1.3) / len(scores)),
            'zero_rate': float(np.sum(scores == 0) / len(scores))
        }

        self.simulation_cache[cache_key] = result
        return result

    def simulate_all_players(self, players: List[PlayerSimulation]) -> Dict[str, Dict]:
        """
        Run Monte Carlo on all players synchronously
        Returns dict mapping player name -> simulation results
        """
        results = {}
        
        logger.info(f"🎲 Running Monte Carlo simulation on {len(players)} players...")
        
        for player in players:
            try:
                sim_result = self.simulate_player(player)
                results[player.name] = sim_result
            except Exception as e:
                logger.warning(f"Monte Carlo failed for {player.name}: {e}")
                # Fallback estimates
                results[player.name] = {
                    'mean': player.base_projection,
                    'std': player.base_projection * 0.3,
                    'floor_10': player.base_projection * 0.5,
                    'ceiling_90': player.base_projection * 1.5,
                    'ceiling_95': player.base_projection * 1.7,
                    'boom_rate': 0.15,
                    'bust_rate': 0.20,
                    'zero_rate': 0.02
                }
        
        logger.info(f"✅ Monte Carlo complete: {len(results)} players analyzed")
        return results

    def get_top_ceiling_players(self, results: Dict[str, Dict], position: str = None, n: int = 10) -> List[Dict]:
        """Get players with highest ceiling potential"""
        players = []
        for name, stats in results.items():
            players.append({
                'name': name,
                'ceiling_90': stats.get('ceiling_90', 0),
                'boom_rate': stats.get('boom_rate', 0),
                'mean': stats.get('mean', 0)
            })
        
        # Sort by ceiling
        players.sort(key=lambda x: x['ceiling_90'], reverse=True)
        return players[:n]

    def get_safest_players(self, results: Dict[str, Dict], n: int = 10) -> List[Dict]:
        """Get players with highest floor (safest plays)"""
        players = []
        for name, stats in results.items():
            players.append({
                'name': name,
                'floor_10': stats.get('floor_10', 0),
                'bust_rate': stats.get('bust_rate', 1),
                'mean': stats.get('mean', 0)
            })
        
        # Sort by floor (highest first) and bust rate (lowest first)
        players.sort(key=lambda x: (x['floor_10'], -x['bust_rate']), reverse=True)
        return players[:n]


def convert_player_data_to_simulation(
    player_data: List[Dict], 
    weather_data: Dict = None,
    vegas_data: Dict = None,
    vegas_multipliers: Dict = None
) -> List[PlayerSimulation]:
    """Convert standard player data to PlayerSimulation objects"""
    
    simulations = []
    
    for player in player_data:
        name = player.get('name', player.get('player_name', ''))
        team = player.get('team', '')
        position = player.get('position', '')
        
        # Normalize defense position
        if position in ['DST', 'D/ST', 'DEF']:
            position = 'D'
        
        # Get weather impact
        weather_impact = 1.0
        if weather_data and team in weather_data:
            weather_impact = weather_data[team].get('weather_factor', 
                           weather_data[team].get('factor', 1.0))
        
        # Get game environment from vegas
        game_total = player.get('game_total', 45.0)
        game_mult = player.get('game_environment_mult', 1.0)
        
        if vegas_multipliers and team in vegas_multipliers:
            game_mult = vegas_multipliers[team]
        
        sim_player = PlayerSimulation(
            name=name,
            position=position,
            team=team,
            salary=player.get('salary', 5000),
            base_projection=player.get('projection', player.get('projected_points', 5.0)),
            weather_impact=weather_impact,
            game_total=game_total,
            game_environment_mult=game_mult,
            injury_risk=0.02
        )
        
        simulations.append(sim_player)
    
    return simulations


def run_monte_carlo_sync(
    player_data: List[Dict],
    weather_data: Dict = None,
    vegas_data: Dict = None,
    vegas_multipliers: Dict = None,
    num_simulations: int = 1000
) -> Dict[str, Dict]:
    """
    MAIN ENTRY POINT: Run Monte Carlo synchronously
    Returns dict of player name -> simulation results
    """
    # Convert to simulation format
    sim_players = convert_player_data_to_simulation(
        player_data, weather_data, vegas_data, vegas_multipliers
    )
    
    # Run simulations
    engine = MonteCarloEngine(num_simulations=num_simulations)
    results = engine.simulate_all_players(sim_players)
    
    return results


def enhance_players_with_monte_carlo(
    players: List[Dict],
    weather_data: Dict = None,
    vegas_data: Dict = None,
    vegas_multipliers: Dict = None,
    num_simulations: int = 1000
) -> List[Dict]:
    """
    Enhance player dicts with Monte Carlo results
    Adds ceiling_90, floor_10, boom_rate, bust_rate to each player
    """
    # Run Monte Carlo
    mc_results = run_monte_carlo_sync(
        players, weather_data, vegas_data, vegas_multipliers, num_simulations
    )
    
    # Merge results back into player dicts
    for player in players:
        name = player.get('name', player.get('player_name', ''))
        if name in mc_results:
            mc = mc_results[name]
            player['ceiling_90'] = mc.get('ceiling_90', player.get('projection', 10) * 1.5)
            player['ceiling_95'] = mc.get('ceiling_95', player.get('projection', 10) * 1.7)
            player['floor_10'] = mc.get('floor_10', player.get('projection', 10) * 0.5)
            player['boom_rate'] = mc.get('boom_rate', 0.15)
            player['bust_rate'] = mc.get('bust_rate', 0.20)
            player['monte_carlo_analyzed'] = True
            player['mc_mean'] = mc.get('mean', player.get('projection', 10))
            player['mc_std'] = mc.get('std', 3.0)
    
    return players
