"""
Analytical helpers for DFS lineup evaluation.

This module implements simple Monte Carlo simulations to estimate the
distribution of lineup scores under projection uncertainty. It uses
``numpy`` to generate normally distributed random variates around
each player's projected fantasy points. The standard deviation is
assumed to be a fraction of the projection (e.g. 30%). The results
include summary statistics such as mean, median and percentile values.
"""

from typing import List, Dict, Any, Tuple
import numpy as np  # type: ignore

from src.lineup_builder import build_lineup


def monte_carlo_simulation(players: List[Dict[str, Any]], iters: int = 500,
                           odds_data: List[Dict[str, Any]] | None = None,
                           weather_data: List[Dict[str, Any]] | None = None,
                           stdev_factor: float = 0.3) -> Dict[str, float]:
    """
    Run a Monte Carlo simulation to estimate lineup score distribution.

    Args:
        players: List of player dicts to simulate.
        iters: Number of iterations. More iterations provide smoother
            distributions but increase runtime.
        odds_data: Optional odds data passed through to ``build_lineup``.
        weather_data: Optional weather data passed through to
            ``build_lineup``.
        stdev_factor: Fraction of a player's projection used as the
            standard deviation in the normal distribution.

    Returns:
        Dictionary of summary statistics: mean, median, 75th percentile,
        90th percentile, and maximum lineup score across the simulations.
    """
    if not players:
        return {}
    scores: List[float] = []
    # Precompute base projections to avoid repeated dict lookups
    base_projs = [p.get('ProjFP', 0.0) for p in players]
    salary_list = [p.get('Salary', 0) for p in players]
    player_templates = [p.copy() for p in players]
    for _ in range(iters):
        # Randomize projections
        rand_projs = np.random.normal(loc=base_projs, scale=np.array(base_projs) * stdev_factor)
        # Build new player list with randomized projections
        sim_players: List[Dict[str, Any]] = []
        for idx, tpl in enumerate(player_templates):
            p_new = tpl.copy()
            p_new['ProjFP'] = max(0.0, rand_projs[idx])  # projections cannot be negative
            p_new['Salary'] = salary_list[idx]
            sim_players.append(p_new)
        lineup, score = build_lineup(sim_players, odds_data=odds_data, weather_data=weather_data)
        if lineup is not None:
            scores.append(score)
    if not scores:
        return {}
    arr = np.array(scores)
    return {
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'max': float(arr.max()),
    }
