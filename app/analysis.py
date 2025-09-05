import numpy as np

class MonteCarloSimulator:
    def __init__(self, num_simulations=10000):
        self.num_simulations = num_simulations

    def simulate_lineup_performance(self, lineup_indices, player_data):
        scores = np.zeros(self.num_simulations)
        for pid in lineup_indices:
            stats = player_data.get(pid, {})
            mu = float(stats.get('projected_points', 0.0))
            sigma = float(stats.get('historical_std_dev', 0.0))
            if sigma <= 0:
                sims = np.full(self.num_simulations, mu)
            else:
                sims = np.random.normal(mu, sigma, self.num_simulations)
                sims = np.maximum(sims, 0)
            scores += sims
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return {
            'mean_score': mean,
            'std_dev': std,
            'percentiles': {
                '50th': float(np.percentile(scores, 50)),
                '90th': float(np.percentile(scores, 90)),
                '95th': float(np.percentile(scores, 95)),
            },
            'sharpe_ratio': float(mean / std) if std != 0 else 0.0
        }
