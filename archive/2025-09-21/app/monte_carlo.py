# app/monte_carlo.py
import numpy as np

class MonteCarlo:
    def simulate_lineup(self, lineup, n=1000):
        """Monte Carlo simulate lineup score distribution to estimate variance."""
        totals = np.zeros(n)
        for player in lineup:
            mean = player.get('projection', 0) or 0
            sd = 0.5 * mean
            if sd <= 0:
                totals += mean
            else:
                samples = np.random.normal(mean, sd, n)
                samples[samples < 0] = 0
                totals += samples
        avg = float(np.mean(totals))
        std = float(np.std(totals))
        p75 = float(np.percentile(totals, 75))
        return {'mean': round(avg, 1), 'stddev': round(std, 1), 'p75': round(p75, 1)}
