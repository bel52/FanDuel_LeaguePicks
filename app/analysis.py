import numpy as np

class MonteCarloSimulator:
    def __init__(self, num_simulations=50000):
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

def build_lineup_analysis_prompt(lineup_indices, players_df, sim_results):
    lines = []
    game_lines = []
    seen_games = set()
    total_salary = 0
    total_proj = 0.0

    for pid in lineup_indices:
        p = players_df.loc[pid]
        name = str(p.get('PLAYER NAME'))
        pos = str(p.get('POS'))
        team = str(p.get('TEAM'))
        opp = str(p.get('OPP')) if 'OPP' in p else ''
        opp_team = opp.replace('@','').replace('vs','').strip()
        proj = float(p.get('PROJ PTS', 0))
        sal = int(p.get('SALARY', 0))
        total_salary += sal
        total_proj += proj
        own = p.get('PROJ ROSTER %', '')
        own_str = f", Ownership ~{own}" if isinstance(own,str) and own else ""
        home_away = "at" if opp.startswith('@') else "vs"
        lines.append(f"{pos} {name} ({team}) {home_away} {opp_team} - ${sal:,}, Proj: {proj:.2f}{own_str}")
        key = tuple(sorted([team, opp_team])) if opp_team else None
        if key and key not in seen_games:
            seen_games.add(key)
            game_lines.append(f"{team} vs {opp_team}")

    sim = sim_results
    sim_txt = (f"Mean: {sim['mean_score']:.1f} | StdDev: {sim['std_dev']:.1f} | "
               f"P50: {sim['percentiles']['50th']:.1f} | P90: {sim['percentiles']['90th']:.1f} | "
               f"P95: {sim['percentiles']['95th']:.1f} | Sharpe: {sim['sharpe_ratio']:.3f}")

    # Instruct a fixed, console-friendly structure.
    constraints = (
        "Return plain text with these sections and nothing else:\n"
        "CORRELATION:\n"
        "- bullet points about stacks (QB-WR/TE) and any bring-backs.\n"
        "LEVERAGE:\n"
        "- bullets about chalk vs low-owned plays.\n"
        "RISK & CEILING:\n"
        "- bullets about volatility and upside.\n"
        "STRENGTHS:\n- top 2–4 strengths.\n"
        "WEAKNESSES:\n- top 2–4 weaknesses.\n"
        "SWAP IDEAS (max 2 legal FanDuel swaps):\n"
        "- each line: 'Out -> In (reason)'. Must preserve roster rules and cap.\n"
    )

    prompt = "You are an expert DFS analyst. Analyze this FanDuel lineup.\n"
    prompt += "LINEUP:\n" + "\n".join(lines) + "\n"
    if game_lines:
        prompt += "GAMES: " + "; ".join(game_lines) + "\n"
    prompt += f"SUMMARY: Salary Used ${total_salary:,}, Total Projection {total_proj:.1f}\n"
    prompt += "SIMULATION: " + sim_txt + "\n"
    prompt += (constraints +
               "Do not restate the entire roster; focus on the sections above.\n"
               "Keep bullets concise (1–2 lines each).")
    return prompt
