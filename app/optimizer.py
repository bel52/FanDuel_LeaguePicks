# app/optimizer.py
import pulp

class Optimizer:
    def __init__(self, salary_cap=60000):
        self.salary_cap = salary_cap

    def generate_lineups(self, players, team_game_info, num_lineups=1, locked_players=None, game_type='gpp'):
        prob = pulp.LpProblem('DFS_Lineup', pulp.LpMaximize)
        x = pulp.LpVariable.dicts('x', range(len(players)), lowBound=0, upBound=1, cat='Binary')
        qb_idx = [i for i,p in enumerate(players) if p['position'] == 'QB']
        rb_idx = [i for i,p in enumerate(players) if p['position'] == 'RB']
        wr_idx = [i for i,p in enumerate(players) if p['position'] == 'WR']
        te_idx = [i for i,p in enumerate(players) if p['position'] == 'TE']
        dst_idx = [i for i,p in enumerate(players) if p['position'] == 'DST']
        locked_idx = []
        remaining_positions = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'FLEX': 1, 'DST': 1}
        remaining_salary_cap = self.salary_cap
        if locked_players:
            for lp in locked_players:
                for i,p in enumerate(players):
                    if p['id'] == lp['id']:
                        locked_idx.append(i)
                        pos = p['position']
                        if pos in ['RB','WR','TE']:
                            if remaining_positions.get(pos, 0) == 0 and remaining_positions['FLEX'] > 0:
                                remaining_positions['FLEX'] -= 1
                            else:
                                remaining_positions[pos] = max(0, remaining_positions[pos] - 1)
                        elif pos in ['QB','DST']:
                            remaining_positions[pos] = max(0, remaining_positions[pos] - 1)
                        remaining_salary_cap -= p['salary']
                        break
        # Objective: maximize total projected points
        prob += pulp.lpSum([players[i]['projection'] * x[i] for i in range(len(players)) if i not in locked_idx])
        # Roster constraints
        prob += pulp.lpSum([x[i] for i in qb_idx if i not in locked_idx]) == remaining_positions['QB']
        prob += pulp.lpSum([x[i] for i in dst_idx if i not in locked_idx]) == remaining_positions['DST']
        prob += pulp.lpSum([x[i] for i in rb_idx if i not in locked_idx]) >= remaining_positions['RB']
        prob += pulp.lpSum([x[i] for i in wr_idx if i not in locked_idx]) >= remaining_positions['WR']
        prob += pulp.lpSum([x[i] for i in te_idx if i not in locked_idx]) >= remaining_positions['TE']
        total_rwt = remaining_positions['RB'] + remaining_positions['WR'] + remaining_positions['TE'] + remaining_positions['FLEX']
        prob += pulp.lpSum([x[i] for i in rb_idx + wr_idx + te_idx if i not in locked_idx]) == total_rwt
        prob += pulp.lpSum([players[i]['salary'] * x[i] for i in range(len(players)) if i not in locked_idx]) <= remaining_salary_cap
        if game_type.lower() == 'gpp':
            # QB stack: if QB selected, at least one same-team WR/TE selected
            for i in qb_idx:
                if i in locked_idx: 
                    continue
                team = players[i]['team']
                receiver_indices = [j for j,p in enumerate(players) if p['team'] == team and p['position'] in ['WR','TE'] and j not in locked_idx]
                if receiver_indices:
                    prob += x[i] <= pulp.lpSum([x[j] for j in receiver_indices])
            # Avoid offensive player against own DST
            for i in dst_idx:
                if i in locked_idx:
                    continue
                team = players[i]['team']
                opp_team = team_game_info.get(team, {}).get('opponent')
                if opp_team:
                    opp_offense = [j for j,p in enumerate(players) if p['team'] == opp_team and p['position'] in ['QB','RB','WR','TE'] and j not in locked_idx]
                    for j in opp_offense:
                        prob += x[i] + x[j] <= 1
        solutions = []
        for lineup_num in range(num_lineups):
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            if pulp.LpStatus[prob.status] != 'Optimal':
                break
            chosen_idx = [i for i in range(len(players)) if i not in locked_idx and pulp.value(x[i]) == 1]
            lineup = locked_players.copy() if locked_players else []
            for idx in chosen_idx:
                lineup.append(players[idx])
            solutions.append(sorted(lineup, key=lambda p: p['position']))
            if lineup_num < num_lineups - 1:
                prob += pulp.lpSum([x[i] for i in chosen_idx]) <= total_rwt - 1
        return solutions
