import itertools
from src import lineup_rules as rules

CAP = 60000

# Search budgets (kept small for speed; can increase)
QB_TOP   = 10
MATE_TOP = 5
RB_TOP   = 12
WR_TOP   = 18
TE_TOP   = 10
DST_TOP  = 10
FLEX_TOP = 20

def build_lineup(players, odds_data=None, weather_data=None):
    """
    Greedy-but-broadened search:
      - Try top N QBs by adjusted score
      - For each, try up to M single-stack mates (WR or TE, same team)
      - Try RB pairs, WR sets, TE (if not stacked), DST
      - Fill FLEX from remaining best that fit under CAP
    Enforces:
      - Single stack (QB + 1 WR/TE)
      - ≤2 non-QB teammates (including the stack mate)
      - Avoid DST directly opposing the QB
    """
    # Filter by value thresholds
    pool = [p for p in players if rules.meets_value(p)]
    QBs  = [p for p in pool if p['Pos']=='QB']
    RBs  = [p for p in pool if p['Pos']=='RB']
    WRs  = [p for p in pool if p['Pos']=='WR']
    TEs  = [p for p in pool if p['Pos']=='TE']
    DSTs = [p for p in pool if p['Pos']=='DST']

    # Implied totals / weather map (team -> info)
    gi = {}
    if odds_data:
        for g in odds_data:
            if g.get('HomeImplied')!='':
                gi[g['Home']] = gi.get(g['Home'], {}) | {'implied_total': float(g['HomeImplied'])}
            if g.get('AwayImplied')!='':
                gi[g['Away']] = gi.get(g['Away'], {}) | {'implied_total': float(g['AwayImplied'])}
    # Weather could be merged similarly if provided.

    def adj(p):
        # defensive: skip any accidental non-player dicts
        if not isinstance(p, dict) or 'Team' not in p:
            return 0.0
        return rules.adjusted_score(p, gi.get(p['Team']))

    # Sort by adjusted score
    QBs.sort(key=adj, reverse=True)
    RBs.sort(key=adj, reverse=True)
    WRs.sort(key=adj, reverse=True)
    TEs.sort(key=adj, reverse=True)
    DSTs.sort(key=adj, reverse=True)

    if not (QBs and len(RBs)>=2 and len(WRs)>=3 and TEs and DSTs):
        return None, -1.0

    # Limit search sets
    QBs  = QBs[:QB_TOP]
    RBsT = RBs[:RB_TOP]
    WRT  = WRs[:WR_TOP]
    TET  = TEs[:TE_TOP]
    DSTT = DSTs[:DST_TOP]

    best = None
    best_score = -1.0

    for qb in QBs:
        # stack mates: WR/TE same team
        mates = [p for p in (WRT + TET) if p['Team']==qb['Team'] and p['Player']!=qb['Player']]
        mates.sort(key=adj, reverse=True)
        if not mates:
            continue

        for mate in mates[:MATE_TOP]:
            used_names = {qb['Player'], mate['Player']}
            salary = qb['Salary'] + mate['Salary']
            roster = {'QB': qb}
            wr_needed = 3
            if mate['Pos']=='WR':
                roster['WR1'] = mate
                wr_needed = 2
            else:
                roster['TE'] = mate

            # RB pair
            for rb1, rb2 in itertools.combinations([r for r in RBsT if r['Player'] not in used_names], 2):
                s_rb = rb1['Salary'] + rb2['Salary']
                if salary + s_rb > CAP: 
                    continue
                used_rb = used_names | {rb1['Player'], rb2['Player']}

                # WR fill
                wr_pool = [w for w in WRT if w['Player'] not in used_rb]
                for wrs in itertools.combinations(wr_pool, wr_needed):
                    s_wr = sum(w['Salary'] for w in wrs)
                    if salary + s_rb + s_wr > CAP:
                        continue
                    roster_wr = {}
                    if 'WR1' in roster:
                        roster_wr['WR2'], roster_wr['WR3'] = wrs
                    else:
                        roster_wr['WR1'], roster_wr['WR2'], roster_wr['WR3'] = wrs

                    used_wr = used_rb | {w['Player'] for w in wrs}

                    # TE if not already stacked
                    s_te = 0
                    roster_te = {}
                    if 'TE' not in roster:
                        te_options = [t for t in TET if t['Player'] not in used_wr]
                        te_pick = None
                        for te in te_options:
                            if salary + s_rb + s_wr + te['Salary'] <= CAP:
                                te_pick = te
                                s_te = te['Salary']
                                break
                        if not te_pick:
                            continue
                        roster_te['TE'] = te_pick

                    # DST (avoid DST vs QB opponent)
                    dst_opts = [d for d in DSTT if d['Player'] not in used_wr and d['Team'] != qb.get('Opp')]
                    dst_pick = None
                    s_dst = 0
                    for d in dst_opts:
                        if salary + s_rb + s_wr + s_te + d['Salary'] <= CAP:
                            dst_pick = d
                            s_dst = d['Salary']
                            break
                    if not dst_pick:
                        continue

                    # FLEX from remaining best RB/WR/TE
                    used = used_wr | ({roster_te['TE']['Player']} if 'TE' in roster_te else set())
                    used |= {dst_pick['Player']}
                    flex_pool = [p for p in (RBsT + WRT + TET) if p['Player'] not in used]
                    flex_pool.sort(key=adj, reverse=True)

                    flex_pick = None
                    for fx in flex_pool[:FLEX_TOP]:
                        total_sal = salary + s_rb + s_wr + s_te + s_dst + fx['Salary']
                        if total_sal <= CAP:
                            flex_pick = fx
                            break
                    if not flex_pick:
                        continue

                    # ≤2 non-QB teammates with QB (including the mate)
                    team_count = 0
                    group = [mate, rb1, rb2] + list(wrs) + ([roster_te['TE']] if 'TE' in roster_te else []) + [flex_pick]
                    for pl in group:
                        if pl['Team'] == qb['Team'] and pl['Pos'] != 'QB':
                            team_count += 1
                    if team_count > 2:
                        continue

                    # Build final lineup object (ONLY player dicts as values)
                    lineup = {'QB': qb, 'RB1': rb1, 'RB2': rb2, 'DST': dst_pick, 'FLEX': flex_pick}
                    lineup.update(roster_wr)
                    if 'WR1' in roster:  # stacked with WR
                        lineup['WR1'] = roster['WR1']
                    if 'TE' in roster_te:
                        lineup['TE'] = roster_te['TE']
                    elif 'TE' in roster:  # stacked with TE
                        lineup['TE'] = roster['TE']

                    # Score
                    total = sum(adj(p) for p in lineup.values())
                    if total > best_score:
                        best = lineup
                        best_score = round(total, 2)

    return best, best_score
