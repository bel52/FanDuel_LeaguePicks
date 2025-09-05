#!/usr/bin/env python3
import os
from src import util, data_fetch, lineup_rules as rules

week = util.current_week()
logf = os.path.join("logs", f"late_swap_week{week}.log")
util.setup_logging(logf)
util.logger.info(f"=== Late Swap (Week {week}) ===")

# Executed lineup
exec_rows = util.read_csv(os.path.join("data","executed","fd_executed.csv"))
line = [r for r in exec_rows if str(r.get('Week'))==str(week)]
if not line:
    util.logger.error("No executed lineup to consider.")
    raise SystemExit(1)
line_by_slot = {r['Position']: r for r in line}
used_pairs = {(r['Player'], r['Team']) for r in line}
total_salary = sum(int(r['Salary']) for r in line)

# Mid-slate status (optional)
status = "BEHIND"
plan_path = os.path.join("data","weekly",f"2025_w{week:02d}","late_swap_plan.json")
try:
    import json
    if os.path.exists(plan_path):
        status = json.load(open(plan_path)).get("status", status)
except Exception as e:
    util.logger.warning(f"No mid-slate plan parsed: {e}")

players = data_fetch.load_projections()  # now includes KickSlot('EARLY'/'LATE') and KickRaw
name_team_to = {(p['Player'], p['Team']): p for p in players}

# Odds for adj score
odds = data_fetch.fetch_odds()
gi = {}
if odds:
    for g in odds:
        if g.get('HomeImplied')!='':
            gi[g['Home']] = gi.get(g['Home'], {}) | {'implied_total': float(g['HomeImplied'])}
        if g.get('AwayImplied')!='':
            gi[g['Away']] = gi.get(g['Away'], {}) | {'implied_total': float(g['AwayImplied'])}

def adj(p): return rules.adjusted_score(p, gi.get(p['Team']))

def is_late_not_started(row):
    # Use executed row -> look up original player entry (has KickSlot)
    p = name_team_to.get((row['Player'], row['Team']))
    if not p: return False
    return p.get('KickSlot') == 'LATE'

def suggest_for_slot(slot_name, row_out):
    # Only swap if that rostered player is in late games
    if not is_late_not_started(row_out):
        return None, "not_late"

    pos_out = row_out.get("Pos","")
    salary_out = int(row_out.get("Salary",0))

    # FLEX can be RB/WR/TE; others must match same Pos
    allowed_pos = [pos_out] if slot_name != "FLEX" else ["RB","WR","TE"]

    # Candidates: LATE only, same allowed pos, not already used
    cand_pool = [p for p in players
                 if p.get('KickSlot')=='LATE'
                 and p['Pos'] in allowed_pos
                 and (p['Player'], p['Team']) not in used_pairs]

    if not cand_pool:
        return None, "no_candidates"

    cand_pool.sort(key=adj, reverse=True)

    cap_left_if_remove = 60000 - (total_salary - salary_out)
    for c in cand_pool:
        if c['Salary'] <= cap_left_if_remove:
            if slot_name == "DST":
                qb_opp = line_by_slot["QB"]["Opp"]
                if c['Team'] == qb_opp:
                    continue
            return c, f"best_{'upside' if status=='BEHIND' else 'stable'}"
    return None, "cap_blocked"

suggestions = []
for slot, row in line_by_slot.items():
    if slot not in {"QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"}:
        continue
    s, reason = suggest_for_slot(slot, row)
    if s:
        suggestions.append((slot, row['Player'], s['Player'], reason, row['Salary'], s['Salary'], s['ProjFP'], s.get('KickRaw','')))

if not suggestions:
    util.logger.info("No late-swap suggestions available (late-only/slot/cap rules).")
    raise SystemExit(0)

for slot, out_name, in_name, why, out_sal, in_sal, in_proj, kick in suggestions:
    util.logger.info(f"[{slot}] OUT {out_name} (${out_sal}) -> IN {in_name} (${in_sal}) [{why}] proj~{in_proj}  kick={kick}")
