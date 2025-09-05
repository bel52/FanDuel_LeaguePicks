#!/usr/bin/env python3
import os
from src import util, data_fetch, lineup_builder

week = util.current_week()
logf = os.path.join("logs", f"process_inactives_week{week}.log")
util.setup_logging(logf)
util.logger.info(f"=== Inactives Processing (Week {week}) ===")

target_rows = util.read_csv(os.path.join("data","targets","fd_target.csv"))
target = {r['Position']: r for r in target_rows if str(r.get('Week'))==str(week)}
if not target:
    util.logger.error("No fd_target.csv for this week.")
    raise SystemExit(1)

players = data_fetch.load_projections()
# OUT = players with ProjFP==0 in current loader
out_pos = []
for pos,row in target.items():
    match = next((p for p in players if p['Player']==row['Player'] and p['Team']==row['Team']), None)
    if match and float(match.get('ProjFP',0)) == 0.0:
        out_pos.append(pos)

fields = ["Week","Position","Player","Team","Opp","Pos","Kick","Salary","ProjFP"]

if not out_pos:
    util.logger.info("No inactives detected in current lineup.")
    util.write_csv(os.path.join("data","executed","fd_executed.csv"), fields, list(target.values()), mode='w')
    raise SystemExit(0)

util.logger.info(f"Detected OUT positions: {out_pos}. Rebuilding without them.")
ban = {target[p]['Player'] for p in out_pos}
pool = [p for p in players if p['Player'] not in ban]
lineup, score = lineup_builder.build_lineup(pool)
if not lineup:
    util.logger.error("Rebuild failed after removing inactives.")
    raise SystemExit(1)

rows = [{"Week":week,"Position":pos,"Player":pl['Player'],"Team":pl['Team'],"Opp":pl['Opp'],
         "Pos":pl['Pos'],"Kick":pl.get('KickRaw',''),"Salary":pl['Salary'],"ProjFP":f"{pl['ProjFP']:.1f}"} for pos,pl in lineup.items()]
util.write_csv(os.path.join("data","executed","fd_executed.csv"), fields, rows, mode='w')
