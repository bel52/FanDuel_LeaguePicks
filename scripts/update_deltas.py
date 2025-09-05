#!/usr/bin/env python3
import os
from src import util, data_fetch, lineup_builder

week = util.current_week()
logf = os.path.join("logs", f"update_deltas_week{week}.log")
util.setup_logging(logf)
util.logger.info(f"=== Update Deltas (Week {week}) ===")

players = data_fetch.load_projections()
if not players:
    util.logger.error("No projections available; skipping.")
    raise SystemExit(0)

odds = data_fetch.fetch_odds()
weather = {}

prior = {}
try:
    rows = util.read_csv(os.path.join("data","targets","fd_target.csv"))
    for r in rows:
        if str(r.get("Week")) == str(week):
            prior[r['Position']] = r
except Exception as e:
    util.logger.warning(f"No prior target lineup: {e}")

lineup, score = lineup_builder.build_lineup(players, odds_data=odds, weather_data=weather)
if not lineup:
    util.logger.error("Rebuild produced no lineup.")
    raise SystemExit(0)

changes = []
for pos,p in lineup.items():
    old = prior.get(pos)
    if old and old.get('Player') != p['Player']:
        changes.append((pos, old.get('Player'), p['Player']))
if changes:
    util.logger.info("Changes since last target:")
    for pos,a,b in changes:
        util.logger.info(f"  {pos}: {a} -> {b}")
else:
    util.logger.info("No changes after deltas.")

fields = ["Week","Position","Player","Team","Opp","Pos","Kick","Salary","ProjFP"]
rows = [{
    "Week":week,"Position":pos,"Player":pl['Player'],"Team":pl['Team'],
    "Opp":pl['Opp'],"Pos":pl['Pos'],"Kick":pl.get('KickRaw',''),
    "Salary":pl['Salary'],"ProjFP":f"{pl['ProjFP']:.1f}"
} for pos,pl in lineup.items()]
util.write_csv(os.path.join("data","targets","fd_target.csv"), fields, rows, mode='w')
