#!/usr/bin/env python3
import os
from src import util, data_fetch, lineup_builder

week = util.current_week()
logf = os.path.join("logs", f"deep_build_week{week}.log")
util.setup_logging(logf)
util.logger.info(f"=== Deep Build (Week {week}) ===")

players = data_fetch.load_projections()
if not players:
    util.logger.error("No FantasyPros CSVs found in data/fantasypros/. Aborting.")
    raise SystemExit(1)

odds = data_fetch.fetch_odds()
weather = {}

lineup, score = lineup_builder.build_lineup(players, odds_data=odds, weather_data=weather)
if not lineup:
    util.logger.error("Failed to build a valid lineup.")
    raise SystemExit(1)

total_salary = sum(p['Salary'] for p in lineup.values())
util.logger.info(f"Lineup (AdjScore={score}, Salary=${total_salary}):")
for pos,p in lineup.items():
    util.logger.info(f"  {pos}: {p['Player']} ({p['Team']}) ${p['Salary']} Proj {p['ProjFP']:.1f}")

# Save target lineup with Pos + Kick
fields = ["Week","Position","Player","Team","Opp","Pos","Kick","Salary","ProjFP"]
rows = []
for pos,p in lineup.items():
    rows.append({
        "Week":week,"Position":pos,"Player":p['Player'],"Team":p['Team'],
        "Opp":p['Opp'],"Pos":p['Pos'],"Kick":p.get('KickRaw',''),
        "Salary":p['Salary'],"ProjFP":f"{p['ProjFP']:.1f}"
    })
util.write_csv(os.path.join("data","targets","fd_target.csv"), fields, rows, mode='w')

# Board
if odds:
    board_dir = os.path.join("data","weekly",f"2025_w{week:02d}")
    os.makedirs(board_dir, exist_ok=True)
    util.write_csv(os.path.join(board_dir,"board.csv"), list(odds[0].keys()), odds, mode='w')
