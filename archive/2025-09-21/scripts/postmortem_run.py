#!/usr/bin/env python3
import os, random
from src import util, postmortem

# Use last completed week
wk_now = util.current_week()
week = max(1, wk_now - 1)

logf = os.path.join("logs", f"postmortem_week{week}.log")
util.setup_logging(logf)
util.logger.info(f"=== Postmortem (Week {week}) ===")

rows = util.read_csv(os.path.join("data","executed","fd_executed.csv"))
line = [r for r in rows if str(r.get('Week'))==str(week)]
if not line:
    util.logger.error("No executed lineup found for week %s", week)
    raise SystemExit(1)

# Simulate actual points (placeholder until real stat ingestion is added)
total = 0.0
for r in line:
    proj = float(r.get('ProjFP',0))
    total += proj * random.uniform(0.8, 1.2)
util.logger.info(f"Bot points: {total:.1f}")

opp = total - random.uniform(-20,20)
util.logger.info(f"Opponent points (simulated): {opp:.1f}")

res = postmortem.log_week_result(week, total, opp)
util.logger.info(f"Result: {res['Result']} (Bot {res['BotPoints']} vs Opp {res['OppPoints']})")

# Simple bankroll sim
entry = 5.00
winnings = 10.00 if total >= 120 else 0.00
bk = postmortem.log_bankroll(entry, winnings)
util.logger.info(f"Bankroll entry: {bk}")
