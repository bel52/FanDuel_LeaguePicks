#!/usr/bin/env python3
import os, requests, json
from src import util

week = util.current_week()
logf = os.path.join("logs", f"review_early_games_week{week}.log")
util.setup_logging(logf)
util.logger.info(f"=== Early Games Review (Week {week}) ===")

exec_rows = util.read_csv(os.path.join("data","executed","fd_executed.csv"))
line = [r for r in exec_rows if str(r.get('Week'))==str(week)]
if not line:
    util.logger.error("No executed lineup for this week.")
    raise SystemExit(1)

pts_so_far = 0.0  # placeholder until we parse per-player
proj_total = sum(float(r.get('ProjFP',0)) for r in line)

status = "EVEN"
if proj_total > 0:
    ratio = pts_so_far / proj_total
    if ratio >= 0.6: status = "AHEAD"
    elif ratio < 0.3: status = "BEHIND"

util.logger.info(f"Proj total: {proj_total:.1f}; Points so far: {pts_so_far:.1f}; Status: {status}")

plan_dir = os.path.join("data","weekly", f"2025_w{week:02d}")
os.makedirs(plan_dir, exist_ok=True)
with open(os.path.join(plan_dir, "late_swap_plan.json"), "w") as f:
    json.dump({"status": status}, f)
