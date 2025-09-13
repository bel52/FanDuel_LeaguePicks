# app/weekly_scheduler.py
import logging
from datetime import datetime, timezone

from app.data_collector import DataCollector
from app.optimizer import Optimizer
from app.ai_integration import AIIntegration
from app.monte_carlo import MonteCarlo
from app import exporter

# Initialize main components
data_collector = DataCollector()
optimizer = Optimizer()
ai_client = AIIntegration()
monte_carlo = MonteCarlo()

# Store last generated lineup for late swap reference
last_final_lineup = []

def run_initial_lineup():
    logging.info("Running initial lineup generation (Tuesday)...")
    data = data_collector.collect_weekly_data()
    if not data:
        logging.error("Initial lineup generation failed: no data.")
        return
    players = data['players']
    team_game_info = data['team_game_info']
    lineup_result = optimizer.generate_lineups(players, team_game_info, num_lineups=1, locked_players=None, game_type='gpp')
    lineup = lineup_result[0] if lineup_result else []
    logging.info(f"Initial lineup: {[p['name'] for p in lineup]}")
    if lineup:
        sim = monte_carlo.simulate_lineup(lineup)
        logging.info(f"Projected total {sim['mean']} ± {sim['stddev']} (75th %: {sim['p75']})")
    week = data.get('week') or datetime.now().strftime('%Y%m%d')
    exporter.export_lineups([lineup], f"initial_lineup_week{week}.csv")
    return lineup

def run_midweek_update():
    logging.info("Running midweek lineup update (Friday)...")
    data = data_collector.collect_weekly_data()
    if not data:
        logging.error("Midweek update failed: no data.")
        return
    players = data['players']
    team_game_info = data['team_game_info']
    lineup_result = optimizer.generate_lineups(players, team_game_info, num_lineups=1, locked_players=None, game_type='gpp')
    lineup = lineup_result[0] if lineup_result else []
    logging.info(f"Midweek updated lineup: {[p['name'] for p in lineup]}")
    if lineup:
        sim = monte_carlo.simulate_lineup(lineup)
        logging.info(f"Projected total {sim['mean']} ± {sim['stddev']} (75th %: {sim['p75']})")
    week = data.get('week') or datetime.now().strftime('%Y%m%d')
    exporter.export_lineups([lineup], f"midweek_lineup_week{week}.csv")
    return lineup

def run_final_lineup():
    logging.info("Running final Sunday AM lineup build...")
    data = data_collector.collect_weekly_data()
    if not data:
        logging.error("Final lineup build failed: no data.")
        return
    players = data['players']
    team_game_info = data['team_game_info']
    lineup_result = optimizer.generate_lineups(players, team_game_info, num_lineups=1, locked_players=None, game_type='gpp')
    lineup = lineup_result[0] if lineup_result else []
    logging.info(f"Final lineup: {[p['name'] for p in lineup]}")
    if lineup:
        sim = monte_carlo.simulate_lineup(lineup)
        logging.info(f"Projected total {sim['mean']} ± {sim['stddev']} (75th %: {sim['p75']})")
    week = data.get('week') or datetime.now().strftime('%Y%m%d')
    exporter.export_lineups([lineup], f"final_lineup_week{week}.csv")
    global last_final_lineup
    last_final_lineup = lineup
    return lineup

def run_late_swap_lineup():
    logging.info("Running late swap adjustments (Sunday mid-games)...")
    global last_final_lineup
    if not last_final_lineup:
        logging.warning("No final lineup stored for late swap.")
        return
    data = data_collector.collect_weekly_data()
    if not data:
        logging.error("Late swap failed: no data.")
        return
    players = data['players']
    team_game_info = data['team_game_info']
    now = datetime.now(timezone.utc)
    locked_players = []
    for p in last_final_lineup:
        start = team_game_info.get(p['team'], {}).get('start_time')
        if start and start <= now:
            locked_players.append(p)
    if not locked_players:
        logging.info("No players locked by game start; no late swap needed.")
        return last_final_lineup
    lineup_result = optimizer.generate_lineups(players, team_game_info, num_lineups=1, locked_players=locked_players, game_type='gpp')
    new_lineup = lineup_result[0] if lineup_result else last_final_lineup
    logging.info(f"Adjusted lineup: {[p['name'] for p in new_lineup]}")
    if new_lineup:
        sim = monte_carlo.simulate_lineup(new_lineup)
        logging.info(f"Projected total {sim['mean']} ± {sim['stddev']} (75th %: {sim['p75']})")
    week = data.get('week') or datetime.now().strftime('%Y%m%d')
    exporter.export_lineups([new_lineup], f"late_swap_lineup_week{week}.csv")
    return new_lineup

def schedule_jobs(scheduler):
    logging.info("Scheduling weekly lineup optimizer tasks...")
    try:
        scheduler.add_job(run_initial_lineup, trigger='cron', day_of_week='tue', hour=3, minute=0, timezone='America/New_York', id='initial_lineup')
        scheduler.add_job(run_midweek_update, trigger='cron', day_of_week='fri', hour=3, minute=0, timezone='America/New_York', id='midweek_update')
        scheduler.add_job(run_final_lineup, trigger='cron', day_of_week='sun', hour=11, minute=30, timezone='America/New_York', id='final_lineup')
        scheduler.add_job(run_late_swap_lineup, trigger='cron', day_of_week='sun', hour=16, minute=5, timezone='America/New_York', id='late_swap')
        logging.info("Weekly tasks scheduled (Tue/Fri/Sun).")
    except Exception as e:
        logging.error(f"Failed to schedule jobs: {e}")
