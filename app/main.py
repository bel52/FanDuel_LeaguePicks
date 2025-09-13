# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import logging

from app.weekly_scheduler import data_collector, optimizer, ai_client, monte_carlo, schedule_jobs
from apscheduler.schedulers.background import BackgroundScheduler

app = FastAPI(title="FanDuel NFL DFS Optimizer")

# Start scheduled weekly tasks on startup
scheduler = BackgroundScheduler()
schedule_jobs(scheduler)
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/optimize")
def optimize_endpoint(game_type: str = "gpp", num_lineups: int = 1):
    data = data_collector.collect_weekly_data()
    if not data:
        raise HTTPException(status_code=500, detail="Failed to retrieve data for optimization")
    players = data['players']
    team_game_info = data['team_game_info']
    lineup_results = optimizer.generate_lineups(players, team_game_info, num_lineups=num_lineups, locked_players=None, game_type=game_type)
    output = []
    for lineup in lineup_results:
        metrics = monte_carlo.simulate_lineup(lineup)
        output.append({
            "players": [{"position": p['position'], "name": p['name'], "team": p['team'], "projection": p['projection'], "salary": p['salary']} for p in lineup],
            "metrics": {"proj": metrics['mean'], "stdev": metrics['stddev'], "p75": metrics['p75']}
        })
    return {"lineups": output}

@app.get("/optimize/text", response_class=PlainTextResponse)
def optimize_text_endpoint(game_type: str = "gpp", num_lineups: int = 1):
    data = data_collector.collect_weekly_data()
    if not data:
        return "Error: failed to retrieve data.\n"
    players = data['players']
    team_game_info = data['team_game_info']
    lineup_results = optimizer.generate_lineups(players, team_game_info, num_lineups=num_lineups, locked_players=None, game_type=game_type)
    if not lineup_results:
        return "No lineup generated.\n"
    lines = []
    for idx, lineup in enumerate(lineup_results, start=1):
        lines.append(f"Lineup {idx}:")
        for p in lineup:
            lines.append(f"{p['position']}: {p['name']} ({p['team']}) - $${p['salary']} proj {p['projection']}")
        metrics = monte_carlo.simulate_lineup(lineup)
        lines.append(f"Projected total: {metrics['mean']} ± {metrics['stddev']} (75th %ile: {metrics['p75']})")
        lines.append("")
    text_output = "\n".join(lines)
    if num_lineups == 1:
        analysis = ai_client.analyze_lineup(lineup_results[0], team_game_info)
        text_output += f"Analysis: {analysis}\n"
    return text_output
