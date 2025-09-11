from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from typing import Dict, Any
import os

from src.data_fetch import load_all_players, fetch_odds, fetch_weather, fetch_injuries
from src.lineup_builder import build_lineup
from src.analysis import monte_carlo_simulation
from src.scheduler import start_scheduler

# Initialize FastAPI application with a descriptive title
app = FastAPI(title="FanDuel NFL DFS Optimizer (Fixed)")

# Start the background scheduler on application startup
@app.on_event("startup")
async def startup_event() -> None:
    # Only start scheduler if enabled
    start_scheduler()

@app.get("/health")
def health():
    """
    Basic health endpoint. Returns the list of fantasypros CSV files detected
    in the ``data/fantasypros`` folder and a version string.
    """
    base_dir = "data/fantasypros"
    files = [os.path.join(base_dir, f) for f in ("qb.csv","rb.csv","wr.csv","te.csv","dst.csv")]
    present = [f for f in files if os.path.exists(f)]
    return {"status": "ok", "version": "3.0-fix1", "data_files": present}

@app.get("/data/status")
def data_status():
    """
    Returns counts of loaded players by position.
    """
    players = load_all_players()
    pos_counts: Dict[str,int] = {}
    for p in players:
        pos_counts[p['Pos']] = pos_counts.get(p['Pos'], 0) + 1
    return {"players": len(players), "by_pos": pos_counts}

def _format_lineup_text(lineup: Dict[str,Any], score: float) -> str:
    """
    Produce a human-readable text representation of a lineup with salary and projected points.
    """
    if not lineup:
        return "No valid lineup could be built. Check your CSVs."
    def row(label: str, p: Dict[str, Any]) -> str:
        return f"{label:>4}  {p['Name']:<24} {p['Team']:<3} {p['Pos']:<3}  ${p['Salary']:>5}  Proj:{p['ProjFP']:>5.1f}"
    ordered_keys = ["QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"]
    lines = []
    total_salary = 0
    for k in ordered_keys:
        p = lineup[k]
        lines.append(row(k, p))
        total_salary += int(p.get('Salary', 0) or 0)
    lines.append("-"*60)
    lines.append(f"Total Salary: ${total_salary}   Score: {score:.2f}")
    return "\n".join(lines)

@app.get("/optimize")
def optimize():
    """
    Build the best lineup given the current fantasypros CSVs and return JSON.
    """
    players = load_all_players()
    # Fetch supplemental data if enabled
    odds_data = fetch_odds()
    weather_data = fetch_weather()
    injuries = fetch_injuries()
    lineup, score = build_lineup(players, odds_data=odds_data, weather_data=weather_data, injuries=injuries)
    if not lineup:
        return {"ok": False, "error": "No lineup built. Ensure CSVs exist in data/fantasypros."}
    out = {k: {kk: vv for kk, vv in v.items()} for k, v in lineup.items()}
    total_salary = sum(int(v.get('Salary',0) or 0) for v in lineup.values())
    return {"ok": True, "score": score, "salary": total_salary, "lineup": out}

@app.get("/optimize_text", response_class=PlainTextResponse)
def optimize_text(width: int = 80):
    """
    Build the best lineup and return a plain-text table. ``width`` is unused but accepted for compatibility.
    """
    players = load_all_players()
    odds_data = fetch_odds()
    weather_data = fetch_weather()
    injuries = fetch_injuries()
    lineup, score = build_lineup(players, odds_data=odds_data, weather_data=weather_data, injuries=injuries)
    text = _format_lineup_text(lineup, score)
    return PlainTextResponse(text)


# Additional endpoint for Monte Carlo analysis
@app.get("/analysis/monte_carlo")
def monte_carlo(iters: int = 500):
    """
    Run a Monte Carlo simulation of lineup scores. Returns summary
    statistics including mean, median, 75th and 90th percentiles and
    maximum score. ``iters`` controls the number of simulations.
    """
    iters = max(10, min(iters, 2000))  # Clamp iterations between 10 and 2000
    players = load_all_players()
    odds_data = fetch_odds()
    weather_data = fetch_weather()
    stats = monte_carlo_simulation(players, iters=iters, odds_data=odds_data, weather_data=weather_data)
    if not stats:
        return {"ok": False, "error": "Unable to compute Monte Carlo stats. No valid lineups."}
    return {"ok": True, "iters": iters, "stats": stats}

@app.get("/schedule/providers")
def schedule_providers():
    """Return available schedule providers. Kept minimal for compatibility."""
    return {"providers": ["local"]}

@app.get("/schedule")
def schedule_list():
    """Return an empty job list for compatibility."""
    return {"jobs": []}
