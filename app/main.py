import os
import logging
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse, JSONResponse

from . import data_ingestion, optimization, analysis, openai_utils
from .formatting import build_text_report
from .player_match import match_names_to_indices
from .kickoff_times import build_kickoff_map, auto_lock_started_players, save_last_lineup, load_last_lineup
from .schedule_providers import fetch_kickoffs_from_oddsapi_events, fetch_kickoffs_from_espn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

def _run_optimization(
    salary_cap: int,
    enforce_stack: bool,
    min_stack_receivers: int,
    lock_names: Optional[List[str]],
    ban_names: Optional[List[str]],
    auto_late_swap: bool = True
):
    players_df = data_ingestion.load_weekly_data()
    if players_df is None or players_df.empty:
        return {"error": "No data available for optimization"}

    # Manual locks/bans by name
    lock_idx_manual, nf_lock = match_names_to_indices(lock_names or [], players_df)
    ban_idx_manual,  nf_ban  = match_names_to_indices(ban_names or [], players_df)

    # Auto-lock via last saved lineup + kickoff map
    kickoff_map = build_kickoff_map(players_df) if auto_late_swap else {}
    last_lineup = load_last_lineup()
    auto_locked_names = auto_lock_started_players(last_lineup, kickoff_map)
    auto_idx, nf_auto = match_names_to_indices(auto_locked_names, players_df)

    lock_ids = sorted(set(lock_idx_manual) | set(auto_idx))
    ban_ids  = sorted(set(ban_idx_manual))

    lineup = optimization.optimize_lineup(
        players_df,
        salary_cap=salary_cap,
        enforce_stack=enforce_stack,
        min_stack_receivers=min_stack_receivers,
        lock_ids=lock_ids,
        ban_ids=ban_ids
    )
    if not lineup:
        return {"error": "No feasible lineup found with given constraints"}

    # Build response
    total_proj = 0.0
    total_salary = 0
    lineup_players = []
    for pid in lineup:
        r = players_df.loc[pid]
        total_proj += float(r["PROJ PTS"])
        total_salary += int(r["SALARY"])
        lineup_players.append({
            "name": str(r["PLAYER NAME"]),
            "position": str(r["POS"]),
            "team": str(r["TEAM"]),
            "opponent": str(r["OPP"]),
            "proj_points": float(r["PROJ PTS"]),
            "salary": int(r["SALARY"]),
            "proj_roster_pct_raw": str(r.get("PROJ ROSTER %","")),
            "own_pct": float(r.get("OWN_PCT")) if (r.get("OWN_PCT")==r.get("OWN_PCT")) else None
        })

    # Monte Carlo
    sim = analysis.MonteCarloSimulator(num_simulations=10000)
    pdata = {pid: {"projected_points": float(players_df.loc[pid, "PROJ PTS"]),
                   "historical_std_dev": max(float(players_df.loc[pid, "PROJ PTS"]) * 0.15, 1.0)}
             for pid in lineup}
    sim_results = sim.simulate_lineup_performance(lineup, pdata)

    # Cheap built-in analysis text (no API)
    prompt = "internal"
    ai_text = openai_utils.analyze_prompt_with_gpt(prompt)

    result = {
        "lineup": lineup_players,
        "total_projected_points": total_proj,
        "cap_usage": {"total_salary": total_salary, "remaining": max(0, salary_cap - total_salary)},
        "simulation": sim_results,
        "analysis": ai_text,
        "constraints": {
            "auto_locked": auto_locked_names,
            "locks": lock_names or [],
            "bans": ban_names or [],
            "not_found": list(set(nf_lock + nf_ban + nf_auto))
        }
    }

    # Persist last lineup (used for auto-lock on future calls)
    try:
        save_last_lineup(lineup_players)
    except Exception:
        pass

    return result

@app.get("/optimize")
def optimize_endpoint(
    salary_cap: int = Query(60000, ge=1000, le=100000),
    enforce_stack: bool = Query(False),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: Optional[List[str]] = Query(default=None),
    ban: Optional[List[str]] = Query(default=None),
    auto_late_swap: bool = Query(True)
):
    result = _run_optimization(salary_cap, enforce_stack, min_stack_receivers, lock, ban, auto_late_swap)
    if "error" in result:
        return JSONResponse(result, status_code=422)
    return result

@app.get("/optimize_text", response_class=PlainTextResponse)
def optimize_text_endpoint(
    salary_cap: int = Query(60000, ge=1000, le=100000),
    enforce_stack: bool = Query(False),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: Optional[List[str]] = Query(default=None),
    ban: Optional[List[str]] = Query(default=None),
    auto_late_swap: bool = Query(True),
    width: int = Query(100, ge=70, le=160)
):
    result = _run_optimization(salary_cap, enforce_stack, min_stack_receivers, lock, ban, auto_late_swap)
    if "error" in result:
        return PlainTextResponse(result["error"], status_code=422)
    return build_text_report(result, width=width)

@app.get("/schedule/providers")
def providers_status():
    tz = os.getenv("TIMEZONE", "America/New_York")
    odds = {}
    espn = {}
    try:
        odds = fetch_kickoffs_from_oddsapi_events(tz)
    except Exception:
        odds = {}
    try:
        espn = fetch_kickoffs_from_espn(10, tz)
    except Exception:
        espn = {}
    def sample(d):
        if not d: return None
        k = sorted(d.keys())[0]
        return {k: d[k]}
    return {
        "oddsapi_events": {"count": len(odds), "sample": sample(odds)},
        "espn": {"count": len(espn), "sample": sample(espn)}
    }

@app.get("/schedule")
def schedule_endpoint():
    players_df = data_ingestion.load_weekly_data()
    ko = build_kickoff_map(players_df if players_df is not None else None)
    tz = os.getenv("TIMEZONE", "America/New_York")
    last = load_last_lineup()
    auto_locked = auto_lock_started_players(last, ko)
    return {"timezone": tz, "kickoffs": ko, "auto_locked_from_last_lineup": auto_locked, "not_found": []}
