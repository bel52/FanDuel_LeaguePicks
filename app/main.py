import os
import logging
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key = os.getenv('OPENAI_API_KEY')

from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from . import data_ingestion, optimization, analysis, openai_utils
from .formatting import build_text_report
from .player_match import match_names_to_indices
from .kickoff_times import build_kickoff_map, auto_lock_started_players, save_last_lineup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def check_openai_api_configuration():
    api_key = os.getenv('OPENAI_API_KEY')
    status = {'api_key_present': bool(api_key), 'connection_success': False}
    if not api_key:
        return status
    try:
        openai.Model.list()
        status['connection_success'] = True
        logging.info("OpenAI API connection successful.")
    except Exception as e:
        logging.error(f"OpenAI API connection failed: {e}")
    return status

openai_status = check_openai_api_configuration()
if not openai_status['api_key_present']:
    logging.warning("OPENAI_API_KEY not found. GPT analysis disabled.")
elif not openai_status['connection_success']:
    logging.warning("OpenAI not reachable. GPT analysis disabled.")

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

def _run_optimization(salary_cap: int, enforce_stack: bool, min_stack_receivers: int,
                      lock_names: list[str]|None, ban_names: list[str]|None, auto_late_swap: bool):
    players_df = data_ingestion.load_weekly_data()
    if players_df is None or players_df.empty:
        return {"error": "No data available for optimization"}

    # Manual lock/ban (optional)
    lock_idx_manual, lock_nf_manual = match_names_to_indices(lock_names or [], players_df)
    ban_idx_manual,  ban_nf_manual  = match_names_to_indices(ban_names or [],  players_df)

    # Auto-lock based on kickoff times from last saved lineup
    kickoff_map = build_kickoff_map(players_df) if auto_late_swap else {}
    lock_idx_auto, auto_locked_names, nf_auto = auto_lock_started_players(players_df, kickoff_map) if kickoff_map else ([], [], [])

    combined_lock_idx = list(set(lock_idx_manual + lock_idx_auto))
    combined_ban_idx  = list(set(ban_idx_manual))

    lineup = optimization.optimize_lineup(
        players_df,
        salary_cap=salary_cap,
        enforce_stack=enforce_stack,
        min_stack_receivers=min_stack_receivers,
        lock_indices=combined_lock_idx,
        ban_indices=combined_ban_idx
    )
    if not lineup:
        return {
            "error": "No feasible lineup found with given constraints",
            "constraints": {
                "locks": lock_names or [],
                "bans": ban_names or [],
                "auto_locked": auto_locked_names,
                "not_found": list(set(lock_nf_manual + ban_nf_manual + nf_auto))
            }
        }

    # Monte Carlo
    sim = analysis.MonteCarloSimulator(num_simulations=10000)
    pdata = {}
    for pid in lineup:
        proj = float(players_df.loc[pid, 'PROJ PTS'])
        pdata[pid] = {'projected_points': proj, 'historical_std_dev': max(proj * 0.15, 1.0)}
    sim_results = sim.simulate_lineup_performance(lineup, pdata)

    # AI analysis
    prompt = analysis.build_lineup_analysis_prompt(lineup, players_df, sim_results)
    ai_text = None
    if openai_status['api_key_present'] and openai_status['connection_success']:
        try:
            ai_text = openai_utils.analyze_prompt_with_gpt(prompt)
        except Exception as e:
            logging.error(f"GPT analysis failed: {e}")

    lineup_players = []
    total_proj = 0.0
    total_salary = 0
    for pid in lineup:
        row = players_df.loc[pid]
        total_proj += float(row['PROJ PTS'])
        total_salary += int(row['SALARY'])
        lineup_players.append({
            'name': row['PLAYER NAME'],
            'position': row['POS'],
            'team': row['TEAM'],
            'opponent': row.get('OPP', ''),
            'proj_points': float(row['PROJ PTS']),
            'salary': int(row['SALARY']),
            'proj_roster_pct_raw': row.get('PROJ ROSTER %', None),
            'own_pct': (None if 'OWN_PCT' not in row else (None if pd.isna(row['OWN_PCT']) else float(row['OWN_PCT'])))
        })

    # Persist last lineup for future auto-locks
    try:
        save_last_lineup(lineup_players, meta={
            "salary_cap": salary_cap,
            "enforce_stack": enforce_stack,
            "min_stack_receivers": min_stack_receivers
        })
    except Exception as e:
        logging.warning(f"Failed to save last lineup: {e}")

    result = {
        "params": {
            "salary_cap": salary_cap,
            "enforce_stack": enforce_stack,
            "min_stack_receivers": min_stack_receivers,
            "auto_late_swap": auto_late_swap
        },
        "constraints": {
            "locks": [players_df.loc[i,'PLAYER NAME'] for i in lock_idx_manual],
            "bans":  [players_df.loc[i,'PLAYER NAME'] for i in ban_idx_manual],
            "auto_locked": auto_locked_names,
            "not_found": list(set(lock_nf_manual + ban_nf_manual + nf_auto))
        },
        "cap_usage": {
            "total_salary": total_salary,
            "remaining": salary_cap - total_salary
        },
        "lineup": lineup_players,
        "total_projected_points": round(total_proj, 2),
        "simulation": sim_results,
        "analysis": ai_text if ai_text else "Analysis not available"
    }
    return result

@app.get("/optimize")
def optimize_endpoint(
    salary_cap: int = Query(60000, ge=1000, le=100000),
    enforce_stack: bool = Query(False),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: list[str] = Query(default=[]),
    ban:  list[str] = Query(default=[]),
    auto_late_swap: bool = Query(True),
):
    return _run_optimization(salary_cap, enforce_stack, min_stack_receivers, lock, ban, auto_late_swap)

@app.get("/optimize_text", response_class=PlainTextResponse)
def optimize_text_endpoint(
    salary_cap: int = Query(60000, ge=1000, le=100000),
    enforce_stack: bool = Query(False),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    width: int = Query(100, ge=70, le=160),
    lock: list[str] = Query(default=[]),
    ban:  list[str] = Query(default=[]),
    auto_late_swap: bool = Query(True),
):
    result = _run_optimization(salary_cap, enforce_stack, min_stack_receivers, lock, ban, auto_late_swap)
    if "error" in result:
        msg = f"ERROR: {result['error']}\n"
        cons = result.get("constraints", {})
        if cons:
            msg += f"Locks: {', '.join(cons.get('locks',[]))}\n"
            msg += f"Bans: {', '.join(cons.get('bans',[]))}\n"
            auto_l = cons.get("auto_locked", [])
            if auto_l:
                msg += f"Auto-locked: {', '.join(auto_l)}\n"
            nf = cons.get('not_found', [])
            if nf:
                msg += f"Not found: {', '.join(nf)}\n"
        return msg
    return build_text_report(result, width=width)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
