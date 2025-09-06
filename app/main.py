import os
import logging
import sys
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from dotenv import load_dotenv

# Load environment first
load_dotenv()

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app import data_ingestion, optimization
from app.formatting import build_text_report
from app.player_match import match_names_to_indices
from app.kickoff_times import save_last_lineup, load_last_lineup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting FanDuel DFS Optimizer...")
    
    # Ensure directories exist
    os.makedirs(settings.input_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")

app = FastAPI(
    title="FanDuel NFL DFS Optimizer",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint with system info"""
    return {
        "app": "FanDuel NFL DFS Optimizer",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "optimize_text": "/optimize_text",
            "schedule": "/schedule",
            "data_status": "/data/status"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "components": {
            "api": "operational",
            "data": "unknown"
        }
    }
    
    # Check data availability
    try:
        df = data_ingestion.load_weekly_data()
        if df is not None and not df.empty:
            health_status["components"]["data"] = f"operational ({len(df)} players)"
        else:
            health_status["components"]["data"] = "no_data"
    except Exception as e:
        health_status["components"]["data"] = f"error: {str(e)}"
    
    return health_status

@app.get("/data/status")
def data_status():
    """Check data availability and freshness"""
    status = {
        "input_files": {},
        "player_count": 0,
        "last_update": None
    }
    
    # Check for input CSV files
    positions = ["qb", "rb", "wr", "te", "dst"]
    for pos in positions:
        file_path = os.path.join(settings.input_dir, f"{pos}.csv")
        status["input_files"][pos] = os.path.exists(file_path)
    
    # Try to load data
    try:
        df = data_ingestion.load_weekly_data()
        if df is not None:
            status["player_count"] = len(df)
            status["positions"] = df["POS"].value_counts().to_dict() if "POS" in df.columns else {}
    except Exception as e:
        status["error"] = str(e)
    
    return status

def _run_optimization(
    salary_cap: int,
    enforce_stack: bool,
    min_stack_receivers: int,
    lock_names: Optional[List[str]],
    ban_names: Optional[List[str]],
    game_type: str = "league"
) -> Dict[str, Any]:
    """Core optimization logic"""
    
    # Try to load data
    players_df = data_ingestion.load_weekly_data()
    if players_df is None or players_df.empty:
        raise HTTPException(status_code=422, detail="No player data available. Please upload CSV files to data/input/")
    
    # Manual locks/bans
    lock_idx_manual, nf_lock = match_names_to_indices(lock_names or [], players_df)
    ban_idx_manual, nf_ban = match_names_to_indices(ban_names or [], players_df)
    
    # Get leverage weight based on game type
    leverage_weight = 0.7 if game_type == "h2h" else 0.3
    
    # Run optimization
    lineup = optimization.optimize_lineup(
        players_df,
        salary_cap=salary_cap,
        enforce_stack=enforce_stack,
        min_stack_receivers=min_stack_receivers,
        lock_indices=lock_idx_manual,
        ban_indices=ban_idx_manual,
        leverage_weight=leverage_weight
    )
    
    if not lineup:
        raise HTTPException(status_code=422, detail="No feasible lineup found with given constraints")
    
    # Build lineup details
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
            "team": str(r.get("TEAM", "")),
            "opponent": str(r.get("OPP", "")),
            "proj_points": float(r["PROJ PTS"]),
            "salary": int(r["SALARY"]),
            "own_pct": float(r.get("OWN_PCT")) if pd.notna(r.get("OWN_PCT")) else None
        })
    
    result = {
        "lineup": lineup_players,
        "total_projected_points": round(total_proj, 2),
        "cap_usage": {
            "total_salary": total_salary,
            "remaining": salary_cap - total_salary
        },
        "game_type": game_type,
        "constraints": {
            "locks": lock_names or [],
            "bans": ban_names or [],
            "not_found": list(set(nf_lock + nf_ban))
        }
    }
    
    # Save last lineup
    try:
        save_last_lineup(lineup_players)
    except Exception as e:
        logger.warning(f"Failed to save lineup: {e}")
    
    return result

@app.get("/optimize")
def optimize_endpoint(
    salary_cap: int = Query(60000, ge=1000, le=100000),
    game_type: str = Query("league", regex="^(league|h2h)$"),
    enforce_stack: bool = Query(True, description="Require QB-WR/TE stack"),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: Optional[List[str]] = Query(default=None, description="Players to lock"),
    ban: Optional[List[str]] = Query(default=None, description="Players to ban")
):
    """Generate optimized lineup"""
    try:
        result = _run_optimization(
            salary_cap, enforce_stack, min_stack_receivers, 
            lock, ban, game_type
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize_text", response_class=PlainTextResponse)
def optimize_text_endpoint(
    salary_cap: int = Query(60000),
    game_type: str = Query("league"),
    enforce_stack: bool = Query(True),
    min_stack_receivers: int = Query(1),
    lock: Optional[List[str]] = Query(default=None),
    ban: Optional[List[str]] = Query(default=None),
    width: int = Query(100, ge=70, le=160)
):
    """Generate optimized lineup as formatted text"""
    try:
        result = _run_optimization(
            salary_cap, enforce_stack, min_stack_receivers,
            lock, ban, game_type
        )
        return build_text_report(result, width=width)
    except HTTPException as e:
        return PlainTextResponse(e.detail, status_code=e.status_code)
    except Exception as e:
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)

@app.get("/schedule")
def schedule_endpoint():
    """Get current game schedule"""
    return {
        "timezone": settings.timezone,
        "message": "Schedule endpoint - integrate with your kickoff_times module"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
