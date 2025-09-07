import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import traceback

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

# Local imports
from app.data_ingestion import load_weekly_data
from app.optimization import optimize_lineup
from app.formatting import build_text_report
from app.kickoff_times import (
    get_schedule,
    build_kickoff_map,
    auto_lock_started_players,
    persist_kickoffs,
    _weekly_csv_path,
    _snapshot_weekly_csv,
)

class Settings:
    def __init__(self) -> None:
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8010"))
        self.data_dir: str = os.getenv("DATA_DIR", "data")
        self.input_dir: str = os.path.join(self.data_dir, "input")
        self.output_dir: str = os.path.join(self.data_dir, "output")
        self.weekly_dir: str = os.path.join(self.data_dir, "weekly")
        self.log_dir: str = os.getenv("LOG_DIR", "logs")
        self.kickoff_api_url: Optional[str] = os.getenv("KICKOFF_API_URL")

settings = Settings()

# Setup logging
os.makedirs(settings.log_dir, exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("app.main")

_REQUIRED_INPUTS = ("qb.csv", "rb.csv", "wr.csv", "te.csv", "dst.csv")

def _data_is_present() -> bool:
    """Check if required CSV files are present"""
    try:
        if not os.path.isdir(settings.input_dir):
            return False
        return all(os.path.isfile(os.path.join(settings.input_dir, f)) for f in _REQUIRED_INPUTS)
    except Exception:
        return False

def _get_player_data() -> Optional[pd.DataFrame]:
    """Load player data and return DataFrame"""
    try:
        df = load_weekly_data(settings.input_dir)
        return df
    except Exception as e:
        log.error(f"Error loading player data: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    log.info("Starting FanDuel DFS Optimizer...")

    # Create required directories
    for d in (settings.data_dir, settings.input_dir, settings.output_dir, 
              settings.weekly_dir, settings.log_dir):
        os.makedirs(d, exist_ok=True)

    # Warm kickoff cache
    try:
        team_map = build_kickoff_map(settings.kickoff_api_url)
        if team_map:
            persist_kickoffs(team_map, output_path=os.path.join(settings.output_dir, "kickoffs.json"))
            csv_path = _weekly_csv_path()
            if not os.path.exists(csv_path):
                _snapshot_weekly_csv(team_map=team_map)
        else:
            log.warning("No kickoff data available at startup")
    except Exception as e:
        log.error("Failed to warm kickoff data: %s", e)

    yield
    log.info("Shutting down FanDuel DFS Optimizer...")

# Create FastAPI app
app = FastAPI(
    title="FanDuel DFS Optimizer",
    description="AI-powered NFL DFS lineup optimization",
    version="3.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health() -> JSONResponse:
    """Health check endpoint"""
    data_status = "operational" if _data_is_present() else "no_data"
    
    try:
        sched = get_schedule(settings.kickoff_api_url)
        kickoff_status = "operational" if sched else "no_kickoffs"
    except Exception as e:
        kickoff_status = f"error: {e}"

    # Check if we can load player data
    try:
        df = _get_player_data()
        data_load_status = "operational" if df is not None else "load_failed"
        player_count = len(df) if df is not None else 0
    except Exception as e:
        data_load_status = f"error: {e}"
        player_count = 0

    body: Dict[str, Any] = {
        "status": "healthy",
        "components": {
            "api": "operational",
            "data_files": data_status,
            "data_loading": data_load_status,
            "kickoffs": kickoff_status
        },
        "player_count": player_count,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    return JSONResponse(body)

@app.get("/schedule")
def schedule() -> JSONResponse:
    """Get NFL game schedule with kickoff times"""
    try:
        games = get_schedule(settings.kickoff_api_url) or []
        
        # Persist normalized kickoffs.json and ensure weekly CSV exists
        if games:
            team_map: Dict[str, str] = {}
            for g in games:
                iso = str(g.get("kickoff"))
                for t in g.get("teams", []):
                    team_map[str(t).upper()] = iso
            persist_kickoffs(team_map, output_path=os.path.join(settings.output_dir, "kickoffs.json"))
            csv_path = _weekly_csv_path()
            if not os.path.exists(csv_path):
                _snapshot_weekly_csv(games=games)
        
        return JSONResponse(games)
    except Exception as e:
        log.error("Schedule fetch failed: %s", e)
        return JSONResponse([])

@app.get("/optimize")
def optimize(
    game_type: str = Query("league", description="Game type: league or h2h"),
    salary_cap: int = Query(60000, description="Salary cap"),
    lock: Optional[str] = Query(None, description="Player name to lock"),
    ban: Optional[str] = Query(None, description="Player name to ban"),
    enforce_stack: bool = Query(True, description="Enforce QB stacking")
) -> JSONResponse:
    """Generate optimized lineup (JSON format)"""
    
    if not _data_is_present():
        raise HTTPException(
            status_code=400, 
            detail="No player data available. Please place CSV files in data/input/"
        )
    
    try:
        # Load player data
        df = _get_player_data()
        if df is None or df.empty:
            raise HTTPException(
                status_code=500,
                detail="Failed to load player data or no players available"
            )
        
        # Handle lock/ban players
        lock_indices = []
        ban_indices = []
        
        if lock:
            lock_mask = df['PLAYER NAME'].str.contains(lock, case=False, na=False)
            lock_indices = df[lock_mask].index.tolist()
            if not lock_indices:
                log.warning(f"Lock player '{lock}' not found")
        
        if ban:
            ban_mask = df['PLAYER NAME'].str.contains(ban, case=False, na=False)
            ban_indices = df[ban_mask].index.tolist()
            if not ban_indices:
                log.warning(f"Ban player '{ban}' not found")
        
        # Run optimization
        lineup_indices, metadata = optimize_lineup(
            df,
            game_type=game_type,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            lock_indices=lock_indices,
            ban_indices=ban_indices
        )
        
        if not lineup_indices:
            error_msg = metadata.get('error', 'Optimization failed to produce a lineup')
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Build response
        lineup_data = []
        total_salary = 0
        total_projection = 0.0
        
        for idx in lineup_indices:
            player = df.loc[idx]
            lineup_data.append({
                'player_name': player['PLAYER NAME'],
                'position': player['POS'],
                'team': player['TEAM'],
                'opponent': player.get('OPP', ''),
                'salary': int(player['SALARY']),
                'projection': float(player['PROJ PTS']),
                'ownership': float(player.get('OWN_PCT', 0)) if pd.notna(player.get('OWN_PCT')) else None
            })
            total_salary += int(player['SALARY'])
            total_projection += float(player['PROJ PTS'])
        
        # Get locked players info
        locked_players = []
        if lock_indices:
            for idx in lock_indices:
                if idx in lineup_indices:
                    locked_players.append(df.loc[idx]['PLAYER NAME'])
        
        response = {
            "success": True,
            "game_type": game_type,
            "lineup": lineup_data,
            "summary": {
                "total_salary": total_salary,
                "salary_remaining": salary_cap - total_salary,
                "total_projection": round(total_projection, 2),
                "average_projection": round(total_projection / len(lineup_data), 2)
            },
            "constraints": {
                "salary_cap": salary_cap,
                "enforce_stack": enforce_stack,
                "locked_players": locked_players,
                "banned_players": [ban] if ban else []
            },
            "metadata": metadata,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return JSONResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Optimization error: {e}")
        log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal optimization error: {str(e)}"
        )

@app.get("/optimize_text")
def optimize_text(
    width: int = Query(110, ge=40, le=160),
    game_type: str = Query("league", description="Game type: league or h2h"),
    salary_cap: int = Query(60000, description="Salary cap"),
    lock: Optional[str] = Query(None, description="Player name to lock"),
    ban: Optional[str] = Query(None, description="Player name to ban")
) -> PlainTextResponse:
    """Generate optimized lineup (formatted text)"""
    
    if not _data_is_present():
        return PlainTextResponse("No player data available. Please place CSV files under data/input/")
    
    try:
        # Load player data
        df = _get_player_data()
        if df is None or df.empty:
            return PlainTextResponse("Failed to load player data or no players available")
        
        # Handle lock/ban players
        lock_indices = []
        ban_indices = []
        
        if lock:
            lock_mask = df['PLAYER NAME'].str.contains(lock, case=False, na=False)
            lock_indices = df[lock_mask].index.tolist()
        
        if ban:
            ban_mask = df['PLAYER NAME'].str.contains(ban, case=False, na=False)
            ban_indices = df[ban_mask].index.tolist()
        
        # Run optimization
        lineup_indices, metadata = optimize_lineup(
            df,
            game_type=game_type,
            salary_cap=salary_cap,
            enforce_stack=True,
            lock_indices=lock_indices,
            ban_indices=ban_indices
        )
        
        if not lineup_indices:
            error_msg = metadata.get('error', 'Optimization failed')
            return PlainTextResponse(f"Optimization failed: {error_msg}")
        
        # Build lineup data for formatting
        lineup_data = []
        total_salary = 0
        total_projection = 0.0
        
        for idx in lineup_indices:
            player = df.loc[idx].to_dict()
            lineup_data.append(player)
            total_salary += int(player['SALARY'])
            total_projection += float(player['PROJ PTS'])
        
        # Create result object for text formatter
        result = {
            "game_type": game_type,
            "lineup": lineup_data,
            "total_projected_points": round(total_projection, 2),
            "cap_usage": {
                "total_salary": total_salary,
                "remaining": salary_cap - total_salary
            }
        }
        
        # Generate formatted text
        formatted_text = build_text_report(result, width=width)
        
        # Add metadata footer
        footer_lines = [
            "",
            f"Optimization Method: {metadata.get('method', 'unknown')}",
            f"Game Type: {game_type.upper()}",
            f"Players Evaluated: {len(df)}",
        ]
        
        if lock:
            footer_lines.append(f"Locked: {lock}")
        if ban:
            footer_lines.append(f"Banned: {ban}")
            
        footer_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        full_text = formatted_text + "\n" + "\n".join(footer_lines)
        
        return PlainTextResponse(full_text)
        
    except Exception as e:
        log.error(f"Text optimization error: {e}")
        return PlainTextResponse(f"Error generating lineup: {str(e)}")

@app.get("/data/status")
def data_status() -> JSONResponse:
    """Get detailed data status information"""
    try:
        status = {
            "data_directory": settings.input_dir,
            "files_present": {},
            "player_counts": {},
            "total_players": 0,
            "data_loaded": False,
            "errors": []
        }
        
        # Check file presence
        for file in _REQUIRED_INPUTS:
            file_path = os.path.join(settings.input_dir, file)
            status["files_present"][file] = os.path.isfile(file_path)
        
        # Try to load data
        try:
            df = _get_player_data()
            if df is not None:
                status["data_loaded"] = True
                status["total_players"] = len(df)
                
                # Count by position
                if 'POS' in df.columns:
                    pos_counts = df['POS'].value_counts().to_dict()
                    status["player_counts"] = pos_counts
                
                # Check for required columns
                required_cols = ['PLAYER NAME', 'POS', 'TEAM', 'SALARY', 'PROJ PTS']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    status["errors"].append(f"Missing required columns: {missing_cols}")
                
                # Check for data quality issues
                if df['SALARY'].isna().any():
                    status["errors"].append("Some players have missing salary data")
                if df['PROJ PTS'].isna().any():
                    status["errors"].append("Some players have missing projection data")
                    
        except Exception as e:
            status["errors"].append(f"Failed to load data: {str(e)}")
        
        return JSONResponse(status)
        
    except Exception as e:
        log.error(f"Data status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/kickoffs/snapshot")
def kickoffs_snapshot(week: Optional[str] = None) -> JSONResponse:
    """Force-create the weekly kickoff CSV and refresh kickoffs.json"""
    try:
        games = get_schedule(settings.kickoff_api_url)
        if not games:
            return JSONResponse(
                {"saved": False, "reason": "no schedule available from API/CSV/JSON"}, 
                status_code=400
            )

        # Persist JSON + CSV
        team_map: Dict[str, str] = {}
        for g in games:
            iso = str(g.get("kickoff"))
            for t in g.get("teams", []):
                team_map[str(t).upper()] = iso
        
        persist_kickoffs(team_map, output_path=os.path.join(settings.output_dir, "kickoffs.json"))
        saved_to = _snapshot_weekly_csv(games=games, week_id=week)

        return JSONResponse({"saved": True, "csv_path": saved_to, "games": len(games)})
    
    except Exception as e:
        log.error(f"Kickoffs snapshot error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root() -> JSONResponse:
    """Root endpoint"""
    return JSONResponse({
        "message": "FanDuel DFS Optimizer API",
        "version": "3.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "schedule": "/schedule", 
            "optimize_json": "/optimize",
            "optimize_text": "/optimize_text",
            "data_status": "/data/status"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
