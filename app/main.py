import logging
import os
import sys
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="FanDuel NFL DFS Optimizer", version="3.0.0")

@app.on_event("startup")
async def startup_event():
    """Log startup"""
    logger.info("FanDuel DFS Optimizer starting up...")

@app.get("/")
def root():
    """Root endpoint providing basic info"""
    return {
        "app": "FanDuel NFL DFS Optimizer",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "optimize_text": "/optimize_text",
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
        input_dir = "/app/data/input"
        if os.path.exists(input_dir):
            files_found = []
            for pos in ["qb", "rb", "wr", "te", "dst"]:
                file_path = os.path.join(input_dir, f"{pos}.csv")
                if os.path.exists(file_path):
                    files_found.append(f"{pos}.csv")
            
            if files_found:
                health_status["components"]["data"] = f"files found: {', '.join(files_found)}"
            else:
                health_status["components"]["data"] = "no CSV files found"
        else:
            health_status["components"]["data"] = "input directory not found"
    except Exception as e:
        health_status["components"]["data"] = f"error: {str(e)}"
    
    return health_status

@app.get("/data/status")
def data_status():
    """Check data availability and basic stats"""
    status = {
        "files_present": {},
        "total_players": 0,
        "input_directory": "/app/data/input",
        "directories_exist": {}
    }
    
    # Check directories
    dirs_to_check = ["/app/data", "/app/data/input", "/app/data/output"]
    for dir_path in dirs_to_check:
        status["directories_exist"][dir_path] = os.path.exists(dir_path)
    
    # Check for CSV files
    input_dir = "/app/data/input"
    if os.path.exists(input_dir):
        for pos in ["qb", "rb", "wr", "te", "dst"]:
            file_path = os.path.join(input_dir, f"{pos}.csv")
            status["files_present"][f"{pos}.csv"] = os.path.exists(file_path)
            
            # Try to count lines if file exists
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        lines = len(f.readlines()) - 1  # Subtract header
                        status[f"{pos}_players"] = max(0, lines)
                        status["total_players"] += max(0, lines)
                except Exception as e:
                    status[f"{pos}_error"] = str(e)
    
    return status

@app.get("/optimize")
async def optimize_endpoint(
    game_type: str = Query("league", regex="^(league|h2h)$"),
    salary_cap: int = Query(60000, ge=1000, le=100000),
    enforce_stack: bool = Query(True, description="Require QB-WR/TE stack"),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: Optional[List[str]] = Query(default=None, description="Player names to lock"),
    ban: Optional[List[str]] = Query(default=None, description="Player names to ban")
):
    """Generate optimal lineup (JSON format)"""
    
    # Try to import and run optimization
    try:
        # Import data ingestion
        from app.data_ingestion import load_data_from_input_dir
        
        # Load data
        df, warnings = load_data_from_input_dir()
        if df is None or df.empty:
            raise HTTPException(
                status_code=422, 
                detail="No player data available. Please upload FantasyPros CSVs to data/input/"
            )
        
        # Try to import optimizer
        try:
            from app.enhanced_optimizer import EnhancedDFSOptimizer
            optimizer = EnhancedDFSOptimizer()
            
            # Convert lock/ban names to indices (simplified)
            lock_indices = []
            ban_indices = []
            
            if lock:
                for name in lock:
                    matches = df.index[df['PLAYER NAME'].str.lower() == name.lower()].tolist()
                    lock_indices.extend(matches)
            
            if ban:
                for name in ban:
                    matches = df.index[df['PLAYER NAME'].str.lower() == name.lower()].tolist()
                    ban_indices.extend(matches)
            
            # Run optimization
            lineup_indices, metadata = await optimizer.optimize_lineup(
                df=df,
                game_type=game_type,
                salary_cap=salary_cap,
                enforce_stack=enforce_stack,
                min_stack_receivers=min_stack_receivers,
                lock_indices=lock_indices if lock_indices else None,
                ban_indices=ban_indices if ban_indices else None
            )
            
            if not lineup_indices:
                raise HTTPException(
                    status_code=422, 
                    detail="No feasible lineup found with the given constraints"
                )
            
            return metadata
            
        except ImportError as e:
            # Fallback to basic optimization
            logger.warning(f"Enhanced optimizer not available: {e}")
            from app.optimization import optimize_lineup
            
            lineup_indices, metadata = optimize_lineup(
                df, game_type, salary_cap, enforce_stack, 
                min_stack_receivers, lock_indices, ban_indices
            )
            
            if not lineup_indices:
                raise HTTPException(
                    status_code=422,
                    detail="No feasible lineup found"
                )
            
            return metadata
            
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Optimization engine not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.get("/optimize_text", response_class=PlainTextResponse)
async def optimize_text_endpoint(
    game_type: str = Query("league", regex="^(league|h2h)$"),
    salary_cap: int = Query(60000),
    enforce_stack: bool = Query(True),
    min_stack_receivers: int = Query(1),
    lock: Optional[List[str]] = Query(default=None),
    ban: Optional[List[str]] = Query(default=None),
    width: int = Query(100, ge=70, le=160)
):
    """Generate optimal lineup (plain text format)"""
    
    try:
        # Try to use the JSON endpoint and format as text
        result = await optimize_endpoint(
            game_type=game_type,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            min_stack_receivers=min_stack_receivers,
            lock=lock,
            ban=ban
        )
        
        # Format as text
        from app.formatting import build_text_report
        
        # Convert result to expected format for text report
        lineup_players = result.get("lineup_players", [])
        if lineup_players:
            text_result = {
                "game_type": game_type,
                "lineup": [],
                "cap_usage": {},
                "total_projected_points": result.get("total_projection", 0)
            }
            
            total_salary = 0
            for p in lineup_players:
                text_result["lineup"].append({
                    "POS": p.get("position", ""),
                    "PLAYER NAME": p.get("name", ""),
                    "TEAM": p.get("team", ""),
                    "OPP": p.get("opponent", ""),
                    "SALARY": int(p.get("salary", 0)),
                    "PROJ PTS": float(p.get("proj_points", 0.0)),
                    "OWN_PCT": p.get("own_pct", None)
                })
                total_salary += int(p.get("salary", 0))
            
            text_result["cap_usage"] = {
                "total_salary": total_salary,
                "remaining": salary_cap - total_salary
            }
            
            return build_text_report(text_result, width=width)
        else:
            return f"No lineup generated for {game_type} strategy."
            
    except HTTPException as e:
        return f"Error: {e.detail}"
    except Exception as e:
        return f"Optimization failed: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)

