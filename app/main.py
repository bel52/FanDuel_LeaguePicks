# cat app/main.py
import logging
import asyncio
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse
from app import data_ingestion
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.formatting import build_text_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(title="FanDuel NFL DFS Optimizer", version="3.0.0")

# Instantiate the optimizer globally for reuse
optimizer = EnhancedDFSOptimizer()

@app.on_event("startup")
def startup_event():
    """
    On startup, log the status of data loading.
    """
    logger.info("Application starting up...")
    df, warnings = data_ingestion.load_data_from_input_dir()
    for warning in warnings:
        if "ERROR" in warning:
            logger.error(warning)
        else:
            logger.info(warning)
    if df is not None:
        logger.info(f"Initial player pool loaded: {len(df)} players")
    logger.info("Application startup complete.")

@app.get("/")
def root():
    """
    Root endpoint providing basic info and available endpoints.
    """
    return {
        "app": "FanDuel NFL DFS Optimizer",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "optimize_json": "/optimize",
            "optimize_text": "/optimize_text",
            "data_status": "/data/status"
        }
    }

@app.get("/health")
def health_check():
    """
    Comprehensive health check endpoint.
    """
    health_status = {"status": "healthy", "components": {"api": "operational", "data": "unknown"}}
    try:
        df, _ = data_ingestion.load_data_from_input_dir()
        if df is not None and not df.empty:
            health_status["components"]["data"] = f"operational ({len(df)} players)"
        else:
            health_status["components"]["data"] = "no_data"
    except Exception as e:
        health_status["components"]["data"] = f"error: {e}"
    return health_status

@app.get("/data/status")
def data_status():
    """
    Check data availability and basic stats.
    """
    df, _ = data_ingestion.load_data_from_input_dir()
    status = {
        "files_present": {},
        "total_players": 0,
        "positions": {}
    }
    # Check presence of each expected file
    for pos in ["qb", "rb", "wr", "te", "dst"]:
        path = f"data/input/{pos}.csv"
        status["files_present"][f"{pos}.csv"] = "yes" if asyncio.get_event_loop().run_in_executor(None, __import__('os').path.exists, path).result() else "no"
    if df is not None:
        status["total_players"] = len(df)
        if not df.empty and "POS" in df.columns:
            status["positions"] = df["POS"].value_counts().to_dict()
    return status

@app.get("/optimize")
async def optimize_endpoint(
    game_type: str = Query("league", regex="^(league|h2h)$"),
    salary_cap: int = Query(60000, ge=1000, le=100000),
    enforce_stack: bool = Query(True, description="Require QB-WR/TE stack"),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: list[str] = Query(default=None, description="Player names to lock into lineup"),
    ban: list[str] = Query(default=None, description="Player names to ban from lineup")
):
    """Generate optimal lineup (JSON format)."""
    # Load and validate player data
    df, warnings = data_ingestion.load_data_from_input_dir()
    if df is None or df.empty:
        raise HTTPException(status_code=422, detail="No player data available. Please upload FantasyPros CSVs to data/input/")
    # Map lock/ban names to DataFrame indices
    lock_indices = []
    ban_indices = []
    if lock:
        for name in lock:
            matches = df.index[df['PLAYER NAME'].str.lower() == name.lower()].tolist()
            if matches:
                lock_indices.extend(matches)
            else:
                logger.warning(f"Lock request ignored, player not found: {name}")
    if ban:
        for name in ban:
            matches = df.index[df['PLAYER NAME'].str.lower() == name.lower()].tolist()
            if matches:
                ban_indices.extend(matches)
            else:
                logger.warning(f"Ban request ignored, player not found: {name}")
    # Run optimization (AI-enhanced)
    try:
        lineup_indices, meta = await optimizer.optimize_lineup(
            df,
            game_type=game_type,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            min_stack_receivers=min_stack_receivers,
            lock_indices=lock_indices or None,
            ban_indices=ban_indices or None
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {e}")
    if not lineup_indices:
        # No feasible lineup found given the constraints
        raise HTTPException(status_code=422, detail="No feasible lineup found with the given constraints.")
    # (Optional) Could save lineup to data/output/last_lineup.json or update fd_target.csv here.
    return meta

@app.get("/optimize_text", response_class=PlainTextResponse)
async def optimize_text_endpoint(
    game_type: str = Query("league", regex="^(league|h2h)$"),
    salary_cap: int = Query(60000),
    enforce_stack: bool = Query(True),
    min_stack_receivers: int = Query(1),
    lock: list[str] = Query(default=None),
    ban: list[str] = Query(default=None),
    width: int = Query(100, ge=70, le=160)
):
    """Generate optimal lineup (plain text format)."""
    # Reuse JSON optimize logic to get lineup metadata
    try:
        result = await optimize_endpoint(
            game_type=game_type,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            min_stack_receivers=min_stack_receivers,
            lock=lock or None,
            ban=ban or None
        )
    except HTTPException as http_exc:
        # Return error message in plain text if any constraint issue or error occurred
        return PlainTextResponse(http_exc.detail, status_code=http_exc.status_code)
    # Build text report from result
    lineup_players = result.get("lineup_players", [])
    text_result = {"game_type": game_type, "lineup": [], "cap_usage": {}}
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
    text_result["cap_usage"] = {"total_salary": total_salary, "remaining": salary_cap - total_salary}
    # Format as table
    return build_text_report(text_result, width=width)
