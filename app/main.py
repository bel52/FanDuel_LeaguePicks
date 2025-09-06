import os
import logging
import sys
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from dotenv import load_dotenv

# Load environment first
load_dotenv()

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app import data_ingestion
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.ai_integration import AIAnalyzer
from app.data_monitor import RealTimeDataMonitor
from app.auto_swap_system import AutoSwapSystem
from app.formatting import build_text_report
from app.player_match import match_names_to_indices
from app.kickoff_times import (
    build_kickoff_map, 
    auto_lock_started_players, 
    save_last_lineup, 
    load_last_lineup
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
optimizer = None
ai_analyzer = None
data_monitor = None
auto_swap_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with enhanced monitoring"""
    global optimizer, ai_analyzer, data_monitor, auto_swap_system
    
    logger.info("Starting Enhanced FanDuel DFS Optimizer...")
    
    # Ensure directories exist
    os.makedirs(settings.input_dir, exist_ok=True)
    os.makedirs(settings.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize enhanced components
    optimizer = EnhancedDFSOptimizer()
    ai_analyzer = AIAnalyzer()
    data_monitor = RealTimeDataMonitor()
    auto_swap_system = AutoSwapSystem()
    
    # Start background monitoring tasks
    monitoring_tasks = []
    
    if os.getenv("USE_NFL_DATA_PY", "true").lower() == "true":
        logger.info("Starting real-time data monitoring...")
        monitoring_task = asyncio.create_task(data_monitor.start_monitoring())
        monitoring_tasks.append(monitoring_task)
    
    if os.getenv("AUTO_SWAP_ENABLED", "true").lower() == "true":
        logger.info("Starting automated player swapping...")
        swap_task = asyncio.create_task(auto_swap_system.start_monitoring())
        monitoring_tasks.append(swap_task)
    
    yield
    
    # Cleanup
    logger.info("Shutting down enhanced monitoring...")
    for task in monitoring_tasks:
        task.cancel()
    
    try:
        await asyncio.gather(*monitoring_tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="Enhanced FanDuel NFL DFS Optimizer",
    version="3.0.0",
    description="AI-powered DFS optimization with real-time monitoring and automated swapping",
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
    """Root endpoint with enhanced system info"""
    return {
        "app": "Enhanced FanDuel NFL DFS Optimizer",
        "version": "3.0.0",
        "status": "operational",
        "features": {
            "ai_analysis": bool(settings.openai_api_key),
            "real_time_monitoring": os.getenv("USE_NFL_DATA_PY", "true").lower() == "true",
            "auto_swapping": os.getenv("AUTO_SWAP_ENABLED", "true").lower() == "true",
            "head_to_head_strategy": True
        },
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "optimize_text": "/optimize_text",
            "schedule": "/schedule",
            "data_status": "/data/status",
            "monitoring": "/monitoring/status",
            "swaps": "/swaps/summary",
            "manual_swap": "/swaps/manual"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    health_status = {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat(),
        "components": {
            "api": "operational",
            "data": "unknown",
            "ai": "unknown",
            "monitoring": "unknown",
            "auto_swap": "unknown"
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
    
    # Check AI system
    if ai_analyzer:
        try:
            cost_summary = ai_analyzer.get_cost_summary()
            health_status["components"]["ai"] = f"operational (${cost_summary['daily_cost']:.4f} spent)"
        except Exception as e:
            health_status["components"]["ai"] = f"error: {str(e)}"
    
    # Check monitoring system
    if data_monitor:
        try:
            recent_updates = await data_monitor.get_recent_updates(hours=1)
            health_status["components"]["monitoring"] = f"operational ({len(recent_updates)} recent updates)"
        except Exception as e:
            health_status["components"]["monitoring"] = f"error: {str(e)}"
    
    # Check auto-swap system
    if auto_swap_system:
        try:
            swap_summary = await auto_swap_system.get_swap_summary()
            health_status["components"]["auto_swap"] = f"operational ({swap_summary['swaps_executed']} swaps today)"
        except Exception as e:
            health_status["components"]["auto_swap"] = f"error: {str(e)}"
    
    return health_status

@app.get("/data/status")
def data_status():
    """Enhanced data status with real-time info"""
    status = {
        "input_files": {},
        "player_count": 0,
        "last_update": None,
        "data_sources": {
            "fantasypros": False,
            "nfl_data_py": False,
            "weather_gov": False,
            "espn_apis": False
        }
    }
    
    # Check for input CSV files
    positions = ["qb", "rb", "wr", "te", "dst"]
    for pos in positions:
        file_path = os.path.join(settings.input_dir, f"{pos}.csv")
        status["input_files"][pos] = os.path.exists(file_path)
        status["data_sources"]["fantasypros"] = any(status["input_files"].values())
    
    # Try to load data
    try:
        df = data_ingestion.load_weekly_data()
        if df is not None:
            status["player_count"] = len(df)
            status["positions"] = df["POS"].value_counts().to_dict() if "POS" in df.columns else {}
            status["last_update"] = pd.Timestamp.now().isoformat()
    except Exception as e:
        status["error"] = str(e)
    
    # Check other data sources
    status["data_sources"]["nfl_data_py"] = os.getenv("USE_NFL_DATA_PY", "true").lower() == "true"
    status["data_sources"]["weather_gov"] = os.getenv("USE_WEATHER_GOV", "true").lower() == "true"
    status["data_sources"]["espn_apis"] = os.getenv("USE_ESPN_HIDDEN_APIS", "true").lower() == "true"
    
    return status

@app.get("/monitoring/status")
async def monitoring_status():
    """Get real-time monitoring status"""
    if not data_monitor:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
    
    try:
        recent_updates = await data_monitor.get_recent_updates(hours=24)
        
        # Categorize updates
        by_type = {}
        by_severity = {"high": 0, "medium": 0, "low": 0}
        
        for update in recent_updates:
            update_type = update['update_type']
            severity = update['severity']
            
            by_type[update_type] = by_type.get(update_type, 0) + 1
            
            if severity >= 0.7:
                by_severity["high"] += 1
            elif severity >= 0.4:
                by_severity["medium"] += 1
            else:
                by_severity["low"] += 1
        
        return {
            "status": "active",
            "last_24_hours": {
                "total_updates": len(recent_updates),
                "by_type": by_type,
                "by_severity": by_severity
            },
            "recent_high_severity": [
                {
                    "player": update['player_name'],
                    "type": update['update_type'],
                    "severity": update['severity'],
                    "description": update['description'],
                    "source": update['source'],
                    "timestamp": update['timestamp']
                }
                for update in recent_updates[:5]
                if update['severity'] >= 0.7
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monitoring error: {str(e)}")

@app.get("/swaps/summary")
async def swap_summary():
    """Get automated swap system summary"""
    if not auto_swap_system:
        raise HTTPException(status_code=503, detail="Auto-swap system not available")
    
    try:
        return await auto_swap_system.get_swap_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Swap summary error: {str(e)}")

@app.post("/swaps/manual")
async def manual_swap(
    player_out: str = Query(..., description="Player to remove from lineup"),
    player_in: str = Query(..., description="Player to add to lineup"),
    reason: str = Query("Manual swap request", description="Reason for swap")
):
    """Execute a manual player swap"""
    if not auto_swap_system:
        raise HTTPException(status_code=503, detail="Auto-swap system not available")
    
    try:
        result = await auto_swap_system.manual_swap_request(player_out, player_in, reason)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result["error"])
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manual swap error: {str(e)}")

async def _run_enhanced_optimization(
    salary_cap: int,
    game_type: str,
    enforce_stack: bool,
    min_stack_receivers: int,
    lock_names: Optional[List[str]],
    ban_names: Optional[List[str]],
    auto_late_swap: bool = True
) -> Dict[str, Any]:
    """Enhanced optimization with AI and real-time data"""
    
    # Try to load data
    players_df = data_ingestion.load_weekly_data()
    if players_df is None or players_df.empty:
        raise HTTPException(status_code=422, detail="No player data available. Please upload CSV files to data/input/")
    
    # Manual locks/bans
    lock_idx_manual, nf_lock = match_names_to_indices(lock_names or [], players_df)
    ban_idx_manual, nf_ban = match_names_to_indices(ban_names or [], players_df)
    
    # Auto-lock for late swap
    auto_locked_names = []
    if auto_late_swap:
        try:
            kickoff_map = build_kickoff_map(players_df)
            last_lineup = load_last_lineup()
            auto_locked_names = auto_lock_started_players(last_lineup, kickoff_map)
            auto_idx, nf_auto = match_names_to_indices(auto_locked_names, players_df)
            lock_idx_manual.extend(auto_idx)
        except Exception as e:
            logger.warning(f"Auto-lock failed: {e}")
    
    # Run enhanced optimization
    lineup_indices, optimization_metadata = await optimizer.optimize_lineup(
        players_df,
        game_type=game_type,
        salary_cap=salary_cap,
        enforce_stack=enforce_stack,
        min_stack_receivers=min_stack_receivers,
        lock_indices=lock_idx_manual,
        ban_indices=ban_idx_manual,
        auto_swap_enabled=True
    )
    
    if not lineup_indices:
        raise HTTPException(status_code=422, detail="No feasible lineup found with given constraints")
    
    # Extract lineup details from metadata (enhanced optimizer returns this)
    result = {
        "lineup": optimization_metadata.get('lineup_players', []),
        "total_projected_points": optimization_metadata.get('total_projection', 0),
        "cap_usage": {
            "total_salary": sum(p['salary'] for p in optimization_metadata.get('lineup_players', [])),
            "remaining": salary_cap - sum(p['salary'] for p in optimization_metadata.get('lineup_players', []))
        },
        "simulation": optimization_metadata.get('simulation_results', {}),
        "analysis": optimization_metadata.get('ai_analysis', 'Analysis not available'),
        "constraints": {
            "auto_locked": auto_locked_names,
            "locks": lock_names or [],
            "bans": ban_names or [],
            "not_found": list(set(nf_lock + nf_ban))
        },
        "optimization_details": {
            "method": optimization_metadata.get('method', 'unknown'),
            "ai_enhanced": optimization_metadata.get('ai_enhanced', False),
            "game_type": game_type,
            "objective_value": optimization_metadata.get('objective_value')
        }
    }
    
    # Save last lineup for auto-lock
    if auto_late_swap:
        try:
            save_last_lineup(optimization_metadata.get('lineup_players', []))
        except Exception as e:
            logger.warning(f"Failed to save lineup: {e}")
    
    return result

@app.get("/optimize")
async def optimize_endpoint(
    salary_cap: int = Query(60000, ge=1000, le=100000),
    game_type: str = Query("league", regex="^(league|h2h)$", description="League or head-to-head strategy"),
    enforce_stack: bool = Query(True, description="Require QB-WR/TE stack"),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: Optional[List[str]] = Query(default=None, description="Players to lock"),
    ban: Optional[List[str]] = Query(default=None, description="Players to ban"),
    auto_late_swap: bool = Query(True, description="Auto-lock started players")
):
    """Generate enhanced optimized lineup"""
    try:
        result = await _run_enhanced_optimization
