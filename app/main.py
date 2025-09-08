import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import pandas as pd
import asyncio
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from app.config import settings
from app.data_ingestion import DataIngestion
from app.optimization_engine import OptimizationEngine
from app.ai_analyzer import AIAnalyzer
from app.cache_manager import CacheManager
from app.data_monitor import DataMonitor
from app.auto_swap_system import AutoSwapSystem
from app.formatting import TextFormatter
from app.kickoff_manager import KickoffManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global components
data_ingestion = None
optimization_engine = None
ai_analyzer = None
cache_manager = None
data_monitor = None
auto_swap_system = None
text_formatter = None
kickoff_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    global data_ingestion, optimization_engine, ai_analyzer, cache_manager, data_monitor, auto_swap_system, text_formatter, kickoff_manager
    
    logger.info("Starting FanDuel DFS Optimizer v3.0...")
    
    # Create required directories
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/targets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize components
    cache_manager = CacheManager()
    data_ingestion = DataIngestion()
    optimization_engine = OptimizationEngine(cache_manager)
    ai_analyzer = AIAnalyzer(cache_manager)
    data_monitor = DataMonitor(cache_manager, ai_analyzer)
    auto_swap_system = AutoSwapSystem(cache_manager, ai_analyzer, optimization_engine)
    text_formatter = TextFormatter()
    kickoff_manager = KickoffManager()
    
    # Start background tasks
    if settings.enable_real_time_monitoring:
        asyncio.create_task(data_monitor.start_monitoring())
    
    if settings.enable_auto_swap:
        asyncio.create_task(auto_swap_system.start_monitoring())
    
    logger.info("FanDuel DFS Optimizer started successfully")
    
    yield
    
    logger.info("Shutting down FanDuel DFS Optimizer...")

# Create FastAPI app
app = FastAPI(
    title="FanDuel DFS Optimizer",
    description="AI-powered NFL DFS lineup optimization with real-time monitoring",
    version="3.0.0",
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

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Check data availability
        data_status = await data_ingestion.check_data_availability()
        
        # Check optimization engine
        opt_status = optimization_engine.health_check()
        
        # Check AI availability
        ai_status = ai_analyzer.health_check()
        
        # Check cache
        cache_status = await cache_manager.health_check()
        
        # Check monitoring systems
        monitor_status = data_monitor.health_check() if data_monitor else "disabled"
        swap_status = auto_swap_system.health_check() if auto_swap_system else "disabled"
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_ingestion": data_status,
                "optimization_engine": opt_status,
                "ai_analyzer": ai_status,
                "cache_manager": cache_status,
                "data_monitor": monitor_status,
                "auto_swap_system": swap_status
            },
            "version": "3.0.0"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=503
        )

@app.get("/schedule")
async def get_schedule():
    """Get NFL game schedule with kickoff times"""
    try:
        schedule = await kickoff_manager.get_schedule()
        return JSONResponse(schedule)
    except Exception as e:
        logger.error(f"Schedule fetch failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/optimize")
async def optimize_lineup(
    game_type: str = Query("league", description="Game type: league or h2h"),
    salary_cap: int = Query(60000, description="Salary cap"),
    lock: Optional[str] = Query(None, description="Player name to lock"),
    ban: Optional[str] = Query(None, description="Player name to ban"),
    enforce_stack: bool = Query(True, description="Enforce QB stacking"),
    use_ai: bool = Query(True, description="Use AI enhancements")
):
    """Generate optimized lineup (JSON format)"""
    
    try:
        # Load player data
        player_data = await data_ingestion.load_weekly_data()
        if player_data is None or player_data.empty:
            raise HTTPException(status_code=400, detail="No player data available")
        
        # Prepare optimization parameters
        optimization_params = {
            "game_type": game_type,
            "salary_cap": salary_cap,
            "enforce_stack": enforce_stack,
            "lock_players": [lock] if lock else [],
            "ban_players": [ban] if ban else [],
            "use_ai": use_ai
        }
        
        # Run optimization
        result = await optimization_engine.optimize_lineup(player_data, **optimization_params)
        
        if not result:
            raise HTTPException(status_code=500, detail="Optimization failed")
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/optimize_text")
async def optimize_lineup_text(
    width: int = Query(110, ge=40, le=160),
    game_type: str = Query("league", description="Game type: league or h2h"),
    salary_cap: int = Query(60000, description="Salary cap"),
    lock: Optional[str] = Query(None, description="Player name to lock"),
    ban: Optional[str] = Query(None, description="Player name to ban"),
    use_ai: bool = Query(True, description="Use AI enhancements")
):
    """Generate optimized lineup (formatted text)"""
    
    try:
        # Load player data
        player_data = await data_ingestion.load_weekly_data()
        if player_data is None or player_data.empty:
            return PlainTextResponse("No player data available")
        
        # Prepare optimization parameters
        optimization_params = {
            "game_type": game_type,
            "salary_cap": salary_cap,
            "enforce_stack": True,
            "lock_players": [lock] if lock else [],
            "ban_players": [ban] if ban else [],
            "use_ai": use_ai
        }
        
        # Run optimization
        result = await optimization_engine.optimize_lineup(player_data, **optimization_params)
        
        if not result:
            return PlainTextResponse("Optimization failed")
        
        # Format as text
        formatted_text = text_formatter.format_lineup(result, width=width)
        
        return PlainTextResponse(formatted_text)
        
    except Exception as e:
        logger.error(f"Text optimization error: {e}")
        return PlainTextResponse(f"Error: {str(e)}")

@app.get("/lineup/analyze/{lineup_id}")
async def analyze_lineup(lineup_id: str):
    """Get AI analysis for a specific lineup"""
    try:
        lineup_data = await cache_manager.get(f"lineup:{lineup_id}")
        if not lineup_data:
            raise HTTPException(status_code=404, detail="Lineup not found")
        
        analysis = await ai_analyzer.analyze_lineup(lineup_data)
        return JSONResponse(analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lineup analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status")
async def monitoring_status():
    """Get real-time monitoring status"""
    try:
        if not data_monitor:
            return JSONResponse({"status": "disabled"})
        
        status = await data_monitor.get_status()
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"Monitoring status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/swaps/summary")
async def swap_summary():
    """Get auto-swap system summary"""
    try:
        if not auto_swap_system:
            return JSONResponse({"status": "disabled"})
        
        summary = await auto_swap_system.get_summary()
        return JSONResponse(summary)
        
    except Exception as e:
        logger.error(f"Swap summary error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/swaps/manual")
async def manual_swap(
    player_out: str = Query(..., description="Player to remove"),
    player_in: str = Query(..., description="Player to add"),
    reason: str = Query("Manual swap", description="Reason for swap")
):
    """Execute a manual player swap"""
    try:
        if not auto_swap_system:
            raise HTTPException(status_code=503, detail="Auto-swap system not available")
        
        result = await auto_swap_system.manual_swap(player_out, player_in, reason)
        
        if result["success"]:
            return JSONResponse(result)
        else:
            raise HTTPException(status_code=400, detail=result["error"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual swap error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/status")
async def data_status():
    """Get detailed data status information"""
    try:
        status = await data_ingestion.get_detailed_status()
        return JSONResponse(status)
        
    except Exception as e:
        logger.error(f"Data status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/admin/refresh_data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh player data from sources"""
    try:
        background_tasks.add_task(data_ingestion.refresh_data)
        return JSONResponse({"message": "Data refresh initiated"})
        
    except Exception as e:
        logger.error(f"Data refresh error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/ai/cost_summary")
async def ai_cost_summary():
    """Get AI usage and cost summary"""
    try:
        summary = ai_analyzer.get_cost_summary()
        return JSONResponse(summary)
        
    except Exception as e:
        logger.error(f"AI cost summary error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return JSONResponse({
        "message": "FanDuel DFS Optimizer API v3.0",
        "status": "operational",
        "features": [
            "AI-powered lineup optimization",
            "Real-time data monitoring",
            "Automated player swapping",
            "Advanced correlation analysis",
            "Weather impact modeling",
            "Multi-strategy optimization"
        ],
        "endpoints": {
            "health": "/health",
            "schedule": "/schedule",
            "optimize": "/optimize",
            "optimize_text": "/optimize_text",
            "monitoring": "/monitoring/status",
            "swaps": "/swaps/summary",
            "data_status": "/data/status"
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
