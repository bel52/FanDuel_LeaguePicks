import os
import logging
import asyncio
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from app.config import INPUT_DIR
from app.data_ingestion import load_data_from_input_dir
from app.services import generate_and_save_lineup
from app.formatting import build_text_report
from app.state_manager import state_manager
from app.auto_swap_system import AutoSwapSystem
from app.data_monitor import RealTimeDataMonitor
from app.cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
auto_swap_system = AutoSwapSystem()
data_monitor = RealTimeDataMonitor()
cache_manager = CacheManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting FanDuel DFS Optimizer with AI...")
    
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Start background tasks
    asyncio.create_task(data_monitor.start_monitoring())
    asyncio.create_task(auto_swap_system.start_monitoring())
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    await cache_manager.close()

app = FastAPI(
    title="FanDuel NFL DFS Optimizer - AI Enhanced",
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

@app.get("/")
def root():
    """Root endpoint with system info"""
    return {
        "app": "FanDuel NFL DFS Optimizer - AI Enhanced",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "AI-powered lineup optimization",
            "Real-time weather analysis",
            "Automated player swapping",
            "News impact analysis",
            "Game script predictions"
        ],
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "optimize_text": "/optimize_text",
            "data_status": "/data/status",
            "monitoring_status": "/monitoring/status",
            "swap_summary": "/swaps/summary"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "components": {
            "api": "operational",
            "data": "unknown",
            "redis": "unknown",
            "ai": "unknown",
            "monitoring": "unknown"
        }
    }
    
    # Check data availability
    try:
        df, warnings = load_data_from_input_dir()
        if df is not None and not df.empty:
            health_status["components"]["data"] = f"operational ({len(df)} players)"
        else:
            health_status["components"]["data"] = "no_data"
    except Exception as e:
        health_status["components"]["data"] = f"error: {str(e)}"
    
    # Check Redis
    try:
        if await cache_manager.exists("health_check"):
            health_status["components"]["redis"] = "operational"
        else:
            await cache_manager.set("health_check", "ok", ttl=10)
            health_status["components"]["redis"] = "operational"
    except:
        health_status["components"]["redis"] = "offline"
    
    # Check AI
    from app.ai_analyzer import AI_CLIENT
    health_status["components"]["ai"] = "operational" if AI_CLIENT else "not_configured"
    
    # Check monitoring
    health_status["components"]["monitoring"] = "operational"
    
    return health_status

@app.get("/data/status")
def data_status():
    """Check data availability and freshness"""
    status = {
        "input_files": {},
        "player_count": 0,
        "positions": {},
        "errors": []
    }
    
    # Check for input CSV files
    positions = ["qb", "rb", "wr", "te", "dst"]
    for pos in positions:
        file_path = os.path.join(INPUT_DIR, f"{pos}.csv")
        status["input_files"][pos] = os.path.exists(file_path)
    
    # Try to load data
    try:
        df, warnings = load_data_from_input_dir()
        if df is not None:
            status["player_count"] = len(df)
            status["positions"] = df["POS"].value_counts().to_dict() if "POS" in df.columns else {}
        status["errors"] = [w for w in warnings if "ERROR" in w]
    except Exception as e:
        status["errors"].append(str(e))
    
    return status

@app.get("/optimize")
async def optimize_endpoint(
    game_type: str = Query("league", regex="^(league|h2h)$", description="Game type: league or h2h"),
    salary_cap: int = Query(60000, ge=1000, le=100000),
    enforce_stack: bool = Query(True, description="Require QB-WR/TE stack"),
    min_stack_receivers: int = Query(1, ge=1, le=3),
    lock: Optional[List[str]] = Query(default=None, description="Players to lock"),
    ban: Optional[List[str]] = Query(default=None, description="Players to ban"),
    use_ai: bool = Query(True, description="Use AI enhancements"),
    use_weather: bool = Query(True, description="Use weather analysis"),
    use_game_scripts: bool = Query(True, description="Use game script predictions")
):
    """Generate AI-optimized lineup"""
    try:
        result = await generate_and_save_lineup(
            game_mode=game_type,
            teams=None,
            use_ai=use_ai,
            use_weather=use_weather,
            use_game_scripts=use_game_scripts,
            lock_players=lock,
            ban_players=ban,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            min_stack_receivers=min_stack_receivers
        )
        
        if not result:
            raise HTTPException(status_code=422, detail="No feasible lineup found")
        
        return result
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize_text", response_class=PlainTextResponse)
async def optimize_text_endpoint(
    game_type: str = Query("league", regex="^(league|h2h)$"),
    salary_cap: int = Query(60000),
    enforce_stack: bool = Query(True),
    min_stack_receivers: int = Query(1),
    lock: Optional[List[str]] = Query(default=None),
    ban: Optional[List[str]] = Query(default=None),
    use_ai: bool = Query(True),
    width: int = Query(100, ge=70, le=160)
):
    """Generate optimized lineup as formatted text"""
    try:
        result = await generate_and_save_lineup(
            game_mode=game_type,
            use_ai=use_ai,
            lock_players=lock,
            ban_players=ban,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            min_stack_receivers=min_stack_receivers
        )
        
        if not result:
            return "No feasible lineup found with given constraints"
        
        return build_text_report(result, width=width)
        
    except Exception as e:
        return PlainTextResponse(f"Error: {str(e)}", status_code=500)

@app.get("/monitoring/status")
async def monitoring_status():
    """Get real-time monitoring status"""
    try:
        recent_updates = await data_monitor.get_recent_updates(hours=4)
        
        high_severity = [u for u in recent_updates if u.get('severity', 0) >= 0.7]
        medium_severity = [u for u in recent_updates if 0.3 <= u.get('severity', 0) < 0.7]
        
        return {
            "status": "operational",
            "total_updates": len(recent_updates),
            "high_severity_count": len(high_severity),
            "medium_severity_count": len(medium_severity),
            "recent_high_severity": high_severity[:5],
            "monitoring_intervals": {
                "news": f"{data_monitor.news_interval}s",
                "weather": f"{data_monitor.weather_interval}s",
                "injuries": f"{data_monitor.injury_interval}s"
            }
        }
    except Exception as e:
        logger.error(f"Monitoring status error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/swaps/summary")
async def swaps_summary():
    """Get automated swap summary"""
    try:
        summary = await auto_swap_system.get_swap_summary()
        return summary
    except Exception as e:
        logger.error(f"Swap summary error: {e}")
        return {"error": str(e)}

@app.post("/swaps/manual")
async def manual_swap(
    player_out: str = Query(..., description="Player to remove"),
    player_in: str = Query(..., description="Player to add"),
    reason: str = Query("Manual swap", description="Reason for swap")
):
    """Execute manual player swap"""
    try:
        result = await auto_swap_system.manual_swap_request(
            player_out, player_in, reason
        )
        return result
    except Exception as e:
        logger.error(f"Manual swap error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/analyze-news")
async def analyze_news(
    player_name: str = Query(..., description="Player name"),
    news_text: str = Query(..., description="News text to analyze"),
    current_projection: float = Query(..., description="Current projection")
):
    """Analyze news impact on player with AI"""
    from app.ai_analyzer import get_ai_projection_adjustment
    
    try:
        adjustment, reasoning = get_ai_projection_adjustment(
            player_name, news_text, current_projection
        )
        
        return {
            "player": player_name,
            "original_projection": current_projection,
            "adjustment": adjustment,
            "new_projection": current_projection + adjustment,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"AI news analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/cost-summary")
def ai_cost_summary():
    """Get AI usage and cost summary"""
    from app.ai_integration import AIAnalyzer
    ai = AIAnalyzer()
    return ai.get_cost_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
