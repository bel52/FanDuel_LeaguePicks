# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from typing import List, Optional, Dict, Any
import logging
import os
import pandas as pd
import asyncio
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configurations
from app.config import settings
from app.cache_manager import CacheManager
from app.data_ingestion import load_weekly_data, load_data_from_input_dir
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.ai_analyzer import AIAnalyzer
from app.auto_swap_system import AutoSwapSystem
from app.data_monitor import RealTimeDataMonitor
from app.formatting import build_text_report
from app.player_match import match_names_to_indices
from app.kickoff_times import get_schedule

# Global instances
cache_manager = None
optimizer = None
ai_analyzer = None
auto_swap = None
data_monitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global cache_manager, optimizer, ai_analyzer, auto_swap, data_monitor
    
    logger.info("Starting FanDuel DFS Optimization API...")
    
    try:
        # Initialize services
        cache_manager = CacheManager()
        optimizer = EnhancedDFSOptimizer()
        ai_analyzer = AIAnalyzer(cache_manager)
        auto_swap = AutoSwapSystem()
        data_monitor = RealTimeDataMonitor()
        
        # Start background monitoring if enabled
        if os.getenv("AUTO_MONITORING", "false").lower() == "true":
            asyncio.create_task(data_monitor.start_monitoring())
            asyncio.create_task(auto_swap.start_monitoring())
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")

app = FastAPI(
    title="FanDuel NFL DFS Optimizer",
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

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "FanDuel NFL DFS Optimizer API v3.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "optimize_text": "/optimize_text",
            "schedule": "/schedule",
            "data_status": "/data/status",
            "monitoring": "/monitoring/status",
            "swaps": "/swaps/summary"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "cache": "healthy" if cache_manager else "not_initialized",
            "optimizer": "healthy" if optimizer else "not_initialized",
            "ai_analyzer": ai_analyzer.health_check() if ai_analyzer else "not_initialized",
            "data_monitor": "healthy" if data_monitor else "not_initialized",
            "auto_swap": "healthy" if auto_swap else "not_initialized"
        }
    }
    
    # Check data availability
    try:
        df, warnings = load_data_from_input_dir()
        health_status["data_available"] = df is not None and not df.empty
        health_status["data_rows"] = len(df) if df is not None else 0
    except Exception as e:
        health_status["data_available"] = False
        health_status["data_error"] = str(e)
    
    return health_status

@app.get("/optimize")
async def optimize_lineup(
    game_type: str = Query("league", description="Game type: 'league' or 'h2h'"),
    lock: Optional[List[str]] = Query(None, description="Player names to lock"),
    ban: Optional[List[str]] = Query(None, description="Player names to ban"),
    enforce_stack: bool = Query(True, description="Enforce QB stacking"),
    min_stack_receivers: int = Query(1, description="Minimum receivers to stack with QB"),
    use_ai: bool = Query(True, description="Use AI analysis"),
    use_weather: bool = Query(True, description="Apply weather adjustments"),
    salary_cap: int = Query(60000, description="Salary cap"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate optimized DFS lineup"""
    try:
        # Load player data
        df, warnings = load_data_from_input_dir()
        
        if df is None or df.empty:
            # Try to load weekly data as fallback
            df = load_weekly_data()
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="No player data available")
        
        # Process lock/ban players
        lock_indices = []
        ban_indices = []
        
        if lock:
            lock_indices, not_found = match_names_to_indices(lock, df)
            if not_found:
                logger.warning(f"Could not find players to lock: {not_found}")
        
        if ban:
            ban_indices, not_found = match_names_to_indices(ban, df)
            if not_found:
                logger.warning(f"Could not find players to ban: {not_found}")
        
        # Run optimization
        lineup_indices, metadata = await optimizer.optimize_lineup(
            df=df,
            game_type=game_type,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            min_stack_receivers=min_stack_receivers,
            lock_indices=lock_indices,
            ban_indices=ban_indices,
            auto_swap_enabled=False  # Manual optimization doesn't use auto-swap
        )
        
        if not lineup_indices:
            raise HTTPException(status_code=400, detail="Failed to generate valid lineup")
        
        # Build lineup data
        lineup = []
        total_salary = 0
        total_projection = 0
        
        for idx in lineup_indices:
            player = df.loc[idx]
            player_data = {
                "player_name": player.get("PLAYER NAME", "Unknown"),
                "position": player.get("POS", ""),
                "team": player.get("TEAM", ""),
                "opponent": player.get("OPP", ""),
                "salary": int(player.get("SALARY", 0)),
                "projection": float(player.get("PROJ PTS", 0)),
                "ownership": float(player.get("OWN_PCT", 0)) if pd.notna(player.get("OWN_PCT")) else None
            }
            lineup.append(player_data)
            total_salary += player_data["salary"]
            total_projection += player_data["projection"]
        
        # Get AI analysis if enabled
        ai_analysis = ""
        if use_ai and ai_analyzer:
            try:
                # Create lineup data for AI analysis
                lineup_data = {
                    "lineup": lineup,
                    "summary": {
                        "total_projection": total_projection,
                        "total_salary": total_salary
                    }
                }
                ai_analysis = await ai_analyzer.analyze_lineup(lineup_data, game_type)
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                ai_analysis = "AI analysis unavailable"
        
        result = {
            "success": True,
            "game_type": game_type,
            "lineup": lineup,
            "total_salary": total_salary,
            "total_projection": round(total_projection, 2),
            "salary_remaining": salary_cap - total_salary,
            "metadata": metadata,
            "ai_analysis": ai_analysis,
            "warnings": warnings
        }
        
        # Cache result in background
        if cache_manager:
            cache_key = f"lineup:{game_type}:{datetime.now().strftime('%Y%m%d%H%M')}"
            background_tasks.add_task(cache_manager.set, cache_key, result, ttl=3600)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize_text", response_class=PlainTextResponse)
async def optimize_lineup_text(
    game_type: str = Query("league"),
    width: int = Query(100),
    lock: Optional[List[str]] = Query(None),
    ban: Optional[List[str]] = Query(None)
):
    """Generate optimized lineup in text format"""
    try:
        # Get JSON result
        result = await optimize_lineup(
            game_type=game_type,
            lock=lock,
            ban=ban
        )
        
        # Format as text
        text_report = build_text_report(result, width=width)
        
        # Add AI analysis if available
        if result.get("ai_analysis"):
            text_report += f"\n\nAI ANALYSIS:\n{result['ai_analysis']}"
        
        return text_report
        
    except HTTPException as e:
        return PlainTextResponse(
            content=f"Error: {e.detail}",
            status_code=e.status_code
        )

@app.get("/schedule")
async def get_game_schedule():
    """Get current NFL game schedule with kickoff times"""
    try:
        schedule = get_schedule()
        return {
            "success": True,
            "games": schedule,
            "count": len(schedule)
        }
    except Exception as e:
        logger.error(f"Failed to get schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/status")
async def get_data_status():
    """Check data availability and freshness"""
    try:
        # Check input directory
        input_status = {}
        data_dir = "data/input"
        
        for pos in ["qb", "rb", "wr", "te", "dst"]:
            file_path = os.path.join(data_dir, f"{pos}.csv")
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                input_status[pos] = {
                    "exists": True,
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
            else:
                input_status[pos] = {"exists": False}
        
        # Load and analyze data
        df, warnings = load_data_from_input_dir()
        
        data_analysis = {}
        if df is not None and not df.empty:
            position_counts = df['POS'].value_counts().to_dict() if 'POS' in df.columns else {}
            salary_range = {
                "min": int(df['SALARY'].min()) if 'SALARY' in df.columns else 0,
                "max": int(df['SALARY'].max()) if 'SALARY' in df.columns else 0,
                "avg": int(df['SALARY'].mean()) if 'SALARY' in df.columns else 0
            }
            
            data_analysis = {
                "total_players": len(df),
                "positions": position_counts,
                "salary_range": salary_range,
                "has_projections": 'PROJ PTS' in df.columns,
                "has_ownership": 'OWN_PCT' in df.columns
            }
        
        return {
            "success": True,
            "input_files": input_status,
            "data_analysis": data_analysis,
            "warnings": warnings,
            "data_available": df is not None and not df.empty
        }
        
    except Exception as e:
        logger.error(f"Failed to get data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/status")
async def get_monitoring_status():
    """Get real-time monitoring status"""
    try:
        if not data_monitor:
            return {"status": "disabled", "message": "Monitoring not initialized"}
        
        recent_updates = await data_monitor.get_recent_updates(hours=24)
        
        # Categorize updates by severity
        high_severity = [u for u in recent_updates if u.get('severity', 0) >= 0.7]
        medium_severity = [u for u in recent_updates if 0.3 <= u.get('severity', 0) < 0.7]
        low_severity = [u for u in recent_updates if u.get('severity', 0) < 0.3]
        
        return {
            "status": "active",
            "total_updates_24h": len(recent_updates),
            "high_severity": len(high_severity),
            "medium_severity": len(medium_severity),
            "low_severity": len(low_severity),
            "recent_high_severity": high_severity[:5],
            "monitoring_intervals": {
                "news": data_monitor.news_interval,
                "weather": data_monitor.weather_interval,
                "injuries": data_monitor.injury_interval
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/swaps/summary")
async def get_swaps_summary():
    """Get automated swap activity summary"""
    try:
        if not auto_swap:
            return {"status": "disabled", "message": "Auto-swap not initialized"}
        
        summary = await auto_swap.get_swap_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get swap summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/swaps/manual")
async def manual_swap(
    player_out: str = Query(..., description="Player to remove"),
    player_in: str = Query(..., description="Player to add"),
    reason: str = Query("Manual swap", description="Reason for swap")
):
    """Execute a manual player swap"""
    try:
        if not auto_swap:
            raise HTTPException(status_code=503, detail="Auto-swap service not available")
        
        result = await auto_swap.manual_swap_request(player_out, player_in, reason)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Swap failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Manual swap failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/cost-summary")
async def get_ai_cost_summary():
    """Get AI usage and cost summary"""
    try:
        if not ai_analyzer:
            return {"status": "disabled", "message": "AI analyzer not initialized"}
        
        return ai_analyzer.get_cost_summary()
        
    except Exception as e:
        logger.error(f"Failed to get AI cost summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/analyze-news")
async def analyze_news_impact(
    player_name: str = Query(..., description="Player name"),
    news_text: str = Query(..., description="News text to analyze"),
    current_projection: float = Query(..., description="Current projection")
):
    """Analyze how news impacts a player's projection"""
    try:
        if not ai_analyzer:
            raise HTTPException(status_code=503, detail="AI analyzer not available")
        
        news_items = [{"title": "User provided", "summary": news_text}]
        
        new_projection, reasoning = await ai_analyzer.analyze_player_news_impact(
            player_name, news_items, current_projection
        )
        
        return {
            "player": player_name,
            "original_projection": current_projection,
            "adjusted_projection": new_projection,
            "adjustment": new_projection - current_projection,
            "reasoning": reasoning
        }
        
    except Exception as e:
        logger.error(f"News analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin endpoints
@app.post("/admin/process-inactives")
async def process_inactives():
    """Process inactive players and rebuild lineup"""
    try:
        # This would integrate with your existing inactive processing
        return {"status": "success", "message": "Inactives processed"}
    except Exception as e:
        logger.error(f"Failed to process inactives: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/mid-slate-review")
async def mid_slate_review():
    """Review mid-slate status and suggest adjustments"""
    try:
        # This would integrate with your existing review process
        return {"status": "success", "message": "Mid-slate review completed"}
    except Exception as e:
        logger.error(f"Failed to complete mid-slate review: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/final-swaps")
async def final_swaps():
    """Execute final swap opportunities"""
    try:
        # This would integrate with your existing swap logic
        return {"status": "success", "message": "Final swaps executed"}
    except Exception as e:
        logger.error(f"Failed to execute final swaps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
