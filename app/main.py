#!/usr/bin/env python3
"""
Production-Grade FanDuel NFL DFS Optimizer
Version: 4.0.0 - Zero FantasyPros Dependency

Complete free data integration with advanced AI analysis
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import httpx
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.data_collector import NFLDataCollector
from app.optimizer import AdvancedDFSOptimizer
from app.ai_integration import AIAnalyzer
from app.cache_manager import CacheManager
from app.monitoring import SystemMonitor
from app.lineup_validator import LineupValidator
from app.export_manager import ExportManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
data_collector = None
optimizer = None
ai_analyzer = None
cache_manager = None
system_monitor = None
lineup_validator = None
export_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global data_collector, optimizer, ai_analyzer, cache_manager, system_monitor, lineup_validator, export_manager
    
    logger.info("🚀 Initializing Production DFS System...")
    
    try:
        # Initialize core components
        cache_manager = CacheManager()
        data_collector = NFLDataCollector(cache_manager)
        optimizer = AdvancedDFSOptimizer(cache_manager)
        ai_analyzer = AIAnalyzer() if settings.OPENAI_API_KEY else None
        system_monitor = SystemMonitor()
        lineup_validator = LineupValidator()
        export_manager = ExportManager()
        
        # Start background monitoring
        asyncio.create_task(background_data_monitoring())
        
        logger.info("✅ System initialization complete!")
        logger.info(f"💰 Estimated weekly cost: ${settings.ESTIMATED_WEEKLY_COST}")
        logger.info("🔄 Starting background data collection...")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down system...")
        # Cleanup resources
        if cache_manager:
            await cache_manager.close()

# Create FastAPI app
app = FastAPI(
    title="Production FanDuel NFL DFS Optimizer",
    description="Zero-cost DFS optimization with 12 free data sources",
    version="4.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class OptimizationRequest(BaseModel):
    game_type: str = Field(default="gpp", description="gpp, cash, or tournament")
    salary_cap: int = Field(default=60000, description="FanDuel salary cap")
    num_lineups: int = Field(default=1, ge=1, le=150, description="Number of lineups to generate")
    lock_players: List[str] = Field(default=[], description="Players to lock in lineup")
    exclude_players: List[str] = Field(default=[], description="Players to exclude")
    stack_teams: List[str] = Field(default=[], description="Teams to stack (QB+pass catchers)")
    min_salary: int = Field(default=58000, description="Minimum total salary")
    unique_players: int = Field(default=3, description="Unique players between lineups")
    
class LineupResponse(BaseModel):
    lineup: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    total_salary: int
    projected_points: float
    ceiling_projection: float
    floor_projection: float
    stack_info: Dict[str, Any]
    
class SystemHealthResponse(BaseModel):
    status: str
    data_sources: Dict[str, bool]
    cache_stats: Dict[str, Any]
    ai_usage: Dict[str, Any]
    last_update: str
    estimated_cost: float

async def background_data_monitoring():
    """Background task to continuously monitor NFL data"""
    while True:
        try:
            # Update player data every 5 minutes
            await data_collector.update_all_sources()
            
            # Check for high-impact updates
            updates = await data_collector.get_recent_updates()
            high_impact = [u for u in updates if u.get('severity', 0) > 7]
            
            if high_impact:
                logger.info(f"🚨 High impact updates detected: {len(high_impact)}")
                # Trigger cache invalidation for affected players
                for update in high_impact:
                    if 'player_name' in update:
                        await cache_manager.invalidate_player(update['player_name'])
            
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            logger.error(f"Background monitoring error: {e}")
            await asyncio.sleep(60)  # Retry in 1 minute

@app.get("/", response_class=JSONResponse)
async def root():
    """System overview and quick start guide"""
    return {
        "system": "Production FanDuel NFL DFS Optimizer v4.0",
        "status": "operational",
        "features": [
            "12 free NFL data sources",
            "Zero FantasyPros dependency", 
            "Advanced AI analysis ($0.10/week)",
            "Real-time injury monitoring",
            "Monte Carlo simulation",
            "Correlation-aware stacking"
        ],
        "endpoints": {
            "optimize": "/optimize - Generate optimal lineups",
            "health": "/health - System health check",
            "players": "/players/current - Current player pool",
            "export": "/export/{format} - Export lineups"
        },
        "quick_start": "curl -X POST 'http://localhost:8000/optimize' -H 'Content-Type: application/json' -d '{\"game_type\":\"gpp\",\"num_lineups\":5}'"
    }

@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """Comprehensive system health check"""
    try:
        # Test all data sources
        data_sources = await data_collector.health_check()
        
        # Cache statistics
        cache_stats = await cache_manager.get_stats()
        
        # AI usage tracking
        ai_usage = {
            "enabled": ai_analyzer is not None,
            "weekly_cost": settings.ESTIMATED_WEEKLY_COST,
            "calls_today": cache_stats.get('ai_calls_today', 0),
            "model": settings.GPT_MODEL
        }
        
        return SystemHealthResponse(
            status="healthy",
            data_sources=data_sources,
            cache_stats=cache_stats,
            ai_usage=ai_usage,
            last_update=datetime.utcnow().isoformat(),
            estimated_cost=settings.ESTIMATED_WEEKLY_COST
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"System unhealthy: {str(e)}")

@app.post("/optimize", response_model=LineupResponse)
async def optimize_lineup(request: OptimizationRequest = None, background_tasks: BackgroundTasks = None):
    """Generate optimized DFS lineups using free data sources"""
    start_time = time.time()
    
    try:
        # Use default request if none provided
        if request is None:
            request = OptimizationRequest()
        
        logger.info(f"🎯 Optimizing {request.num_lineups} lineup(s) for {request.game_type}")
        
        # Get current player pool from all free sources
        player_data = await data_collector.get_current_player_pool()
        if not player_data or len(player_data) < 100:
            raise HTTPException(
                status_code=503, 
                detail="Insufficient player data. Free APIs may be updating."
            )
        
        # Apply AI analysis if available
        if ai_analyzer and len(player_data) > 0:
            enhanced_data = await ai_analyzer.enhance_projections(
                player_data, 
                request.game_type
            )
            player_data = enhanced_data
        
        # Generate optimal lineup(s)
        lineups = await optimizer.optimize_lineups(
            player_data=player_data,
            game_type=request.game_type,
            num_lineups=request.num_lineups,
            constraints={
                'salary_cap': request.salary_cap,
                'min_salary': request.min_salary,
                'lock_players': request.lock_players,
                'exclude_players': request.exclude_players,
                'stack_teams': request.stack_teams,
                'unique_players': request.unique_players
            }
        )
        
        if not lineups:
            raise HTTPException(status_code=400, detail="No valid lineups found")
        
        # Use first lineup for single request
        lineup = lineups[0] if len(lineups) == 1 else lineups
        
        # Calculate projections and metadata
        total_salary = sum(p['salary'] for p in lineup['players'])
        projected_points = sum(p['projection'] for p in lineup['players'])
        ceiling = sum(p.get('ceiling', p['projection'] * 1.3) for p in lineup['players'])
        floor = sum(p.get('floor', p['projection'] * 0.7) for p in lineup['players'])
        
        optimization_time = time.time() - start_time
        
        # Export lineup for FanDuel upload
        if background_tasks:
            background_tasks.add_task(
                export_manager.save_for_upload, 
                lineup['players'], 
                f"optimized_{int(time.time())}"
            )
        
        response = LineupResponse(
            lineup=lineup['players'],
            metadata={
                "optimization_time": f"{optimization_time:.2f}s",
                "data_sources": len(await data_collector.get_active_sources()),
                "ai_enhanced": ai_analyzer is not None,
                "game_type": request.game_type,
                "generated_at": datetime.utcnow().isoformat()
            },
            total_salary=total_salary,
            projected_points=round(projected_points, 2),
            ceiling_projection=round(ceiling, 2),
            floor_projection=round(floor, 2),
            stack_info=lineup.get('stack_info', {})
        )
        
        logger.info(f"✅ Optimization complete: {projected_points:.1f} pts, ${total_salary}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.get("/optimize/text")
async def optimize_text_format(
    game_type: str = Query("gpp"),
    num_lineups: int = Query(1),
    width: int = Query(100)
):
    """Generate lineup in text format for easy reading"""
    request = OptimizationRequest(game_type=game_type, num_lineups=num_lineups)
    result = await optimize_lineup(request)
    
    # Format as readable text
    text_output = f"""
╔═══════════════════════════════════════════════════════════════╗
║                  FANDUEL NFL LINEUP OPTIMIZER                ║
║                     Production v4.0                          ║
╚═══════════════════════════════════════════════════════════════╝

🎯 Game Type: {game_type.upper()}
💰 Total Salary: ${result.total_salary:,}
⚡ Projected Points: {result.projected_points:.1f}
📈 Ceiling: {result.ceiling_projection:.1f} | Floor: {result.floor_projection:.1f}
🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

LINEUP:
"""
    
    for i, player in enumerate(result.lineup, 1):
        pos = player['position'].ljust(4)
        name = player['name'][:20].ljust(20)
        team = player.get('team', 'UNK')[:3].ljust(3)
        salary = f"${player['salary']:,}".rjust(8)
        proj = f"{player['projection']:.1f}".rjust(5)
        
        text_output += f"{i:2}. {pos} {name} {team} {salary} {proj} pts\n"
    
    if result.stack_info and result.stack_info.get('primary_stack'):
        stack = result.stack_info['primary_stack']
        text_output += f"\n📊 Primary Stack: {stack['team']} ({', '.join(stack['positions'])})"
    
    return PlainTextResponse(text_output)

@app.get("/players/current")
async def get_current_players():
    """Get current player pool with projections"""
    try:
        players = await data_collector.get_current_player_pool()
        active_sources = await data_collector.get_active_sources()
        
        return {
            "players": len(players),
            "active_sources": len(active_sources),
            "sources": list(active_sources.keys()),
            "sample_players": players[:10] if players else [],
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/players/injuries")
async def get_injury_report():
    """Current injury report with DFS impact analysis"""
    try:
        injuries = await data_collector.get_injury_report()
        return {
            "injuries": injuries,
            "high_impact": [i for i in injuries if i.get('dfs_impact', 0) > 5],
            "updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather/current")
async def get_weather_conditions():
    """Current weather for all NFL games"""
    try:
        weather = await data_collector.get_game_weather()
        return {
            "weather_impacts": weather,
            "alerts": [w for w in weather if w.get('dfs_impact', 0) > 3],
            "updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export/fanduel")
async def export_fanduel_format(lineup_data: Dict):
    """Export lineup in FanDuel CSV format"""
    try:
        file_path = await export_manager.export_fanduel_csv(lineup_data['lineup'])
        return {"file_path": file_path, "status": "exported"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/costs")
async def get_cost_monitoring():
    """Track actual vs estimated costs"""
    try:
        usage_stats = await system_monitor.get_api_usage_stats()
        return {
            "estimated_weekly": settings.ESTIMATED_WEEKLY_COST,
            "actual_usage": usage_stats,
            "cost_breakdown": {
                "ai_analysis": usage_stats.get('openai_cost', 0),
                "odds_api": usage_stats.get('odds_cost', 0),
                "other_apis": 0.0  # All other APIs are free
            },
            "status": "under_budget" if usage_stats.get('total_cost', 0) < 15 else "review_needed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/force-update")
async def force_data_update():
    """Manually trigger data update from all sources"""
    try:
        results = await data_collector.force_update_all()
        return {
            "status": "completed",
            "sources_updated": len([r for r in results if r['success']]),
            "sources_failed": len([r for r in results if not r['success']]),
            "details": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
