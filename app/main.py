"""
Simplified FastAPI app that actually works
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from typing import List, Optional
import logging
import os
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our working components
from app.simple_optimizer import SimpleDFSOptimizer
from app.working_ai_analyzer import SimpleAIAnalyzer
from app.data_ingestion import load_weekly_data, load_data_from_input_dir

app = FastAPI(
    title="FanDuel NFL DFS Optimizer",
    description="Simple, working NFL DFS lineup optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
optimizer = SimpleDFSOptimizer()
ai_analyzer = SimpleAIAnalyzer()

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "FanDuel NFL DFS Optimizer v1.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "optimize": "/optimize",
            "optimize_text": "/optimize_text",
            "data_status": "/data/status"
        }
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "optimizer": "ready",
        "ai_analyzer": ai_analyzer.health_check(),
        "api_key_configured": ai_analyzer.api_available
    }

@app.get("/optimize")
async def optimize_lineup(
    game_type: str = Query("league", description="Game type: 'league' or 'h2h'"),
    lock: Optional[List[str]] = Query(None, description="Player names to lock"),
    ban: Optional[List[str]] = Query(None, description="Player names to ban"),
    enforce_stack: bool = Query(True, description="Enforce QB stacking"),
    use_ai: bool = Query(True, description="Use AI analysis"),
    salary_cap: int = Query(60000, description="Salary cap")
):
    """Generate optimized DFS lineup"""
    try:
        # Load player data
        logger.info("Loading player data...")
        df, warnings = load_data_from_input_dir()
        
        if df is None or df.empty:
            # Try weekly data as fallback
            df = load_weekly_data()
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail="No player data available. Please add CSV files to data/input/")
        
        logger.info(f"Loaded {len(df)} players")
        
        # Run optimization
        logger.info("Running optimization...")
        result = await optimizer.optimize_lineup(
            df=df,
            game_type=game_type,
            salary_cap=salary_cap,
            enforce_stack=enforce_stack,
            lock_players=lock,
            ban_players=ban
        )
        
        if not result.get('success'):
            error_msg = result.get('error', 'Optimization failed')
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Add AI analysis if requested
        ai_analysis = ""
        if use_ai:
            logger.info("Generating AI analysis...")
            try:
                ai_analysis = await ai_analyzer.analyze_lineup(result['lineup'], game_type)
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")
                ai_analysis = "AI analysis unavailable"
        
        # Add analysis to result
        result['ai_analysis'] = ai_analysis
        result['warnings'] = warnings
        
        logger.info(f"Optimization successful - {result['total_projection']:.1f} projected points")
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
        text_report = format_lineup_text(result, width)
        
        return text_report
        
    except HTTPException as e:
        return PlainTextResponse(
            content=f"Error: {e.detail}",
            status_code=e.status_code
        )

@app.get("/data/status")
async def get_data_status():
    """Check data availability"""
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
        
        # Try to load data
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

def format_lineup_text(result: dict, width: int = 100) -> str:
    """Format lineup as readable text"""
    if not result.get('success'):
        return f"Error: {result.get('error', 'Unknown error')}"
    
    game_type = result.get("game_type", "DFS").upper()
    title = f"FANDUEL NFL DFS LINEUP - {game_type} STRATEGY"
    line = "=" * len(title)
    
    header = f"{'POS':<4} {'PLAYER':<22} {'TEAM':<4} {'SALARY':>7} {'PROJ':>6} {'VALUE':>6}"
    separator = "-" * len(header)
    
    rows = [title, line, "OPTIMIZED LINEUP:", separator, header, separator]
    
    lineup = result.get("lineup", [])
    if not lineup:
        rows.append("No lineup generated.")
        return "\n".join(rows)

    total_proj = 0.0
    total_salary = 0
    
    for player in lineup:
        pos = player.get("position", "N/A")
        name = str(player.get("player_name", "N/A"))[:22]
        team = player.get("team", "N/A")
        salary = player.get("salary", 0)
        proj = float(player.get("projection", 0.0))
        
        total_proj += proj
        total_salary += salary
        
        value = (proj / (salary / 1000)) if salary > 0 else 0
        
        rows.append(
            f"{pos:<4} {name:<22} {team:<4} ${salary:>6} {proj:>6.1f} {value:>6.2f}"
        )

    rows.append(separator)
    rows.append(f"TOTALS:{'':<26} ${total_salary:>6}  {total_proj:>6.1f}")
    rows.append(f"SALARY REMAINING: ${result.get('salary_remaining', 0)}")
    
    # Add AI analysis if available
    ai_analysis = result.get('ai_analysis')
    if ai_analysis and ai_analysis != "AI analysis unavailable":
        rows.append(separator)
        rows.append("AI ANALYSIS:")
        rows.append(ai_analysis)
    
    rows.append(separator)
    
    return "\n".join(rows)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
