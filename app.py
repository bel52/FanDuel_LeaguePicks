"""
FastAPI web interface for DFS optimization system
Provides REST endpoints for data access and lineup generation
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import json
from datetime import datetime
from loguru import logger
import traceback

try:
    from data_collector import get_fresh_data
    from optimizer import optimize_dfs_lineups
    from scheduler import get_scheduler, start_background_scheduler
    from config import API_HOST, API_PORT, DATA_DIR
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure all modules are properly installed")
    # Create fallback objects to prevent startup failure
    DATA_DIR = Path("./data")
    API_HOST = "0.0.0.0"
    API_PORT = 8020

# Initialize FastAPI app
app = FastAPI(
    title="NFL DFS Optimizer",
    description="Automated NFL Daily Fantasy Sports lineup optimization system",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class OptimizationRequest(BaseModel):
    contest_type: str = "gpp"  # gpp, cash, single_game, or contrarian
    num_lineups: int = 10
    avoid_high_ownership: bool = True
    force_stacks: bool = True
    max_salary: int = 60000
    single_game_id: Optional[str] = None  # For single game contests

class LineupResponse(BaseModel):
    players: List[str]
    total_salary: int
    projected_points: float
    ownership_total: float
    correlation_score: float

# Global variables
scheduler = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global scheduler
    
    logger.info("Starting NFL DFS Optimizer API")
    
    try:
        # Start the background scheduler
        scheduler = start_background_scheduler()
        logger.info("Background scheduler started successfully")
    except Exception as e:
        logger.error(f"Failed to start background scheduler: {e}")
        scheduler = None
    
    logger.info("API startup complete")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NFL DFS Optimizer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .status-card { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .button:hover { background: #2980b9; }
            .success { color: #27ae60; }
            .error { color: #e74c3c; }
            .lineup-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; margin: 10px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏈 NFL DFS Optimizer</h1>
                <p>Enhanced lineup optimization with FLEX position support</p>
            </div>
            
            <div class="status-card">
                <h2>Quick Actions</h2>
                <button class="button" onclick="generateLineups()">Generate Tournament Lineups</button>
                <button class="button" onclick="generateCash()">Generate Cash Game Lineups</button>
                <button class="button" onclick="forceUpdate()">Force Data Update</button>
                
                <h3>Contest Type</h3>
                <select id="contestType" style="margin: 10px; padding: 8px;">
                    <option value="gpp">Tournament/GPP - High ceiling lineups</option>
                    <option value="cash">Cash Game - Consistent, safe lineups</option>
                    <option value="contrarian">Contrarian - Low ownership plays</option>
                    <option value="single_game">Single Game - MVP + 5 FLEX</option>
                </select>
                
                <label style="margin: 10px;">
                    Number of lineups: 
                    <input type="number" id="numLineups" value="5" min="1" max="20" style="margin: 5px; padding: 5px; width: 60px;">
                </label>
            </div>
            
            <div id="lineups-section" style="display: none;">
                <h2>Latest Lineups</h2>
                <div id="lineups-content"></div>
            </div>
        </div>
        
        <script>
            async function forceUpdate() {
                try {
                    const response = await fetch('/update', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message || 'Update completed');
                } catch (error) {
                    alert('Error forcing update: ' + error.message);
                }
            }
            
            async function generateLineups() {
                await generateLineupsType('gpp');
            }
            
            async function generateCash() {
                await generateLineupsType('cash');
            }
            
            async function generateLineupsType(type) {
                try {
                    const numLineups = parseInt(document.getElementById('numLineups').value);
                    
                    const requestBody = { 
                        contest_type: type, 
                        num_lineups: numLineups,
                        avoid_high_ownership: type === 'gpp' || type === 'contrarian',
                        force_stacks: type !== 'cash'
                    };
                    
                    const response = await fetch('/optimize', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });
                    const lineups = await response.json();
                    displayLineups(lineups, type);
                } catch (error) {
                    alert('Error generating lineups: ' + error.message);
                }
            }
            
            function displayLineups(lineups, contestType) {
                const section = document.getElementById('lineups-section');
                const content = document.getElementById('lineups-content');
                
                if (!lineups || lineups.length === 0) {
                    content.innerHTML = '<div class="error">No lineups generated</div>';
                    section.style.display = 'block';
                    return;
                }
                
                content.innerHTML = lineups.map((lineup, index) => {
                    const playersHtml = lineup.players.map(player => 
                        `<li>${player}</li>`
                    ).join('');
                    
                    return `
                        <div class="lineup-card">
                            <h3>Lineup ${index + 1} (${contestType.toUpperCase()})</h3>
                            <div><strong>Salary:</strong> $${lineup.total_salary.toLocaleString()}</div>
                            <div><strong>Projected:</strong> ${lineup.projected_points.toFixed(1)} pts</div>
                            <div><strong>Ownership:</strong> ${lineup.ownership_total.toFixed(1)}%</div>
                            <div><strong>Players:</strong></div>
                            <ul>${playersHtml}</ul>
                        </div>
                    `;
                }).join('');
                
                section.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/optimize")
async def optimize_lineups(request: OptimizationRequest):
    """Generate optimized lineups with FLEX position support"""
    try:
        # Get fresh data
        data = await get_fresh_data()
        
        if not data or not data.get('players'):
            raise HTTPException(status_code=400, detail="No player data available. Try updating data first.")
        
        lineups = optimize_dfs_lineups(
            player_data=data['players'],
            weather_data=data.get('weather', {}),
            num_lineups=request.num_lineups,
            contest_type=request.contest_type
        )
        
        if not lineups:
            raise HTTPException(status_code=400, detail="Optimization failed to generate valid lineups")
        
        response_lineups = []
        for lineup in lineups:
            response_lineups.append(LineupResponse(
                players=[f"{p.name} (${p.salary:,}) - {p.position}-{p.team}" for p in lineup.players],
                total_salary=lineup.total_salary,
                projected_points=round(lineup.projected_points, 2),
                ownership_total=round(lineup.ownership_total, 1),
                correlation_score=round(lineup.correlation_score, 3)
            ))
        
        return response_lineups
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def force_data_update():
    """Force an immediate data update"""
    try:
        data = await get_fresh_data()
        if data and data.get('players'):
            return {"message": f"Data updated successfully - {len(data['players'])} players loaded", "status": "success"}
        else:
            return {"message": "Data update completed but no players found", "status": "warning"}
    except Exception as e:
        logger.error(f"Force update failed: {e}")
        return {"message": f"Update failed: {str(e)}", "status": "error"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting DFS Optimizer API on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
