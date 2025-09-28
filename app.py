"""
FastAPI web interface for DFS optimization system
FIXED: Proper JavaScript integration and async handling
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
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

class NewsRequest(BaseModel):
    news: str

# Global variables
scheduler = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application WITHOUT broken scheduler"""
    global scheduler

    logger.info("Starting NFL DFS Optimizer API")

    # SKIP broken scheduler for now - just focus on web interface
    logger.info("Skipping scheduler initialization - web-only mode")
    scheduler = None

    logger.info("API startup complete")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the FIXED main dashboard"""
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
            .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; font-size: 14px; }
            .button:hover { background: #2980b9; }
            .success { color: #27ae60; }
            .error { color: #e74c3c; }
            .lineup-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; margin: 10px 0; }
            .loading { color: #f39c12; }
            #output { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; min-height: 100px; }
            .controls { display: flex; align-items: center; gap: 15px; margin: 15px 0; flex-wrap: wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏈 NFL DFS Optimizer</h1>
                <p>Enhanced lineup optimization for friends league domination</p>
            </div>
            
            <div class="status-card">
                <h2>Quick Actions</h2>
                <div class="controls">
                    <select id="contestType" style="margin: 10px; padding: 8px;">
                        <option value="gpp">Tournament/GPP - High ceiling lineups</option>
                        <option value="cash">Cash Game - Consistent, safe lineups</option>
                        <option value="contrarian">Contrarian - Low ownership plays</option>
                        <option value="bestball">Best Ball - Highest scoring regardless of ownership</option>
                        <option value="single_game">Single Game - MVP + 5 FLEX</option>
                    </select>
                    
                    <label>
                        Number of lineups: 
                        <input type="number" id="numLineups" value="5" min="1" max="20" style="margin: 5px; padding: 5px; width: 60px;">
                    </label>
                    
                    <button class="button" onclick="generateLineups()">Generate Lineups</button>
                    <button class="button" onclick="forceUpdate()">Force Data Update</button>
                    <button class="button" onclick="testSystem()">Test System</button>
                </div>
            </div>
            
            <div id="output">
                <p>Ready to generate lineups! Make sure you have data/fanduel_salaries_manual.csv file.</p>
            </div>
        </div>
        
        <script>
            function log(message, type = 'info') {
                const output = document.getElementById('output');
                const timestamp = new Date().toLocaleTimeString();
                let className = '';
                let emoji = '📋';
                
                if (type === 'success') { className = 'success'; emoji = '✅'; }
                else if (type === 'error') { className = 'error'; emoji = '❌'; }
                else if (type === 'loading') { className = 'loading'; emoji = '⏳'; }
                
                output.innerHTML += `<div class="${className}">${emoji} [${timestamp}] ${message}</div>`;
                output.scrollTop = output.scrollHeight;
            }
            
            async function forceUpdate() {
                try {
                    log('🔄 Forcing data update...', 'loading');
                    
                    const response = await fetch('/update', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        log(`Data update: ${result.message}`, 'success');
                    } else {
                        log(`Update failed: ${result.message || 'Unknown error'}`, 'error');
                    }
                } catch (error) {
                    log(`Update error: ${error.message}`, 'error');
                }
            }
            
            async function generateLineups() {
                try {
                    const contestType = document.getElementById('contestType').value;
                    const numLineups = parseInt(document.getElementById('numLineups').value);
                    
                    log(`🧠 Generating ${numLineups} ${contestType.toUpperCase()} lineups...`, 'loading');
                    
                    const requestBody = { 
                        contest_type: contestType, 
                        num_lineups: numLineups,
                        avoid_high_ownership: contestType === 'gpp' || contestType === 'contrarian',
                        force_stacks: contestType !== 'cash'
                    };
                    
                    const response = await fetch('/optimize', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP ${response.status}`);
                    }
                    
                    const lineups = await response.json();
                    displayLineups(lineups, contestType);
                    
                } catch (error) {
                    log(`❌ Lineup generation failed: ${error.message}`, 'error');
                }
            }
            
            async function testSystem() {
                try {
                    log('🧪 Testing system components...', 'loading');
                    
                    const response = await fetch('/health');
                    const result = await response.json();
                    
                    if (response.ok) {
                        log(`✅ System health check passed: ${result.status}`, 'success');
                        
                        // Test data collection
                        log('Testing data collection...', 'loading');
                        await forceUpdate();
                        
                    } else {
                        log('❌ System health check failed', 'error');
                    }
                } catch (error) {
                    log(`❌ System test failed: ${error.message}`, 'error');
                }
            }
            
            function displayLineups(lineups, contestType) {
                if (!lineups || lineups.length === 0) {
                    log('No lineups generated', 'error');
                    return;
                }
                
                log(`✅ Generated ${lineups.length} ${contestType.toUpperCase()} lineups!`, 'success');
                
                lineups.slice(0, 3).forEach((lineup, index) => {
                    const playersHtml = lineup.players.map(player => 
                        `<li>${player}</li>`
                    ).join('');
                    
                    const lineupHtml = `
                        <div class="lineup-card">
                            <h3>Lineup ${index + 1} (${contestType.toUpperCase()})</h3>
                            <div><strong>Salary:</strong> $${lineup.total_salary.toLocaleString()}</div>
                            <div><strong>Projected:</strong> ${lineup.projected_points.toFixed(1)} pts</div>
                            <div><strong>Ownership:</strong> ${lineup.ownership_total.toFixed(1)}%</div>
                            <div><strong>Players:</strong></div>
                            <ul>${playersHtml}</ul>
                        </div>
                    `;
                    
                    document.getElementById('output').innerHTML += lineupHtml;
                });
                
                log(`💾 Lineups exported to CSV for FanDuel upload`, 'success');
            }
            
            // Initial status check
            window.addEventListener('load', function() {
                log('🚀 NFL DFS Optimizer loaded and ready!');
                log('📁 Make sure data/fanduel_salaries_manual.csv exists before generating lineups');
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/optimize")
async def optimize_lineups(request: OptimizationRequest):
    """Generate optimized lineups with FIXED error handling"""
    try:
        log(f"🧠 Starting {request.contest_type} optimization for {request.num_lineups} lineups...")

        # Get fresh data
        data = await get_fresh_data()

        if not data or not data.get('players'):
            logger.error("No player data available")
            raise HTTPException(status_code=400, detail="No player data available. Make sure data/fanduel_salaries_manual.csv exists!")

        logger.info(f"✅ Loaded {len(data['players'])} players")

        lineups = optimize_dfs_lineups(
            player_data=data['players'],
            weather_data=data.get('weather', {}),
            vegas_multipliers=data.get('vegas_multipliers', {}),
            num_lineups=request.num_lineups,
            contest_type=request.contest_type
        )

        if not lineups:
            raise HTTPException(status_code=400, detail="Optimization failed to generate valid lineups")

        logger.info(f"✅ Generated {len(lineups)} lineups successfully")

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
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def force_data_update():
    """Force an immediate data update"""
    try:
        logger.info("🔄 Force data update requested")
        data = await get_fresh_data()

        if data and data.get('players'):
            player_count = len(data['players'])
            quality = data.get('data_quality', {})
            current_week = quality.get('current_week', 'Unknown')

            message = f"Data updated successfully - {player_count} players loaded for Week {current_week}"
            logger.info(f"✅ {message}")
            return {"message": message, "status": "success"}
        else:
            message = "Data update completed but no players found. Check data/fanduel_salaries_manual.csv"
            logger.warning(message)
            return {"message": message, "status": "warning"}

    except Exception as e:
        logger.error(f"Force update failed: {e}")
        logger.error(traceback.format_exc())
        return {"message": f"Update failed: {str(e)}", "status": "error"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }

        # Check if we can access data
        try:
            data_dir_exists = DATA_DIR.exists()
            csv_file_exists = (DATA_DIR / "fanduel_salaries_manual.csv").exists()

            health_status.update({
                "data_dir_exists": data_dir_exists,
                "csv_file_exists": csv_file_exists,
                "data_status": "ready" if csv_file_exists else "missing_csv"
            })
        except Exception as e:
            health_status["data_error"] = str(e)

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}

@app.post("/process-news")
async def process_breaking_news(request: NewsRequest):
    """Process breaking news input"""
    try:
        news_text = request.news.strip()

        if not news_text:
            return {"status": "error", "message": "No news provided"}

        logger.info(f"📰 Processing news: {news_text[:100]}...")

        # Get current data
        fresh_data = await get_fresh_data()
        if not fresh_data or not fresh_data.get('players'):
            return {"status": "error", "message": "No player data available"}

        # Use AI to analyze the news (if available)
        try:
            from ai_analyzer import DualAIDFSAnalyzer
            analyzer = DualAIDFSAnalyzer()

            news_events = [{'text': news_text}]
            analysis = await analyzer.analyze_breaking_news(news_events, fresh_data['players'][:20])

            ai_response = analysis.get('full_analysis', 'No analysis available')

        except Exception as ai_error:
            logger.warning(f"AI analysis failed: {ai_error}")
            ai_response = f"Manual analysis needed for: {news_text}"

        return {
            "status": "success",
            "news": news_text,
            "ai_analysis": ai_response,
            "message": "News analyzed - check response for recommendations"
        }

    except Exception as e:
        logger.error(f"News processing failed: {e}")
        return {"status": "error", "message": f"Failed: {str(e)}"}

def log(message: str):
    """Helper function for logging"""
    logger.info(message)