"""
FastAPI web interface for DFS optimization system
Provides REST endpoints for data access and lineup generation
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups, DFSOptimizer
from scheduler import get_scheduler, start_background_scheduler
from config import API_HOST, API_PORT, DATA_DIR

# Initialize FastAPI app
app = FastAPI(
    title="NFL DFS Optimizer",
    description="Automated NFL Daily Fantasy Sports lineup optimization system",
    version="2.0.0"
)

# Pydantic models for API requests/responses
class OptimizationRequest(BaseModel):
    contest_type: str = "gpp"  # gpp, cash, single_game, or contrarian
    num_lineups: int = 10
    avoid_high_ownership: bool = True
    force_stacks: bool = True
    max_salary: int = 60000

class PlayerData(BaseModel):
    name: str
    position: str
    team: str
    salary: int
    projection: float
    ownership: Optional[float] = None

class LineupResponse(BaseModel):
    players: List[str]
    total_salary: int
    projected_points: float
    ownership_total: float
    correlation_score: float

class StatusResponse(BaseModel):
    status: str
    data_freshness: Dict[str, Any]
    last_update: Dict[str, str]
    current_data_summary: Dict[str, int]
    scheduler_running: bool

# Global variables
scheduler = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global scheduler
    
    # Setup logging
    logger.add(
        "logs/api_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    
    logger.info("Starting NFL DFS Optimizer API")
    
    # Start the background scheduler
    scheduler = start_background_scheduler()
    
    # Create static directories if they don't exist
    (Path(__file__).parent / "static").mkdir(exist_ok=True)
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if scheduler:
        scheduler.stop_scheduler()
    logger.info("API shutdown complete")

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
            .warning { color: #f39c12; }
            .error { color: #e74c3c; }
            .lineup-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
            .lineup-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏈 NFL DFS Optimizer</h1>
                <p>Automated lineup optimization for FanDuel NFL contests</p>
            </div>
            
            <div class="status-card">
                <h2>System Status</h2>
                <div id="status-content">Loading...</div>
                <button class="button" onclick="refreshStatus()">Refresh Status</button>
                <button class="button" onclick="forceUpdate()">Force Data Update</button>
                <button class="button" onclick="generateLineups()">Generate New Lineups</button>
            </div>
            
            <div class="status-card">
                <h2>Quick Actions</h2>
                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                    <button class="button" onclick="viewData()">View Current Data</button>
                    <button class="button" onclick="downloadCSV()">Download Lineups CSV</button>
                    <button class="button" onclick="viewWeather()">Weather Report</button>
                    <button class="button" onclick="viewInjuries()">Injury Reports</button>
                </div>
                
                <h3>Contest Type</h3>
                <select id="contestType" style="margin: 10px; padding: 8px;" onchange="handleContestTypeChange()">
                    <option value="gpp">Tournament/League (GPP) - High ceiling lineups</option>
                    <option value="cash">Cash Game - Consistent, safe lineups</option>
                    <option value="single_game">Single Game - One game only (MVP + 5 FLEX)</option>
                    <option value="contrarian">Contrarian - Low ownership plays</option>
                </select>
                
                <!-- Single Game Selection (hidden by default) -->
                <div id="singleGameSelection" style="display: none; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <h4>Select Single Game Contest</h4>
                    <select id="singleGame" style="width: 100%; padding: 8px;">
                        <option value="">Select a game...</option>
                        <option value="game_1">Sunday 1:00 PM - PHI @ WAS (Entry: $1-$25)</option>
                        <option value="game_2">Sunday 1:00 PM - BAL @ BUF (Entry: $1-$25)</option>
                        <option value="game_3">Sunday 1:00 PM - DET @ GB (Entry: $1-$25)</option>
                        <option value="game_4">Sunday 4:25 PM - KC @ LAC (Entry: $1-$25)</option>
                        <option value="game_5">Sunday 8:20 PM - SF @ DAL (Entry: $1-$25)</option>
                        <option value="game_6">Monday 8:15 PM - TEN @ MIA (Entry: $1-$25)</option>
                    </select>
                    <p style="margin: 10px 0; font-size: 14px; color: #666;">
                        <strong>Single Game Format:</strong> 1 MVP (1.5x points) + 5 FLEX positions (players from selected game only)
                    </p>
                </div>
                
                <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 14px;">
                    <strong>Contest Type Explanations:</strong><br>
                    <strong>Tournament/League:</strong> High-upside lineups for large tournaments. Uses correlation stacking.<br>
                    <strong>Cash Game:</strong> Consistent, high-floor players for 50/50 or double-up contests.<br>
                    <strong>Single Game:</strong> Pick one NFL game and build lineup from only those two teams.<br>
                    <strong>Contrarian:</strong> Low-ownership players to differentiate from the field.
                </div>
                
                <label style="margin: 10px;">
                    Number of lineups: 
                    <input type="number" id="numLineups" value="10" min="1" max="50" style="margin: 5px; padding: 5px; width: 60px;">
                </label>
            </div>
            
            <div id="lineups-section" style="display: none;">
                <h2>Latest Lineups</h2>
                <div id="lineups-content" class="lineup-grid"></div>
            </div>
            
            <div class="status-card">
                <h2>API Endpoints</h2>
                <ul>
                    <li><code>GET /status</code> - System status and data freshness</li>
                    <li><code>GET /data</code> - Current player data and projections</li>
                    <li><code>POST /optimize</code> - Generate custom lineups</li>
                    <li><code>GET /lineups</code> - Get latest generated lineups</li>
                    <li><code>GET /weather</code> - Current weather conditions</li>
                    <li><code>GET /injuries</code> - Latest injury reports</li>
                </ul>
            </div>
        </div>
        
        <script>
            function handleContestTypeChange() {
                const contestType = document.getElementById('contestType').value;
                const singleGameSelection = document.getElementById('singleGameSelection');
                
                if (contestType === 'single_game') {
                    singleGameSelection.style.display = 'block';
                    document.getElementById('numLineups').value = 5;
                } else {
                    singleGameSelection.style.display = 'none';
                    document.getElementById('numLineups').value = 10;
                }
            }
            
            async function refreshStatus() {
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    document.getElementById('status-content').innerHTML = `
                        <div class="${status.scheduler_running ? 'success' : 'error'}">
                            Scheduler: ${status.scheduler_running ? 'Running' : 'Stopped'}
                        </div>
                        <div>Players: ${status.current_data_summary.player_count || 0}</div>
                        <div>Weather Locations: ${status.current_data_summary.weather_locations || 0}</div>
                        <div>Injury Reports: ${status.current_data_summary.injury_reports || 0}</div>
                        <div>Data Age: ${status.data_freshness.data_age || 'Unknown'}</div>
                        <div>Lineups Age: ${status.data_freshness.lineups_age || 'Unknown'}</div>
                    `;
                } catch (error) {
                    document.getElementById('status-content').innerHTML = '<div class="error">Error loading status</div>';
                }
            }
            
            async function forceUpdate() {
                try {
                    const response = await fetch('/update', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message);
                    refreshStatus();
                } catch (error) {
                    alert('Error forcing update: ' + error.message);
                }
            }
            
            async function generateLineups() {
                try {
                    const contestType = document.getElementById('contestType').value;
                    const numLineups = parseInt(document.getElementById('numLineups').value);
                    
                    if (contestType === 'single_game') {
                        const selectedGame = document.getElementById('singleGame').value;
                        if (!selectedGame) {
                            alert('Please select a Single Game contest first');
                            return;
                        }
                    }
                    
                    const response = await fetch('/optimize', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            contest_type: contestType, 
                            num_lineups: numLineups,
                            avoid_high_ownership: contestType === 'gpp' || contestType === 'contrarian',
                            force_stacks: contestType !== 'single_game' && contestType !== 'cash'
                        })
                    });
                    const lineups = await response.json();
                    displayLineups(lineups, contestType);
                } catch (error) {
                    alert('Error generating lineups: ' + error.message);
                }
            }
            
            function displayLineups(lineups, contestType) {
                const section = document.getElementById('lineups-section');
                const content = document.getElementById('lineups-content');
                
                if (lineups.length === 0) {
                    content.innerHTML = '<div class="error">No lineups generated</div>';
                    return;
                }
                
                const formatTitle = contestType === 'single_game' ? 'Single Game Lineups (MVP + 5 FLEX)' : 'Latest Lineups';
                section.querySelector('h2').textContent = formatTitle;
                
                content.innerHTML = lineups.slice(0, 6).map((lineup, index) => {
                    let playersHtml = '';
                    
                    if (contestType === 'single_game') {
                        lineup.players.forEach((p, idx) => {
                            if (idx === 0) {
                                playersHtml += `<li><strong>MVP (1.5x):</strong> ${p}</li>`;
                            } else {
                                playersHtml += `<li><strong>FLEX${idx}:</strong> ${p}</li>`;
                            }
                        });
                    } else {
                        const playersByPosition = {};
                        
                        lineup.players.forEach(playerStr => {
                            const match = playerStr.match(/- (\w+)-/);
                            if (match) {
                                const position = match[1];
                                if (!playersByPosition[position]) {
                                    playersByPosition[position] = [];
                                }
                                playersByPosition[position].push(playerStr);
                            }
                        });
                        
                        // Display in FanDuel order
                        if (playersByPosition['QB'] && playersByPosition['QB'][0]) {
                            playersHtml += `<li><strong>QB:</strong> ${playersByPosition['QB'][0]}</li>`;
                        }
                        
                        if (playersByPosition['RB']) {
                            if (playersByPosition['RB'][0]) {
                                playersHtml += `<li><strong>RB:</strong> ${playersByPosition['RB'][0]}</li>`;
                            }
                            if (playersByPosition['RB'][1]) {
                                playersHtml += `<li><strong>RB:</strong> ${playersByPosition['RB'][1]}</li>`;
                            }
                        }
                        
                        if (playersByPosition['WR']) {
                            if (playersByPosition['WR'][0]) {
                                playersHtml += `<li><strong>WR:</strong> ${playersByPosition['WR'][0]}</li>`;
                            }
                            if (playersByPosition['WR'][1]) {
                                playersHtml += `<li><strong>WR:</strong> ${playersByPosition['WR'][1]}</li>`;
                            }
                            if (playersByPosition['WR'][2]) {
                                playersHtml += `<li><strong>WR:</strong> ${playersByPosition['WR'][2]}</li>`;
                            }
                        }
                        
                        if (playersByPosition['TE'] && playersByPosition['TE'][0]) {
                            playersHtml += `<li><strong>TE:</strong> ${playersByPosition['TE'][0]}</li>`;
                        }
                        
                        const usedPlayers = [];
                        if (playersByPosition['RB']) usedPlayers.push(...playersByPosition['RB'].slice(0, 2));
                        if (playersByPosition['WR']) usedPlayers.push(...playersByPosition['WR'].slice(0, 3));
                        if (playersByPosition['TE']) usedPlayers.push(...playersByPosition['TE'].slice(0, 1));
                        
                        const allFlexEligible = [
                            ...(playersByPosition['RB'] || []),
                            ...(playersByPosition['WR'] || []),
                            ...(playersByPosition['TE'] || [])
                        ];
                        
                        const flexPlayer = allFlexEligible.find(player => !usedPlayers.includes(player));
                        if (flexPlayer) {
                            playersHtml += `<li><strong>FLEX:</strong> ${flexPlayer}</li>`;
                        }
                        
                        if (playersByPosition['K'] && playersByPosition['K'][0]) {
                            playersHtml += `<li><strong>K:</strong> ${playersByPosition['K'][0]}</li>`;
                        }
                        
                        const defPlayer = (playersByPosition['DST'] && playersByPosition['DST'][0]) ||
                                         (playersByPosition['DEF'] && playersByPosition['DEF'][0]) ||
                                         (playersByPosition['D'] && playersByPosition['D'][0]);
                        if (defPlayer) {
                            playersHtml += `<li><strong>DEF:</strong> ${defPlayer}</li>`;
                        }
                    }
                    
                    return `
                        <div class="lineup-card">
                            <h3>Lineup ${index + 1} ${contestType === 'single_game' ? '(Single Game)' : ''}</h3>
                            <div><strong>Projected Points:</strong> ${lineup.projected_points.toFixed(1)} ${contestType === 'single_game' ? '(with MVP 1.5x)' : ''}</div>
                            <div><strong>Salary:</strong> $${lineup.total_salary.toLocaleString()}</div>
                            <div><strong>Ownership:</strong> ${lineup.ownership_total.toFixed(1)}%</div>
                            <div><strong>Players:</strong></div>
                            <ul style="font-size: 14px;">
                                ${playersHtml}
                            </ul>
                        </div>
                    `;
                }).join('');
                
                section.style.display = 'block';
            }
            
            async function viewData() {
                try {
                    const response = await fetch('/data');
                    const data = await response.json();
                    
                    showDataSection('data-display', `Current Player Data (${data.total_players} players)`);
                    
                    let html = `<p>Last Updated: ${data.last_updated}</p>
                        <div style="max-height: 400px; overflow-y: auto;">
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="background-color: #f2f2f2;">
                                <th style="border: 1px solid #ddd; padding: 8px;">Name</th>
                                <th style="border: 1px solid #ddd; padding: 8px;">Position</th>
                                <th style="border: 1px solid #ddd; padding: 8px;">Team</th>
                                <th style="border: 1px solid #ddd; padding: 8px;">Salary</th>
                                <th style="border: 1px solid #ddd; padding: 8px;">Projection</th>
                                <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
                            </tr>`;
                    
                    data.players.forEach(player => {
                        html += `
                            <tr style="border: 1px solid #ddd;">
                                <td style="border: 1px solid #ddd; padding: 8px;">${player.player_name || player.name || 'Unknown'}</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">${player.position}</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">${player.team}</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">$${player.salary ? player.salary.toLocaleString() : 'N/A'}</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">${player.projection ? player.projection.toFixed(1) : 'N/A'}</td>
                                <td style="border: 1px solid #ddd; padding: 8px;">${player.value ? player.value.toFixed(2) : 'N/A'}</td>
                            </tr>
                        `;
                    });
                    
                    html += '</table></div>' + getHideButton('data-display');
                    document.getElementById('data-display').innerHTML += html;
                } catch (error) {
                    alert('Error loading data: ' + error.message);
                }
            }
            
            async function viewWeather() {
                try {
                    const response = await fetch('/weather');
                    const data = await response.json();
                    
                    showDataSection('weather-display', 'Weather Report');
                    
                    let html = '';
                    Object.entries(data).forEach(([team, weather]) => {
                        const forecast = weather.forecast || {};
                        const isDome = ['ATL', 'DET', 'HOU', 'IND', 'LV', 'LAR', 'MIN', 'NO', 'ARI'].includes(team);
                        
                        html += `
                            <div style="border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px; background-color: ${isDome ? '#d1ecf1' : '#fff3cd'};">
                                <h3>${team} - ${weather.stadium}</h3>
                                ${isDome ? '<p><strong>DOME - No weather impact</strong></p>' : ''}
                                <p><strong>Conditions:</strong> ${forecast.name || 'Unknown'}</p>
                                <p><strong>Temperature:</strong> ${forecast.temperature || 'N/A'}°F</p>
                                <p><strong>Wind:</strong> ${forecast.windSpeed || 'N/A'}</p>
                                <p><strong>Forecast:</strong> ${forecast.detailedForecast || 'No details available'}</p>
                            </div>
                        `;
                    });
                    
                    document.getElementById('weather-display').innerHTML += html + getHideButton('weather-display');
                } catch (error) {
                    alert('Error loading weather: ' + error.message);
                }
            }
            
            async function viewInjuries() {
                try {
                    const response = await fetch('/injuries');
                    const data = await response.json();
                    
                    showDataSection('injury-display', `Latest Injury Reports (${data.total_reports} reports)`);
                    
                    let html = `<p>Last Updated: ${data.last_updated}</p>`;
                    
                    data.injury_reports.forEach(report => {
                        html += `
                            <div style="border: 1px solid #ddd; margin: 10px; padding: 15px; border-radius: 5px;">
                                <h3>${report.headline}</h3>
                                <p>${report.description}</p>
                                <p><small>Published: ${report.published}</small></p>
                                ${report.link ? `<p><a href="${report.link}" target="_blank">Read More</a></p>` : ''}
                            </div>
                        `;
                    });
                    
                    document.getElementById('injury-display').innerHTML += html + getHideButton('injury-display');
                } catch (error) {
                    alert('Error loading injuries: ' + error.message);
                }
            }
            
            function showDataSection(sectionId, title) {
                ['data-display', 'weather-display', 'injury-display'].forEach(id => {
                    const section = document.getElementById(id);
                    if (section) section.remove();
                });
                
                const newSection = document.createElement('div');
                newSection.id = sectionId;
                newSection.style.cssText = 'margin-top: 20px; padding: 20px; background: white; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1);';
                newSection.innerHTML = `<h2>${title}</h2>`;
                document.querySelector('.container').appendChild(newSection);
                newSection.scrollIntoView({ behavior: 'smooth' });
            }
            
            function getHideButton(sectionId) {
                return `<button onclick="hideDataDisplay('${sectionId}')" style="margin-top: 10px; padding: 8px 16px; background: #e74c3c; color: white; border: none; border-radius: 3px;">Hide</button>`;
            }
            
            function hideDataDisplay(sectionId) {
                const dataSection = document.getElementById(sectionId || 'data-display');
                if (dataSection) {
                    dataSection.remove();
                }
            }
            
            async function downloadCSV() {
                window.open('/lineups/csv', '_blank');
            }
            
            setInterval(refreshStatus, 30000);
            refreshStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status"""
    try:
        scheduler = get_scheduler()
        status_data = scheduler.get_status()
        
        return StatusResponse(
            status="operational",
            data_freshness=status_data['data_freshness'],
            last_update={k: v.isoformat() if isinstance(v, datetime) else str(v) 
                        for k, v in status_data['last_updates'].items()},
            current_data_summary=status_data['current_data_summary'],
            scheduler_running=status_data['is_running']
        )
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data")
async def get_current_data():
    """Get current player data and projections"""
    try:
        scheduler = get_scheduler()
        current_data = scheduler.current_data
        
        if not current_data:
            current_data = await get_fresh_data()
        
        return {
            "players": current_data.get('players', []),
            "last_updated": current_data.get('last_updated'),
            "data_quality": current_data.get('data_quality', {}),
            "total_players": len(current_data.get('players', []))
        }
    except Exception as e:
        logger.error(f"Error getting current data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_lineups(request: OptimizationRequest):
    """Generate optimized lineups"""
    try:
        scheduler = get_scheduler()
        current_data = scheduler.current_data
        
        if not current_data or not current_data.get('players'):
            raise HTTPException(status_code=400, detail="No player data available. Try updating data first.")
        
        lineups = optimize_dfs_lineups(
            player_data=current_data['players'],
            weather_data=current_data.get('weather', {}),
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

@app.get("/lineups")
async def get_latest_lineups():
    """Get the latest generated lineups"""
    try:
        scheduler = get_scheduler()
        
        if not scheduler.latest_lineups:
            raise HTTPException(status_code=404, detail="No lineups available. Generate lineups first.")
        
        response = {}
        for lineup_type, lineups in scheduler.latest_lineups.items():
            if lineups:
                response[lineup_type] = [
                    {
                        "players": [f"{p.name} (${p.salary:,}) - {p.position}-{p.team}" for p in lineup.players],
                        "total_salary": lineup.total_salary,
                        "projected_points": round(lineup.projected_points, 2),
                        "ownership_total": round(lineup.ownership_total, 1)
                    }
                    for lineup in lineups[:5]
                ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting lineups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lineups/csv")
async def download_lineups_csv():
    """Download latest lineups as CSV"""
    try:
        lineup_dir = DATA_DIR / "lineups"
        csv_files = list(lineup_dir.glob("*.csv"))
        
        if not csv_files:
            raise HTTPException(status_code=404, detail="No lineup CSV files available")
        
        latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)
        
        return FileResponse(
            path=str(latest_csv),
            filename=f"dfs_lineups_{datetime.now().strftime('%Y%m%d')}.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Error downloading CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/weather")
async def get_weather_data():
    """Get current weather conditions for NFL stadiums"""
    try:
        scheduler = get_scheduler()
        weather_data = scheduler.current_data.get('weather', {})
