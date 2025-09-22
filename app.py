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
    from optimizer import optimize_dfs_lineups, DFSOptimizer
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

class GameInfo(BaseModel):
    id: str
    home_team: str
    away_team: str
    time: str
    entry_range: str
    total_points: Optional[float] = None

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
    
    try:
        # Start the background scheduler
        scheduler = start_background_scheduler()
        logger.info("Background scheduler started successfully")
    except Exception as e:
        logger.error(f"Failed to start background scheduler: {e}")
        scheduler = None
    
    # Create static directories if they don't exist
    (Path(__file__).parent / "static").mkdir(exist_ok=True)
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        if scheduler:
            scheduler.stop_scheduler()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
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
                        <option value="">Loading games...</option>
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
            // Global variable to store available games
            let availableGames = [];
            
            function handleContestTypeChange() {
                const contestType = document.getElementById('contestType').value;
                const singleGameSelection = document.getElementById('singleGameSelection');
                
                if (contestType === 'single_game') {
                    singleGameSelection.style.display = 'block';
                    document.getElementById('numLineups').value = 5;
                    loadAvailableGames();
                } else {
                    singleGameSelection.style.display = 'none';
                    document.getElementById('numLineups').value = 10;
                }
            }
            
            async function loadAvailableGames() {
                try {
                    const response = await fetch('/games');
                    const games = await response.json();
                    const select = document.getElementById('singleGame');
                    
                    select.innerHTML = '<option value="">Select a game...</option>';
                    
                    if (!games || games.length === 0) {
                        select.innerHTML = '<option value="">No games available</option>';
                        return;
                    }
                    
                    console.log(`Loading ${games.length} games:`, games);
                    
                    // Sort games by time (Thursday, then Sunday morning, afternoon, night, then Monday)
                    const sortedGames = games.sort((a, b) => {
                        const timeOrder = {
                            'Thursday': 1,
                            'Sunday 1:00': 2,
                            'Sunday 4:05': 3,
                            'Sunday 4:25': 4,
                            'Sunday 8:20': 5,
                            'Monday': 6
                        };
                        
                        const getTimeKey = (time) => {
                            if (time.includes('Thursday')) return 'Thursday';
                            if (time.includes('Monday')) return 'Monday';
                            if (time.includes('Sunday 1:00')) return 'Sunday 1:00';
                            if (time.includes('Sunday 4:05')) return 'Sunday 4:05';
                            if (time.includes('Sunday 4:25')) return 'Sunday 4:25';
                            if (time.includes('Sunday 8:20')) return 'Sunday 8:20';
                            return 'Sunday 1:00'; // default
                        };
                        
                        return (timeOrder[getTimeKey(a.time)] || 99) - (timeOrder[getTimeKey(b.time)] || 99);
                    });
                    
                    sortedGames.forEach(game => {
                        const option = document.createElement('option');
                        option.value = game.id;
                        option.textContent = `${game.away_team} @ ${game.home_team} - ${game.time} (${game.entry_range})`;
                        select.appendChild(option);
                    });
                    
                    availableGames = sortedGames;
                    console.log(`Successfully loaded ${availableGames.length} games for single game selection`);
                    
                } catch (error) {
                    console.error('Error loading games:', error);
                    document.getElementById('singleGame').innerHTML = '<option value="">Error loading games - check console</option>';
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
                    alert(result.message || 'Update completed');
                    refreshStatus();
                } catch (error) {
                    alert('Error forcing update: ' + error.message);
                }
            }
            
            async function generateLineups() {
                try {
                    const contestType = document.getElementById('contestType').value;
                    const numLineups = parseInt(document.getElementById('numLineups').value);
                    
                    let requestBody = { 
                        contest_type: contestType, 
                        num_lineups: numLineups,
                        avoid_high_ownership: contestType === 'gpp' || contestType === 'contrarian',
                        force_stacks: contestType !== 'single_game' && contestType !== 'cash'
                    };
                    
                    if (contestType === 'single_game') {
                        const selectedGame = document.getElementById('singleGame').value;
                        if (!selectedGame) {
                            alert('Please select a Single Game contest first');
                            return;
                        }
                        requestBody.single_game_id = selectedGame;
                    }
                    
                    const response = await fetch('/optimize', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
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
                        // Single Game format: MVP + 5 FLEX
                        lineup.players.forEach((p, idx) => {
                            if (idx === 0) {
                                playersHtml += `<li><strong>MVP (1.5x):</strong> ${p}</li>`;
                            } else {
                                playersHtml += `<li><strong>FLEX${idx}:</strong> ${p}</li>`;
                            }
                        });
                    } else {
                        // Regular FanDuel format: QB, RB, RB, WR, WR, WR, TE, FLEX, DEF
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
                        
                        // Calculate FLEX (remaining RB/WR/TE)
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
                        
                        // Defense (DST/DEF)
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
                            <div><strong>Total Salary:</strong> ${lineup.total_salary.toLocaleString()}</div>
                            <div><strong>Projected Points:</strong> ${lineup.projected_points.toFixed(1)} ${contestType === 'single_game' ? '(with MVP 1.5x)' : ''}</div>
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
                    alert(`Current data: ${data.total_players} players loaded`);
                } catch (error) {
                    alert('Error loading data: ' + error.message);
                }
            }
            
            async function downloadCSV() {
                window.open('/lineups/csv', '_blank');
            }
            
            async function viewWeather() {
                try {
                    const response = await fetch('/weather');
                    const data = await response.json();
                    alert(`Weather data loaded for ${Object.keys(data).length} stadiums`);
                } catch (error) {
                    alert('Error loading weather: ' + error.message);
                }
            }
            
            async function viewInjuries() {
                try {
                    const response = await fetch('/injuries');
                    const data = await response.json();
                    alert(`${data.total_reports} injury reports loaded`);
                } catch (error) {
                    alert('Error loading injuries: ' + error.message);
                }
            }
            
            setInterval(refreshStatus, 30000);
            refreshStatus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/games")
async def get_available_games():
    """Get available single game contests for the current week"""
    try:
        # This would typically fetch from ESPN API or your data source
        # For now, we'll return sample games that update weekly
        current_week_games = [
            {
                "id": "game_1",
                "away_team": "PHI",
                "home_team": "WAS", 
                "time": "Sunday 1:00 PM ET",
                "entry_range": "$1-$25",
                "total_points": 47.5
            },
            {
                "id": "game_2", 
                "away_team": "BAL",
                "home_team": "BUF",
                "time": "Sunday 1:00 PM ET", 
                "entry_range": "$1-$25",
                "total_points": 51.0
            },
            {
                "id": "game_3",
                "away_team": "DET", 
                "home_team": "GB",
                "time": "Sunday 1:00 PM ET",
                "entry_range": "$1-$25", 
                "total_points": 49.5
            },
            {
                "id": "game_4",
                "away_team": "KC",
                "home_team": "LAC", 
                "time": "Sunday 4:25 PM ET",
                "entry_range": "$1-$25",
                "total_points": 53.0
            },
            {
                "id": "game_5",
                "away_team": "SF", 
                "home_team": "DAL",
                "time": "Sunday 8:20 PM ET",
                "entry_range": "$1-$25",
                "total_points": 46.0
            },
            {
                "id": "game_6",
                "away_team": "TEN",
                "home_team": "MIA",
                "time": "Monday 8:15 PM ET", 
                "entry_range": "$1-$25",
                "total_points": 44.5
            }
        ]
        
        # TODO: Replace with actual ESPN API call to get current week's games
        # scheduler = get_scheduler()
        # if scheduler and scheduler.current_data:
        #     espn_data = scheduler.current_data.get('espn_data', {})
        #     if 'scoreboard' in espn_data:
        #         games = parse_espn_games(espn_data['scoreboard'])
        #         return games
        
        return current_week_games
        
    except Exception as e:
        logger.error(f"Error getting available games: {e}")
        # Return fallback games even if API fails
        return [
            {
                "id": "fallback_1",
                "away_team": "TBD",
                "home_team": "TBD", 
                "time": "TBD",
                "entry_range": "$1-$25",
                "total_points": 45.0
            }
        ]

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
        
        # Handle single game optimization
        if request.contest_type == 'single_game' and request.single_game_id:
            # Filter players to only those from the selected game
            game_teams = get_teams_from_game_id(request.single_game_id)
            if game_teams:
                filtered_players = [
                    player for player in current_data['players'] 
                    if player.get('team') in game_teams
                ]
                if len(filtered_players) < 6:
                    raise HTTPException(status_code=400, detail=f"Not enough players available for selected game. Found {len(filtered_players)} players.")
            else:
                filtered_players = current_data['players']
        else:
            filtered_players = current_data['players']
        
        lineups = optimize_dfs_lineups(
            player_data=filtered_players,
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

def get_teams_from_game_id(game_id: str) -> List[str]:
    """Get team codes from game ID"""
    # This maps game IDs to team codes
    game_team_mapping = {
        "game_1": ["PHI", "WAS"],
        "game_2": ["BAL", "BUF"], 
        "game_3": ["DET", "GB"],
        "game_4": ["KC", "LAC"],
        "game_5": ["SF", "DAL"],
        "game_6": ["TEN", "MIA"]
    }
    
    return game_team_mapping.get(game_id, [])

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
        
        if not weather_data:
            raise HTTPException(status_code=404, detail="No weather data available")
        
        return weather_data
        
    except Exception as e:
        logger.error(f"Error getting weather data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/injuries")
async def get_injury_reports():
    """Get latest injury reports"""
    try:
        scheduler = get_scheduler()
        injury_data = scheduler.current_data.get('injuries', [])
        
        return {
            "injury_reports": injury_data,
            "total_reports": len(injury_data),
            "last_updated": scheduler.current_data.get('last_updated', 'Unknown')
        }
        
    except Exception as e:
        logger.error(f"Error getting injury reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update")
async def force_data_update(background_tasks: BackgroundTasks):
    """Force an immediate data update"""
    try:
        scheduler = get_scheduler()
        
        # Run update in background
        background_tasks.add_task(scheduler.force_update)
        
        return {"message": "Data update started", "status": "initiated"}
        
    except Exception as e:
        logger.error(f"Error initiating data update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
