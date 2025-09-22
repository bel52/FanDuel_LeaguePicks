"""
FastAPI web interface with proper syntax and enhanced current week game detection
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
from datetime import datetime, timedelta
from loguru import logger
import traceback

try:
    from data_collector import get_fresh_data
    from optimizer import optimize_dfs_lineups, EnhancedDFSOptimizer
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
    description="Enhanced NFL Daily Fantasy Sports lineup optimization system",
    version="2.1.0"
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
    contest_type: str = "gpp"  # gpp, cash, single_game, contrarian
    num_lineups: int = 10
    avoid_high_ownership: bool = True
    force_stacks: bool = True
    max_salary: int = 60000
    single_game_id: Optional[str] = None

class GameInfo(BaseModel):
    id: str
    home_team: str
    away_team: str
    time: str
    entry_range: str
    total_points: Optional[float] = None
    week: int = 1

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
    contest_type: str
    ceiling_score: float = 0.0
    floor_score: float = 0.0

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
    
    logger.info("Starting Enhanced NFL DFS Optimizer API")
    
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
    """Enhanced dashboard with improved game detection"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NFL DFS Optimizer v2.1</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .header { text-align: center; color: #2c3e50; margin-bottom: 30px; }
            .status-card { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .button { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .button:hover { background: #2980b9; }
            .button.contrarian { background: #e74c3c; }
            .button.cash { background: #27ae60; }
            .success { color: #27ae60; }
            .warning { color: #f39c12; }
            .error { color: #e74c3c; }
            .lineup-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-top: 20px; }
            .lineup-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
            .lineup-card.cash { border-left-color: #27ae60; }
            .lineup-card.contrarian { border-left-color: #e74c3c; }
            .lineup-card.single_game { border-left-color: #9b59b6; }
            .game-selection { margin: 15px 0; padding: 15px; background: #fff; border: 1px solid #ddd; border-radius: 5px; }
            .contest-explanation { margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏈 NFL DFS Optimizer v2.1</h1>
                <p>Enhanced lineup optimization with proper contest differentiation</p>
            </div>
            
            <div class="status-card">
                <h2>System Status</h2>
                <div id="status-content">Loading...</div>
                <button class="button" onclick="refreshStatus()">Refresh Status</button>
                <button class="button" onclick="forceUpdate()">Force Data Update</button>
            </div>
            
            <div class="status-card">
                <h2>Lineup Generation</h2>
                
                <h3>Contest Type</h3>
                <select id="contestType" style="margin: 10px; padding: 8px; width: 100%;" onchange="handleContestTypeChange()">
                    <option value="gpp">🏆 Tournament/League (GPP) - High ceiling, correlation stacking</option>
                    <option value="cash">💰 Cash Game - Consistent, safe, high-floor lineups</option>
                    <option value="contrarian">🎯 Contrarian - Low ownership, leverage plays</option>
                    <option value="single_game">⚡ Single Game - One game only (MVP + 5 FLEX)</option>
                </select>
                
                <div class="contest-explanation" id="contestExplanation">
                    <strong>Tournament/League Strategy:</strong> Maximizes ceiling potential with correlation stacking (QB-WR combos), 
                    accepts higher risk for tournament upside. Uses ownership leverage to find differentiating plays.
                </div>
                
                <!-- Single Game Selection -->
                <div id="singleGameSelection" style="display: none;">
                    <h4>Select Single Game Contest</h4>
                    <div class="game-selection">
                        <select id="singleGame" style="width: 100%; padding: 8px; margin-bottom: 10px;">
                            <option value="">Loading current week games...</option>
                        </select>
                        <p style="margin: 5px 0; font-size: 14px; color: #666;">
                            <strong>Single Game Format:</strong> 1 MVP (1.5x points) + 5 FLEX positions (players from selected game only)
                        </p>
                    </div>
                </div>
                
                <div style="margin: 15px 0;">
                    <label style="margin-right: 20px;">
                        Number of lineups: 
                        <input type="number" id="numLineups" value="10" min="1" max="50" style="margin: 5px; padding: 5px; width: 60px;">
                    </label>
                    <button class="button" onclick="generateLineups()">Generate Optimized Lineups</button>
                </div>
            </div>
            
            <div id="lineups-section" style="display: none;">
                <h2>Latest Lineups</h2>
                <div id="lineups-content" class="lineup-grid"></div>
                <div style="margin: 20px 0;">
                    <button class="button" onclick="downloadCSV()">📄 Download Lineups CSV</button>
                    <button class="button" onclick="viewDetailedStats()">📊 View Detailed Stats</button>
                </div>
            </div>
            
            <div class="status-card">
                <h2>Quick Actions & Data</h2>
                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px;">
                    <button class="button" onclick="viewData()">View Current Data</button>
                    <button class="button" onclick="viewWeather()">🌤️ Weather Report</button>
                    <button class="button" onclick="viewInjuries()">🏥 Injury Reports</button>
                    <button class="button" onclick="checkCurrentWeek()">📅 Check Current Week</button>
                </div>
            </div>
            
            <div class="status-card">
                <h2>API Endpoints</h2>
                <ul>
                    <li><code>GET /status</code> - System status and data freshness</li>
                    <li><code>GET /data</code> - Current player data and projections</li>
                    <li><code>GET /games/current-week</code> - Current week games for single game contests</li>
                    <li><code>POST /optimize</code> - Generate custom lineups</li>
                    <li><code>GET /lineups</code> - Get latest generated lineups</li>
                    <li><code>GET /weather</code> - Current weather conditions</li>
                    <li><code>GET /injuries</code> - Latest injury reports</li>
                </ul>
            </div>
        </div>
        
        <script>
            // Global variables
            let availableGames = [];
            let currentWeekInfo = null;
            
            function handleContestTypeChange() {
                const contestType = document.getElementById('contestType').value;
                const singleGameSelection = document.getElementById('singleGameSelection');
                const contestExplanation = document.getElementById('contestExplanation');
                
                // Update explanation
                const explanations = {
                    'gpp': '<strong>Tournament/League Strategy:</strong> Maximizes ceiling potential with correlation stacking (QB-WR combos), accepts higher risk for tournament upside. Uses ownership leverage to find differentiating plays.',
                    'cash': '<strong>Cash Game Strategy:</strong> Prioritizes high floor and consistency. Focuses on safe, reliable plays with good value. Minimal stacking, emphasizes volume and matchup advantages.',
                    'contrarian': '<strong>Contrarian Strategy:</strong> Heavy ownership fade with ceiling chasing. Targets low-owned players with high upside. Unconventional stacks and leverage spots for large tournaments.',
                    'single_game': '<strong>Single Game Strategy:</strong> Game-specific correlation plays. MVP selection crucial (1.5x scoring). Stack players from high-scoring games with bring-back opportunities.'
                };
                contestExplanation.innerHTML = explanations[contestType];
                
                // Handle single game selection
                if (contestType === 'single_game') {
                    singleGameSelection.style.display = 'block';
                    document.getElementById('numLineups').value = 5;
                    loadCurrentWeekGames();
                } else {
                    singleGameSelection.style.display = 'none';
                    const defaultLineups = {'gpp': 20, 'cash': 5, 'contrarian': 15};
                    document.getElementById('numLineups').value = defaultLineups[contestType] || 10;
                }
            }
            
            async function loadCurrentWeekGames() {
                try {
                    const response = await fetch('/games/current-week');
                    const games = await response.json();
                    const select = document.getElementById('singleGame');
                    
                    select.innerHTML = '<option value="">Select a game...</option>';
                    
                    if (!games || games.length === 0) {
                        select.innerHTML = '<option value="">No games available for current week</option>';
                        return;
                    }
                    
                    console.log(`Loading ${games.length} games for week ${games[0]?.week || 'unknown'}:`, games);
                    
                    games.forEach(game => {
                        const option = document.createElement('option');
                        option.value = game.id;
                        option.textContent = `${game.away_team} @ ${game.home_team} - ${game.time} (Total: ${game.total_points || 'N/A'})`;
                        select.appendChild(option);
                    });
                    
                    availableGames = games;
                    console.log(`Successfully loaded ${availableGames.length} games for single game selection`);
                    
                } catch (error) {
                    console.error('Error loading current week games:', error);
                    document.getElementById('singleGame').innerHTML = '<option value="">Error loading games - check console</option>';
                }
            }
            
            async function checkCurrentWeek() {
                try {
                    const response = await fetch('/games/current-week');
                    const games = await response.json();
                    
                    if (games && games.length > 0) {
                        const week = games[0].week || 'Unknown';
                        const teams = [...new Set(games.flatMap(g => [g.home_team, g.away_team]))];
                        alert(`Current NFL Week: ${week}\\n\\nTeams Playing: ${teams.join(', ')}\\n\\nTotal Games: ${games.length}`);
                    } else {
                        alert('No current week data available');
                    }
                } catch (error) {
                    alert('Error checking current week: ' + error.message);
                }
            }
            
            async function refreshStatus() {
                try {
                    const response = await fetch('/status');
                    const status = await response.json();
                    document.getElementById('status-content').innerHTML = `
                        <div class="${status.scheduler_running ? 'success' : 'error'}">
                            Scheduler: ${status.scheduler_running ? '✅ Running' : '❌ Stopped'}
                        </div>
                        <div>Players: <strong>${status.current_data_summary.player_count || 0}</strong></div>
                        <div>Weather Locations: <strong>${status.current_data_summary.weather_locations || 0}</strong></div>
                        <div>Injury Reports: <strong>${status.current_data_summary.injury_reports || 0}</strong></div>
                        <div>Data Age: <em>${status.data_freshness.data_age || 'Unknown'}</em></div>
                        <div>Lineups Age: <em>${status.data_freshness.lineups_age || 'Unknown'}</em></div>
                    `;
                } catch (error) {
                    document.getElementById('status-content').innerHTML = '<div class="error">Error loading status</div>';
                }
            }
            
            async function forceUpdate() {
                try {
                    document.getElementById('status-content').innerHTML = '<div class="warning">⏳ Updating data...</div>';
                    const response = await fetch('/update', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message || 'Update completed');
                    refreshStatus();
                } catch (error) {
                    alert('Error forcing update: ' + error.message);
                    refreshStatus();
                }
            }
            
            async function generateLineups() {
                try {
                    const contestType = document.getElementById('contestType').value;
                    const numLineups = parseInt(document.getElementById('numLineups').value);
                    
                    // Show loading state
                    const button = event.target;
                    button.textContent = '⏳ Generating...';
                    button.disabled = true;
                    
                    let requestBody = { 
                        contest_type: contestType, 
                        num_lineups: numLineups,
                        avoid_high_ownership: contestType !== 'cash',
                        force_stacks: contestType !== 'cash' && contestType !== 'single_game'
                    };
                    
                    if (contestType === 'single_game') {
                        const selectedGame = document.getElementById('singleGame').value;
                        if (!selectedGame) {
                            alert('Please select a Single Game contest first');
                            button.textContent = 'Generate Optimized Lineups';
                            button.disabled = false;
                            return;
                        }
                        requestBody.single_game_id = selectedGame;
                    }
                    
                    const response = await fetch('/optimize', { 
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    const lineups = await response.json();
                    displayLineups(lineups, contestType);
                    
                    // Reset button
                    button.textContent = 'Generate Optimized Lineups';
                    button.disabled = false;
                    
                } catch (error) {
                    console.error('Lineup generation error:', error);
                    alert('Error generating lineups: ' + error.message);
                    // Reset button
                    event.target.textContent = 'Generate Optimized Lineups';
                    event.target.disabled = false;
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
                
                const contestTitles = {
                    'gpp': '🏆 Tournament Lineups (High Ceiling)',
                    'cash': '💰 Cash Game Lineups (High Floor)', 
                    'contrarian': '🎯 Contrarian Lineups (Low Ownership)',
                    'single_game': '⚡ Single Game Lineups (MVP + 5 FLEX)'
                };
                
                section.querySelector('h2').textContent = contestTitles[contestType] || 'Generated Lineups';
                
                content.innerHTML = lineups.slice(0, 8).map((lineup, index) => {
                    let playersHtml = '';
                    let cardClass = `lineup-card ${contestType}`;
                    
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
                        const positionOrder = ['QB', 'RB', 'WR', 'TE'];
                        positionOrder.forEach(pos => {
                            if (playersByPosition[pos]) {
                                if (pos === 'RB') {
                                    playersByPosition[pos].slice(0, 2).forEach((player, i) => {
                                        playersHtml += `<li><strong>${pos}${i > 0 ? i + 1 : ''}:</strong> ${player}</li>`;
                                    });
                                } else if (pos === 'WR') {
                                    playersByPosition[pos].slice(0, 3).forEach((player, i) => {
                                        playersHtml += `<li><strong>${pos}${i > 0 ? i + 1 : ''}:</strong> ${player}</li>`;
                                    });
                                } else if (pos === 'TE') {
                                    playersHtml += `<li><strong>TE:</strong> ${playersByPosition[pos][0]}</li>`;
                                } else if (pos === 'QB') {
                                    playersHtml += `<li><strong>QB:</strong> ${playersByPosition[pos][0]}</li>`;
                                }
                            }
                        });
                        
                        // Calculate FLEX
                        const usedPlayers = [];
                        ['RB', 'WR', 'TE'].forEach(pos => {
                            if (playersByPosition[pos]) {
                                const count = pos === 'RB' ? 2 : pos === 'WR' ? 3 : 1;
                                usedPlayers.push(...playersByPosition[pos].slice(0, count));
                            }
                        });
                        
                        const allFlexEligible = [
                            ...(playersByPosition['RB'] || []),
                            ...(playersByPosition['WR'] || []),
                            ...(playersByPosition['TE'] || [])
                        ];
                        
                        const flexPlayer = allFlexEligible.find(player => !usedPlayers.includes(player));
                        if (flexPlayer) {
                            playersHtml += `<li><strong>FLEX:</strong> ${flexPlayer}</li>`;
                        }
                        
                        // Defense
                        const defPlayer = (playersByPosition['DST'] && playersByPosition['DST'][0]) ||
                                         (playersByPosition['DEF'] && playersByPosition['DEF'][0]) ||
                                         (playersByPosition['D'] && playersByPosition['D'][0]);
                        if (defPlayer) {
                            playersHtml += `<li><strong>DEF:</strong> ${defPlayer}</li>`;
                        }
                    }
                    
                    // Contest-specific metrics
                    let metricsHtml = `
                        <div><strong>Total Salary:</strong> ${lineup.total_salary.toLocaleString()}</div>
                        <div><strong>Projected:</strong> ${lineup.projected_points.toFixed(1)} pts</div>
                        <div><strong>Ownership:</strong> ${lineup.ownership_total.toFixed(1)}%</div>
                    `;
                    
                    if (lineup.ceiling_score > 0) {
                        metricsHtml += `<div><strong>Ceiling:</strong> ${lineup.ceiling_score.toFixed(1)} pts</div>`;
                    }
                    if (lineup.floor_score > 0) {
                        metricsHtml += `<div><strong>Floor:</strong> ${lineup.floor_score.toFixed(1)} pts</div>`;
                    }
                    
                    return `
                        <div class="${cardClass}">
                            <h3>Lineup ${index + 1}</h3>
                            ${metricsHtml}
                            <div style="margin-top: 10px;"><strong>Players:</strong></div>
                            <ul style="font-size: 13px; margin: 5px 0;">
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
                    alert(`Current data: ${data.total_players} players loaded\\nLast updated: ${data.last_updated || 'Unknown'}`);
                } catch (error) {
                    alert('Error loading data: ' + error.message);
                }
            }
            
            async function downloadCSV() {
                try {
                    const response = await fetch('/lineups/csv');
                    if (response.ok) {
                        window.open('/lineups/csv', '_blank');
                    } else {
                        alert('No CSV files available to download');
                    }
                } catch (error) {
                    alert('Error downloading CSV: ' + error.message);
                }
            }
            
            async function viewWeather() {
                try {
                    const response = await fetch('/weather');
                    const data = await response.json();
                    const stadiumCount = Object.keys(data).length;
                    let weatherSummary = `Weather data loaded for ${stadiumCount} stadiums:\\n\\n`;
                    
                    Object.entries(data).slice(0, 5).forEach(([team, info]) => {
                        const forecast = info.forecast || {};
                        weatherSummary += `${team}: ${forecast.temperature || 'N/A'}°F, ${forecast.shortForecast || 'N/A'}\\n`;
                    });
                    
                    alert(weatherSummary);
                } catch (error) {
                    alert('Error loading weather: ' + error.message);
                }
            }
            
            async function viewInjuries() {
                try {
                    const response = await fetch('/injuries');
                    const data = await response.json();
                    alert(`${data.total_reports || 0} injury reports loaded\\nLast updated: ${data.last_updated || 'Unknown'}`);
                } catch (error) {
                    alert('Error loading injuries: ' + error.message);
                }
            }
            
            async function viewDetailedStats() {
                try {
                    const response = await fetch('/lineups');
                    const data = await response.json();
                    console.log('Detailed lineup stats:', data);
                    alert('Detailed stats logged to console (F12 to view)');
                } catch (error) {
                    alert('Error loading detailed stats: ' + error.message);
                }
            }
            
            // Auto-refresh status every 30 seconds
            setInterval(refreshStatus, 30000);
            
            // Initialize
            refreshStatus();
            handleContestTypeChange();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/games/current-week")
async def get_current_week_games():
    """Get current NFL week games with proper team filtering"""
    try:
        # Try to get current week info from scheduler first
        scheduler = get_scheduler()
        if scheduler and scheduler.current_data:
            current_data = scheduler.current_data
            
            # Check if we have ESPN data with current week games
            espn_data = current_data.get('espn_data', {})
            if 'scoreboard' in espn_data:
                games = await parse_espn_scoreboard_games(espn_data['scoreboard'])
                if games:
                    logger.info(f"Returning {len(games)} current week games from scheduler data")
                    return games
        
        # Fallback: collect fresh data
        from data_collector import EnhancedDataCollector
        async with EnhancedDataCollector() as collector:
            week_info = await collector.get_current_nfl_week()
            games = []
            
            if week_info and week_info.get('games'):
                for game in week_info['games']:
                    if len(game.get('teams', [])) >= 2:
                        teams = game['teams']
                        home_team = next((t['abbreviation'] for t in teams if t.get('is_home')), 'TBD')
                        away_team = next((t['abbreviation'] for t in teams if not t.get('is_home')), 'TBD')
                        
                        games.append({
                            "id": f"game_{game['id']}",
                            "away_team": away_team,
                            "home_team": home_team,
                            "time": game.get('date', 'TBD'),
                            "entry_range": "$1-$25",
                            "total_points": 47.5,  # Default total
                            "week": week_info['current_week']
                        })
            
            if games:
                logger.info(f"Generated {len(games)} games for week {week_info['current_week']}")
                return games
        
        # Final fallback: return sample current week games
        logger.warning("Using fallback sample games")
        return get_fallback_current_week_games()
        
    except Exception as e:
        logger.error(f"Error getting current week games: {e}")
        return get_fallback_current_week_games()

async def parse_espn_scoreboard_games(scoreboard_data: Dict) -> List[Dict]:
    """Parse ESPN scoreboard data into game format"""
    games = []
    
    try:
        current_week = scoreboard_data.get('week', {}).get('number', 1)
        
        for event in scoreboard_data.get('events', []):
            try:
                game_week = event.get('week', {}).get('number', current_week)
                if game_week != current_week:
                    continue  # Skip games from other weeks
                
                teams = []
                if 'competitions' in event:
                    for comp in event['competitions']:
                        if 'competitors' in comp:
                            for competitor in comp['competitors']:
                                team_info = competitor.get('team', {})
                                teams.append({
                                    'abbreviation': team_info.get('abbreviation', ''),
                                    'is_home': competitor.get('homeAway') == 'home'
                                })
                
                if len(teams) >= 2:
                    home_team = next((t['abbreviation'] for t in teams if t.get('is_home')), 'TBD')
                    away_team = next((t['abbreviation'] for t in teams if not t.get('is_home')), 'TBD')
                    
                    games.append({
                        "id": f"game_{event.get('id')}",
                        "away_team": away_team,
                        "home_team": home_team,
                        "time": event.get('date', 'TBD'),
                        "entry_range": "$1-$25",
                        "total_points": 47.5,
                        "week": current_week
                    })
                    
            except Exception as e:
                logger.warning(f"Error parsing individual game: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error parsing ESPN scoreboard: {e}")
    
    return games

def get_fallback_current_week_games() -> List[Dict]:
    """Fallback games for current week"""
    # This should be updated weekly - in production, you'd pull this from a database
    return [
        {
            "id": "game_1",
            "away_team": "PHI", 
            "home_team": "WAS",
            "time": "Sunday 1:00 PM ET",
            "entry_range": "$1-$25",
            "total_points": 47.5,
            "week": 3
        },
        {
            "id": "game_2",
            "away_team": "BAL",
            "home_team": "BUF", 
            "time": "Sunday 1:00 PM ET",
            "entry_range": "$1-$25",
            "total_points": 51.0,
            "week": 3
        },
        {
            "id": "game_3",
            "away_team": "DET",
            "home_team": "GB",
            "time": "Sunday 1:00 PM ET", 
            "entry_range": "$1-$25",
            "total_points": 49.5,
            "week": 3
        },
        {
            "id": "game_4",
            "away_team": "TEN",
            "home_team": "MIA",
            "time": "Monday 8:15 PM ET",
            "entry_range": "$1-$25", 
            "total_points": 44.5,
            "week": 3
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
    """Generate optimized lineups with enhanced contest differentiation"""
    try:
        scheduler = get_scheduler()
        current_data = scheduler.current_data
        
        if not current_data or not current_data.get('players'):
            raise HTTPException(status_code=400, detail="No player data available. Try updating data first.")
        
        # Handle single game optimization
        single_game_teams = None
        if request.contest_type == 'single_game' and request.single_game_id:
            single_game_teams = get_teams_from_game_id(request.single_game_id)
            if not single_game_teams:
                raise HTTPException(status_code=400, detail=f"Invalid game ID: {request.single_game_id}")
            
            # Filter players to only those from the selected game teams
            filtered_players = [
                player for player in current_data['players'] 
                if player.get('team') in single_game_teams
            ]
            if len(filtered_players) < 6:
                raise HTTPException(status_code=400, detail=f"Not enough players ({len(filtered_players)}) available for selected game. Need at least 6 players.")
        else:
            filtered_players = current_data['players']
        
        # Log optimization details
        logger.info(f"Optimizing {request.num_lineups} {request.contest_type} lineups with {len(filtered_players)} players")
        if single_game_teams:
            logger.info(f"Single game teams: {single_game_teams}")
        
        # Run optimization with enhanced parameters
        lineups = optimize_dfs_lineups(
            player_data=filtered_players,
            weather_data=current_data.get('weather', {}),
            num_lineups=request.num_lineups,
            contest_type=request.contest_type,
            single_game_teams=single_game_teams
        )
        
        if not lineups:
            raise HTTPException(status_code=400, detail="Optimization failed to generate valid lineups")
        
        # Format response
        response_lineups = []
        for lineup in lineups:
            response_lineups.append(LineupResponse(
                players=[f"{p.name} (${p.salary:,}) - {p.position}-{p.team}" for p in lineup.players],
                total_salary=lineup.total_salary,
                projected_points=round(lineup.projected_points, 2),
                ownership_total=round(lineup.ownership_total, 1),
                correlation_score=round(lineup.correlation_score, 3),
                contest_type=lineup.contest_type,
                ceiling_score=round(lineup.ceiling_score, 2),
                floor_score=round(lineup.floor_score, 2)
            ))
        
        logger.info(f"Successfully generated {len(response_lineups)} {request.contest_type} lineups")
        return response_lineups
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def get_teams_from_game_id(game_id: str) -> List[str]:
    """Get team codes from game ID - enhanced with current week logic"""
    # This mapping should be updated weekly in production
    game_team_mapping = {
        "game_1": ["PHI", "WAS"],
        "game_2": ["BAL", "BUF"],
        "game_3": ["DET", "GB"], 
        "game_4": ["TEN", "MIA"],
        # Add more games as needed
    }
    
    teams = game_team_mapping.get(game_id, [])
    logger.info(f"Game ID {game_id} maps to teams: {teams}")
    return teams

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
                        "ownership_total": round(lineup.ownership_total, 1),
                        "contest_type": lineup.contest_type,
                        "ceiling_score": round(lineup.ceiling_score, 2),
                        "floor_score": round(lineup.floor_score, 2)
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
        "version": "2.1.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Enhanced DFS Optimizer API on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
