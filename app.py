"""
FastAPI web interface for DFS optimization system
FIXED: H2H mode now properly loads games and players from correct CSV
"""
from fastapi import Request
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import os
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
    DATA_DIR = Path("./data")
    API_HOST = "0.0.0.0"
    API_PORT = 8020

# Initialize FastAPI app
app = FastAPI(
    title="NFL DFS Optimizer",
    description="Enhanced NFL Daily Fantasy Sports lineup optimization system",
    version="2.2.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models
class OptimizationRequest(BaseModel):
    contest_type: str = "gpp"
    num_lineups: int = 10
    locked_players: List[str] = []
    excluded_players: List[str] = []
    avoid_high_ownership: bool = True
    force_stacks: bool = True
    max_salary: int = 60000
    use_ai: bool = True
    selected_game: Optional[str] = None

class LineupResponse(BaseModel):
    players: List[str]
    total_salary: int
    projected_points: float
    ownership_total: float
    correlation_score: float

# Global variable to store current player data
current_player_data = None


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Enhanced dashboard with H2H support"""
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>NFL DFS Optimizer Pro</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1600px; margin: 0 auto; background: white; min-height: 100vh; box-shadow: 0 0 20px rgba(0,0,0,0.1); }
            .header { background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color: white; padding: 20px; text-align: center; }
            .header h1 { margin: 0; font-size: 2.2em; font-weight: 300; }
            .header p { margin: 8px 0 0 0; opacity: 0.9; }

            .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; padding: 20px; }

            .left-panel { }
            .controls-section { background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .main-controls { display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center; margin-bottom: 15px; }
            .control-group { display: flex; flex-direction: column; align-items: center; }
            .control-group label { font-weight: 600; margin-bottom: 5px; color: #495057; font-size: 14px; }
            select, input[type="number"] { padding: 8px; border: 2px solid #e9ecef; border-radius: 6px; font-size: 14px; }
            select:focus, input:focus { outline: none; border-color: #007bff; }
            .button { background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.3s; }
            .button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,123,255,0.3); }
            .button:disabled { background: #6c757d; cursor: not-allowed; transform: none; }

            .search-section { text-align: center; margin: 15px 0; }
            .search-box { padding: 10px; border: 2px solid #e9ecef; border-radius: 6px; width: 250px; font-size: 14px; }

            .lineup-display { background: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
            .lineup-tabs { display: flex; gap: 10px; margin-bottom: 15px; }
            .lineup-tab { padding: 8px 15px; background: #e9ecef; border-radius: 6px; cursor: pointer; font-size: 14px; }
            .lineup-tab.active { background: #007bff; color: white; }
            .fanduel-lineup { display: none; }
            .fanduel-lineup.active { display: block; }
            .position-slots { display: grid; grid-template-columns: 1fr; gap: 8px; }
            .position-slot { display: flex; justify-content: space-between; align-items: center; background: white; padding: 12px; border-radius: 6px; border-left: 4px solid #007bff; }
            .position-label { font-weight: 600; color: #495057; width: 50px; }
            .player-info { flex: 1; margin-left: 15px; }
            .player-name { font-weight: 600; color: #2c3e50; }
            .player-details { font-size: 12px; color: #6c757d; }
            .player-salary { font-weight: 600; color: #28a745; }

            .right-panel { }
            .players-section { background: #f8f9fa; border-radius: 10px; padding: 20px; }
            .position-group { margin-bottom: 20px; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden; }
            .position-header { background: linear-gradient(135deg, #495057 0%, #6c757d 100%); color: white; padding: 12px 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
            .player-table { width: 100%; border-collapse: collapse; display: none; font-size: 13px; }
            .player-table.active { display: table; }
            .player-table th { background: #f8f9fa; padding: 8px; text-align: left; font-weight: 600; border-bottom: 2px solid #dee2e6; }
            .player-table td { padding: 6px 8px; border-bottom: 1px solid #dee2e6; }
            .player-row:hover { background: #f8f9fa; }
            .lock-controls { display: flex; gap: 8px; align-items: center; }
            .lock-checkbox { width: 16px; height: 16px; cursor: pointer; }
            .locked { background-color: #d4edda !important; }
            .excluded { background-color: #f8d7da !important; }
            .output-section { background: #f8f9fa; border-radius: 10px; padding: 15px; margin-top: 20px; max-height: 300px; overflow-y: auto; }
            .success { color: #28a745; }
            .error { color: #dc3545; }
            .loading { color: #ffc107; }
            .hidden { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="text-3xl font-bold text-center mb-2">🏈 LeathAI NFL Optimizer Pro 🦁</h1>
                <p class="text-center text-gray-600 mb-6">Taking Their Money - One Slate At A Time</p>
            </div>

            <div class="main-content">
                <!-- LEFT PANEL -->
                <div class="left-panel">
                    <div class="controls-section">
                        <div class="main-controls">
                            <div class="control-group">
                                <label>Contest Type</label>
                                <select id="contestType" onchange="handleContestTypeChange()">
                                    <option value="friends_league" selected>Friends League (12-Person)</option>
                                    <option value="h2h">Head-to-Head Single Game</option>
                                    <option value="gpp">Tournament/GPP</option>
                                    <option value="cash">Cash Game</option>
                                    <option value="contrarian">Contrarian</option>
                                </select>
                            </div>

                            <!-- H2H Game Selector -->
                            <div class="control-group" id="gameSelector" style="display: none;">
                                <label>Select Game</label>
                                <select id="selectedGame">
                                    <option value="">-- Select Game --</option>
                                </select>
                            </div>

                            <div class="control-group">
                                <label>Lineups</label>
                                <input type="number" id="numLineups" value="3" min="1" max="10">
                            </div>
                            
                            <div class="control-group">
                                <label style="display: flex; align-items: center; gap: 5px;">
                                    <input type="checkbox" id="useAI" checked style="width: auto; margin: 0;">
                                    <span>Use AI</span>
                                </label>
                            </div>

                            <button class="button" onclick="generateLineups()" id="generateBtn">Generate</button>
                            <button class="button" onclick="refreshData()" id="refreshBtn">Refresh</button>
                        </div>

                        <div class="search-section">
                            <input type="text" class="search-box" id="playerSearch" placeholder="Search players..." onkeyup="searchPlayers()">
                            <div style="margin-top: 8px; font-size: 12px; color: #6c757d;">
                                <span id="lockStats">Locked: 0 | Excluded: 0</span>
                            </div>
                        </div>
                    </div>

                    <div class="lineup-display" id="lineupDisplay" style="display: none;">
                        <div class="lineup-tabs" id="lineupTabs"></div>
                        <div id="lineupContainer"></div>
                    </div>

                    <div class="output-section" id="output">
                        <p style="text-align: center; color: #6c757d; padding: 15px;">
                            📋 Ready to generate lineups!<br>
                            1. Select contest type<br>
                            2. (H2H only) Select a game<br>
                            3. Click Refresh to load players<br>
                            4. Lock/exclude as needed<br>
                            5. Generate lineups
                        </p>
                    </div>
                </div>

                <!-- RIGHT PANEL -->
                <div class="right-panel">
                    <div class="players-section" id="playersSection">
                        <div style="text-align: center; padding: 30px; color: #6c757d;">
                            <p>Click "Refresh" to load players</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let playerData = {};
            let lockedPlayers = new Set();
            let excludedPlayers = new Set();
            let currentLineups = [];

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

            // FIX #1: Get contestType from DOM instead of relying on scope
            function handleContestTypeChange() {
                const contestType = document.getElementById('contestType').value;
                const gameSelector = document.getElementById('gameSelector');
                
                if (contestType === 'h2h') {
                    gameSelector.style.display = 'flex';
                    loadAvailableGames();
                } else {
                    gameSelector.style.display = 'none';
                }
            }

            async function loadAvailableGames() {
                try {
                    // FIX #1: Read contestType from DOM
                    const contestType = document.getElementById('contestType').value;
                    const endpoint = contestType === 'h2h' ? '/h2h-games' : '/games';
                    
                    log(`Loading games from ${endpoint}...`, 'loading');
                    
                    const response = await fetch(endpoint);
                    const data = await response.json();
                    
                    const gameSelect = document.getElementById('selectedGame');
                    gameSelect.innerHTML = '<option value="">-- Select Game --</option>';
                    
                    if (data.games && data.games.length > 0) {
                        data.games.forEach(game => {
                            const option = document.createElement('option');
                            option.value = game.game_id;
                            option.textContent = game.display;
                            gameSelect.appendChild(option);
                        });
                        log(`📋 Loaded ${data.games.length} available games`, 'success');
                    } else {
                        log('⚠️ No games found in CSV', 'error');
                    }
                } catch (error) {
                    log(`❌ Failed to load games: ${error.message}`, 'error');
                }
            }

            async function refreshData() {
                try {
                    document.getElementById('refreshBtn').disabled = true;
                    
                    // FIX #3: Pass contest type to backend
                    const contestType = document.getElementById('contestType').value;
                    log(`🔄 Loading ${contestType} player data...`, 'loading');

                    const response = await fetch(`/players?contest_type=${contestType}`);
                    const data = await response.json();

                    if (response.ok && data.players) {
                        playerData = data.players;
                        displayPlayers();
                        log(`✅ Loaded ${Object.values(playerData).flat().length} players`, 'success');
                    } else {
                        throw new Error(data.message || 'Failed to load players');
                    }
                } catch (error) {
                    log(`❌ Failed to load players: ${error.message}`, 'error');
                } finally {
                    document.getElementById('refreshBtn').disabled = false;
                }
            }

            function displayPlayers() {
                const section = document.getElementById('playersSection');
                const positions = ['QB', 'RB', 'WR', 'TE', 'D'];

                let html = '';
                positions.forEach(pos => {
                    const players = playerData[pos] || [];
                    if (players.length === 0) return;

                    html += `
                        <div class="position-group">
                            <div class="position-header" onclick="togglePosition('${pos}')">
                                <span><strong>${pos}</strong> (${players.length})</span>
                                <span id="arrow-${pos}">▼</span>
                            </div>
                            <table class="player-table" id="table-${pos}">
                                <thead>
                                    <tr>
                                        <th>Controls</th>
                                        <th>Player</th>
                                        <th>Team</th>
                                        <th>Salary</th>
                                        <th>FPPG</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody id="tbody-${pos}"></tbody>
                            </table>
                        </div>
                    `;
                });

                section.innerHTML = html;
                positions.forEach(pos => {
                    if (playerData[pos] && playerData[pos].length > 0) {
                        updatePlayerTable(pos);
                    }
                });
            }

            function updatePlayerTable(position) {
                const players = playerData[position] || [];
                const tbody = document.getElementById(`tbody-${position}`);

                let html = '';
                players.forEach(player => {
                    const isLocked = lockedPlayers.has(player.name);
                    const isExcluded = excludedPlayers.has(player.name);
                    const rowClass = isLocked ? 'locked' : (isExcluded ? 'excluded' : '');
                    const value = (player.projected_points / (player.salary / 1000)).toFixed(2);

                    html += `
                        <tr class="player-row ${rowClass}">
                            <td>
                                <div class="lock-controls">
                                    <label><input type="checkbox" class="lock-checkbox" onchange="toggleLock('${player.name}')" ${isLocked ? 'checked' : ''}>🔒</label>
                                    <label><input type="checkbox" class="lock-checkbox" onchange="toggleExclude('${player.name}')" ${isExcluded ? 'checked' : ''}>❌</label>
                                </div>
                            </td>
                            <td><strong>${player.name}</strong></td>
                            <td>${player.team}</td>
                            <td>$${player.salary.toLocaleString()}</td>
                            <td>${player.projected_points.toFixed(1)}</td>
                            <td>${value}x</td>
                        </tr>
                    `;
                });

                tbody.innerHTML = html;
            }

            function togglePosition(position) {
                const table = document.getElementById(`table-${position}`);
                const arrow = document.getElementById(`arrow-${position}`);
                if (table.classList.contains('active')) {
                    table.classList.remove('active');
                    arrow.textContent = '▼';
                } else {
                    table.classList.add('active');
                    arrow.textContent = '▲';
                }
            }

            function toggleLock(playerName) {
                if (lockedPlayers.has(playerName)) {
                    lockedPlayers.delete(playerName);
                } else {
                    lockedPlayers.add(playerName);
                    excludedPlayers.delete(playerName);
                }
                updatePlayerDisplay();
                updateLockStats();
            }

            function toggleExclude(playerName) {
                if (excludedPlayers.has(playerName)) {
                    excludedPlayers.delete(playerName);
                } else {
                    excludedPlayers.add(playerName);
                    lockedPlayers.delete(playerName);
                }
                updatePlayerDisplay();
                updateLockStats();
            }

            function updatePlayerDisplay() {
                Object.keys(playerData).forEach(pos => {
                    if (playerData[pos] && playerData[pos].length > 0) {
                        updatePlayerTable(pos);
                    }
                });
            }

            function updateLockStats() {
                document.getElementById('lockStats').textContent = 
                    `Locked: ${lockedPlayers.size} | Excluded: ${excludedPlayers.size}`;
            }

            function searchPlayers() {
                const searchTerm = document.getElementById('playerSearch').value.toLowerCase();
                const rows = document.querySelectorAll('.player-row');
                rows.forEach(row => {
                    const playerName = row.querySelector('strong').textContent.toLowerCase();
                    row.style.display = playerName.includes(searchTerm) ? '' : 'none';
                });
            }

            async function generateLineups() {
                try {
                    document.getElementById('generateBtn').disabled = true;

                    const contestType = document.getElementById('contestType').value;
                    const numLineups = parseInt(document.getElementById('numLineups').value);
                    const useAI = document.getElementById('useAI').checked;

                    const requestBody = {
                        contest_type: contestType,
                        num_lineups: numLineups,
                        locked_players: Array.from(lockedPlayers),
                        excluded_players: Array.from(excludedPlayers),
                        use_ai: useAI
                    };

                    if (contestType === 'h2h') {
                        const selectedGame = document.getElementById('selectedGame').value;
                        if (!selectedGame) {
                            log('❌ Please select a game for H2H mode', 'error');
                            document.getElementById('generateBtn').disabled = false;
                            return;
                        }
                        requestBody.selected_game = selectedGame;
                        log(`🎯 Generating H2H lineup for ${selectedGame}`, 'loading');
                    }

                    log(`🧠 Generating ${numLineups} ${contestType.toUpperCase()} lineups...`, 'loading');

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
                    currentLineups = lineups;
                    displayFanDuelLineups(lineups, contestType);

                } catch (error) {
                    log(`❌ Lineup generation failed: ${error.message}`, 'error');
                } finally {
                    document.getElementById('generateBtn').disabled = false;
                }
            }

            function displayFanDuelLineups(lineups, contestType) {
                if (!lineups || lineups.length === 0) {
                    log('No lineups generated', 'error');
                    return;
                }

                log(`✅ Generated ${lineups.length} ${contestType.toUpperCase()} lineups!`, 'success');

                const lineupDisplay = document.getElementById('lineupDisplay');
                const tabsContainer = document.getElementById('lineupTabs');
                const container = document.getElementById('lineupContainer');

                let tabsHtml = '';
                lineups.forEach((lineup, index) => {
                    const activeClass = index === 0 ? 'active' : '';
                    tabsHtml += `<div class="lineup-tab ${activeClass}" onclick="showLineup(${index})">Lineup ${index + 1}</div>`;
                });
                tabsContainer.innerHTML = tabsHtml;

                let lineupsHtml = '';
                lineups.forEach((lineup, index) => {
                    const activeClass = index === 0 ? 'active' : '';
                    const players = parsePlayerStrings(lineup.players);

                    lineupsHtml += `
                        <div class="fanduel-lineup ${activeClass}" id="lineup-${index}">
                            <div style="text-align: center; margin-bottom: 15px;">
                                <h3>Lineup ${index + 1} (${contestType.toUpperCase()})</h3>
                                <div><strong>${lineup.total_salary.toLocaleString()}</strong> | <strong>${lineup.projected_points.toFixed(1)} pts</strong></div>
                            </div>
                            <div class="position-slots">
                                ${createPositionSlots(players, contestType)}
                            </div>
                        </div>
                    `;
                });

                container.innerHTML = lineupsHtml;
                lineupDisplay.style.display = 'block';
            }

            function parsePlayerStrings(playerStrings) {
                return playerStrings.map(playerStr => {
                    const match = playerStr.match(/^(.+?) \\(\\$([0-9,]+)\\) - ([A-Z]+)-([A-Z]+)$/);
                    if (match) {
                        return { name: match[1], salary: match[2], position: match[3], team: match[4] };
                    }
                    return { name: playerStr, salary: '0', position: '', team: '' };
                });
            }

            function createPositionSlots(players, contestType) {
                let positions;
                if (contestType === 'h2h') {
                    positions = ['MVP', 'FLEX', 'FLEX', 'FLEX', 'FLEX', 'FLEX'];
                } else {
                    positions = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF'];
                }

                let html = '';
                positions.forEach((posLabel, index) => {
                    const player = players[index] || { name: 'No Player', salary: '0', position: '', team: '' };
                    html += `
                        <div class="position-slot">
                            <div class="position-label">${posLabel}</div>
                            <div class="player-info">
                                <div class="player-name">${player.name}</div>
                                <div class="player-details">${player.position} - ${player.team}</div>
                            </div>
                            <div class="player-salary">${player.salary}</div>
                        </div>
                    `;
                });
                return html;
            }

            function showLineup(index) {
                document.querySelectorAll('.lineup-tab').forEach((tab, i) => {
                    tab.classList.toggle('active', i === index);
                });
                document.querySelectorAll('.fanduel-lineup').forEach((lineup, i) => {
                    lineup.classList.toggle('active', i === index);
                });
            }

            window.addEventListener('load', function() {
                log('🚀 NFL DFS Optimizer Pro loaded!', 'success');
                updateLockStats();
            });
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)


@app.get("/players")
async def get_players(contest_type: str = Query("gpp")):
    """Get formatted player data using the same system as CLI"""
    global current_player_data

    try:
        # Use the SAME data collection as CLI
        from data_collector import get_fresh_data

        logger.info(f"📋 Loading {contest_type} players using CLI data collection...")

        # Get data the same way as CLI
        fresh_data = await get_fresh_data()

        if not fresh_data or not fresh_data.get('players'):
            raise HTTPException(status_code=400, detail="No player data available")

        players = fresh_data['players']

        # Group by position for web UI display
        players_by_position = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'D': []}

        for player in players:
            position = player.get('position', '')
            if position in players_by_position:
                players_by_position[position].append({
                    'id': player.get('player_id', ''),
                    'name': player.get('name', ''),
                    'team': player.get('team', ''),
                    'position': position,
                    'salary': player.get('salary', 0),
                    'projected_points': player.get('projected_points', 0),
                    'game': player.get('game', '')
                })

        # Sort by salary within each position
        for pos in players_by_position:
            players_by_position[pos].sort(key=lambda p: p['salary'], reverse=True)

        # Cache for optimizer
        current_player_data = fresh_data

        return {
            "players": players_by_position,
            "total_players": sum(len(players) for players in players_by_position.values()),
            "contest_type": contest_type,
            "injury_filtering": "enabled"
        }

    except Exception as e:
        logger.error(f"Error getting players: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/h2h-games")
async def get_h2h_games():
    """FIX #2: Read H2H games directly from CSV"""
    try:
        h2h_csv = DATA_DIR / "fanduel_h2h_salaries.csv"

        if not h2h_csv.exists():
            raise HTTPException(status_code=400, detail=f"H2H CSV not found: {h2h_csv}")

        import pandas as pd
        df = pd.read_csv(h2h_csv)

        games_set = set()
        for _, row in df.iterrows():
            game = str(row.get('Game', '')).strip()
            if game and '@' in game:
                games_set.add(game)

        if not games_set:
            raise HTTPException(status_code=400, detail="No games found in H2H CSV")

        games = []
        for game_str in sorted(games_set):
            parts = game_str.split('@')
            if len(parts) == 2:
                away_team = parts[0].strip()
                home_team = parts[1].strip()
                games.append({
                    'game_id': game_str,
                    'away_team': away_team,
                    'home_team': home_team,
                    'display': f"{away_team} @ {home_team}"
                })

        logger.info(f"📋 Found {len(games)} H2H games")
        return {"games": games, "total_games": len(games)}

    except Exception as e:
        logger.error(f"Error loading H2H games: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_lineups(request: OptimizationRequest):
    """Generate optimized lineups"""
    global current_player_data

    try:
        os.environ['AI_ENABLED'] = 'true' if request.use_ai else 'false'
        logger.info(f"🎯 Optimize: contest={request.contest_type}, game={getattr(request, 'selected_game', None)}")

        # Auto-refresh using CLI data collection
        if not current_player_data:
            logger.info("📡 Auto-refreshing using CLI data collection...")
            from data_collector import get_fresh_data

            fresh_data = await get_fresh_data()
            if not fresh_data or not fresh_data.get('players'):
                raise HTTPException(status_code=400, detail="No player data available")

            current_player_data = fresh_data
            logger.info(f"✅ Auto-refresh: {len(fresh_data['players'])} players with injury filtering")

        all_players = current_player_data.get('players', [])
        if isinstance(all_players, dict):
            # Flatten if it's position-grouped
            all_players = [p for pos_players in all_players.values() for p in pos_players]

        # H2H: Filter to selected game
        single_game_teams = None

        if request.contest_type == "h2h":
            selected_game = getattr(request, "selected_game", None)

            if not selected_game:
                raise HTTPException(status_code=400, detail="H2H requires a selected game")

            if "@" in selected_game:
                parts = selected_game.split("@")
                single_game_teams = [parts[0].strip(), parts[1].strip()]
                logger.info(f"🎯 H2H parsed teams: {single_game_teams} from {selected_game}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid game format: {selected_game}")

        # Apply locks/exclusions
        filtered_players = []
        for player in all_players:
            player_name = str(player.get('name', ''))

            if player_name in request.locked_players:
                player['locked'] = True
                logger.info(f"🔒 LOCKED: {player_name}")
            else:
                player['locked'] = False

            if player_name in request.excluded_players:
                logger.info(f"❌ EXCLUDED: {player_name}")
                continue

            filtered_players.append(player)

        logger.info(f"✅ Filtered: {len(filtered_players)} players")

        # Optimize
        lineups = optimize_dfs_lineups(
            player_data=filtered_players,
            weather_data=current_player_data.get('weather', {}),
            vegas_multipliers=current_player_data.get('vegas_multipliers', {}),
            num_lineups=request.num_lineups,
            contest_type=request.contest_type
        )

        if not lineups:
            raise HTTPException(status_code=400, detail="No lineups generated")

        lineup_dicts = []
        for lineup in lineups:
            lineup_dicts.append({
                'players': [f"{p.name} (${p.salary:,}) - {p.position}-{p.team}" for p in lineup.players],
                'total_salary': lineup.total_salary,
                'projected_points': round(lineup.projected_points, 1),
                'ownership_total': round(lineup.ownership_total, 1),
                'correlation_score': round(lineup.correlation_score, 2),
                'contest_type': lineup.contest_type
            })

        return lineup_dicts

    except Exception as e:
        logger.error(f"Optimization error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.1"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)