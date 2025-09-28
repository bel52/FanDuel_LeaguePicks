"""
FastAPI web interface for DFS optimization system
ENHANCED: FanDuel-style lineup display with player controls
"""
from fastapi import Request
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
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
    description="Enhanced NFL Daily Fantasy Sports lineup optimization system",
    version="2.2.0"
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
    """Enhanced dashboard with FanDuel-style lineup display"""
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

            /* LEFT SIDE - Controls and Lineups */
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
            .news-alert { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px; padding: 10px; margin: 10px 0; }
            .news-critical { background: #f8d7da; border: 1px solid #f5c6cb; }
            .news-modal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; display: none; }
            .news-content { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border-radius: 10px; padding: 20px; max-width: 600px; max-height: 70vh; overflow-y: auto; }
            .close-news { float: right; font-size: 24px; cursor: pointer; }

            .search-section { text-align: center; margin: 15px 0; }
            .search-box { padding: 10px; border: 2px solid #e9ecef; border-radius: 6px; width: 250px; font-size: 14px; }
            .search-box:focus { border-color: #007bff; outline: none; }

            /* FANDUEL STYLE LINEUP DISPLAY */
            .lineup-display { background: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
            .lineup-tabs { display: flex; gap: 10px; margin-bottom: 15px; }
            .lineup-tab { padding: 8px 15px; background: #e9ecef; border-radius: 6px; cursor: pointer; font-size: 14px; }
            .lineup-tab.active { background: #007bff; color: white; }
            .fanduel-lineup { display: none; }
            .fanduel-lineup.active { display: block; }
            .lineup-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
            .lineup-stats { display: flex; gap: 15px; font-size: 13px; }
            .stat-item { background: white; padding: 6px 10px; border-radius: 4px; }

            .position-slots { display: grid; grid-template-columns: 1fr; gap: 8px; }
            .position-slot { display: flex; justify-content: space-between; align-items: center; background: white; padding: 12px; border-radius: 6px; border-left: 4px solid #007bff; }
            .position-label { font-weight: 600; color: #495057; width: 50px; }
            .player-info { flex: 1; margin-left: 15px; }
            .player-name { font-weight: 600; color: #2c3e50; }
            .player-details { font-size: 12px; color: #6c757d; }
            .player-salary { font-weight: 600; color: #28a745; }

            /* RIGHT SIDE - Player Tables */
            .right-panel { }
            .players-section { background: #f8f9fa; border-radius: 10px; padding: 20px; }
            .position-group { margin-bottom: 20px; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden; }
            .position-header { background: linear-gradient(135deg, #495057 0%, #6c757d 100%); color: white; padding: 12px 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
            .position-header:hover { background: linear-gradient(135deg, #343a40 0%, #495057 100%); }
            .player-table { width: 100%; border-collapse: collapse; display: none; font-size: 13px; }
            .player-table.active { display: table; }
            .player-table th { background: #f8f9fa; padding: 8px; text-align: left; font-weight: 600; border-bottom: 2px solid #dee2e6; }
            .player-table td { padding: 6px 8px; border-bottom: 1px solid #dee2e6; }
            .player-row:hover { background: #f8f9fa; }
            .injury-status { padding: 3px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; }
            .injury-q { background: #fff3cd; color: #856404; }
            .injury-o { background: #f8d7da; color: #721c24; }
            .injury-healthy { background: #d4edda; color: #155724; }
            .lock-controls { display: flex; gap: 8px; align-items: center; }
            .lock-checkbox { width: 16px; height: 16px; cursor: pointer; }
            .locked { background-color: #d4edda !important; }
            .excluded { background-color: #f8d7da !important; }
            .pagination { text-align: center; margin: 15px 0; }
            .pagination button { margin: 0 3px; padding: 6px 10px; border: 1px solid #dee2e6; background: white; cursor: pointer; border-radius: 4px; font-size: 12px; }
            .pagination button.active { background: #007bff; color: white; }

            .output-section { background: #f8f9fa; border-radius: 10px; padding: 15px; margin-top: 20px; max-height: 300px; overflow-y: auto; }
            .success { color: #28a745; }
            .error { color: #dc3545; }
            .loading { color: #ffc107; }
            .hidden { display: none; }

            @media (max-width: 1200px) {
                .main-content { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏈 NFL DFS Optimizer Pro</h1>
                <p>Advanced lineup optimization with FanDuel-style display</p>
            </div>

            <div class="main-content">
                <!-- LEFT PANEL - Controls and Lineups -->
                <div class="left-panel">
                    <div class="controls-section">
                        <div class="main-controls">
                            <div class="control-group">
                                <label>Contest Type</label>
                                <select id="contestType">
                                    <option value="gpp">Tournament/GPP</option>
                                    <option value="cash">Cash Game</option>
                                    <option value="contrarian">Contrarian</option>
                                    <option value="bestball">Best Ball</option>
                                </select>
                            </div>

                            <div class="control-group">
                                <label>Lineups</label>
                                <input type="number" id="numLineups" value="3" min="1" max="10">
                            </div>

                            <button class="button" onclick="generateLineups()" id="generateBtn">Generate</button>
                            <button class="button" onclick="refreshData()" id="refreshBtn">Refresh</button>
                            <button class="button" onclick="checkBreakingNews()" id="newsBtn">📰 News</button>
                            <button class="button" onclick="analyzeLiveSlate()" id="liveBtn">📊 Live Analysis</button>
                        </div>

                        <div class="search-section">
                            <input type="text" class="search-box" id="playerSearch" placeholder="Search for player..." onkeyup="searchPlayers()">
                            <div style="margin-top: 8px; font-size: 12px; color: #6c757d;">
                                <span id="lockStats">Locked: 0 | Excluded: 0</span>
                            </div>
                        </div>
                    </div>

                    <!-- FANDUEL STYLE LINEUP DISPLAY -->
                    <div class="lineup-display" id="lineupDisplay" style="display: none;">
                        <div class="lineup-tabs" id="lineupTabs"></div>
                        <div id="lineupContainer"></div>
                    </div>

                    <div class="output-section" id="output">
                        <p style="text-align: center; color: #6c757d; padding: 15px;">
                            📋 Ready to generate lineups!<br>
                            1. Refresh data to load players<br>
                            2. Lock/exclude players as needed<br>
                            3. Generate optimized lineups
                        </p>
                    </div>
                </div>

                <!-- RIGHT PANEL - Player Tables -->
                <div class="right-panel">
                    <div class="players-section" id="playersSection">
                        <div style="text-align: center; padding: 30px; color: #6c757d;">
                            <p>Click "Refresh" to load players</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- NEWS MODAL -->
        <div id="newsModal" class="news-modal">
            <div class="news-content">
                <span class="close-news" onclick="closeNewsModal()">&times;</span>
                <h3>📰 Breaking NFL News</h3>
                <div id="newsContent">Loading...</div>
            </div>
        </div>

        <script>
            let playerData = {};
            let lockedPlayers = new Set();
            let excludedPlayers = new Set();
            let currentPage = {};
            let currentLineups = [];

            function log(message, type = 'info') {
                const output = document.getElementById('output');
                const timestamp = new Date().toLocaleTimeString();
                let className = '';
                let emoji = '📋';

                if (type === 'success') { className = 'success'; emoji = '✅'; }
                else if (type === 'error') { className = 'error'; emoji = '❌'; }
                else if (type === 'loading') { className = 'loading'; emoji = '⏳'; }

                output.innerHTML += `<div class="${className}" style="margin: 3px 0; font-size: 13px;">${emoji} [${timestamp}] ${message}</div>`;
                output.scrollTop = output.scrollHeight;
            }

            async function refreshData() {
                try {
                    document.getElementById('refreshBtn').disabled = true;
                    log('🔄 Loading fresh player data...', 'loading');

                    const response = await fetch('/players');
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

                    currentPage[pos] = 0;

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
                                        <th>Status</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody id="tbody-${pos}">
                                </tbody>
                            </table>
                            <div class="pagination" id="pagination-${pos}"></div>
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
    const page = currentPage[position] || 0;
    const pageSize = 20;
    const start = page * pageSize;
    const end = start + pageSize;
    const pageData = players.slice(start, end);

    let html = '';
    pageData.forEach(player => {
        const isLocked = lockedPlayers.has(player.name);
        const isExcluded = excludedPlayers.has(player.name);
        const rowClass = isLocked ? 'locked' : (isExcluded ? 'excluded' : '');

        const injuryClass = getInjuryClass(player.injury_status);
        const value = (player.projected_points / (player.salary / 1000)).toFixed(2);

        html += `
            <tr class="player-row ${rowClass}" data-player-name="${player.name.toLowerCase()}">
                <td>
                    <div class="lock-controls">
                        <label title="Lock">
                            <input type="checkbox" class="lock-checkbox" 
                                   onchange="toggleLock('${player.name}')" 
                                   ${isLocked ? 'checked' : ''}>
                            🔒
                        </label>
                        <label title="Exclude">
                            <input type="checkbox" class="lock-checkbox" 
                                   onchange="toggleExclude('${player.name}')" 
                                   ${isExcluded ? 'checked' : ''}>
                            ❌
                        </label>
                    </div>
                </td>
                <td><strong>${player.name}</strong></td>
                <td>${player.team}</td>
                <td>$${player.salary.toLocaleString()}</td>
                <td>${player.projected_points.toFixed(1)}</td>
                <td><span class="injury-status ${injuryClass}">${player.injury_status || 'Healthy'}</span></td>
                <td>${value}x</td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
    updatePagination(position, players.length, pageSize);
}

function getInjuryClass(status) {
    if (!status || status === 'Healthy') return 'injury-healthy';
    if (status.includes('Q')) return 'injury-q';
    if (status.includes('O')) return 'injury-o';
    return 'injury-healthy';
}

function updatePagination(position, totalPlayers, pageSize) {
    const totalPages = Math.ceil(totalPlayers / pageSize);
    const currentPageNum = currentPage[position] || 0;
    const pagination = document.getElementById(`pagination-${position}`);

    if (totalPages <= 1) {
        pagination.innerHTML = '';
        return;
    }

    let html = '';
    for (let i = 0; i < totalPages; i++) {
        const activeClass = i === currentPageNum ? 'active' : '';
        html += `<button class="${activeClass}" onclick="changePage('${position}', ${i})">${i + 1}</button>`;
    }

    pagination.innerHTML = html;
}

function changePage(position, page) {
    currentPage[position] = page;
    updatePlayerTable(position);
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
        const playerName = row.getAttribute('data-player-name');
        if (playerName.includes(searchTerm)) {
            row.style.display = '';
            if (searchTerm && playerName.includes(searchTerm)) {
                row.style.backgroundColor = '#fff3cd';
            }
        } else {
            row.style.display = searchTerm ? 'none' : '';
        }
    });
}

            async function generateLineups() {
                try {
                    document.getElementById('generateBtn').disabled = true;

                    const contestType = document.getElementById('contestType').value;
                    const numLineups = parseInt(document.getElementById('numLineups').value);

                    if (lockedPlayers.size > 8) {
    throw new Error(`Too many locked players (${lockedPlayers.size}). Maximum is 8.`);
}

                    log(`🧠 Generating ${numLineups} ${contestType.toUpperCase()} lineups...`, 'loading');

                    const requestBody = {
    contest_type: contestType,
    num_lineups: parseInt(document.getElementById('numLineups').value),
    locked_players: Array.from(lockedPlayers),
    excluded_players: Array.from(excludedPlayers),
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

                // Create tabs
                let tabsHtml = '';
                lineups.forEach((lineup, index) => {
                    const activeClass = index === 0 ? 'active' : '';
                    tabsHtml += `<div class="lineup-tab ${activeClass}" onclick="showLineup(${index})">Lineup ${index + 1}</div>`;
                });
                tabsContainer.innerHTML = tabsHtml;

                // Create lineup displays
                let lineupsHtml = '';
                lineups.forEach((lineup, index) => {
                    const activeClass = index === 0 ? 'active' : '';
                    const players = parsePlayerStrings(lineup.players);

                    lineupsHtml += `
                        <div class="fanduel-lineup ${activeClass}" id="lineup-${index}">
                            <div class="lineup-header">
                                <h3>Lineup ${index + 1} (${contestType.toUpperCase()})</h3>
                                <div class="lineup-stats">
                                    <div class="stat-item"><strong>$${lineup.total_salary.toLocaleString()}</strong></div>
                                    <div class="stat-item"><strong>${lineup.projected_points.toFixed(1)} pts</strong></div>
                                    <div class="stat-item"><strong>${lineup.ownership_total.toFixed(1)}% owned</strong></div>
                                </div>
                            </div>
                            <div class="position-slots">
                                ${createPositionSlots(players)}
                            </div>
                        </div>
                    `;
                });

                container.innerHTML = lineupsHtml;
                lineupDisplay.style.display = 'block';

                log(`💾 Lineups saved to CSV files`, 'success');
            }

            function parsePlayerStrings(playerStrings) {
                return playerStrings.map(playerStr => {
                    const match = playerStr.match(/^(.+?) \\(\\$([0-9,]+)\\) - ([A-Z]+)-([A-Z]+)$/);
                    if (match) {
                        return {
                            name: match[1],
                            salary: match[2],
                            position: match[3],
                            team: match[4]
                        };
                    }
                    return { name: playerStr, salary: '0', position: '', team: '' };
                });
            }

            function createPositionSlots(players) {
                const positions = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF'];
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
                            <div class="player-salary">$${player.salary}</div>
                        </div>
                    `;
                });

                return html;
            }

            function showLineup(index) {
                // Update tabs
                document.querySelectorAll('.lineup-tab').forEach((tab, i) => {
                    tab.classList.toggle('active', i === index);
                });

                // Update lineup display
                document.querySelectorAll('.fanduel-lineup').forEach((lineup, i) => {
                    lineup.classList.toggle('active', i === index);
                });
            }

            async function checkBreakingNews() {
                try {
                    document.getElementById('newsBtn').disabled = true;
                    log('📰 Checking breaking news...', 'loading');

                    const response = await fetch('/breaking-news');
                    const data = await response.json();

                    if (response.ok) {
                        showNewsModal(data);
                        if (data.news_events && data.news_events.length > 0) {
                            log(`📰 Found ${data.news_events.length} news items`, 'success');
                        } else {
                            log('📰 No breaking news found', 'success');
                        }
                    } else {
                        throw new Error(data.message || 'Failed to get news');
                    }
                } catch (error) {
                    log(`❌ News check failed: ${error.message}`, 'error');
                } finally {
                    document.getElementById('newsBtn').disabled = false;
                }
            }

            function showNewsModal(newsData) {
    const modal = document.getElementById('newsModal');
    const content = document.getElementById('newsContent');

    let html = '';

    if (newsData.news_events && newsData.news_events.length > 0) {
        newsData.news_events.forEach(news => {
            // Clean up source name
            const sourceName = cleanSourceName(news.source);
            
            // Get impact level with emoji
            const impactInfo = getImpactDisplay(news.dfs_impact);
            
            // Format timestamp
            const timeAgo = formatTimeAgo(news.timestamp);
            
            // Choose alert style based on impact
            const alertClass = news.dfs_impact >= 7 ? 'news-critical' : 'news-alert';
            
            html += `
                <div class="${alertClass}" style="margin-bottom: 12px; padding: 12px; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                        <span style="font-weight: 600; color: #2c3e50; font-size: 14px;">${sourceName}</span>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="background: ${impactInfo.color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: 600;">
                                ${impactInfo.emoji} ${impactInfo.level}
                            </span>
                            <span style="color: #6c757d; font-size: 11px;">${timeAgo}</span>
                        </div>
                    </div>
                    <div style="font-size: 14px; line-height: 1.4; color: #2c3e50;">
                        ${news.headline}
                    </div>
                    ${news.summary ? `<div style="font-size: 12px; color: #6c757d; margin-top: 4px; line-height: 1.3;">${news.summary}</div>` : ''}
                </div>
            `;
        });

        // Show AI analysis if available
        if (newsData.impact_analysis && newsData.impact_analysis.ai_analysis) {
            html += `
                <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #007bff;">
                    <div style="font-weight: 600; color: #007bff; margin-bottom: 6px; font-size: 13px;">
                        🤖 DFS Impact Analysis
                    </div>
                    <div style="font-size: 13px; line-height: 1.4; color: #495057;">
                        ${newsData.impact_analysis.ai_analysis.substring(0, 200)}...
                    </div>
                </div>
            `;
        }
    } else {
        html = `
            <div style="text-align: center; padding: 20px; color: #6c757d;">
                <div style="font-size: 48px; margin-bottom: 10px;">📰</div>
                <div style="font-size: 16px; margin-bottom: 5px;">No breaking news at this time</div>
                <div style="font-size: 13px;">We'll notify you when DFS-relevant updates are available</div>
            </div>
        `;
    }

    content.innerHTML = html;
    modal.style.display = 'block';
}

function cleanSourceName(source) {
    const sourceMap = {
        'espn_nfl': 'ESPN',
        'cbs_sports': 'CBS Sports',
        'yahoo_sports': 'Yahoo Sports', 
        'nfl_com': 'NFL.com',
        'usa_today': 'USA Today',
        'nfl_official': 'NFL.com',
        'rotoworld': 'Rotoworld', 
        'fantasypros': 'FantasyPros'
    };
    return sourceMap[source] || source.toUpperCase();
}

function getImpactDisplay(impact) {
    if (impact >= 8) {
        return { level: 'HIGH', emoji: '🚨', color: '#dc3545' };
    } else if (impact >= 6) {
        return { level: 'MED', emoji: '⚠️', color: '#ffc107' };
    } else if (impact >= 4) {
        return { level: 'LOW', emoji: 'ℹ️', color: '#17a2b8' };
    } else {
        return { level: 'INFO', emoji: '📝', color: '#6c757d' };
    }
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return 'Unknown';
    
    try {
        const newsTime = new Date(timestamp);
        const now = new Date();
        const diffMs = now - newsTime;
        const diffMins = Math.floor(diffMs / (1000 * 60));
        const diffHours = Math.floor(diffMins / 60);
        
        if (diffMins < 60) {
            return `${diffMins}m ago`;
        } else if (diffHours < 24) {
            return `${diffHours}h ago`;
        } else {
            return newsTime.toLocaleDateString();
        }
    } catch (e) {
        return 'Recent';
    }
}

function cleanSourceName(source) {
    const sourceMap = {
        'espn_nfl': 'ESPN',
        'nfl_official': 'NFL.com',
        'rotoworld': 'Rotoworld', 
        'fantasypros': 'FantasyPros'
    };
    return sourceMap[source] || source.toUpperCase();
}

function getImpactDisplay(impact) {
    if (impact >= 8) {
        return { level: 'HIGH', emoji: '🚨', color: '#dc3545' };
    } else if (impact >= 6) {
        return { level: 'MED', emoji: '⚠️', color: '#ffc107' };
    } else if (impact >= 4) {
        return { level: 'LOW', emoji: 'ℹ️', color: '#17a2b8' };
    } else {
        return { level: 'INFO', emoji: '📝', color: '#6c757d' };
    }
}

function formatTimeAgo(timestamp) {
    if (!timestamp) return 'Unknown';
    
    try {
        const newsTime = new Date(timestamp);
        const now = new Date();
        const diffMs = now - newsTime;
        const diffMins = Math.floor(diffMs / (1000 * 60));
        const diffHours = Math.floor(diffMins / 60);
        
        if (diffMins < 60) {
            return `${diffMins}m ago`;
        } else if (diffHours < 24) {
            return `${diffHours}h ago`;
        } else {
            return newsTime.toLocaleDateString();
        }
    } catch (e) {
        return 'Recent';
    }
}

            function closeNewsModal() {
                document.getElementById('newsModal').style.display = 'none';
            }
async function analyzeLiveSlate() {
    try {
        document.getElementById('liveBtn').disabled = true;
        log('📊 Analyzing live slate impact...', 'loading');

        // Get currently locked players from UI state
const currentlyLocked = Array.from(lockedPlayers).map(playerId => {
    // Find player data for each locked ID
    for (const pos of ['QB', 'RB', 'WR', 'TE', 'D']) {
        const posPlayers = playerData[pos] || [];
        const player = posPlayers.find(p => p.id === playerId);
        if (player) {
            return {
                name: player.name,
                team: player.team,
                position: player.position,
                projected_points: player.projected_points || 0
            };
        }
    }
    return null;
}).filter(p => p !== null);

const response = await fetch('/analyze-live-slate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ locked_players: currentlyLocked })
});

        const data = await response.json();

        if (response.ok) {
            displayLiveAnalysis(data);
            log('📊 Live analysis completed', 'success');
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        log(`❌ Live analysis failed: ${error.message}`, 'error');
    } finally {
        document.getElementById('liveBtn').disabled = false;
    }
}

function displayLiveAnalysis(data) {
    let html = '<h3>📊 Live Slate Analysis</h3>';
    
    // Show locked player performance
    const locked = data.locked_performance || {};
    for (const [player, perf] of Object.entries(locked)) {
        html += `<div style="margin: 5px 0; padding: 8px; background: #f0f8ff; border-radius: 4px;">`;
        html += `<strong>${player}</strong>: ${perf.game_script} | `;
        html += `Projected: ${perf.projected_final} pts (${perf.vs_expectation > 0 ? '+' : ''}${perf.vs_expectation})`;
        html += `</div>`;
    }
    
    // Show strategy changes
    const strategy = data.strategy_changes || [];
    strategy.forEach(change => {
        html += `<div style="margin: 5px 0; padding: 8px; background: #fff3cd; border-radius: 4px;">${change}</div>`;
    });
    
    document.getElementById('output').innerHTML += `<div style="margin-top: 15px;">${html}</div>`;
}
            // Auto-check for news on Sunday every 15 minutes
            function startNewsMonitoring() {
                const now = new Date();
                if (now.getDay() === 0) { // Sunday
                    setInterval(async () => {
                        try {
                            const response = await fetch('/breaking-news');
                            const data = await response.json();

                            // Check for compelling changes that warrant auto-adjustment
                            if (data.impact_analysis && data.impact_analysis.confidence > 0.8) {
                                const removeCount = (data.impact_analysis.remove_players || []).length;
                                const addCount = (data.impact_analysis.add_players || []).length;

                                if (removeCount > 0 || addCount >= 2) {
                                    log(`🚨 COMPELLING NEWS: Auto-adjustment recommended!`, 'error');
                                    log(`Suggested: Remove ${removeCount}, Add ${addCount} players`, 'loading');

                                    // Show news modal automatically
                                    showNewsModal(data);
                                }
                            }
                        } catch (error) {
                            console.error('Auto news check failed:', error);
                        }
                    }, 15 * 60 * 1000); // 15 minutes

                    log('📰 Sunday news monitoring active (15min intervals)', 'success');
                }
            }

            // Initialize
            window.addEventListener('load', function() {
                log('🚀 NFL DFS Optimizer Pro loaded!', 'success');
                updateLockStats();
                startNewsMonitoring();
            });
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content)


@app.get("/players")
async def get_players():
    """Get formatted player data for the enhanced UI"""
    global current_player_data

    try:
        # Get fresh data
        data = await get_fresh_data()

        if not data or not data.get('players'):
            raise HTTPException(status_code=400, detail="No player data available")

        # Group players by position and format for UI
        players_by_position = {
            'QB': [],
            'RB': [],
            'WR': [],
            'TE': [],
            'D': []
        }

        for player in data['players']:
            position = player.get('position', '')

            # Skip if position not recognized
            if position not in players_by_position:
                continue

            # Format player for UI
            formatted_player = {
                'id': player.get('player_id', player.get('name', '')),
                'name': player.get('name', ''),
                'team': player.get('team', ''),
                'position': position,
                'salary': player.get('salary', 0),
                'projected_points': player.get('projected_points', 0),
                'injury_status': player.get('injury_status', ''),
                'ownership': player.get('ownership', 0),
                'value': (player.get('projected_points', 0) / (player.get('salary', 5000) / 1000))
            }

            players_by_position[position].append(formatted_player)

        # Sort each position by salary (highest first)
        for position in players_by_position:
            players_by_position[position].sort(key=lambda p: p['salary'], reverse=True)

        current_player_data = data

        return {
            "players": players_by_position,
            "total_players": sum(len(players) for players in players_by_position.values()),
            "week": data.get('data_quality', {}).get('current_week', 'Unknown')
        }

    except Exception as e:
        logger.error(f"Error getting players: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_lineups(request: OptimizationRequest):
    """Generate optimized lineups using the request data"""
    try:
        logger.info(f"DEBUG: Received locked_players: {request.locked_players}")
        logger.info(f"DEBUG: Received excluded_players: {request.excluded_players}")
        logger.info(f"🧠 Starting {request.contest_type} optimization with locks/exclusions...")

        # Get current players
        if not current_player_data or not current_player_data.get('players'):
            raise HTTPException(status_code=400, detail="No player data available. Please refresh first.")

        all_players = current_player_data['players']

        # DEBUG: Show actual player IDs vs requested locked IDs
        actual_ids = [str(player.get('id', '')) for player in all_players[:10]]
        logger.info(f"🔍 DEBUG: First 10 actual player IDs: {actual_ids}")
        logger.info(f"🔍 DEBUG: Requested locked IDs: {request.locked_players}")

        # Apply locks/exclusions with debug logging
        filtered_players = []
        locked_count = 0
        excluded_count = 0

        for player in all_players:
            player_id = str(player.get('id', ''))
            player_name = str(player.get('name', ''))

            # FIXED: Use player name if ID is empty (which it always is)
            lookup_key = player_name if not player_id else player_id

            # Mark locked players - check against player names from UI
            if lookup_key in request.locked_players or player_name in request.locked_players:
                if isinstance(player, dict):
                    player['locked'] = True
                    locked_count += 1
                    logger.info(f"🔒 MARKING LOCKED: {player.get('name')} (using name: {player_name})")
                else:
                    logger.error(f"Player is not a dict: {type(player)} - {player}")
            else:
                if isinstance(player, dict):
                    player['locked'] = False

            # Skip excluded players
            if lookup_key in request.excluded_players or player_name in request.excluded_players:
                excluded_count += 1
                logger.info(f"❌ EXCLUDING: {player.get('name')} (using name: {player_name})")
                continue

            filtered_players.append(player)

        logger.info(f"✅ Filtered players: {len(filtered_players)} total, {locked_count} locked, {excluded_count} excluded")

        # Validate we have enough locked players
        if len(request.locked_players) > 8:
            raise HTTPException(status_code=400,
                                detail=f"Too many locked players ({len(request.locked_players)}). Maximum is 8.")

            # Optimize lineups - convert to Player objects first
            try:
                from optimizer import EnhancedDFSOptimizer
                optimizer = EnhancedDFSOptimizer()

                # Convert dict data to Player objects using the prepare_players method
                player_objects = await optimizer.prepare_players(filtered_players)

                lineups = await optimizer.generate_multiple_lineups(
                    players=player_objects,
                    num_lineups=request.num_lineups,
                    contest_type=request.contest_type
                )

                if not lineups:
                    raise Exception("No lineups generated")

                logger.info(f"✅ Generated {len(lineups)} lineups with player constraints")

                # Convert LineupResult objects to dictionaries for JSON serialization
                lineup_dicts = []
                for lineup in lineups:
                    lineup_dict = {
                        'players': [f"{p.name} (${p.salary:,}) - {p.position}-{p.team}" for p in lineup.players],
                        'total_salary': lineup.total_salary,
                        'projected_points': round(lineup.projected_points, 1),
                        'ownership_total': round(lineup.ownership_total, 1),
                        'correlation_score': round(lineup.correlation_score, 2),
                        'contest_type': lineup.contest_type
                    }
                    lineup_dicts.append(lineup_dict)

            except Exception as opt_error:
                logger.error(f"Optimization failed: {opt_error}")
                raise HTTPException(status_code=400, detail="Optimization failed to generate valid lineups")

            # Save to CSV
            week_num = current_player_data.get('week', 1)
            csv_path = save_lineups_to_csv(lineups, request.contest_type, week_num)

            return {
                'lineups': lineup_dicts,  # Use serializable dicts
                'contest_type': request.contest_type,
                'num_lineups': len(lineups),
                'csv_path': str(csv_path) if csv_path else None,
                'locked_count': locked_count,
                'excluded_count': excluded_count
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in optimization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")
@app.post("/analyze-live-slate")
async def analyze_live_slate(request: Request):
    """Analyze locked players vs late slate strategy"""
    try:
        # Get locked players from request body
        request_data = await request.json()
        locked_players = request_data.get('locked_players', [])

        # Mock early game results for testing
        early_results = {
            'LAC': {'score_differential': 14, 'time_remaining_pct': 25},
            'TB': {'score_differential': -7, 'time_remaining_pct': 30},
            'BUF': {'score_differential': 3, 'time_remaining_pct': 40}
        }

        late_slate_players = current_player_data.get('players', [])[:20] if current_player_data else []

        from ai_analyzer import DualAIDFSAnalyzer
        analyzer = DualAIDFSAnalyzer()

        analysis = await analyzer.analyze_live_game_impact(
            locked_players, early_results, late_slate_players
        )

        return {
            'live_analysis': analysis,
            'locked_performance': analysis.get('locked_performance', {}),
            'strategy_changes': analysis.get('strategy_changes', []),
            'late_slate_adjustments': analysis.get('late_slate_adjustments', [])
        }

    except Exception as e:
        logger.error(f"Live slate analysis failed: {e}")
        return {'error': str(e)}
@app.get("/breaking-news")
async def get_breaking_news_endpoint():
    """Get breaking news for the GUI"""
    try:
        # Import news functions
        try:
            from news_monitor import get_breaking_news
            from ai_analyzer import DualAIDFSAnalyzer

            # Get breaking news - FIXED: removed force_check parameter
            news_events = await get_breaking_news()

            # If we have current player data, analyze impact
            impact_analysis = {}
            if current_player_data and news_events:
                try:
                    analyzer = DualAIDFSAnalyzer()
                    impact_analysis = await analyzer.analyze_breaking_news(
                        news_events,
                        current_player_data.get('players', [])
                    )
                except Exception as e:
                    logger.warning(f"News analysis failed: {e}")
                    impact_analysis = {'ai_analysis': 'Analysis unavailable'}

            return {
                'news_events': news_events,
                'impact_analysis': impact_analysis,
                'news_count': len(news_events),
                'last_check': datetime.now().isoformat()
            }

        except ImportError:
            return {
                'news_events': [],
                'impact_analysis': {'ai_analysis': 'News monitoring not available'},
                'news_count': 0,
                'error': 'News monitoring module not installed'
            }

    except Exception as e:
        logger.error(f"Breaking news endpoint failed: {e}")
        return {
            'news_events': [],
            'impact_analysis': {},
            'news_count': 0,
            'error': str(e)
        }


@app.post("/update")
async def force_data_update():
    """Force an immediate data refresh"""
    global current_player_data

    try:
        logger.info("🔄 Force data refresh requested")
        data = await get_fresh_data()

        if data and data.get('players'):
            current_player_data = data
            player_count = len(data['players'])
            quality = data.get('data_quality', {})
            current_week = quality.get('current_week', 'Unknown')

            message = f"Data refreshed successfully - {player_count} players loaded for Week {current_week}"
            logger.info(f"✅ {message}")
            return {"message": message, "status": "success"}
        else:
            current_player_data = None
            message = "Data refresh completed but no players found. Check data/fanduel_salaries_manual.csv"
            logger.warning(message)
            return {"message": message, "status": "warning"}

    except Exception as e:
        logger.error(f"Force refresh failed: {e}")
        logger.error(traceback.format_exc())
        current_player_data = None
        return {"message": f"Refresh failed: {str(e)}", "status": "error"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "mode": "enhanced-gui"
        }

        # Check if we can access data
        try:
            data_dir_exists = DATA_DIR.exists()
            csv_file_exists = (DATA_DIR / "fanduel_salaries_manual.csv").exists()

            health_status.update({
                "data_dir_exists": data_dir_exists,
                "csv_file_exists": csv_file_exists,
                "data_status": "ready" if csv_file_exists else "missing_csv",
                "cached_players": current_player_data is not None
            })
        except Exception as e:
            health_status["data_error"] = str(e)

        return health_status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}


async def save_lineups_to_csv(lineups, contest_type, data_quality):
    """Save lineups to organized CSV files"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create organized directory structure
        lineup_dir = Path("data/lineups")
        week_dir = lineup_dir / f"week_{data_quality.get('current_week', 'unknown')}"
        week_dir.mkdir(parents=True, exist_ok=True)

        csv_file = week_dir / f"{contest_type}_lineups_{timestamp}.csv"

        # Create CSV export
        lineup_data = []
        for i, lineup in enumerate(lineups):
            lineup_row = {
                'Lineup': i + 1,
                'QB': f"{lineup.players[0].name}",
                'RB1': f"{lineup.players[1].name}",
                'RB2': f"{lineup.players[2].name}",
                'WR1': f"{lineup.players[3].name}",
                'WR2': f"{lineup.players[4].name}",
                'WR3': f"{lineup.players[5].name}",
                'TE': f"{lineup.players[6].name}",
                'FLEX': f"{lineup.players[7].name}",
                'DEF': f"{lineup.players[8].name}",
                'Salary': lineup.total_salary,
                'Projected': round(lineup.projected_points, 1),
                'Ownership': round(lineup.ownership_total, 1)
            }
            lineup_data.append(lineup_row)

        import pandas as pd
        df = pd.DataFrame(lineup_data)
        df.to_csv(csv_file, index=False)

        logger.info(f"💾 Exported {len(lineups)} lineups to: {csv_file}")

    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")


def log(message: str):
    """Helper function for logging"""
    logger.info(message)