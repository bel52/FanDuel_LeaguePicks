"""
FastAPI web interface for DFS optimization system
IMPROVED UI: Lineups below controls, players by position on right with collapsible headers
"""
from late_swap import LateSwapEngine, filter_for_late_swap
from fastapi import Request
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import os
import math
from late_swap import LateSwapEngine, filter_for_late_swap
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
    logger.error(f"Import error in app.py: {e}")
    raise


def sanitize_for_json(obj):
    """Recursively sanitize float values for JSON serialization (handles inf/nan)"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


app = FastAPI(title="FanDuel DFS Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OptimizationRequest(BaseModel):
    contest_type: str
    num_lineups: int = 3
    locked_players: List[str] = []
    excluded_players: List[str] = []
    use_ai: bool = True
    selected_game: Optional[str] = None


class LateSwapRequest(BaseModel):
    contest_type: str
    num_lineups: int = 3
    locked_players: List[str] = []
    excluded_players: List[str] = []
    use_ai: bool = True
    started_games: List[str] = []
    original_lineups: List[Dict[str, Any]] = []


class LineupResponse(BaseModel):
    players: List[str]
    total_salary: int
    projected_points: float
    ownership_total: float
    correlation_score: float


current_player_data = None


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Enhanced dashboard with improved layout"""
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>NFL DFS Optimizer Pro</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: #1a1a2e; min-height: 100vh; color: #eee; }
            .container { max-width: 1800px; margin: 0 auto; padding: 15px; }
            
            .header { background: linear-gradient(135deg, #16213e 0%, #0f3460 100%); padding: 15px 25px; border-radius: 12px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }
            .header h1 { margin: 0; font-size: 1.6em; font-weight: 500; color: #e94560; }
            .header p { margin: 0; opacity: 0.7; font-size: 0.9em; }
            
            .main-layout { display: grid; grid-template-columns: 1fr 380px; gap: 15px; }
            
            /* Left Panel - Controls + Lineups */
            .left-panel { display: flex; flex-direction: column; gap: 15px; }
            
            .controls-card { background: #16213e; padding: 15px 20px; border-radius: 10px; }
            .controls-row { display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
            .control-group { display: flex; flex-direction: column; gap: 4px; }
            .control-group label { font-size: 11px; text-transform: uppercase; color: #888; font-weight: 600; }
            .control-group select, .control-group input { padding: 8px 12px; border: 1px solid #0f3460; border-radius: 6px; background: #1a1a2e; color: #eee; font-size: 14px; }
            .control-group select:focus, .control-group input:focus { outline: none; border-color: #e94560; }
            
            .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; transition: all 0.2s; }
            .btn-primary { background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%); color: white; }
            .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(233,69,96,0.4); }
            .btn-primary:disabled { background: #444; cursor: not-allowed; transform: none; box-shadow: none; }
            .btn-secondary { background: #0f3460; color: #eee; }
            .btn-secondary:hover { background: #1a4a7a; }
            
            /* Lineups Section */
            .lineups-section { background: #16213e; border-radius: 10px; padding: 15px; flex: 1; }
            .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid #0f3460; }
            .section-title { font-size: 1.1em; font-weight: 600; color: #e94560; }
            
            .lineups-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; max-height: calc(100vh - 280px); overflow-y: auto; }
            .lineup-card { background: #1a1a2e; border-radius: 8px; padding: 12px; border: 1px solid #0f3460; }
            .lineup-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #0f3460; }
            .lineup-num { font-weight: 700; color: #e94560; }
            .lineup-stats { display: flex; gap: 12px; font-size: 12px; color: #888; }
            .lineup-stats span { }
            .lineup-stats .salary { color: #4ade80; }
            .lineup-stats .proj { color: #60a5fa; }
            
            .lineup-players { font-size: 12px; }
            .lineup-player { padding: 3px 0; display: flex; justify-content: space-between; border-bottom: 1px solid #0f346033; }
            .lineup-player:last-child { border-bottom: none; }
            .lineup-player .pos { color: #e94560; font-weight: 600; width: 35px; }
            .lineup-player .name { flex: 1; }
            .lineup-player .salary { color: #4ade80; }
            .lineup-player .team { color: #888; width: 35px; text-align: right; }
            
            /* Right Panel - Players by Position */
            .right-panel { background: #16213e; border-radius: 10px; padding: 15px; max-height: calc(100vh - 100px); overflow-y: auto; }
            .right-panel .section-title { margin-bottom: 10px; }
            
            .search-box { width: 100%; padding: 8px 12px; border: 1px solid #0f3460; border-radius: 6px; background: #1a1a2e; color: #eee; margin-bottom: 12px; }
            .search-box:focus { outline: none; border-color: #e94560; }
            
            .position-group { margin-bottom: 8px; }
            .position-header { background: #0f3460; padding: 8px 12px; border-radius: 6px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; user-select: none; }
            .position-header:hover { background: #1a4a7a; }
            .position-header .pos-name { font-weight: 600; color: #e94560; }
            .position-header .pos-count { font-size: 12px; color: #888; }
            .position-header .chevron { transition: transform 0.2s; }
            .position-header.collapsed .chevron { transform: rotate(-90deg); }
            
            .position-players { padding: 5px 0; }
            .position-players.hidden { display: none; }
            
            .player-row { display: flex; align-items: center; padding: 6px 8px; border-radius: 4px; margin: 2px 0; background: #1a1a2e; font-size: 12px; gap: 8px; }
            .player-row:hover { background: #0f3460; }
            .player-row .player-name { flex: 1; font-weight: 500; }
            .player-row .player-team { color: #888; width: 30px; }
            .player-row .player-salary { color: #4ade80; width: 50px; text-align: right; }
            .player-row .player-proj { color: #60a5fa; width: 40px; text-align: right; }
            .player-row .player-own { color: #888; width: 40px; text-align: right; font-size: 11px; }
            
            .player-tags { display: flex; gap: 4px; }
            .tag { padding: 2px 6px; border-radius: 10px; font-size: 9px; font-weight: 700; text-transform: uppercase; }
            .tag-must { background: #065f46; color: #34d399; }
            .tag-fade { background: #7f1d1d; color: #fca5a5; }
            .tag-inj { background: #78350f; color: #fbbf24; }
            
            .player-actions { display: flex; gap: 4px; }
            .player-actions button { padding: 3px 8px; border: none; border-radius: 4px; font-size: 10px; cursor: pointer; font-weight: 600; }
            .btn-lock { background: #1e40af; color: #93c5fd; }
            .btn-lock.active { background: #2563eb; color: white; }
            .btn-exclude { background: #7f1d1d; color: #fca5a5; }
            .btn-exclude.active { background: #dc2626; color: white; }
            
            /* Log Section */
            .log-section { background: #16213e; border-radius: 10px; padding: 12px; margin-top: 15px; }
            .log-output { background: #0a0a15; padding: 10px; border-radius: 6px; font-family: 'Monaco', 'Consolas', monospace; font-size: 11px; max-height: 120px; overflow-y: auto; }
            .log-output .info { color: #888; }
            .log-output .success { color: #4ade80; }
            .log-output .error { color: #f87171; }
            .log-output .loading { color: #60a5fa; }
            
            /* AI Insights */
            .insights-card { background: #0f3460; border-radius: 8px; padding: 12px; margin-top: 12px; }
            .insights-card h4 { margin: 0 0 8px 0; color: #e94560; font-size: 12px; }
            .insights-card pre { margin: 0; font-size: 11px; white-space: pre-wrap; color: #aaa; max-height: 250px; overflow-y: auto; line-height: 1.4; }
            
            @media (max-width: 1200px) {
                .main-layout { grid-template-columns: 1fr; }
                .right-panel { max-height: 400px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>🏈 NFL DFS Optimizer Pro</h1>
                    <p>Friends League Domination Mode</p>
                </div>
                <div id="dataStatus" style="font-size: 12px; color: #888;">Loading...</div>
            </div>
            
            <div class="main-layout">
                <div class="left-panel">
                    <!-- Controls -->
                    <div class="controls-card">
                        <div class="controls-row">
                            <div class="control-group">
                                <label>Contest Type</label>
                                <select id="contestType" onchange="handleContestTypeChange()">
                                    <option value="friends_league" selected>Friends League</option>
                                    <option value="gpp">Tournament/GPP</option>
                                    <option value="cash">Cash Game</option>
                                    <option value="h2h">Head-to-Head</option>
                                </select>
                            </div>
                            <div class="control-group">
                                <label>Game/Slate</label>
                                <select id="gameSelect" style="min-width: 150px;">
                                    <option value="">Full Slate</option>
                                </select>
                            </div>
                            <div class="control-group">
                                <label># Lineups</label>
                                <select id="numLineups">
                                    <option value="1">1</option>
                                    <option value="3" selected>3</option>
                                    <option value="5">5</option>
                                    <option value="10">10</option>
                                    <option value="15">15</option>
                                    <option value="20">20</option>
                                </select>
                            </div>
                            <div class="control-group">
                                <label>AI Analysis</label>
                                <select id="useAI">
                                    <option value="true" selected>Enabled</option>
                                    <option value="false">Disabled</option>
                                </select>
                            </div>
                            <div class="control-group" style="margin-top: 16px;">
                                <button class="btn btn-primary" id="buildBtn" onclick="generateLineups()">🚀 Build Lineups</button>
                            </div>
                            <div class="control-group" style="margin-top: 16px;">
                                <button class="btn btn-secondary" onclick="downloadCSV()">📥 Export CSV</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Generated Lineups -->
                    <div class="lineups-section">
                        <div class="section-header">
                            <span class="section-title">Generated Lineups</span>
                            <span id="lineupCount" style="font-size: 12px; color: #888;">0 lineups</span>
                        </div>
                        <div id="lineupsGrid" class="lineups-grid">
                            <div style="color: #666; padding: 40px; text-align: center;">
                                Click "Build Lineups" to generate optimized lineups
                            </div>
                        </div>
                    </div>
                    
                    <!-- Log -->
                    <div class="log-section">
                        <div class="section-header" style="margin-bottom: 8px; padding-bottom: 8px;">
                            <span class="section-title" style="font-size: 0.9em;">Activity Log</span>
                        </div>
                        <div id="logOutput" class="log-output"></div>
                    </div>
                </div>
                
                <!-- Right Panel - Players -->
                <div class="right-panel">
                    <div class="section-title">Player Pool</div>
                    <input type="text" class="search-box" id="playerSearch" placeholder="Search players..." onkeyup="filterPlayers()">
                    
                    <div id="playersByPosition">
                        <div style="color: #666; padding: 20px; text-align: center;">Loading players...</div>
                    </div>
                    
                    <div class="insights-card" id="aiInsightsCard" style="display: none;">
                        <h4>🧠 AI Insights</h4>
                        <pre id="aiInsights"></pre>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let playerData = {};
            let playersByPosition = {};
            let lockedPlayers = new Set();
            let excludedPlayers = new Set();
            let currentLineups = [];
            let collapsedPositions = new Set();
            
            const POSITION_ORDER = ['QB', 'RB', 'WR', 'TE', 'DEF'];
            const POSITION_NAMES = { 'QB': 'Quarterbacks', 'RB': 'Running Backs', 'WR': 'Wide Receivers', 'TE': 'Tight Ends', 'DEF': 'Defense' };
            
            function log(message, type = 'info') {
                const output = document.getElementById('logOutput');
                const time = new Date().toLocaleTimeString();
                const icons = { info: '📋', success: '✅', error: '❌', loading: '⏳' };
                output.innerHTML += `<div class="${type}">${icons[type] || '📋'} [${time}] ${message}</div>`;
                output.scrollTop = output.scrollHeight;
            }
            
            function handleContestTypeChange() {
                lockedPlayers.clear();
                excludedPlayers.clear();
                loadAvailableGames();
                loadPlayers();
            }
            
            async function loadAvailableGames() {
                try {
                    const contestType = document.getElementById('contestType').value;
                    const endpoint = contestType === 'h2h' ? '/h2h-games' : '/games';
                    const response = await fetch(endpoint);
                    const data = await response.json();
                    
                    const gameSelect = document.getElementById('gameSelect');
                    gameSelect.innerHTML = '<option value="">Full Slate</option>';
                    
                    if (data.games && data.games.length > 0) {
                        data.games.forEach(game => {
                            const option = document.createElement('option');
                            option.value = game.game_id || game.id || game.display;
                            option.textContent = game.display || `${game.away_team} @ ${game.home_team}`;
                            gameSelect.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Error loading games:', error);
                }
            }
            
            async function loadPlayers() {
                try {
                    const contestType = document.getElementById('contestType').value;
                    log(`Loading ${contestType} player data...`, 'loading');
                    document.getElementById('dataStatus').textContent = 'Loading players...';
                    
                    const response = await fetch(`/players?contest_type=${encodeURIComponent(contestType)}`);
                    if (!response.ok) throw new Error(`Failed: ${response.status}`);
                    
                    const data = await response.json();
                    playerData = data.players || {};
                    
                    // Group by position
                    playersByPosition = {};
                    const players = Array.isArray(playerData) ? playerData : Object.values(playerData);
                    
                    players.forEach(p => {
                        let pos = p.position || 'OTHER';
                        // Normalize all defense positions to DEF
                        if (pos === 'DST' || pos === 'D/ST' || pos === 'D') pos = 'DEF';
                        if (!playersByPosition[pos]) playersByPosition[pos] = [];
                        playersByPosition[pos].push(p);
                    });
                    
                    // Sort each position by salary
                    Object.keys(playersByPosition).forEach(pos => {
                        playersByPosition[pos].sort((a, b) => (b.salary || 0) - (a.salary || 0));
                    });
                    
                    renderPlayersByPosition();
                    updateAIInsights(data.ai_recommendations);
                    
                    const totalPlayers = players.length;
                    document.getElementById('dataStatus').textContent = `${totalPlayers} players loaded`;
                    log(`Loaded ${totalPlayers} players`, 'success');
                    
                } catch (error) {
                    console.error(error);
                    log(`Error: ${error.message}`, 'error');
                    document.getElementById('dataStatus').textContent = 'Error loading';
                }
            }
            
            function renderPlayersByPosition(filter = '') {
                const container = document.getElementById('playersByPosition');
                container.innerHTML = '';
                
                const filterLower = filter.toLowerCase();
                
                POSITION_ORDER.forEach(pos => {
                    let players = playersByPosition[pos] || [];
                    
                    if (players.length === 0) return;
                    
                    // Filter players
                    const filteredPlayers = filter ? players.filter(p => 
                        p.name.toLowerCase().includes(filterLower) ||
                        p.team.toLowerCase().includes(filterLower)
                    ) : players;
                    
                    if (filteredPlayers.length === 0 && filter) return;
                    
                    const group = document.createElement('div');
                    group.className = 'position-group';
                    
                    const isCollapsed = collapsedPositions.has(pos);
                    
                    group.innerHTML = `
                        <div class="position-header ${isCollapsed ? 'collapsed' : ''}" onclick="togglePosition('${pos}')">
                            <span class="pos-name">${POSITION_NAMES[pos] || pos}</span>
                            <span>
                                <span class="pos-count">${filteredPlayers.length} players</span>
                                <span class="chevron">▼</span>
                            </span>
                        </div>
                        <div class="position-players ${isCollapsed ? 'hidden' : ''}" id="players-${pos}">
                            ${filteredPlayers.map(p => renderPlayerRow(p)).join('')}
                        </div>
                    `;
                    
                    container.appendChild(group);
                });
            }
            
            function renderPlayerRow(p) {
                const isLocked = lockedPlayers.has(p.name);
                const isExcluded = excludedPlayers.has(p.name);
                
                let tags = '';
                if (p.ai_must_play) tags += '<span class="tag tag-must">PLAY</span>';
                if (p.ai_must_fade) tags += '<span class="tag tag-fade">FADE</span>';
                if (p.injury_status && p.injury_status !== 'healthy' && p.injury_status !== 'HEALTHY') {
                    tags += `<span class="tag tag-inj">${p.injury_status}</span>`;
                }
                
                const proj = (p.projection || 0).toFixed(1);
                const own = p.ownership ? p.ownership.toFixed(0) + '%' : '--';
                
                return `
                    <div class="player-row" data-name="${p.name}">
                        <span class="player-name">${p.name}</span>
                        <span class="player-team">${p.team}</span>
                        <div class="player-tags">${tags}</div>
                        <span class="player-salary">$${(p.salary || 0).toLocaleString()}</span>
                        <span class="player-proj">${proj}</span>
                        <span class="player-own">${own}</span>
                        <div class="player-actions">
                            <button class="btn-lock ${isLocked ? 'active' : ''}" onclick="toggleLock('${p.name.replace(/'/g, "\\'")}')">🔒</button>
                            <button class="btn-exclude ${isExcluded ? 'active' : ''}" onclick="toggleExclude('${p.name.replace(/'/g, "\\'")}')">❌</button>
                        </div>
                    </div>
                `;
            }
            
            function togglePosition(pos) {
                if (collapsedPositions.has(pos)) {
                    collapsedPositions.delete(pos);
                } else {
                    collapsedPositions.add(pos);
                }
                renderPlayersByPosition(document.getElementById('playerSearch').value);
            }
            
            function toggleLock(playerName) {
                if (lockedPlayers.has(playerName)) {
                    lockedPlayers.delete(playerName);
                    log(`Unlocked: ${playerName}`, 'info');
                } else {
                    lockedPlayers.add(playerName);
                    excludedPlayers.delete(playerName);
                    log(`Locked: ${playerName}`, 'success');
                }
                renderPlayersByPosition(document.getElementById('playerSearch').value);
            }
            
            function toggleExclude(playerName) {
                if (excludedPlayers.has(playerName)) {
                    excludedPlayers.delete(playerName);
                    log(`Removed exclusion: ${playerName}`, 'info');
                } else {
                    excludedPlayers.add(playerName);
                    lockedPlayers.delete(playerName);
                    log(`Excluded: ${playerName}`, 'error');
                }
                renderPlayersByPosition(document.getElementById('playerSearch').value);
            }
            
            function filterPlayers() {
                const query = document.getElementById('playerSearch').value;
                renderPlayersByPosition(query);
            }
            
            function updateAIInsights(recommendations) {
                const card = document.getElementById('aiInsightsCard');
                const pre = document.getElementById('aiInsights');
                
                if (recommendations && Object.keys(recommendations).length > 0) {
                    card.style.display = 'block';
                    
                    // Build formatted display
                    let display = '';
                    
                    // Primary Stack
                    if (recommendations.primary_stack && recommendations.primary_stack.qb) {
                        const ps = recommendations.primary_stack;
                        display += `🎯 PRIMARY STACK: ${ps.qb}`;
                        if (ps.targets && ps.targets.length > 0) {
                            display += ` + ${ps.targets.join(', ')}`;
                        }
                        display += '\n';
                        if (ps.reasoning) {
                            display += `   └─ ${ps.reasoning}\n`;
                        }
                        display += '\n';
                    }
                    
                    // Key Insight
                    if (recommendations.key_insight) {
                        display += `💡 KEY INSIGHT: ${recommendations.key_insight}\n\n`;
                    }
                    
                    // Must Play
                    if (recommendations.must_play && recommendations.must_play.length > 0) {
                        display += `✅ MUST PLAY (${recommendations.must_play.length}):\n`;
                        display += `   ${recommendations.must_play.slice(0, 8).join(', ')}`;
                        if (recommendations.must_play.length > 8) display += '...';
                        display += '\n\n';
                    }
                    
                    // Must Fade
                    if (recommendations.must_fade && recommendations.must_fade.length > 0) {
                        display += `❌ MUST FADE (${recommendations.must_fade.length}):\n`;
                        display += `   ${recommendations.must_fade.slice(0, 6).join(', ')}`;
                        if (recommendations.must_fade.length > 6) display += '...';
                        display += '\n\n';
                    }
                    
                    // Value Plays
                    if (recommendations.value_plays && recommendations.value_plays.length > 0) {
                        display += `💰 VALUE PLAYS:\n`;
                        recommendations.value_plays.slice(0, 4).forEach(vp => {
                            display += `   • ${vp.name}: ${vp.reason || 'Good value'}\n`;
                        });
                        display += '\n';
                    }
                    
                    // Lineup Advice
                    if (recommendations.lineup_advice) {
                        display += `📋 LINEUP ADVICE:\n   ${recommendations.lineup_advice}\n`;
                    }
                    
                    // Confidence
                    if (recommendations.confidence) {
                        const pct = Math.round(recommendations.confidence * 100);
                        display += `\n🎯 AI Confidence: ${pct}%`;
                    }
                    
                    pre.textContent = display || JSON.stringify(recommendations, null, 2);
                } else {
                    card.style.display = 'none';
                }
            }
            
            async function generateLineups() {
                const btn = document.getElementById('buildBtn');
                btn.disabled = true;
                btn.textContent = '⏳ Building...';
                
                try {
                    const contestType = document.getElementById('contestType').value;
                    const numLineups = parseInt(document.getElementById('numLineups').value);
                    const useAI = document.getElementById('useAI').value === 'true';
                    const selectedGame = document.getElementById('gameSelect').value;
                    
                    log(`Generating ${numLineups} ${contestType.toUpperCase()} lineups...`, 'loading');
                    
                    const response = await fetch('/optimize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            contest_type: contestType,
                            num_lineups: numLineups,
                            locked_players: Array.from(lockedPlayers),
                            excluded_players: Array.from(excludedPlayers),
                            use_ai: useAI,
                            selected_game: contestType === 'h2h' ? selectedGame : null
                        })
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(errorText);
                    }
                    
                    currentLineups = await response.json();
                    renderLineups(currentLineups);
                    log(`Generated ${currentLineups.length} lineups`, 'success');
                    
                } catch (error) {
                    console.error(error);
                    log(`Error: ${error.message}`, 'error');
                } finally {
                    btn.disabled = false;
                    btn.textContent = '🚀 Build Lineups';
                }
            }
            
            function renderLineups(lineups) {
                const container = document.getElementById('lineupsGrid');
                document.getElementById('lineupCount').textContent = `${lineups.length} lineup${lineups.length !== 1 ? 's' : ''}`;
                
                if (lineups.length === 0) {
                    container.innerHTML = '<div style="color: #666; padding: 40px; text-align: center;">No lineups generated</div>';
                    return;
                }
                
                container.innerHTML = lineups.map((lineup, idx) => {
                    const players = lineup.players.map(p => {
                        // Parse "Name ($Salary) - POS-TEAM"
                        const match = p.match(/(.+?) \(\$([0-9,]+)\) - ([A-Z]+)-([A-Z]+)/);
                        if (match) {
                            return { name: match[1], salary: match[2].replace(/,/g, ''), pos: match[3], team: match[4] };
                        }
                        return { name: p, salary: '0', pos: '??', team: '??' };
                    });
                    
                    return `
                        <div class="lineup-card">
                            <div class="lineup-header">
                                <span class="lineup-num">Lineup ${idx + 1}</span>
                                <div class="lineup-stats">
                                    <span class="salary">$${lineup.total_salary.toLocaleString()}</span>
                                    <span class="proj">${lineup.projected_points.toFixed(1)} pts</span>
                                </div>
                            </div>
                            <div class="lineup-players">
                                ${players.map(p => `
                                    <div class="lineup-player">
                                        <span class="pos">${p.pos}</span>
                                        <span class="name">${p.name}</span>
                                        <span class="salary">$${parseInt(p.salary).toLocaleString()}</span>
                                        <span class="team">${p.team}</span>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }).join('');
            }
            
            function downloadCSV() {
                if (!currentLineups || currentLineups.length === 0) {
                    log('No lineups to export', 'error');
                    return;
                }
                
                let csv = 'Lineup,Position,Name,Salary,Team\\n';
                currentLineups.forEach((lineup, idx) => {
                    lineup.players.forEach(p => {
                        const match = p.match(/(.+?) \(\$([0-9,]+)\) - ([A-Z]+)-([A-Z]+)/);
                        if (match) {
                            csv += `${idx + 1},${match[3]},"${match[1]}",${match[2].replace(/,/g, '')},${match[4]}\\n`;
                        }
                    });
                });
                
                const blob = new Blob([csv], { type: 'text/csv' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `fanduel_lineups_${new Date().toISOString().slice(0,10)}.csv`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
                log('Exported lineups to CSV', 'success');
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', () => {
                loadAvailableGames();
                loadPlayers();
            });
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/games")
async def get_games():
    """Return basic slate info (non-H2H)"""
    try:
        global current_player_data
        if current_player_data and 'games_info' in current_player_data:
            games_info = current_player_data['games_info']
            games = games_info.get('all_games', [])
            pretty_games = []
            for g in games:
                teams = g.get('teams', [])
                if len(teams) == 2:
                    pretty_games.append({
                        'id': g.get('id'),
                        'away_team': teams[0],
                        'home_team': teams[1],
                        'display': f"{teams[0]} @ {teams[1]} ({g.get('time', '')})"
                    })
            return {"games": pretty_games, "total_games": len(pretty_games)}
        else:
            return {"games": [], "total_games": 0}
    except Exception as e:
        logger.error(f"Error in /games: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/h2h-games")
async def get_h2h_games():
    """Read H2H games directly from CSV"""
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


@app.get("/players")
async def get_players(contest_type: str = Query("friends_league")):
    """Return player pool and AI recommendations for the web UI."""
    global current_player_data

    try:
        if current_player_data and current_player_data.get('contest_type') == contest_type:
            logger.info(f"♻️ Reusing cached player data for {contest_type}")
            return sanitize_for_json({
                "players": current_player_data.get('players', []),
                "ai_recommendations": current_player_data.get('ai_recommendations', {}),
                "data_quality": current_player_data.get('data_quality', {})
            })

        logger.info(f"📋 Loading {contest_type} players using CLI data collection...")

        fresh_data = await get_fresh_data(contest_type)

        if not fresh_data or not fresh_data.get('players'):
            raise HTTPException(status_code=400, detail="No player data available")

        fresh_data['contest_type'] = contest_type
        current_player_data = fresh_data

        return sanitize_for_json({
            "players": current_player_data.get('players', []),
            "ai_recommendations": current_player_data.get('ai_recommendations', {}),
            "data_quality": current_player_data.get('data_quality', {})
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /players: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_lineups(request: OptimizationRequest):
    """Generate optimized lineups"""
    global current_player_data

    try:
        os.environ['AI_ENABLED'] = 'true' if request.use_ai else 'false'
        logger.info(f"🎯 Optimize: contest={request.contest_type}, game={request.selected_game}")

        if (not current_player_data) or (current_player_data.get('contest_type') != request.contest_type):
            logger.info("📡 Auto-refreshing using CLI data collection...")
            fresh_data = await get_fresh_data(request.contest_type)
            if not fresh_data or not fresh_data.get('players'):
                raise HTTPException(status_code=400, detail="No player data available")

            fresh_data['contest_type'] = request.contest_type
            current_player_data = fresh_data
            logger.info(f"✅ Auto-refresh: {len(fresh_data['players'])} players with injury filtering")

        all_players = current_player_data.get('players', [])
        if isinstance(all_players, dict):
            all_players = [p for pos_players in all_players.values() for p in pos_players]

        logger.info(f"📦 Total players before filters: {len(all_players)}")

        single_game_teams = None
        if request.contest_type == "h2h":
            if not request.selected_game:
                raise HTTPException(status_code=400, detail="H2H requires a selected game")

            if "@" in request.selected_game:
                parts = request.selected_game.split("@")
                single_game_teams = [parts[0].strip(), parts[1].strip()]
                logger.info(f"🎯 H2H parsed teams: {single_game_teams} from {request.selected_game}")
            else:
                raise HTTPException(status_code=400, detail=f"Invalid game format: {request.selected_game}")

        filtered_players = []
        for player in all_players:
            player_name = str(player.get('name', ''))

            if request.contest_type == "h2h" and single_game_teams:
                if player.get('team') not in single_game_teams:
                    continue

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

        lineups = optimize_dfs_lineups(
            player_data=filtered_players,
            weather_data=current_player_data.get('weather', {}),
            vegas_multipliers=current_player_data.get('vegas_multipliers', {}),
            vegas_data=current_player_data.get('vegas_odds', {}),
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/late-swap-optimize")
async def late_swap_optimize(request: LateSwapRequest):
    """Late swap optimization"""
    global current_player_data

    try:
        if not current_player_data or not current_player_data.get('players'):
            raise HTTPException(status_code=400, detail="No base slate loaded for late swap")

        started_teams = set(t.upper() for t in request.started_games)
        explicit_locks = set(request.locked_players)
        excluded = set(request.excluded_players)

        all_players = current_player_data['players']
        if isinstance(all_players, dict):
            all_players = [p for pos_players in all_players.values() for p in pos_players]

        locked_players_data = []
        available_players = []

        for p in all_players:
            name = str(p.get('name', ''))
            team = str(p.get('team', '')).upper()

            game_started = team in started_teams
            is_locked = game_started or (name in explicit_locks)

            if name in excluded:
                continue

            if is_locked:
                p['locked'] = True
                locked_players_data.append(p)
            else:
                p['locked'] = False
                available_players.append(p)

        logger.info(f"⏰ Late-swap: {len(available_players)} available, {len(locked_players_data)} locked")
        logger.info(f"🔒 Started teams: {sorted(started_teams)}")

        if not available_players and not locked_players_data:
            raise HTTPException(status_code=400, detail="No players available - all games may have started")

        combined_players = locked_players_data + available_players

        lineups = optimize_dfs_lineups(
            player_data=combined_players,
            weather_data=current_player_data.get('weather', {}),
            vegas_multipliers=current_player_data.get('vegas_multipliers', {}),
            vegas_data=current_player_data.get('vegas_odds', {}),
            num_lineups=request.num_lineups,
            contest_type=request.contest_type
        )

        if not lineups:
            raise HTTPException(status_code=400, detail="Late-swap optimization generated no lineups")

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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Late-swap error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.3.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)