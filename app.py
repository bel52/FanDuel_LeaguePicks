"""
FastAPI web interface for DFS optimization system
VERSION 3.0 - COMPLETE DATA PIPELINE

Flow:
1. Page Load → Fast CSV load (UI display only)
2. Build Lineups → FULL enrichment pipeline:
   - Vegas odds → Game totals, multipliers
   - Weather data → Outdoor game adjustments
   - Monte Carlo → Ceiling/floor/boom rates
   - AI Analysis → Must-play/fade, stacks, strategy
   - Smart filters → Remove backups
   - Optimizer → Generate winning lineups
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
import os
import math
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
import traceback

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from config import API_HOST, API_PORT, DATA_DIR
except ImportError:
    API_HOST = "0.0.0.0"
    API_PORT = 8020
    DATA_DIR = Path("data")

try:
    from optimizer import optimize_dfs_lineups
    OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Optimizer not available: {e}")
    OPTIMIZER_AVAILABLE = False

try:
    from vegas_data_collector import VegasDataCollector
    VEGAS_AVAILABLE = True
    logger.info("✅ Vegas data collector available")
except ImportError:
    VEGAS_AVAILABLE = False
    logger.warning("⚠️ Vegas data collector not available")

try:
    from monte_carlo_engine import run_monte_carlo_sync
    MONTE_CARLO_AVAILABLE = True
    logger.info("✅ Monte Carlo engine available")
except ImportError:
    MONTE_CARLO_AVAILABLE = False
    logger.warning("⚠️ Monte Carlo engine not available")

try:
    from ai_analyzer_enhanced import EnhancedAIAnalyzer, run_enhanced_ai_analysis
    AI_AVAILABLE = True
    logger.info("✅ AI analyzer available")
except ImportError:
    AI_AVAILABLE = False
    logger.warning("⚠️ AI analyzer not available")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_for_json(obj):
    """Recursively sanitize float values for JSON serialization"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


def safe_float(val, default=0.0):
    """Safely convert to float"""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except:
        return default


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(title="FanDuel DFS Optimizer - Tournament Winning Edition")

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


# Global cache
current_player_data = None


# =============================================================================
# PLAYER FILTERING - REMOVE BACKUPS
# =============================================================================

def filter_backup_players(players: List[Dict], contest_type: str = "friends_league") -> List[Dict]:
    """
    Filter out backup/practice squad players that will never score

    Rules:
    - QB: Must be $6,500+ salary (starters are $7,000+)
    - RB: Must be $5,000+ OR have FPPG >= 8
    - WR: Must be $4,500+ OR have FPPG >= 6
    - TE: Must be $4,000+ OR have FPPG >= 5
    - DEF: $3,000+ is fine
    """
    filtered = []
    removed_count = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DEF': 0}

    for p in players:
        pos = p.get('position', '')
        salary = p.get('salary', 0)
        fppg = safe_float(p.get('fppg', 0), 0)
        name = p.get('name', '')

        keep = False

        if pos == 'QB':
            # QBs must be $6,500+ (starter threshold)
            keep = salary >= 6500
        elif pos == 'RB':
            # RBs: $5,000+ OR high FPPG (bellcow backups)
            keep = salary >= 5000 or fppg >= 8
        elif pos == 'WR':
            # WRs: $4,500+ OR decent FPPG
            keep = salary >= 4500 or fppg >= 6
        elif pos == 'TE':
            # TEs: $4,000+ OR decent FPPG
            keep = salary >= 4000 or fppg >= 5
        elif pos == 'DEF':
            # DEF: All are fine
            keep = salary >= 3000
        else:
            keep = False

        if keep:
            filtered.append(p)
        else:
            removed_count[pos] = removed_count.get(pos, 0) + 1

    logger.info(f"🔍 Filtered backups: {sum(removed_count.values())} removed")
    logger.info(f"   QB: -{removed_count['QB']}, RB: -{removed_count['RB']}, WR: -{removed_count['WR']}, TE: -{removed_count['TE']}")
    logger.info(f"✅ Remaining: {len(filtered)} viable players")

    return filtered


# =============================================================================
# CSV LOADING (Fast - for UI display)
# =============================================================================

def load_players_from_csv(contest_type: str = "friends_league") -> Dict[str, Any]:
    """
    FAST: Load players directly from CSV file
    Used for initial UI display - no external API calls
    """
    try:
        if contest_type == 'h2h':
            csv_path = DATA_DIR / "fanduel_h2h_salaries.csv"
            if not csv_path.exists():
                csv_path = DATA_DIR / "fanduel_salaries_manual.csv"
        else:
            csv_path = DATA_DIR / "fanduel_salaries_manual.csv"

        if not csv_path.exists():
            logger.error(f"CSV not found: {csv_path}")
            return {"players": [], "error": f"CSV not found: {csv_path}"}

        logger.info(f"📂 Loading players from {csv_path}")
        df = pd.read_csv(csv_path)

        players = []
        for _, row in df.iterrows():
            try:
                position = str(row.get('Position', '')).strip().upper()
                if position == 'D':
                    position = 'DEF'

                if not position or position not in ['QB', 'RB', 'WR', 'TE', 'DEF']:
                    continue

                nickname = row.get('Nickname', '')
                first_name = row.get('First Name', '')
                last_name = row.get('Last Name', '')

                if nickname and str(nickname).strip():
                    name = str(nickname).strip()
                elif first_name and last_name:
                    name = f"{first_name} {last_name}".strip()
                else:
                    continue

                if not name or len(name) < 2:
                    continue

                try:
                    salary = int(row.get('Salary', 0))
                except:
                    salary = 0

                if salary < 3000:
                    continue

                team = str(row.get('Team', '')).strip().upper()
                if not team:
                    continue

                try:
                    fppg = float(row.get('FPPG', 0) or 0)
                except:
                    fppg = 0

                # Use FPPG as projection, fallback to salary estimate
                projection = fppg if fppg > 0 else salary / 1000

                injury = str(row.get('Injury Indicator', '')).strip()
                game_info = str(row.get('Game', '')).strip()

                player = {
                    'name': name,
                    'position': position,
                    'team': team,
                    'salary': salary,
                    'projection': round(projection, 1),
                    'fppg': fppg,
                    'ownership': estimate_ownership(salary, projection, position),
                    'injury_status': injury,
                    'game_info': game_info,
                    # These will be enriched later
                    'game_environment_mult': 1.0,
                    'weather_factor': 1.0,
                    'ai_must_play': False,
                    'ai_must_fade': False,
                    'ceiling_90': round(projection * 1.5, 1),
                    'floor_10': round(projection * 0.5, 1),
                    'boom_rate': 0.15,
                    'bust_rate': 0.20,
                }

                players.append(player)

            except Exception as e:
                continue

        logger.info(f"✅ Loaded {len(players)} players from CSV")

        # Extract games from CSV
        games_info = extract_games_from_csv(df)

        return {
            'players': players,
            'games_info': games_info,
            'data_quality': {
                'source': 'csv',
                'player_count': len(players),
            },
        }

    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return {"players": [], "error": str(e)}


def estimate_ownership(salary: int, projection: float, position: str) -> float:
    """Estimate ownership based on salary and value"""
    if salary <= 0:
        return 5.0
    value = projection / (salary / 1000) if salary > 0 else 0
    base_own = 5.0
    if salary >= 9000:
        base_own = 18.0
    elif salary >= 7500:
        base_own = 12.0
    elif salary >= 6000:
        base_own = 8.0
    if value >= 3.0:
        base_own *= 1.4
    elif value >= 2.5:
        base_own *= 1.2
    elif value < 2.0:
        base_own *= 0.8
    return min(round(base_own, 1), 35.0)


def extract_games_from_csv(df: pd.DataFrame) -> Dict:
    """Extract game info from CSV"""
    games = []
    games_set = set()

    if 'Game' in df.columns:
        for game_str in df['Game'].dropna().unique():
            game_str = str(game_str).strip()
            if '@' in game_str and game_str not in games_set:
                games_set.add(game_str)
                parts = game_str.split('@')
                if len(parts) == 2:
                    away = parts[0].strip()
                    home = parts[1].strip()
                    games.append({
                        'id': game_str,
                        'teams': [away, home],
                        'away_team': away,
                        'home_team': home,
                    })

    return {
        'all_games': games,
        'main_slate': games,
        'total_games': len(games)
    }


# =============================================================================
# FULL DATA ENRICHMENT PIPELINE
# =============================================================================

async def fetch_vegas_data() -> tuple:
    """Fetch Vegas odds and calculate multipliers"""
    if not VEGAS_AVAILABLE:
        logger.warning("⚠️ Vegas collector not available")
        return {}, {}

    try:
        collector = VegasDataCollector()
        vegas_data = await asyncio.wait_for(
            collector.get_nfl_odds_data(),
            timeout=15.0
        )
        multipliers = collector.get_game_environment_factors(vegas_data)

        high_total_count = len(vegas_data.get('high_total_games', []))
        logger.info(f"🎰 Vegas data: {len(multipliers)} team multipliers, {high_total_count} high-total games")

        return vegas_data, multipliers

    except asyncio.TimeoutError:
        logger.warning("⏰ Vegas API timeout - using fallback")
        return {}, {}
    except Exception as e:
        logger.warning(f"Vegas fetch failed: {e}")
        return {}, {}


async def fetch_weather_data(games_info: Dict) -> Dict[str, Dict]:
    """Fetch weather for outdoor stadiums"""
    # Simplified weather - could be enhanced
    weather = {}

    # Indoor stadiums (no weather impact)
    indoor_teams = {'LV', 'LAC', 'LAR', 'ARI', 'DAL', 'HOU', 'IND', 'NO', 'ATL', 'DET', 'MIN'}

    for game in games_info.get('all_games', []):
        for team in game.get('teams', []):
            if team in indoor_teams:
                weather[team] = {'weather_factor': 1.0, 'dome': True}
            else:
                # Assume neutral outdoor weather for now
                weather[team] = {'weather_factor': 1.0, 'dome': False}

    return weather


def apply_vegas_to_players(players: List[Dict], vegas_data: Dict, multipliers: Dict) -> List[Dict]:
    """Apply Vegas game environment multipliers to players"""

    high_total_teams = set()
    for game in vegas_data.get('high_total_games', []):
        high_total_teams.update(game.get('teams', []))

    games = vegas_data.get('games', {})

    for p in players:
        team = p.get('team', '')

        # Apply multiplier
        mult = multipliers.get(team, 1.0)
        p['game_environment_mult'] = mult

        # Get game total for this team
        game_total = 45.0
        for game_id, game_info in games.items():
            if team in [game_info.get('home_team'), game_info.get('away_team')]:
                game_total = game_info.get('total_points', 45.0)
                break
        p['game_total'] = game_total

        # Flag high-total game players
        p['in_high_total_game'] = team in high_total_teams

        # Boost projection for high-total games
        if mult > 1.0:
            original_proj = p.get('projection', 0)
            p['projection'] = round(original_proj * mult, 1)

    high_total_player_count = sum(1 for p in players if p.get('in_high_total_game'))
    logger.info(f"🔥 {high_total_player_count} players in high-total games")

    return players


def apply_weather_to_players(players: List[Dict], weather_data: Dict) -> List[Dict]:
    """Apply weather factors to players"""
    for p in players:
        team = p.get('team', '')
        if team in weather_data:
            factor = weather_data[team].get('weather_factor', 1.0)
            p['weather_factor'] = factor
            if factor < 1.0:
                p['projection'] = round(p['projection'] * factor, 1)
    return players


def run_monte_carlo_on_players(players: List[Dict], vegas_data: Dict, weather_data: Dict, multipliers: Dict) -> List[Dict]:
    """Run Monte Carlo simulation on all players"""
    if not MONTE_CARLO_AVAILABLE:
        logger.warning("⚠️ Monte Carlo not available - using estimates")
        for p in players:
            proj = p.get('projection', 10)
            p['ceiling_90'] = round(proj * 1.5, 1)
            p['ceiling_95'] = round(proj * 1.7, 1)
            p['floor_10'] = round(proj * 0.5, 1)
            p['boom_rate'] = 0.15
            p['bust_rate'] = 0.20
            p['monte_carlo_analyzed'] = False
        return players

    try:
        logger.info(f"🎲 Running Monte Carlo on {len(players)} players...")

        mc_results = run_monte_carlo_sync(
            player_data=players,
            weather_data=weather_data,
            vegas_data=vegas_data,
            vegas_multipliers=multipliers,
            num_simulations=1000
        )

        for p in players:
            name = p.get('name', '')
            if name in mc_results:
                mc = mc_results[name]
                p['ceiling_90'] = mc.get('ceiling_90', p['projection'] * 1.5)
                p['ceiling_95'] = mc.get('ceiling_95', p['projection'] * 1.7)
                p['floor_10'] = mc.get('floor_10', p['projection'] * 0.5)
                p['boom_rate'] = mc.get('boom_rate', 0.15)
                p['bust_rate'] = mc.get('bust_rate', 0.20)
                p['monte_carlo_analyzed'] = True

        mc_count = sum(1 for p in players if p.get('monte_carlo_analyzed'))
        logger.info(f"✅ Monte Carlo complete: {mc_count}/{len(players)} players")

    except Exception as e:
        logger.warning(f"Monte Carlo failed: {e}")

    return players


async def run_ai_analysis_on_players(
    players: List[Dict],
    vegas_data: Dict,
    weather_data: Dict,
    contest_type: str,
    use_ai: bool
) -> List[Dict]:
    """Run AI analysis and apply recommendations"""

    if not use_ai:
        logger.info("🤖 AI disabled by user")
        return players

    if not AI_AVAILABLE:
        logger.warning("⚠️ AI analyzer not available")
        return players

    if os.getenv('AI_ENABLED', 'true').lower() == 'false':
        logger.info("🤖 AI disabled via environment")
        return players

    try:
        logger.info("🤖 Running AI strategic analysis...")

        # Build MC results dict for AI
        mc_results = {}
        for p in players:
            if p.get('monte_carlo_analyzed'):
                mc_results[p['name']] = {
                    'ceiling_90': p.get('ceiling_90', 0),
                    'floor_10': p.get('floor_10', 0),
                    'boom_rate': p.get('boom_rate', 0),
                    'bust_rate': p.get('bust_rate', 0),
                }

        analysis = await asyncio.wait_for(
            run_enhanced_ai_analysis(
                players=players,
                monte_carlo_results=mc_results,
                vegas_data=vegas_data,
                weather_data=weather_data,
                news_items=[],
                contest_type=contest_type
            ),
            timeout=30.0
        )

        if analysis:
            logger.info(f"🎯 AI Results:")
            logger.info(f"   Must-play: {len(analysis.must_play)} players")
            logger.info(f"   Must-fade: {len(analysis.must_fade)} players")
            if analysis.primary_stack:
                logger.info(f"   Primary stack: {analysis.primary_stack.qb} + {analysis.primary_stack.targets}")

            # Apply AI analysis
            analyzer = EnhancedAIAnalyzer()
            players = analyzer.apply_analysis_to_players(players, analysis)

            # Apply projection boosts/penalties
            for p in players:
                boost = p.get('ai_boost', 0)
                if boost != 0:
                    original = p['projection']
                    p['projection'] = round(original * (1 + boost), 1)

        ai_must_play = sum(1 for p in players if p.get('ai_must_play'))
        ai_must_fade = sum(1 for p in players if p.get('ai_must_fade'))
        logger.info(f"✅ AI applied: {ai_must_play} must-play, {ai_must_fade} must-fade")

    except asyncio.TimeoutError:
        logger.warning("⏰ AI analysis timeout")
    except Exception as e:
        logger.warning(f"AI analysis failed: {e}")
        logger.debug(traceback.format_exc())

    return players


async def enrich_players_for_optimization(
    players: List[Dict],
    games_info: Dict,
    contest_type: str,
    use_ai: bool
) -> tuple:
    """
    FULL DATA ENRICHMENT PIPELINE

    Returns: (enriched_players, vegas_data, vegas_multipliers, weather_data)
    """
    logger.info("=" * 60)
    logger.info("🚀 STARTING FULL DATA ENRICHMENT PIPELINE")
    logger.info("=" * 60)

    # Step 1: Fetch Vegas data
    logger.info("📊 Step 1: Fetching Vegas odds...")
    vegas_data, vegas_multipliers = await fetch_vegas_data()

    # Step 2: Fetch weather
    logger.info("🌤️ Step 2: Fetching weather data...")
    weather_data = await fetch_weather_data(games_info)

    # Step 3: Apply Vegas multipliers
    logger.info("🎰 Step 3: Applying Vegas multipliers...")
    players = apply_vegas_to_players(players, vegas_data, vegas_multipliers)

    # Step 4: Apply weather
    logger.info("☀️ Step 4: Applying weather adjustments...")
    players = apply_weather_to_players(players, weather_data)

    # Step 5: Filter backup players
    logger.info("🔍 Step 5: Filtering backup players...")
    players = filter_backup_players(players, contest_type)

    # Step 6: Run Monte Carlo
    logger.info("🎲 Step 6: Running Monte Carlo simulation...")
    players = run_monte_carlo_on_players(players, vegas_data, weather_data, vegas_multipliers)

    # Step 7: Run AI analysis
    logger.info("🤖 Step 7: Running AI analysis...")
    players = await run_ai_analysis_on_players(players, vegas_data, weather_data, contest_type, use_ai)

    # Summary
    logger.info("=" * 60)
    logger.info("✅ ENRICHMENT COMPLETE")
    logger.info(f"   Players: {len(players)}")
    logger.info(f"   Vegas multipliers: {len(vegas_multipliers)} teams")
    logger.info(f"   High-total players: {sum(1 for p in players if p.get('in_high_total_game'))}")
    logger.info(f"   AI must-play: {sum(1 for p in players if p.get('ai_must_play'))}")
    logger.info(f"   AI must-fade: {sum(1 for p in players if p.get('ai_must_fade'))}")
    logger.info(f"   Monte Carlo: {sum(1 for p in players if p.get('monte_carlo_analyzed'))}")
    logger.info("=" * 60)

    return players, vegas_data, vegas_multipliers, weather_data


# =============================================================================
# HTML TEMPLATE
# =============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>NFL DFS Optimizer Pro</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 0; background: #1a1a2e; min-height: 100vh; color: #eee; }
        .container { max-width: 1800px; margin: 0 auto; padding: 15px; }
        .header { background: linear-gradient(135deg, #16213e 0%, #0f3460 100%); padding: 15px 25px; border-radius: 12px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { margin: 0; font-size: 1.6em; color: #e94560; }
        .header p { margin: 0; opacity: 0.7; font-size: 0.9em; }
        .main-layout { display: grid; grid-template-columns: 1fr 380px; gap: 15px; }
        .left-panel { display: flex; flex-direction: column; gap: 15px; }
        .controls-card { background: #16213e; padding: 15px 20px; border-radius: 10px; }
        .controls-row { display: flex; align-items: center; gap: 15px; flex-wrap: wrap; }
        .control-group { display: flex; flex-direction: column; gap: 4px; }
        .control-group label { font-size: 11px; text-transform: uppercase; color: #888; }
        .control-group select { padding: 8px 12px; border: 1px solid #0f3460; border-radius: 6px; background: #1a1a2e; color: #eee; }
        .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 14px; }
        .btn-primary { background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%); color: white; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(233,69,96,0.4); }
        .btn-primary:disabled { background: #444; cursor: not-allowed; }
        .btn-secondary { background: #0f3460; color: #eee; }
        .lineups-section { background: #16213e; border-radius: 10px; padding: 15px; flex: 1; }
        .section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid #0f3460; }
        .section-title { font-size: 1.1em; font-weight: 600; color: #e94560; }
        .lineups-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; max-height: calc(100vh - 300px); overflow-y: auto; }
        .lineup-card { background: #1a1a2e; border-radius: 8px; padding: 12px; border: 1px solid #0f3460; }
        .lineup-header { display: flex; justify-content: space-between; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #0f3460; }
        .lineup-num { font-weight: 700; color: #e94560; }
        .lineup-stats { display: flex; gap: 12px; font-size: 12px; }
        .lineup-stats .salary { color: #4ade80; }
        .lineup-stats .proj { color: #60a5fa; }
        .lineup-players { font-size: 12px; }
        .lineup-player { padding: 3px 0; display: flex; justify-content: space-between; border-bottom: 1px solid #0f346033; }
        .lineup-player:last-child { border-bottom: none; }
        .lineup-player .pos { color: #e94560; font-weight: 600; width: 35px; }
        .lineup-player .name { flex: 1; }
        .lineup-player .salary { color: #4ade80; }
        .lineup-player .team { color: #888; width: 35px; text-align: right; }
        .right-panel { background: #16213e; border-radius: 10px; padding: 15px; max-height: calc(100vh - 100px); overflow-y: auto; }
        .search-box { width: 100%; padding: 8px 12px; border: 1px solid #0f3460; border-radius: 6px; background: #1a1a2e; color: #eee; margin-bottom: 12px; }
        .position-group { margin-bottom: 8px; }
        .position-header { background: #0f3460; padding: 8px 12px; border-radius: 6px; cursor: pointer; display: flex; justify-content: space-between; }
        .position-header:hover { background: #1a4a7a; }
        .position-header .pos-name { font-weight: 600; color: #e94560; }
        .position-players { padding: 5px 0; }
        .position-players.hidden { display: none; }
        .player-row { display: flex; align-items: center; padding: 6px 8px; border-radius: 4px; margin: 2px 0; background: #1a1a2e; font-size: 12px; gap: 8px; }
        .player-row:hover { background: #0f3460; }
        .player-row .player-name { flex: 1; }
        .player-row .player-team { color: #888; width: 30px; }
        .player-row .player-salary { color: #4ade80; width: 50px; text-align: right; }
        .player-row .player-proj { color: #60a5fa; width: 40px; text-align: right; }
        .player-actions button { padding: 3px 8px; border: none; border-radius: 4px; font-size: 10px; cursor: pointer; margin-left: 4px; }
        .btn-lock { background: #1e40af; color: #93c5fd; }
        .btn-lock.active { background: #2563eb; color: white; }
        .btn-exclude { background: #7f1d1d; color: #fca5a5; }
        .btn-exclude.active { background: #dc2626; color: white; }
        .log-section { background: #16213e; border-radius: 10px; padding: 12px; margin-top: 15px; }
        .log-output { background: #0a0a15; padding: 10px; border-radius: 6px; font-family: monospace; font-size: 11px; max-height: 150px; overflow-y: auto; }
        .log-output .success { color: #4ade80; }
        .log-output .error { color: #f87171; }
        .log-output .loading { color: #60a5fa; }
        @media (max-width: 1200px) { .main-layout { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🏈 NFL DFS Optimizer Pro</h1>
                <p>Tournament Winning Mode - Full Data Pipeline</p>
            </div>
            <div id="dataStatus" style="font-size: 12px; color: #888;">Loading...</div>
        </div>
        <div class="main-layout">
            <div class="left-panel">
                <div class="controls-card">
                    <div class="controls-row">
                        <div class="control-group">
                            <label>Contest Type</label>
                            <select id="contestType" onchange="handleContestTypeChange()">
                                <option value="friends_league" selected>Friends League</option>
                                <option value="gpp">Tournament/GPP</option>
                                <option value="cash">Cash Game</option>
                            </select>
                        </div>
                        <div class="control-group">
                            <label># Lineups</label>
                            <select id="numLineups">
                                <option value="1">1</option>
                                <option value="3" selected>3</option>
                                <option value="5">5</option>
                                <option value="10">10</option>
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
                <div class="lineups-section">
                    <div class="section-header">
                        <span class="section-title">Generated Lineups</span>
                        <span id="lineupCount" style="font-size: 12px; color: #888;">0 lineups</span>
                    </div>
                    <div id="lineupsGrid" class="lineups-grid">
                        <div style="color: #666; padding: 40px; text-align: center;">
                            Click "Build Lineups" to generate optimized lineups<br>
                            <small style="color: #444;">Full pipeline: Vegas + Weather + Monte Carlo + AI</small>
                        </div>
                    </div>
                </div>
                <div class="log-section">
                    <div class="section-header" style="margin-bottom: 8px;">
                        <span class="section-title" style="font-size: 0.9em;">Activity Log</span>
                    </div>
                    <div id="logOutput" class="log-output"></div>
                </div>
            </div>
            <div class="right-panel">
                <div class="section-title">Player Pool</div>
                <input type="text" class="search-box" id="playerSearch" placeholder="Search players..." onkeyup="filterPlayers()">
                <div id="playersByPosition"><div style="color: #666; padding: 20px; text-align: center;">Loading players...</div></div>
            </div>
        </div>
    </div>
    <script>
        let playerData = [];
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
            loadPlayers();
        }
        
        async function loadPlayers() {
            try {
                const contestType = document.getElementById('contestType').value;
                log(`Loading ${contestType} player data...`, 'loading');
                document.getElementById('dataStatus').textContent = 'Loading...';
                const response = await fetch(`/players?contest_type=${encodeURIComponent(contestType)}`);
                if (!response.ok) throw new Error(`Server error: ${response.status}`);
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                playerData = data.players || [];
                playersByPosition = {};
                playerData.forEach(p => {
                    let pos = p.position || 'OTHER';
                    if (pos === 'D') pos = 'DEF';
                    if (!playersByPosition[pos]) playersByPosition[pos] = [];
                    playersByPosition[pos].push(p);
                });
                Object.keys(playersByPosition).forEach(pos => {
                    playersByPosition[pos].sort((a, b) => (b.salary || 0) - (a.salary || 0));
                });
                renderPlayersByPosition();
                document.getElementById('dataStatus').textContent = `${playerData.length} players loaded`;
                log(`Loaded ${playerData.length} players`, 'success');
            } catch (error) {
                console.error('Load error:', error);
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
                const filteredPlayers = filter ? players.filter(p => p.name.toLowerCase().includes(filterLower) || p.team.toLowerCase().includes(filterLower)) : players;
                if (filteredPlayers.length === 0 && filter) return;
                const group = document.createElement('div');
                group.className = 'position-group';
                const isCollapsed = collapsedPositions.has(pos);
                group.innerHTML = `
                    <div class="position-header" onclick="togglePosition('${pos}')">
                        <span class="pos-name">${POSITION_NAMES[pos] || pos}</span>
                        <span style="font-size: 12px; color: #888;">${filteredPlayers.length} players</span>
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
            return `
                <div class="player-row">
                    <span class="player-name">${p.name}</span>
                    <span class="player-team">${p.team}</span>
                    <span class="player-salary">$${(p.salary || 0).toLocaleString()}</span>
                    <span class="player-proj">${(p.projection || 0).toFixed(1)}</span>
                    <div class="player-actions">
                        <button class="btn-lock ${isLocked ? 'active' : ''}" onclick="toggleLock('${p.name.replace(/'/g, "\\'")}')">🔒</button>
                        <button class="btn-exclude ${isExcluded ? 'active' : ''}" onclick="toggleExclude('${p.name.replace(/'/g, "\\'")}')">❌</button>
                    </div>
                </div>
            `;
        }
        
        function togglePosition(pos) {
            if (collapsedPositions.has(pos)) collapsedPositions.delete(pos);
            else collapsedPositions.add(pos);
            renderPlayersByPosition(document.getElementById('playerSearch').value);
        }
        
        function toggleLock(name) {
            if (lockedPlayers.has(name)) { lockedPlayers.delete(name); log(`Unlocked: ${name}`); }
            else { lockedPlayers.add(name); excludedPlayers.delete(name); log(`Locked: ${name}`, 'success'); }
            renderPlayersByPosition(document.getElementById('playerSearch').value);
        }
        
        function toggleExclude(name) {
            if (excludedPlayers.has(name)) { excludedPlayers.delete(name); log(`Removed exclusion: ${name}`); }
            else { excludedPlayers.add(name); lockedPlayers.delete(name); log(`Excluded: ${name}`, 'error'); }
            renderPlayersByPosition(document.getElementById('playerSearch').value);
        }
        
        function filterPlayers() { renderPlayersByPosition(document.getElementById('playerSearch').value); }
        
        async function generateLineups() {
            const btn = document.getElementById('buildBtn');
            const contestType = document.getElementById('contestType').value;
            const numLineups = parseInt(document.getElementById('numLineups').value);
            const useAI = document.getElementById('useAI').value === 'true';
            try {
                btn.disabled = true;
                btn.textContent = '⏳ Running Pipeline...';
                log(`Starting full pipeline for ${numLineups} ${contestType} lineups...`, 'loading');
                log('Fetching Vegas + Weather + Monte Carlo + AI...', 'loading');
                const response = await fetch('/optimize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        contest_type: contestType,
                        num_lineups: numLineups,
                        locked_players: Array.from(lockedPlayers),
                        excluded_players: Array.from(excludedPlayers),
                        use_ai: useAI
                    })
                });
                if (!response.ok) throw new Error(await response.text());
                currentLineups = await response.json();
                renderLineups(currentLineups);
                log(`Generated ${currentLineups.length} optimized lineups!`, 'success');
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
                    const match = p.match(/(.+?) \\(\\$([0-9,]+)\\) - ([A-Z]+)-([A-Z]+)/);
                    if (match) return { name: match[1], salary: match[2].replace(/,/g, ''), pos: match[3], team: match[4] };
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
            if (!currentLineups || currentLineups.length === 0) { log('No lineups to export', 'error'); return; }
            let csv = 'Lineup,Position,Name,Salary,Team\\n';
            currentLineups.forEach((lineup, idx) => {
                lineup.players.forEach(p => {
                    const match = p.match(/(.+?) \\(\\$([0-9,]+)\\) - ([A-Z]+)-([A-Z]+)/);
                    if (match) csv += `${idx + 1},${match[3]},"${match[1]}",${match[2].replace(/,/g, '')},${match[4]}\\n`;
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
            log('Exported lineups to CSV', 'success');
        }
        
        document.addEventListener('DOMContentLoaded', () => { loadPlayers(); });
    </script>
</body>
</html>
'''


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.get("/players")
async def get_players(contest_type: str = Query("friends_league")):
    """Return player pool - fast CSV load for UI"""
    global current_player_data
    try:
        data = load_players_from_csv(contest_type)
        if not data.get('players'):
            return {"players": [], "error": data.get('error', 'No players found')}
        data['contest_type'] = contest_type
        current_player_data = data
        return sanitize_for_json({
            "players": data['players'],
            "data_quality": data.get('data_quality', {})
        })
    except Exception as e:
        logger.error(f"Error in /players: {e}")
        return {"players": [], "error": str(e)}


@app.post("/optimize")
async def optimize_lineups(request: OptimizationRequest):
    """
    Generate optimized lineups with FULL DATA PIPELINE
    """
    global current_player_data

    try:
        logger.info(f"🎯 Optimize request: {request.contest_type}, {request.num_lineups} lineups, AI={request.use_ai}")

        if not OPTIMIZER_AVAILABLE:
            raise HTTPException(status_code=500, detail="Optimizer not available")

        # Load base players from CSV
        if not current_player_data or not current_player_data.get('players'):
            current_player_data = load_players_from_csv(request.contest_type)

        players = current_player_data.get('players', [])
        if not players:
            raise HTTPException(status_code=400, detail="No players available")

        games_info = current_player_data.get('games_info', {})

        # Make a copy to avoid modifying cache
        players = [p.copy() for p in players]

        # ============================================================
        # FULL DATA ENRICHMENT PIPELINE
        # ============================================================
        players, vegas_data, vegas_multipliers, weather_data = await enrich_players_for_optimization(
            players=players,
            games_info=games_info,
            contest_type=request.contest_type,
            use_ai=request.use_ai
        )

        if not players:
            raise HTTPException(status_code=400, detail="No viable players after filtering")

        # Apply locks and exclusions
        filtered_players = []
        for p in players:
            name = p.get('name', '')
            if name in request.excluded_players:
                continue
            p['locked'] = name in request.locked_players
            filtered_players.append(p)

        logger.info(f"📦 Final player pool: {len(filtered_players)} players")

        # Set AI flag
        os.environ['AI_ENABLED'] = 'true' if request.use_ai else 'false'

        # ============================================================
        # RUN OPTIMIZER
        # ============================================================
        lineups = optimize_dfs_lineups(
            player_data=filtered_players,
            weather_data=weather_data,
            vegas_multipliers=vegas_multipliers,
            vegas_data=vegas_data,
            num_lineups=request.num_lineups,
            contest_type=request.contest_type
        )

        if not lineups:
            raise HTTPException(status_code=400, detail="Optimizer failed to generate lineups")

        # Format response
        lineup_dicts = []
        for lineup in lineups:
            lineup_dicts.append({
                'players': [f"{p.name} (${p.salary:,}) - {p.position}-{p.team}" for p in lineup.players],
                'total_salary': lineup.total_salary,
                'projected_points': round(lineup.projected_points, 1),
                'ownership_total': round(lineup.ownership_total, 1),
                'correlation_score': round(getattr(lineup, 'correlation_score', 0), 2),
            })

        logger.info(f"✅ Returning {len(lineup_dicts)} optimized lineups")
        return lineup_dicts

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0-full-pipeline",
        "features": {
            "vegas": VEGAS_AVAILABLE,
            "monte_carlo": MONTE_CARLO_AVAILABLE,
            "ai": AI_AVAILABLE,
            "optimizer": OPTIMIZER_AVAILABLE
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)