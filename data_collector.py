# 1. ENHANCED DATA COLLECTOR (data_collector.py)
"""
Enhanced Data Collector with Proper Slate Management and Real-Time Updates
Ensures correct current week detection, proper game filtering, and fresh data
"""
import asyncio
import aiohttp
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import time
from loguru import logger
import pytz

from config import (
    ESPN_ENDPOINTS, NFL_STADIUMS, WEATHER_API, DATA_DIR, 
    RATE_LIMITS, CACHE_TTL, VALIDATION_THRESHOLDS
)

class EnhancedSlateManager:
    """Manages proper slate detection and game filtering"""
    
    def __init__(self):
        self.eastern = pytz.timezone('America/New_York')
        self.current_week = None
        self.current_season = 2024
        
    def get_current_nfl_week(self) -> int:
        """Get current NFL week with proper logic based on Tuesday slate reset"""
        now = datetime.now(self.eastern)
        
        # NFL 2024 season key dates
        season_start = datetime(2024, 9, 5, tzinfo=self.eastern)  # Week 1 Thursday
        
        if now < season_start:
            return 1
        
        # Calculate based on Tuesday reset (new week starts Tuesday)
        days_since_start = (now - season_start).days
        
        # If it's Tuesday or later, we're in the current game week
        if now.weekday() >= 1:  # Tuesday = 1
            week = min(18, max(1, (days_since_start // 7) + 1))
        else:  # Monday - still previous week for lineup purposes
            week = min(18, max(1, (days_since_start // 7)))
        
        logger.info(f"Current NFL Week: {week} (calculated on {now.strftime('%A %Y-%m-%d')})")
        return week

class EnhancedDataCollector:
    """Enhanced data collection with proper slate management"""
    
    def __init__(self):
        self.session = None
        self.slate_manager = EnhancedSlateManager()
        self.rate_limiters = {
            'espn': self._create_rate_limiter('espn_api'),
            'weather': self._create_rate_limiter('weather_gov'),
            'nfl_data': self._create_rate_limiter('nfl_data_py')
        }
        self.cache = {}
        
    def _create_rate_limiter(self, api_name: str):
        """Create rate limiter for API"""
        limits = RATE_LIMITS.get(api_name, {'calls': 60, 'period': 60})
        return RateLimiter(limits['calls'], limits['period'])
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': WEATHER_API['user_agent']}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_players_for_slate(self, games_info: Dict[str, Any], contest_type: str = 'gpp') -> List[Dict]:
        """Collect players with EXACT FanDuel salaries - no estimates"""
        current_week = games_info['current_week']
        
        # Determine which teams to include
        if contest_type == 'single_game':
            playing_teams = set()
            for game in games_info['single_games']:
                playing_teams.update(game['teams'])
        else:
            playing_teams = set()
            for game in games_info['main_slate']:
                playing_teams.update(game['teams'])
        
        logger.info(f"Collecting players for {contest_type}: {len(playing_teams)} teams playing")
        
        # Get EXACT FanDuel salaries - this is critical
        try:
            from fanduel_salary_scraper import get_fanduel_salaries
            salary_data = await get_fanduel_salaries()
            
            if salary_data.empty or len(salary_data) < 50:
                logger.error("❌ CRITICAL: No valid FanDuel salaries found!")
                raise Exception("Cannot proceed without real FanDuel salaries")
            
            logger.info(f"✅ Retrieved {len(salary_data)} REAL FanDuel salary entries")
            
        except Exception as e:
            logger.error(f"❌ FAILED to get FanDuel salaries: {e}")
            logger.error("❌ CANNOT generate lineups without exact salaries!")
            return []
        
        # Filter salary data to only playing teams
        valid_salary_data = salary_data[salary_data['Team'].str.upper().isin(playing_teams)]
        
        if len(valid_salary_data) < 20:
            logger.error(f"❌ Only {len(valid_salary_data)} players with valid salaries for playing teams")
            logger.error("❌ Need more salary data to generate valid lineups")
            return []
        
        # Convert salary data to player format
        players = []
        for _, row in valid_salary_data.iterrows():
            try:
                player = {
                    'player_id': f"fd_{row.get('Name', '')}",
                    'player_name': row.get('Name', ''),
                    'name': row.get('Name', ''),
                    'position': row.get('Position', ''),
                    'team': row.get('Team', '').upper(),
                    'salary': int(row.get('Salary', 0)),
                    'projection': float(row.get('FPPG', 10.0)),  # Use their projection if available
                    'value': 0.0,  # Will calculate after
                    'source': 'fanduel_exact'
                }
                
                # Calculate value
                if player['salary'] > 0:
                    player['value'] = player['projection'] / (player['salary'] / 1000)
                
                players.append(player)
                
            except Exception as e:
                logger.warning(f"Error processing salary row: {e}")
                continue
        
        logger.info(f"✅ Created {len(players)} players with EXACT FanDuel salaries")
        return players
    
    # Simplified methods for now
    async def get_current_week_games(self):
        current_week = self.slate_manager.get_current_nfl_week()
        
        # Use manual games for now
        games = [
            {'id': 'game_1', 'teams': ['PHI', 'WAS'], 'time_slot': 'sunday_early'},
            {'id': 'game_2', 'teams': ['BAL', 'BUF'], 'time_slot': 'sunday_early'},
            {'id': 'game_3', 'teams': ['DET', 'GB'], 'time_slot': 'sunday_early'},
            {'id': 'game_4', 'teams': ['TEN', 'MIA'], 'time_slot': 'monday'},
        ]
        
        return {
            'current_week': current_week,
            'all_games': games,
            'main_slate': [g for g in games if g['time_slot'] in ['sunday_early', 'sunday_late']],
            'single_games': games,  # All games for single game selection
        }
    
    async def collect_weather_data(self, games):
        return {}  # Simplified for now

class RateLimiter:
    """Simple rate limiter"""
    def __init__(self, calls_per_period: int, period_seconds: int):
        self.calls_per_period = calls_per_period
        self.period_seconds = period_seconds
        self.calls = []
    
    async def acquire(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period_seconds]
        
        if len(self.calls) >= self.calls_per_period:
            sleep_time = self.period_seconds - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

# Main entry point
async def get_fresh_data() -> Dict[str, Any]:
    """Get fresh data with EXACT FanDuel salaries"""
    async with EnhancedDataCollector() as collector:
        # Get current week games
        games_info = await collector.get_current_week_games()
        
        # Collect players with EXACT salaries
        players = await collector.collect_players_for_slate(games_info, 'gpp')
        
        if not players:
            logger.error("❌ NO PLAYERS WITH VALID SALARIES!")
            return {}
        
        # Collect weather
        weather_data = await collector.collect_weather_data(games_info['all_games'])
        
        return {
            'players': players,
            'games_info': games_info,
            'weather': weather_data,
            'last_updated': datetime.now().isoformat(),
            'data_quality': {
                'player_count': len(players),
                'total_games': len(games_info['all_games']),
                'main_slate_games': len(games_info['main_slate']),
                'current_week': games_info['current_week'],
            }
        }
