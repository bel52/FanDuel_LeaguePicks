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
    
    def get_slate_type(self) -> str:
        """Determine current slate type for optimization focus"""
        now = datetime.now(self.eastern)
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        hour = now.hour
        
        if day_of_week == 3:  # Thursday
            return 'thursday_single' if hour >= 18 else 'thursday_build'
        elif day_of_week == 6:  # Sunday
            if hour < 13:
                return 'sunday_main_build'  # Before 1PM - main slate focus
            elif 13 <= hour < 20:
                return 'sunday_main_live'   # 1PM-8PM - late swap available
            else:
                return 'sunday_night_single'  # SNF single game
        elif day_of_week == 0:  # Monday
            return 'monday_single' if hour >= 18 else 'monday_build'
        else:
            return 'midweek_build'  # Building for upcoming slate
    
    def get_target_games(self, contest_type: str) -> str:
        """Get target games based on contest type and timing"""
        slate_type = self.get_slate_type()
        
        if contest_type == 'single_game':
            return 'all_individual_games'  # All games available for single game
        elif contest_type in ['gpp', 'cash', 'contrarian']:
            if 'sunday_main' in slate_type:
                return 'main_slate_only'  # Sunday 1PM + 4PM games only
            else:
                return 'all_games'  # Include all games
        
        return 'main_slate_only'  # Default to main slate

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
    
    async def get_current_week_games(self) -> Dict[str, Any]:
        """Get current week games with proper categorization"""
        current_week = self.slate_manager.get_current_nfl_week()
        slate_type = self.slate_manager.get_slate_type()
        
        try:
            await self.rate_limiters['espn'].acquire()
            async with self.session.get(ESPN_ENDPOINTS['scoreboard']) as response:
                if response.status == 200:
                    scoreboard_data = await response.json()
                    games = await self._parse_espn_games(scoreboard_data, current_week)
                else:
                    logger.warning(f"ESPN API error: {response.status}")
                    games = self._get_manual_games(current_week)
        except Exception as e:
            logger.error(f"Error fetching ESPN data: {e}")
            games = self._get_manual_games(current_week)
        
        # Categorize games
        categorized = self._categorize_games_by_slate(games)
        
        return {
            'current_week': current_week,
            'slate_type': slate_type,
            'all_games': games,
            'main_slate': categorized['main_slate'],
            'single_games': categorized['single_games'],
            'prime_time': categorized['prime_time'],
            'games_by_day': categorized['by_day']
        }
    
    async def _parse_espn_games(self, scoreboard_data: Dict, target_week: int) -> List[Dict]:
        """Parse ESPN scoreboard for current week only"""
        games = []
        
        for event in scoreboard_data.get('events', []):
            try:
                event_week = event.get('week', {}).get('number', target_week)
                if event_week != target_week:
                    continue
                
                teams = []
                for comp in event.get('competitions', []):
                    for competitor in comp.get('competitors', []):
                        team_info = competitor.get('team', {})
                        teams.append({
                            'abbreviation': team_info.get('abbreviation', '').upper(),
                            'name': team_info.get('displayName', ''),
                            'is_home': competitor.get('homeAway') == 'home'
                        })
                
                if len(teams) >= 2:
                    home_team = next((t['abbreviation'] for t in teams if t.get('is_home')), teams[-1]['abbreviation'])
                    away_team = next((t['abbreviation'] for t in teams if not t.get('is_home')), teams[0]['abbreviation'])
                    
                    game_time = self._parse_game_time(event.get('date', ''))
                    time_slot = self._determine_time_slot(game_time)
                    
                    games.append({
                        'id': f"game_{event.get('id')}",
                        'espn_id': event.get('id'),
                        'week': target_week,
                        'home_team': home_team,
                        'away_team': away_team,
                        'teams': [away_team, home_team],
                        'game_time': game_time,
                        'time_slot': time_slot,
                        'status': event.get('status', {}).get('type', {}).get('description', 'Scheduled'),
                        'weather_relevant': self._is_outdoor_game(home_team),
                        'display_name': f"{away_team} @ {home_team}"
                    })
                    
            except Exception as e:
                logger.warning(f"Error parsing ESPN event: {e}")
                continue
        
        logger.info(f"Parsed {len(games)} games from ESPN for week {target_week}")
        return games
    
    def _get_manual_games(self, week: int) -> List[Dict]:
        """Manual game schedule when ESPN fails - UPDATE THIS WEEKLY"""
        # THIS NEEDS TO BE UPDATED WEEKLY FOR CURRENT WEEK
        # Week 3 (Sept 19-23, 2024) Schedule:
        manual_games = [
            # Thursday Night Football
            {
                'id': 'tnf_1', 'week': week, 'time_slot': 'thursday',
                'away_team': 'NYJ', 'home_team': 'NE', 'teams': ['NYJ', 'NE'],
                'display_name': 'NYJ @ NE - Thu 8:15 PM',
                'weather_relevant': True
            },
            # Sunday 1:00 PM ET Games
            {
                'id': 'early_1', 'week': week, 'time_slot': 'sunday_early',
                'away_team': 'CAR', 'home_team': 'LV', 'teams': ['CAR', 'LV'],
                'display_name': 'CAR @ LV - Sun 1:00 PM',
                'weather_relevant': False  # Dome
            },
            {
                'id': 'early_2', 'week': week, 'time_slot': 'sunday_early',
                'away_team': 'CHI', 'home_team': 'IND', 'teams': ['CHI', 'IND'],
                'display_name': 'CHI @ IND - Sun 1:00 PM',
                'weather_relevant': False  # Dome
            },
            {
                'id': 'early_3', 'week': week, 'time_slot': 'sunday_early',
                'away_team': 'HOU', 'home_team': 'MIN', 'teams': ['HOU', 'MIN'],
                'display_name': 'HOU @ MIN - Sun 1:00 PM',
                'weather_relevant': False  # Dome
            },
            {
                'id': 'early_4', 'week': week, 'time_slot': 'sunday_early',
                'away_team': 'GB', 'home_team': 'TEN', 'teams': ['GB', 'TEN'],
                'display_name': 'GB @ TEN - Sun 1:00 PM',
                'weather_relevant': True
            },
            {
                'id': 'early_5', 'week': week, 'time_slot': 'sunday_early',
                'away_team': 'PHI', 'home_team': 'NO', 'teams': ['PHI', 'NO'],
                'display_name': 'PHI @ NO - Sun 1:00 PM',
                'weather_relevant': False  # Dome
            },
            {
                'id': 'early_6', 'week': week, 'time_slot': 'sunday_early',
                'away_team': 'DEN', 'home_team': 'TB', 'teams': ['DEN', 'TB'],
                'display_name': 'DEN @ TB - Sun 1:00 PM',
                'weather_relevant': True
            },
            {
                'id': 'early_7', 'week': week, 'time_slot': 'sunday_early',
                'away_team': 'MIA', 'home_team': 'SEA', 'teams': ['MIA', 'SEA'],
                'display_name': 'MIA @ SEA - Sun 1:00 PM',
                'weather_relevant': True
            },
            # Sunday 4:00 PM ET Games
            {
                'id': 'late_1', 'week': week, 'time_slot': 'sunday_late',
                'away_team': 'PIT', 'home_team': 'LAC', 'teams': ['PIT', 'LAC'],
                'display_name': 'PIT @ LAC - Sun 4:05 PM',
                'weather_relevant': False  # Dome
            },
            {
                'id': 'late_2', 'week': week, 'time_slot': 'sunday_late',
                'away_team': 'BAL', 'home_team': 'DAL', 'teams': ['BAL', 'DAL'],
                'display_name': 'BAL @ DAL - Sun 4:25 PM',
                'weather_relevant': False  # Retractable roof
            },
            {
                'id': 'late_3', 'week': week, 'time_slot': 'sunday_late',
                'away_team': 'DET', 'home_team': 'ARI', 'teams': ['DET', 'ARI'],
                'display_name': 'DET @ ARI - Sun 4:25 PM',
                'weather_relevant': False  # Retractable roof
            },
            {
                'id': 'late_4', 'week': week, 'time_slot': 'sunday_late',
                'away_team': 'KC', 'home_team': 'ATL', 'teams': ['KC', 'ATL'],
                'display_name': 'KC @ ATL - Sun 4:25 PM',
                'weather_relevant': False  # Dome
            },
            # Sunday Night Football
            {
                'id': 'snf_1', 'week': week, 'time_slot': 'sunday_night',
                'away_team': 'BUF', 'home_team': 'MIA', 'teams': ['BUF', 'MIA'],
                'display_name': 'BUF @ MIA - Sun 8:20 PM',
                'weather_relevant': True
            },
            # Monday Night Football
            {
                'id': 'mnf_1', 'week': week, 'time_slot': 'monday',
                'away_team': 'WAS', 'home_team': 'CIN', 'teams': ['WAS', 'CIN'],
                'display_name': 'WAS @ CIN - Mon 8:15 PM',
                'weather_relevant': True
            },
        ]
        
        logger.info(f"Using manual schedule: {len(manual_games)} games for week {week}")
        return manual_games
    
    def _categorize_games_by_slate(self, games: List[Dict]) -> Dict[str, List]:
        """Categorize games by slate type"""
        categorized = {
            'main_slate': [],      # Sunday 1PM + 4PM for tournaments
            'single_games': [],    # All games for single game contests  
            'prime_time': [],      # TNF, SNF, MNF
            'by_day': {
                'thursday': [],
                'sunday': [], 
                'monday': []
            }
        }
        
        for game in games:
            time_slot = game.get('time_slot', 'unknown')
            
            # All games available for single game contests
            categorized['single_games'].append(game)
            
            # Main slate: Sunday early + late games only
            if time_slot in ['sunday_early', 'sunday_late']:
                categorized['main_slate'].append(game)
                categorized['by_day']['sunday'].append(game)
            
            # Prime time games
            elif time_slot in ['thursday', 'sunday_night', 'monday']:
                categorized['prime_time'].append(game)
                if time_slot == 'thursday':
                    categorized['by_day']['thursday'].append(game)
                elif time_slot == 'monday':
                    categorized['by_day']['monday'].append(game)
                else:  # sunday_night
                    categorized['by_day']['sunday'].append(game)
        
        logger.info(f"Categorized: {len(categorized['main_slate'])} main slate, "
                   f"{len(categorized['single_games'])} single games, "
                   f"{len(categorized['prime_time'])} prime time")
        
        return categorized
    
    def _parse_game_time(self, date_str: str) -> Optional[datetime]:
        """Parse ESPN game time"""
        try:
            if date_str:
                dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return dt.astimezone(self.slate_manager.eastern)
        except:
            pass
        return None
    
    def _determine_time_slot(self, game_time: Optional[datetime]) -> str:
        """Determine time slot from game time"""
        if not game_time:
            return 'unknown'
            
        day = game_time.weekday()
        hour = game_time.hour
        
        if day == 3:  # Thursday
            return 'thursday'
        elif day == 6:  # Sunday
            if hour < 16:
                return 'sunday_early'  # 1PM games
            elif hour < 20:
                return 'sunday_late'   # 4PM games
            else:
                return 'sunday_night'  # SNF
        elif day == 0:  # Monday
            return 'monday'
        
        return 'other'
    
    def _is_outdoor_game(self, home_team: str) -> bool:
        """Check if game is outdoor/weather relevant"""
        stadium_info = NFL_STADIUMS.get(home_team, {})
        stadium_type = stadium_info.get('type', 'outdoor')
        return stadium_type in ['outdoor', 'retractable_roof']
    
    async def collect_weather_data(self, games: List[Dict]) -> Dict[str, Any]:
        """Collect weather for all relevant games"""
        weather_data = {}
        
        for game in games:
            if not game.get('weather_relevant', False):
                continue
                
            home_team = game['home_team']
            if home_team in NFL_STADIUMS:
                weather_data[home_team] = await self._get_stadium_weather(home_team, game)
        
        logger.info(f"Collected weather for {len(weather_data)} stadiums")
        return weather_data
    
    async def _get_stadium_weather(self, team: str, game: Dict) -> Dict[str, Any]:
        """Get weather for specific stadium"""
        try:
            stadium = NFL_STADIUMS[team]
            await self.rate_limiters['weather'].acquire()
            
            points_url = f"{WEATHER_API['base_url']}/points/{stadium['lat']},{stadium['lon']}"
            
            async with self.session.get(points_url) as response:
                if response.status == 200:
                    points_data = await response.json()
                    forecast_url = points_data['properties']['forecast']
                    
                    async with self.session.get(forecast_url) as forecast_response:
                        if forecast_response.status == 200:
                            forecast_data = await forecast_response.json()
                            periods = forecast_data['properties']['periods']
                            
                            return {
                                'team': team,
                                'stadium': stadium['name'],
                                'game_time': game.get('game_time', '').isoformat() if game.get('game_time') else None,
                                'current_forecast': periods[0] if periods else {},
                                'game_forecast': self._find_game_time_forecast(periods, game.get('game_time')),
                                'last_updated': datetime.now().isoformat(),
                                'alerts': await self._get_weather_alerts(points_data)
                            }
                        
        except Exception as e:
            logger.error(f"Error getting weather for {team}: {e}")
        
        return {}
    
    def _find_game_time_forecast(self, periods: List[Dict], game_time: Optional[datetime]) -> Dict:
        """Find forecast period closest to game time"""
        if not game_time or not periods:
            return periods[0] if periods else {}
        
        # Find period that includes game time
        for period in periods:
            try:
                start_time = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(period['endTime'].replace('Z', '+00:00'))
                
                if start_time <= game_time <= end_time:
                    return period
            except:
                continue
        
        return periods[0]  # Fallback to current forecast
    
    async def _get_weather_alerts(self, points_data: Dict) -> List[Dict]:
        """Get weather alerts for location"""
        try:
            alerts_url = points_data['properties'].get('alerts')
            if alerts_url:
                async with self.session.get(alerts_url) as response:
                    if response.status == 200:
                        alerts_data = await response.json()
                        return alerts_data.get('features', [])
        except:
            pass
        return []
    
    async def collect_players_for_slate(self, games_info: Dict[str, Any], contest_type: str = 'gpp') -> List[Dict]:
        """Collect players filtered by contest type and slate"""
        current_week = games_info['current_week']
        
        # Determine which teams to include
        if contest_type == 'single_game':
            # All teams playing this week (for single game selection)
            playing_teams = set()
            for game in games_info['single_games']:
                playing_teams.update(game['teams'])
        else:
            # Main slate teams only for tournaments/cash/contrarian
            playing_teams = set()
            for game in games_info['main_slate']:
                playing_teams.update(game['teams'])
        
        logger.info(f"Collecting players for {contest_type}: {len(playing_teams)} teams playing")
        
        # Get NFL data
        try:
            await self.rate_limiters['nfl_data'].acquire()
            weekly_data = nfl.import_weekly_data([2024])
            
            if not weekly_data.empty:
                # Filter for current week and relevant teams
                relevant_data = weekly_data[
                    (weekly_data['week'] == current_week) & 
                    (weekly_data['recent_team'].isin(playing_teams))
                ].copy()
                
                # If insufficient data, include recent weeks
                if len(relevant_data) < 50:
                    recent_weeks = [current_week - 1, current_week - 2]
                    backup_data = weekly_data[
                        (weekly_data['week'].isin(recent_weeks)) & 
                        (weekly_data['recent_team'].isin(playing_teams))
                    ]
                    relevant_data = pd.concat([relevant_data, backup_data]).drop_duplicates('player_id')
                
                players = self._process_player_data(relevant_data, playing_teams)
            else:
                players = self._generate_fallback_players(playing_teams)
                
        except Exception as e:
            logger.error(f"Error collecting NFL data: {e}")
            players = self._generate_fallback_players(playing_teams)
        
        logger.info(f"Collected {len(players)} players for {contest_type}")
        return players
    
    def _process_player_data(self, data: pd.DataFrame, teams: set) -> List[Dict]:
        """Process NFL data into player objects"""
        players = []
        
        for _, row in data.iterrows():
            try:
                team = row.get('recent_team', '').upper()
                if team not in teams:
                    continue
                
                projection = self._calculate_projection(row)
                salary = self._estimate_salary(row.get('position', ''), projection)
                
                player = {
                    'player_id': row.get('player_id', ''),
                    'player_name': row.get('player_name', ''),
                    'name': row.get('player_name', ''),
                    'position': row.get('position', ''),
                    'team': team,
                    'projection': max(0, projection),
                    'salary': salary,
                    'value': projection / (salary / 1000) if salary > 0 else 0,
                    'week': int(row.get('week', 0))
                }
                
                players.append(player)
                
            except Exception as e:
                logger.warning(f"Error processing player: {e}")
                continue
        
        # Ensure all positions for all teams
        return self._ensure_complete_rosters(players, teams)
    
    def _calculate_projection(self, player: pd.Series) -> float:
        """Calculate fantasy projection"""
        # Try fantasy points columns first
        for col in ['fantasy_points_ppr', 'fantasy_points', 'fantasy_points_half_ppr']:
            if col in player.index and pd.notna(player[col]):
                return float(player[col])
        
        # Calculate from stats
        points = 0.0
        points += player.get('passing_yards', 0) * 0.04
        points += player.get('passing_tds', 0) * 4
        points += player.get('rushing_yards', 0) * 0.1
        points += player.get('rushing_tds', 0) * 6
        points += player.get('receptions', 0) * 1.0  # PPR
        points += player.get('receiving_yards', 0) * 0.1
        points += player.get('receiving_tds', 0) * 6
        
        return max(0, points)
    
    def _estimate_salary(self, position: str, projection: float) -> int:
        """Estimate FanDuel salary"""
        base_salaries = {
            'QB': 8000, 'RB': 6500, 'WR': 6000, 'TE': 5000, 
            'K': 4500, 'DST': 4500, 'DEF': 4500
        }
        
        base = base_salaries.get(position, 5000)
        
        # Adjust for projection
        if projection > 20:
            base += 2500
        elif projection > 15:
            base += 1500
        elif projection > 10:
            base += 500
        elif projection < 6:
            base -= 1000
        
        return max(3000, min(12000, base))
    
    def _ensure_complete_rosters(self, players: List[Dict], teams: set) -> List[Dict]:
        """Ensure every team has all positions"""
        team_positions = {}
        for player in players:
            team = player['team']
            pos = player['position']
            if team not in team_positions:
                team_positions[team] = set()
            team_positions[team].add(pos)
        
        required_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
        additional_players = []
        
        for team in teams:
            existing = team_positions.get(team, set())
            for pos in required_positions:
                if pos not in existing:
                    # Add placeholder
                    proj, sal = (18, 8000) if pos == 'QB' else (12, 6000) if pos == 'RB' else (10, 5500) if pos == 'WR' else (8, 5000) if pos == 'TE' else (7, 4500)
                    
                    additional_players.append({
                        'player_id': f'{pos}_{team}',
                        'player_name': f'{team} {pos}',
                        'name': f'{team} {pos}',
                        'position': pos,
                        'team': team,
                        'projection': proj,
                        'salary': sal,
                        'value': proj / (sal / 1000)
                    })
        
        return players + additional_players
    
    def _generate_fallback_players(self, teams: set) -> List[Dict]:
        """Generate fallback player data when API fails"""
        players = []
        positions = [
            ('QB', 8000, 18), ('RB', 6500, 12), ('RB', 5500, 10),
            ('WR', 6000, 11), ('WR', 5500, 9), ('WR', 5000, 8),
            ('TE', 5000, 8), ('K', 4500, 7), ('DST', 4500, 8)
        ]
        
        for team in list(teams)[:10]:  # Limit for fallback
            for i, (pos, sal, proj) in enumerate(positions):
                players.append({
                    'player_id': f'{team}_{pos}_{i}',
                    'player_name': f'{team} {pos}',
                    'name': f'{team} {pos}',
                    'position': pos,
                    'team': team,
                    'projection': proj,
                    'salary': sal,
                    'value': proj / (sal / 1000)
                })
        
        return players

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
    """Get fresh data with proper slate filtering"""
    async with EnhancedDataCollector() as collector:
        # Get current week games with categorization
        games_info = await collector.get_current_week_games()
        
        # Collect players (default to main slate for tournaments)
        players = await collector.collect_players_for_slate(games_info, 'gpp')
        
        # Collect weather for all relevant games
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
                'single_games': len(games_info['single_games']),
                'weather_locations': len(weather_data),
                'current_week': games_info['current_week'],
                'slate_type': games_info['slate_type']
            }
        }
