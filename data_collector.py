"""
Enhanced data collection system with proper week detection and game filtering
Ensures we get the correct players for the current NFL week
"""
import asyncio
import aiohttp
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import time
from loguru import logger

from config import (
    ESPN_ENDPOINTS, NFL_STADIUMS, WEATHER_API, DATA_DIR, 
    RATE_LIMITS, CACHE_TTL, VALIDATION_THRESHOLDS
)

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_period: int, period_seconds: int):
        self.calls_per_period = calls_per_period
        self.period_seconds = period_seconds
        self.calls = []
    
    async def acquire(self):
        now = time.time()
        # Remove calls older than the period
        self.calls = [call_time for call_time in self.calls if now - call_time < self.period_seconds]
        
        if len(self.calls) >= self.calls_per_period:
            sleep_time = self.period_seconds - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class EnhancedDataCollector:
    """Enhanced data collection with proper current week detection"""
    
    def __init__(self):
        self.session = None
        self.rate_limiters = {
            'espn': RateLimiter(RATE_LIMITS['espn_api']['calls'], RATE_LIMITS['espn_api']['period']),
            'weather': RateLimiter(RATE_LIMITS['weather_gov']['calls'], RATE_LIMITS['weather_gov']['period']),
            'nfl_data': RateLimiter(RATE_LIMITS['nfl_data_py']['calls'], RATE_LIMITS['nfl_data_py']['period'])
        }
        self.cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': WEATHER_API['user_agent']}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _is_cache_valid(self, cache_key: str, ttl_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp', 0)
        ttl = CACHE_TTL.get(ttl_key, 3600)
        return time.time() - cache_time < ttl
    
    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Store data in cache with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    async def get_current_nfl_week(self) -> Dict[str, Any]:
        """Get current NFL week and active games"""
        try:
            await self.rate_limiters['espn'].acquire()
            async with self.session.get(ESPN_ENDPOINTS['scoreboard']) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract current week info
                    week_info = {
                        'current_week': 1,
                        'season_type': 2,  # Regular season
                        'year': 2024,
                        'games': []
                    }
                    
                    # Parse scoreboard data
                    if 'week' in data:
                        week_info['current_week'] = data['week'].get('number', 1)
                    
                    if 'season' in data:
                        week_info['season_type'] = data['season'].get('type', 2)
                        week_info['year'] = data['season'].get('year', 2024)
                    
                    # Extract current week games
                    if 'events' in data:
                        for event in data['events']:
                            try:
                                game_info = {
                                    'id': event.get('id'),
                                    'date': event.get('date'),
                                    'name': event.get('name', ''),
                                    'short_name': event.get('shortName', ''),
                                    'week': event.get('week', {}).get('number', week_info['current_week']),
                                    'teams': []
                                }
                                
                                # Extract team info
                                if 'competitions' in event:
                                    for comp in event['competitions']:
                                        if 'competitors' in comp:
                                            for team in comp['competitors']:
                                                game_info['teams'].append({
                                                    'id': team.get('id'),
                                                    'abbreviation': team.get('team', {}).get('abbreviation', ''),
                                                    'display_name': team.get('team', {}).get('displayName', ''),
                                                    'is_home': team.get('homeAway') == 'home'
                                                })
                                
                                # Only include games for current week
                                if game_info['week'] == week_info['current_week']:
                                    week_info['games'].append(game_info)
                                    
                            except Exception as e:
                                logger.warning(f"Error parsing game event: {e}")
                                continue
                    
                    logger.info(f"Current NFL Week: {week_info['current_week']}, Games: {len(week_info['games'])}")
                    return week_info
                    
        except Exception as e:
            logger.error(f"Error getting current NFL week: {e}")
        
        # Fallback to manual calculation
        return self._calculate_current_week()
    
    def _calculate_current_week(self) -> Dict[str, Any]:
        """Calculate current NFL week based on date"""
        now = datetime.now()
        
        # NFL 2024 season start (approximate)
        season_start = datetime(2024, 9, 5)  # First Thursday night game
        
        if now < season_start:
            current_week = 1
        else:
            days_since_start = (now - season_start).days
            current_week = min(18, max(1, (days_since_start // 7) + 1))
        
        logger.info(f"Calculated current NFL week: {current_week}")
        
        return {
            'current_week': current_week,
            'season_type': 2,
            'year': 2024,
            'games': []
        }
    
    def _get_teams_playing_this_week(self, week_info: Dict[str, Any]) -> List[str]:
        """Extract list of teams playing in the current week"""
        teams_playing = set()
        
        for game in week_info.get('games', []):
            for team in game.get('teams', []):
                team_abbr = team.get('abbreviation', '').upper()
                if team_abbr:
                    teams_playing.add(team_abbr)
        
        # If no games found from API, use day-based logic
        if not teams_playing:
            teams_playing = self._get_teams_by_day()
        
        logger.info(f"Teams playing this week: {sorted(teams_playing)}")
        return list(teams_playing)
    
    def _get_teams_by_day(self) -> set:
        """Get teams playing based on current day of week"""
        now = datetime.now()
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        
        teams_playing = set()
        
        # Thursday games (day 3)
        if day_of_week == 3:
            thursday_teams = ['KC', 'BAL', 'DET', 'GB', 'TB', 'NO']  # Example Thursday teams
            teams_playing.update(thursday_teams[:2])  # Typically 2 teams on Thursday
        
        # Sunday games (day 6) - main slate
        elif day_of_week == 6 or (day_of_week >= 0 and day_of_week <= 2):
            # All teams except Monday night teams
            all_nfl_teams = {
                'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
                'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LAR', 'MIA',
                'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB',
                'TEN', 'WAS'
            }
            
            # Exclude typical Monday night teams (this would need weekly updates)
            monday_teams = {'TEN', 'MIA'}  # Example for current week
            sunday_teams = all_nfl_teams - monday_teams
            teams_playing.update(sunday_teams)
        
        # Monday games (day 0)
        elif day_of_week == 0:
            monday_teams = ['TEN', 'MIA']  # Example Monday teams
            teams_playing.update(monday_teams)
        
        return teams_playing
    
    async def collect_nfl_data_py_stats(self, year: int = 2024) -> pd.DataFrame:
        """Collect NFL data with proper week filtering"""
        cache_key = f"nfl_data_py_{year}"
        
        if self._is_cache_valid(cache_key, 'player_projections'):
            return self.cache[cache_key]['data']
        
        try:
            await self.rate_limiters['nfl_data'].acquire()
            logger.info(f"Collecting NFL data for {year}")
            
            # Get current week info
            week_info = await self.get_current_nfl_week()
            current_week = week_info['current_week']
            teams_playing = self._get_teams_playing_this_week(week_info)
            
            # Get weekly data for current and recent weeks
            weekly_data = nfl.import_weekly_data([year])
            
            if not weekly_data.empty:
                # Filter for current week and teams playing
                current_week_data = weekly_data[
                    (weekly_data['week'] == current_week) & 
                    (weekly_data['recent_team'].isin(teams_playing))
                ].copy()
                
                # If current week data is sparse, include previous week
                if len(current_week_data) < 100:
                    logger.info("Current week data sparse, including previous week")
                    prev_week_data = weekly_data[
                        (weekly_data['week'] == max(1, current_week - 1)) & 
                        (weekly_data['recent_team'].isin(teams_playing))
                    ].copy()
                    current_week_data = pd.concat([current_week_data, prev_week_data]).drop_duplicates(
                        subset=['player_id'], keep='first'
                    )
            
            # Get seasonal data for projections
            seasonal_data = nfl.import_seasonal_data([year])
            
            # Process the filtered data
            processed_data = self._process_nfl_data(current_week_data, seasonal_data, week_info, teams_playing)
            
            self._cache_data(cache_key, processed_data)
            logger.info(f"Successfully collected data for {len(processed_data)} players from week {current_week}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error collecting NFL data: {e}")
            return pd.DataFrame()
    
    def _process_nfl_data(self, weekly_data: pd.DataFrame, seasonal_data: pd.DataFrame, 
                         week_info: Dict[str, Any], teams_playing: List[str]) -> pd.DataFrame:
        """Process and clean NFL data with team filtering"""
        try:
            if weekly_data.empty:
                logger.warning("Weekly data is empty")
                return self._create_fallback_data(teams_playing)
            
            # Filter for relevant positions and teams
            relevant_positions = ['QB', 'RB', 'WR', 'TE', 'K']
            
            # Ensure we only have players from teams playing this week
            filtered_data = weekly_data[
                (weekly_data['position'].isin(relevant_positions)) &
                (weekly_data['recent_team'].isin(teams_playing))
            ].copy()
            
            logger.info(f"Filtered to {len(filtered_data)} players from playing teams: {teams_playing}")
            
            # Group by player and calculate projections
            player_stats = []
            
            for player_id in filtered_data['player_id'].unique():
                player_data = filtered_data[filtered_data['player_id'] == player_id]
                
                if player_data.empty:
                    continue
                
                latest_game = player_data.iloc[-1]
                
                # Calculate projection based on recent performance
                fantasy_points_cols = ['fantasy_points_ppr', 'fantasy_points', 'fantasy_points_half_ppr']
                fantasy_points_col = None
                
                for col in fantasy_points_cols:
                    if col in player_data.columns:
                        fantasy_points_col = col
                        break
                
                if fantasy_points_col:
                    projection = player_data[fantasy_points_col].fillna(0).mean()
                else:
                    # Calculate basic fantasy points
                    projection = self._calculate_basic_fantasy_points(player_data)
                
                # Create player record
                player_record = {
                    'player_id': latest_game.get('player_id', ''),
                    'player_name': latest_game.get('player_name', latest_game.get('player_display_name', 'Unknown')),
                    'position': latest_game.get('position', 'UNKNOWN'),
                    'team': latest_game.get('recent_team', latest_game.get('team', 'UNK')),
                    'fantasy_points_ppr': projection,
                    'projection': max(projection, 3.0),  # Minimum projection
                    'salary': self._estimate_salary(latest_game.get('position', 'UNKNOWN'), projection),
                    'value': 0,
                    'week': week_info['current_week']
                }
                
                # Calculate value
                if player_record['salary'] > 0:
                    player_record['value'] = player_record['projection'] / (player_record['salary'] / 1000)
                
                player_stats.append(player_record)
            
            if not player_stats:
                logger.warning("No player stats generated, creating fallback data")
                return self._create_fallback_data(teams_playing)
            
            df = pd.DataFrame(player_stats)
            
            # Add missing positions (DST/Kickers) for teams playing
            df = self._add_missing_positions(df, teams_playing)
            
            # Clean and validate
            df = df.drop_duplicates(subset=['player_name', 'team'])
            df = df[df['projection'] > 0].reset_index(drop=True)
            
            # Ensure name compatibility
            if 'name' not in df.columns and 'player_name' in df.columns:
                df['name'] = df['player_name']
            
            logger.info(f"Processed {len(df)} players for current week slate")
            return df
            
        except Exception as e:
            logger.error(f"Error processing NFL data: {e}")
            return self._create_fallback_data(teams_playing)
    
    def _create_fallback_data(self, teams_playing: List[str]) -> pd.DataFrame:
        """Create fallback data when API fails"""
        logger.info("Creating fallback player data")
        
        fallback_players = []
        
        for team in teams_playing[:16]:  # Limit to reasonable number
            # Add QB
            fallback_players.append({
                'player_id': f'QB_{team}',
                'player_name': f'{team} Quarterback',
                'name': f'{team} Quarterback',
                'position': 'QB',
                'team': team,
                'projection': 18.0,
                'salary': 8000,
                'value': 2.25
            })
            
            # Add RBs
            for i in range(2):
                fallback_players.append({
                    'player_id': f'RB_{team}_{i+1}',
                    'player_name': f'{team} RB{i+1}',
                    'name': f'{team} RB{i+1}',
                    'position': 'RB',
                    'team': team,
                    'projection': 12.0 - (i * 2),
                    'salary': 7000 - (i * 1000),
                    'value': 1.8 - (i * 0.2)
                })
            
            # Add WRs
            for i in range(3):
                fallback_players.append({
                    'player_id': f'WR_{team}_{i+1}',
                    'player_name': f'{team} WR{i+1}',
                    'name': f'{team} WR{i+1}',
                    'position': 'WR',
                    'team': team,
                    'projection': 11.0 - (i * 1.5),
                    'salary': 6500 - (i * 500),
                    'value': 1.7 - (i * 0.1)
                })
            
            # Add TE
            fallback_players.append({
                'player_id': f'TE_{team}',
                'player_name': f'{team} Tight End',
                'name': f'{team} Tight End',
                'position': 'TE',
                'team': team,
                'projection': 8.5,
                'salary': 5500,
                'value': 1.55
            })
            
            # Add K
            fallback_players.append({
                'player_id': f'K_{team}',
                'player_name': f'{team} Kicker',
                'name': f'{team} Kicker',
                'position': 'K',
                'team': team,
                'projection': 7.0,
                'salary': 4500,
                'value': 1.56
            })
            
            # Add DST
            fallback_players.append({
                'player_id': f'DST_{team}',
                'player_name': f'{team} Defense',
                'name': f'{team} Defense',
                'position': 'DST',
                'team': team,
                'projection': 8.0,
                'salary': 4500,
                'value': 1.78
            })
        
        return pd.DataFrame(fallback_players)
    
    def _calculate_basic_fantasy_points(self, player_data: pd.DataFrame) -> float:
        """Calculate basic fantasy points from stats"""
        points = 0
        
        # Passing
        if 'passing_yards' in player_data.columns:
            points += player_data['passing_yards'].fillna(0).mean() * 0.04
        if 'passing_tds' in player_data.columns:
            points += player_data['passing_tds'].fillna(0).mean() * 4
        
        # Rushing
        if 'rushing_yards' in player_data.columns:
            points += player_data['rushing_yards'].fillna(0).mean() * 0.1
        if 'rushing_tds' in player_data.columns:
            points += player_data['rushing_tds'].fillna(0).mean() * 6
        
        # Receiving
        if 'receiving_yards' in player_data.columns:
            points += player_data['receiving_yards'].fillna(0).mean() * 0.1
        if 'receiving_tds' in player_data.columns:
            points += player_data['receiving_tds'].fillna(0).mean() * 6
        if 'receptions' in player_data.columns:
            points += player_data['receptions'].fillna(0).mean() * 1  # PPR
        
        return max(0, points)
    
    def _add_missing_positions(self, df: pd.DataFrame, teams_playing: List[str]) -> pd.DataFrame:
        """Add missing positions for teams playing this week"""
        try:
            existing_positions = set(df['position'].unique()) if not df.empty else set()
            additional_players = []
            
            # Add DST if missing
            if 'DST' not in existing_positions and 'DEF' not in existing_positions:
                for team in teams_playing:
                    dst_player = {
                        'player_id': f'DST_{team}',
                        'player_name': f'{team} Defense',
                        'name': f'{team} Defense',
                        'position': 'DST',
                        'team': team,
                        'fantasy_points_ppr': 8.0,
                        'projection': 8.0,
                        'salary': 4500,
                        'value': 1.78
                    }
                    additional_players.append(dst_player)
            
            # Add kickers for teams playing
            existing_kickers = set(df[df['position'] == 'K']['team'].values) if not df.empty else set()
            missing_kicker_teams = set(teams_playing) - existing_kickers
            
            for team in missing_kicker_teams:
                kicker_player = {
                    'player_id': f'K_{team}',
                    'player_name': f'{team} Kicker',
                    'name': f'{team} Kicker',
                    'position': 'K',
                    'team': team,
                    'fantasy_points_ppr': 7.0,
                    'projection': 7.0,
                    'salary': 4500,
                    'value': 1.56
                }
                additional_players.append(kicker_player)
            
            if additional_players:
                additional_df = pd.DataFrame(additional_players)
                df = pd.concat([df, additional_df], ignore_index=True)
                logger.info(f"Added {len(additional_players)} missing position players")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding missing positions: {e}")
            return df
    
    def _estimate_salary(self, position: str, projection: float) -> int:
        """Estimate salary based on position and projection"""
        base_salaries = {
            'QB': 8000,
            'RB': 7000,
            'WR': 6500,
            'TE': 5500,
            'K': 4500,
            'DST': 4500,
            'DEF': 4500
        }
        
        base = base_salaries.get(position, 5000)
        
        # Adjust based on projection
        if projection > 20:
            base += 2500
        elif projection > 15:
            base += 1500
        elif projection > 10:
            base += 500
        elif projection < 6:
            base -= 1000
        
        # Add variance
        import random
        variance = random.randint(-300, 300)
        
        return max(3000, min(15000, base + variance))
    
    # Include all other methods from original DataCollector
    async def collect_espn_data(self) -> Dict[str, Any]:
        """Collect data from ESPN's free API"""
        cache_key = "espn_data"
        
        if self._is_cache_valid(cache_key, 'player_projections'):
            return self.cache[cache_key]['data']
        
        try:
            await self.rate_limiters['espn'].acquire()
            
            espn_data = {}
            
            # Get scoreboard for current games
            async with self.session.get(ESPN_ENDPOINTS['scoreboard']) as response:
                if response.status == 200:
                    espn_data['scoreboard'] = await response.json()
            
            # Get news
            async with self.session.get(ESPN_ENDPOINTS['news']) as response:
                if response.status == 200:
                    espn_data['news'] = await response.json()
            
            # Get teams
            async with self.session.get(ESPN_ENDPOINTS['teams']) as response:
                if response.status == 200:
                    espn_data['teams'] = await response.json()
            
            self._cache_data(cache_key, espn_data)
            logger.info("Successfully collected ESPN data")
            
            return espn_data
            
        except Exception as e:
            logger.error(f"Error collecting ESPN data: {e}")
            return {}
    
    async def collect_weather_data(self) -> Dict[str, Any]:
        """Collect weather data for NFL stadiums"""
        cache_key = "weather_data"
        
        if self._is_cache_valid(cache_key, 'weather_data'):
            return self.cache[cache_key]['data']
        
        weather_data = {}
        
        for team, stadium in NFL_STADIUMS.items():
            try:
                await self.rate_limiters['weather'].acquire()
                
                # Get weather grid point
                points_url = f"{WEATHER_API['base_url']}/points/{stadium['lat']},{stadium['lon']}"
                
                async with self.session.get(points_url) as response:
                    if response.status == 200:
                        points_data = await response.json()
                        forecast_url = points_data['properties']['forecast']
                        
                        # Get forecast
                        async with self.session.get(forecast_url) as response:
                            if response.status == 200:
                                forecast_data = await response.json()
                                weather_data[team] = {
                                    'stadium': stadium['name'],
                                    'forecast': forecast_data['properties']['periods'][0],
                                    'lat': stadium['lat'],
                                    'lon': stadium['lon']
                                }
                
            except Exception as e:
                logger.error(f"Error getting weather for {team}: {e}")
                continue
        
        self._cache_data(cache_key, weather_data)
        logger.info(f"Successfully collected weather for {len(weather_data)} stadiums")
        
        return weather_data
    
    async def collect_injury_reports(self) -> List[Dict[str, Any]]:
        """Collect injury reports from ESPN news"""
        try:
            espn_data = await self.collect_espn_data()
            
            if 'news' not in espn_data:
                return []
            
            injury_reports = []
            
            for article in espn_data['news'].get('articles', []):
                headline = article.get('headline', '').lower()
                description = article.get('description', '').lower()
                
                # Look for injury-related keywords
                injury_keywords = ['injury', 'injured', 'out', 'questionable', 'doubtful', 'ruled out']
                
                if any(keyword in headline or keyword in description for keyword in injury_keywords):
                    injury_reports.append({
                        'headline': article.get('headline'),
                        'description': article.get('description'),
                        'published': article.get('published'),
                        'link': article.get('links', {}).get('web', {}).get('href')
                    })
            
            logger.info(f"Found {len(injury_reports)} injury-related news items")
            return injury_reports
            
        except Exception as e:
            logger.error(f"Error collecting injury reports: {e}")
            return []
    
    def validate_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate player data meets requirements"""
        try:
            if df.empty:
                logger.warning("Empty dataframe provided for validation")
                return pd.DataFrame()
            
            # Check required fields
            required_fields = ['player_name', 'position', 'team', 'salary', 'projection']
            missing_fields = [field for field in required_fields if field not in df.columns]
            
            if missing_fields:
                logger.warning(f"Missing required fields: {missing_fields}")
                # Try to add missing fields if possible
                if 'name' not in df.columns and 'player_name' in df.columns:
                    df['name'] = df['player_name']
                elif 'player_name' not in df.columns and 'name' in df.columns:
                    df['player_name'] = df['name']
                
                # Check again after attempted fix
                missing_fields = [field for field in required_fields if field not in df.columns]
                if missing_fields:
                    logger.error(f"Still missing required fields: {missing_fields}")
                    return pd.DataFrame()
            
            # Filter valid salary range
            df = df[
                (df['salary'] >= VALIDATION_THRESHOLDS['min_salary']) &
                (df['salary'] <= VALIDATION_THRESHOLDS['max_salary'])
            ]
            
            # Filter valid projection range
            df = df[
                (df['projection'] >= VALIDATION_THRESHOLDS['min_projection']) &
                (df['projection'] <= VALIDATION_THRESHOLDS['max_projection'])
            ]
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['player_name', 'team'])
            
            # Ensure we have the 'name' field for compatibility
            if 'name' not in df.columns and 'player_name' in df.columns:
                df['name'] = df['player_name']
            
            logger.info(f"Validation complete: {len(df)} valid players")
            return df
            
        except Exception as e:
            logger.error(f"Error validating player data: {e}")
            return df
    
    async def collect_all_data(self) -> Dict[str, Any]:
        """Collect all data sources with current week filtering"""
        logger.info("Starting comprehensive data collection with week filtering")
        
        try:
            # Collect data concurrently
            tasks = [
                self.collect_nfl_data_py_stats(),
                self.collect_espn_data(),
                self.collect_weather_data(),
                self.collect_injury_reports()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            nfl_data, espn_data, weather_data, injury_reports = results
            
            # Validate the main player data
            if isinstance(nfl_data, pd.DataFrame) and not nfl_data.empty:
                nfl_data = self.validate_player_data(nfl_data)
            
            comprehensive_data = {
                'players': nfl_data,
                'espn_data': espn_data if not isinstance(espn_data, Exception) else {},
                'weather': weather_data if not isinstance(weather_data, Exception) else {},
                'injuries': injury_reports if not isinstance(injury_reports, Exception) else [],
                'last_updated': datetime.now().isoformat(),
                'data_quality': {
                    'player_count': len(nfl_data) if isinstance(nfl_data, pd.DataFrame) else 0,
                    'weather_stadiums': len(weather_data) if isinstance(weather_data, dict) else 0,
                    'injury_reports': len(injury_reports) if isinstance(injury_reports, list) else 0
                }
            }
            
            # Save to file for persistence
            output_file = DATA_DIR / f"nfl_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert DataFrame to dict for JSON serialization
            if isinstance(comprehensive_data['players'], pd.DataFrame):
                comprehensive_data['players'] = comprehensive_data['players'].to_dict('records')
            
            with open(output_file, 'w') as f:
                json.dump(comprehensive_data, f, indent=2, default=str)
            
            logger.info(f"Data collection complete. Saved to {output_file}")
            logger.info(f"Data quality: {comprehensive_data['data_quality']}")
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection: {e}")
            return {
                'players': [],
                'espn_data': {},
                'weather': {},
                'injuries': [],
                'last_updated': datetime.now().isoformat(),
                'error': str(e)
            }

# Utility function for external use
async def get_fresh_data() -> Dict[str, Any]:
    """Get fresh NFL data with current week filtering - main entry point"""
    async with EnhancedDataCollector() as collector:
        return await collector.collect_all_data()
