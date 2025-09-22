"""
Comprehensive data collection system for NFL DFS optimization
Collects data from multiple free sources with proper rate limiting and error handling
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

class DataCollector:
    """Main data collection class that aggregates NFL data from multiple sources"""
    
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
    
    async def get_current_week(self) -> int:
        """Get current NFL week"""
        try:
            await self.rate_limiters['espn'].acquire()
            async with self.session.get(ESPN_ENDPOINTS['scoreboard']) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract week from scoreboard data
                    if 'week' in data:
                        return data['week']['number']
                    return 1
        except Exception as e:
            logger.error(f"Error getting current week: {e}")
        return 1
    
    async def collect_nfl_data_py_stats(self, year: int = 2024) -> pd.DataFrame:
        """Collect comprehensive NFL data using nfl_data_py"""
        cache_key = f"nfl_data_py_{year}"
        
        if self._is_cache_valid(cache_key, 'player_projections'):
            return self.cache[cache_key]['data']
        
        try:
            await self.rate_limiters['nfl_data'].acquire()
            logger.info(f"Collecting NFL data for {year}")
            
            # Get weekly data (most recent)
            weekly_data = nfl.import_weekly_data([year])
            
            # Get seasonal stats
            seasonal_data = nfl.import_seasonal_data([year])
            
            # Get schedules to identify current week
            schedules = nfl.import_schedules([year])
            
            # Process the data to create player projections
            processed_data = self._process_nfl_data(weekly_data, seasonal_data, schedules)
            
            self._cache_data(cache_key, processed_data)
            logger.info(f"Successfully collected data for {len(processed_data)} players")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error collecting NFL data: {e}")
            # Return empty DataFrame but don't fail completely
            return pd.DataFrame()
    
    def _process_nfl_data(self, weekly_data: pd.DataFrame, seasonal_data: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
        """Process and clean NFL data from nfl_data_py"""
        try:
            if weekly_data.empty:
                logger.warning("Weekly data is empty")
                return pd.DataFrame()
            
            # Filter for relevant positions
            relevant_positions = ['QB', 'RB', 'WR', 'TE', 'K']
            
            # Get the most recent week's data
            if 'week' in weekly_data.columns:
                current_week = weekly_data['week'].max()
                recent_data = weekly_data[weekly_data['week'] >= max(1, current_week - 2)].copy()
            else:
                recent_data = weekly_data.copy()
            
            # Filter for relevant positions
            if 'position' in recent_data.columns:
                recent_data = recent_data[recent_data['position'].isin(relevant_positions)].copy()
            
            # Group by player and calculate projections
            player_stats = []
            
            for player_id in recent_data['player_id'].unique():
                player_data = recent_data[recent_data['player_id'] == player_id]
                
                if player_data.empty:
                    continue
                
                # Get the most recent game data
                latest_game = player_data.iloc[-1]
                
                # Calculate average fantasy points from recent games
                fantasy_points_col = None
                for col in ['fantasy_points_ppr', 'fantasy_points', 'fantasy_points_half_ppr']:
                    if col in player_data.columns:
                        fantasy_points_col = col
                        break
                
                if fantasy_points_col is None:
                    # Calculate basic fantasy points if not available
                    points = 0
                    if 'passing_yards' in player_data.columns:
                        points += player_data['passing_yards'].fillna(0).mean() * 0.04
                    if 'passing_tds' in player_data.columns:
                        points += player_data['passing_tds'].fillna(0).mean() * 4
                    if 'rushing_yards' in player_data.columns:
                        points += player_data['rushing_yards'].fillna(0).mean() * 0.1
                    if 'rushing_tds' in player_data.columns:
                        points += player_data['rushing_tds'].fillna(0).mean() * 6
                    if 'receiving_yards' in player_data.columns:
                        points += player_data['receiving_yards'].fillna(0).mean() * 0.1
                    if 'receiving_tds' in player_data.columns:
                        points += player_data['receiving_tds'].fillna(0).mean() * 6
                    if 'receptions' in player_data.columns:
                        points += player_data['receptions'].fillna(0).mean() * 1  # PPR
                    
                    projection = max(0, points)
                else:
                    # Use rolling average of recent games
                    projection = player_data[fantasy_points_col].fillna(0).mean()
                
                # Create player record
                player_record = {
                    'player_id': latest_game.get('player_id', ''),
                    'player_name': latest_game.get('player_name', latest_game.get('player_display_name', 'Unknown')),
                    'position': latest_game.get('position', 'UNKNOWN'),
                    'team': latest_game.get('recent_team', latest_game.get('team', 'UNK')),
                    'fantasy_points_ppr': projection,
                    'projection': max(projection, 5.0),  # Minimum projection of 5 points
                    'salary': self._estimate_salary(latest_game.get('position', 'UNKNOWN'), projection),
                    'value': 0  # Will calculate after salary
                }
                
                # Calculate value
                if player_record['salary'] > 0:
                    player_record['value'] = player_record['projection'] / (player_record['salary'] / 1000)
                
                player_stats.append(player_record)
            
            if not player_stats:
                logger.warning("No player stats generated")
                return pd.DataFrame()
            
            df = pd.DataFrame(player_stats)
            
            # Add missing positions (DST/Kickers) if needed
            df = self._add_missing_positions(df)
            
            # Remove duplicates and invalid entries
            df = df.drop_duplicates(subset=['player_name', 'team'])
            df = df[df['projection'] > 0].reset_index(drop=True)
            
            logger.info(f"Processed {len(df)} players from NFL data")
            return df
            
        except Exception as e:
            logger.error(f"Error processing NFL data: {e}")
            return pd.DataFrame()
    
    def _add_missing_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing positions needed for FanDuel lineups"""
        try:
            # Check what positions we have
            existing_positions = set(df['position'].unique())
            
            additional_players = []
            
            # Add DST/Defense if missing
            if 'DST' not in existing_positions and 'DEF' not in existing_positions:
                logger.info("Adding dummy DST players for optimization")
                nfl_teams_dst = {
                    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 
                    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 
                    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys', 
                    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers', 
                    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 
                    'KC': 'Kansas City Chiefs', 'LV': 'Las Vegas Raiders', 'LAC': 'Los Angeles Chargers', 
                    'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings', 
                    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants', 
                    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 
                    'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers', 
                    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
                }
                
                for i, (team, full_name) in enumerate(nfl_teams_dst.items()):
                    dst_player = {
                        'player_id': f'DST_{team}',
                        'player_name': f'{full_name} Defense',
                        'position': 'DST',
                        'team': team,
                        'fantasy_points_ppr': 8.0 + (i % 3),  # 8-10 points
                        'projection': 8.0 + (i % 3),
                        'salary': 4000 + (i * 100),  # $4000-$7100
                        'value': 0
                    }
                    dst_player['value'] = dst_player['projection'] / (dst_player['salary'] / 1000)
                    additional_players.append(dst_player)
            
            # Add real kickers if we don't have enough
            kicker_count = len(df[df['position'] == 'K'])
            if kicker_count < 10:
                logger.info(f"Adding real NFL kickers (currently have {kicker_count})")
                nfl_kickers = {
                    'ARI': 'Matt Prater', 'ATL': 'Younghoe Koo', 'BAL': 'Justin Tucker', 
                    'BUF': 'Tyler Bass', 'CAR': 'Eddy Pineiro', 'CHI': 'Cairo Santos', 
                    'CIN': 'Evan McPherson', 'CLE': 'Dustin Hopkins', 'DAL': 'Brandon Aubrey', 
                    'DEN': 'Wil Lutz', 'DET': 'Jake Bates', 'GB': 'Brandon McManus', 
                    'HOU': 'Ka\'imi Fairbairn', 'IND': 'Matt Gay', 'JAX': 'Cam Little', 
                    'KC': 'Harrison Butker', 'LV': 'Daniel Carlson', 'LAC': 'Cameron Dicker', 
                    'LAR': 'Joshua Karty', 'MIA': 'Jason Sanders', 'MIN': 'Will Reichard', 
                    'NE': 'Joey Slye', 'NO': 'Blake Grupe', 'NYG': 'Graham Gano', 
                    'NYJ': 'Greg Zuerlein', 'PHI': 'Jake Elliott', 'PIT': 'Chris Boswell', 
                    'SF': 'Jake Moody', 'SEA': 'Jason Myers', 'TB': 'Chase McLaughlin', 
                    'TEN': 'Nick Folk', 'WAS': 'Austin Seibert'
                }
                
                for i, (team, kicker_name) in enumerate(nfl_kickers.items()):
                    if f'K_{team}' not in df['player_id'].values:
                        # Estimate kicker performance based on team offense
                        base_points = 7.5
                        if team in ['KC', 'BUF', 'SF', 'DAL', 'PHI']:  # High-scoring offenses
                            base_points = 9.0
                        elif team in ['BAL', 'MIA', 'DET', 'LAR']:  # Good offenses
                            base_points = 8.5
                        
                        kicker_player = {
                            'player_id': f'K_{team}',
                            'player_name': kicker_name,
                            'position': 'K',
                            'team': team,
                            'fantasy_points_ppr': base_points,
                            'projection': base_points,
                            'salary': 4200 + (i * 50),  # $4200-$5750
                            'value': 0
                        }
                        kicker_player['value'] = kicker_player['projection'] / (kicker_player['salary'] / 1000)
                        additional_players.append(kicker_player)
            
            if additional_players:
                additional_df = pd.DataFrame(additional_players)
                df = pd.concat([df, additional_df], ignore_index=True)
                logger.info(f"Added {len(additional_players)} additional players")
            
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
            base += 2000
        elif projection > 15:
            base += 1000
        elif projection < 8:
            base -= 1000
        
        # Add some variance
        import random
        variance = random.randint(-500, 500)
        
        return max(3000, min(15000, base + variance))
    
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
                                    'forecast': forecast_data['properties']['periods'][0],  # Today's forecast
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
            
            # Check required fields - use 'player_name' instead of 'name'
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
            
            # Remove duplicates - use player_name instead of name
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
        """Collect all data sources"""
        logger.info("Starting comprehensive data collection")
        
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
    """Get fresh NFL data - main entry point"""
    async with DataCollector() as collector:
        return await collector.collect_all_data()
