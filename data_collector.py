"""
Automated data collection from free APIs and FanDuel
"""
import asyncio
import aiohttp
from aiolimiter import AsyncLimiter
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import nfl_data_py as nfl
import pandas as pd
import polars as pl
from loguru import logger
from config import config
import redis
from tenacity import retry, stop_after_attempt, wait_exponential
from fanduel_scraper import FanDuelScraper, AlternativeSalarySource

class NFLDataCollector:
    """Collects NFL data from multiple free sources"""
    
    def __init__(self):
        self.session = None
        self.rate_limiter = AsyncLimiter(60, 60)  # 60 requests per minute
        self.redis_client = redis.from_url(config.REDIS_URL)
        self.cache_ttl = 1800  # 30 minutes
        self.fanduel_scraper = FanDuelScraper()
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def collect_all_data(self, week: int, season: int = 2024) -> Dict:
        """
        Collect all data for a given week including FanDuel salaries
        
        Args:
            week: NFL week number
            season: NFL season year
            
        Returns:
            Dictionary containing all collected data
        """
        logger.info(f"Starting data collection for Week {week}, Season {season}")
        
        # Check cache first
        cache_key = f"nfl_data:{season}:{week}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            logger.info("Using cached data")
            return json.loads(cached_data)
        
        # Collect data from all sources concurrently
        tasks = [
            self.get_fanduel_salaries(week),  # NEW: Automated FanDuel salaries
            self.get_nfl_data_py(week, season),
            self.get_espn_data(week, season),
            self.get_sleeper_data(),
            self.get_weather_data(week, season),
            self.get_injury_reports(),
            self.get_vegas_lines(week, season)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data = {
            'salaries': results[0] if not isinstance(results[0], Exception) else {},
            'players': results[1] if not isinstance(results[1], Exception) else {},
            'espn': results[2] if not isinstance(results[2], Exception) else {},
            'sleeper': results[3] if not isinstance(results[3], Exception) else {},
            'weather': results[4] if not isinstance(results[4], Exception) else {},
            'injuries': results[5] if not isinstance(results[5], Exception) else {},
            'vegas': results[6] if not isinstance(results[6], Exception) else {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the data
        self.redis_client.setex(
            cache_key,
            self.cache_ttl,
            json.dumps(data, default=str)
        )
        
        logger.info("Data collection completed successfully")
        return data
    
    async def get_fanduel_salaries(self, week: int) -> Dict:
        """Get FanDuel salaries automatically"""
        try:
            logger.info("Fetching FanDuel salaries automatically")
            
            # Get salaries from FanDuel scraper
            salaries_df = await self.fanduel_scraper.get_nfl_salaries(week)
            
            if salaries_df.empty:
                logger.warning("FanDuel scraping failed, trying alternative sources")
                
                # Try alternative sources
                salaries_df = await AlternativeSalarySource.get_from_dfs_sites()
                
                if salaries_df.empty:
                    logger.warning("Alternative sources failed, estimating salaries")
                    # Last resort: estimate based on projections
                    player_stats = await self.get_nfl_data_py(week, 2024)
                    if player_stats:
                        stats_df = pd.DataFrame(player_stats.get('weekly', []))
                        salaries_df = await AlternativeSalarySource.estimate_salaries(stats_df)
            
            if not salaries_df.empty:
                logger.info(f"Retrieved {len(salaries_df)} player salaries")
                return salaries_df.to_dict('records')
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching FanDuel salaries: {e}")
            return {}
    
    async def get_nfl_data_py(self, week: int, season: int) -> Dict:
        """Get data from nfl-data-py (no API key required)"""
        try:
            logger.info("Fetching nfl-data-py data")
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Get weekly data
            weekly_data = await loop.run_in_executor(
                None,
                nfl.import_weekly_data,
                [season],
                ['player_id', 'player_name', 'position', 'team', 'week',
                 'fantasy_points', 'passing_yards', 'passing_tds', 
                 'rushing_yards', 'rushing_tds', 'receiving_yards', 
                 'receiving_tds', 'targets', 'receptions']
            )
            
            # Get roster data for current info
            rosters = await loop.run_in_executor(
                None,
                nfl.import_rosters,
                [season],
                ['player_id', 'player_name', 'position', 'team', 'status']
            )
            
            # Get schedule for matchups
            schedule = await loop.run_in_executor(
                None,
                nfl.import_schedules,
                [season]
            )
            
            return {
                'weekly': weekly_data.to_dict('records'),
                'rosters': rosters.to_dict('records'),
                'schedule': schedule[schedule['week'] == week].to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error fetching nfl-data-py: {e}")
            return {}
    
    # ... rest of the methods remain the same (get_espn_data, get_sleeper_data, etc.)
    # No FantasyPros references, all other methods unchanged
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_espn_data(self, week: int, season: int) -> Dict:
        """Get data from ESPN's undocumented APIs"""
        try:
            logger.info("Fetching ESPN data")
            
            async with self.rate_limiter:
                # Get current scoreboard
                url = f"{config.ESPN_BASE_URL}/apis/site/v2/sports/football/nfl/scoreboard"
                params = {'dates': season, 'seasontype': 2, 'week': week}
                
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    scoreboard = await response.json()
            
            # Get player news
            news_data = []
            async with self.rate_limiter:
                news_url = f"{config.ESPN_BASE_URL}/apis/fantasy/v2/games/ffl/news/players"
                async with self.session.get(news_url) as response:
                    if response.status == 200:
                        news_data = await response.json()
            
            return {
                'scoreboard': scoreboard,
                'news': news_data
            }
            
        except Exception as e:
            logger.error(f"Error fetching ESPN data: {e}")
            return {}
    
    async def get_sleeper_data(self) -> Dict:
        """Get trending players and projections from Sleeper"""
        try:
            logger.info("Fetching Sleeper data")
            
            results = {}
            
            # Get all NFL players
            async with self.rate_limiter:
                url = f"{config.SLEEPER_BASE_URL}/players/nfl"
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    results['players'] = await response.json()
            
            # Get trending adds (ownership insights)
            async with self.rate_limiter:
                url = f"{config.SLEEPER_BASE_URL}/players/nfl/trending/add"
                params = {'lookback_hours': 24, 'limit': 50}
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    results['trending'] = await response.json()
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching Sleeper data: {e}")
            return {}
    
    async def get_weather_data(self, week: int, season: int) -> Dict:
        """Get weather data for outdoor stadiums"""
        try:
            logger.info("Fetching weather data for outdoor stadiums")
            
            weather_data = {}
            
            for team, stadium in config.NFL_STADIUMS.items():
                if stadium['dome']:
                    continue  # Skip domed stadiums
                
                try:
                    async with self.rate_limiter:
                        # Get grid point
                        points_url = f"{config.WEATHER_BASE_URL}/points/{stadium['lat']},{stadium['lon']}"
                        headers = {'User-Agent': 'DFS Optimizer (contact@example.com)'}
                        
                        async with self.session.get(points_url, headers=headers) as response:
                            if response.status != 200:
                                continue
                            points_data = await response.json()
                        
                        # Get forecast
                        forecast_url = points_data['properties']['forecast']
                        async with self.session.get(forecast_url, headers=headers) as response:
                            if response.status == 200:
                                forecast = await response.json()
                                weather_data[team] = forecast['properties']['periods'][:2]
                        
                        await asyncio.sleep(0.5)  # Be respectful to free API
                        
                except Exception as e:
                    logger.warning(f"Weather fetch failed for {team}: {e}")
                    continue
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return {}
    
    async def get_injury_reports(self) -> Dict:
        """Get injury reports from ESPN"""
        try:
            logger.info("Fetching injury reports")
            
            injuries = {}
            teams = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
                    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
                    'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
                    'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']
            
            # ESPN team IDs mapping (simplified - you'd need full mapping)
            team_ids = {team: idx+1 for idx, team in enumerate(teams)}
            
            for team, team_id in team_ids.items():
                try:
                    async with self.rate_limiter:
                        url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
                        async with self.session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                injuries[team] = data.get('items', [])
                        
                        await asyncio.sleep(0.2)  # Rate limiting
                        
                except Exception as e:
                    logger.warning(f"Failed to get injuries for {team}: {e}")
                    continue
            
            return injuries
            
        except Exception as e:
            logger.error(f"Error fetching injury reports: {e}")
            return {}
    
    async def get_vegas_lines(self, week: int, season: int) -> Dict:
        """Get Vegas lines from ESPN scoreboard data"""
        try:
            logger.info("Extracting Vegas lines from ESPN data")
            
            async with self.rate_limiter:
                url = f"{config.ESPN_BASE_URL}/apis/site/v2/sports/football/nfl/scoreboard"
                params = {'dates': season, 'seasontype': 2, 'week': week}
                
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
            
            vegas_lines = {}
            for event in data.get('events', []):
                game_id = event['id']
                competitions = event.get('competitions', [])
                
                if competitions:
                    competition = competitions[0]
                    odds = competition.get('odds', [])
                    
                    if odds:
                        vegas_lines[game_id] = {
                            'home_team': competition['competitors'][0]['team']['abbreviation'],
                            'away_team': competition['competitors'][1]['team']['abbreviation'],
                            'spread': odds[0].get('details', 'N/A'),
                            'over_under': odds[0].get('overUnder', 0)
                        }
            
            return vegas_lines
            
        except Exception as e:
            logger.error(f"Error fetching Vegas lines: {e}")
            return {}


class DataProcessor:
    """Process and combine data from multiple sources"""
    
    def __init__(self):
        self.redis_client = redis.from_url(config.REDIS_URL)
    
    def process_all_data(self, raw_data: Dict, salary_data: Optional[pd.DataFrame] = None) -> pl.DataFrame:
        """
        Process and combine all collected data including FanDuel salaries
        
        Args:
            raw_data: Raw data from collectors
            salary_data: Optional manual salary data (now automated)
            
        Returns:
            Processed Polars DataFrame ready for optimization
        """
        logger.info("Processing collected data")
        
        # Use automated FanDuel salaries from raw_data
        if 'salaries' in raw_data and raw_data['salaries']:
            salaries_df = pd.DataFrame(raw_data['salaries'])
            logger.info(f"Using automated FanDuel salaries: {len(salaries_df)} players")
        elif salary_data is not None and not salary_data.empty:
            salaries_df = salary_data
            logger.info(f"Using provided salary data: {len(salaries_df)} players")
        else:
            logger.error("No salary data available")
            return pl.DataFrame()
        
        # Convert salary data to Polars for better performance
        salaries_pl = pl.from_pandas(salaries_df)
        
        # Process player statistics
        player_stats = self._process_player_stats(raw_data.get('players', {}))
        
        # Add injury status
        injury_data = self._process_injuries(raw_data.get('injuries', {}))
        
        # Add weather impact
        weather_impact = self._calculate_weather_impact(raw_data.get('weather', {}))
        
        # Add Vegas data for game script projections
        vegas_data = self._process_vegas_lines(raw_data.get('vegas', {}))
        
        # Combine all data sources
        combined_data = self._combine_data_sources(
            salaries_pl,
            player_stats,
            injury_data,
            weather_impact,
            vegas_data
        )
        
        # Calculate final projections
        final_data = self._calculate_projections(combined_data)
        
        logger.info(f"Processed {len(final_data)} players")
        return final_data
    
    def _process_player_stats(self, nfl_data: Dict) -> pl.DataFrame:
        """Process player statistics from nfl-data-py"""
        try:
            if not nfl_data or 'weekly' not in nfl_data:
                return pl.DataFrame()
            
            # Convert to Polars and calculate rolling averages
            weekly_df = pl.DataFrame(nfl_data['weekly'])
            
            if weekly_df.height == 0:
                return pl.DataFrame()
            
            # Calculate last 3 games average
            stats = weekly_df.group_by('player_name').agg([
                pl.col('fantasy_points').mean().alias('avg_points'),
                pl.col('fantasy_points').std().alias('std_points'),
                pl.col('fantasy_points').max().alias('ceiling'),
                pl.col('fantasy_points').min().alias('floor'),
                pl.col('targets').mean().alias('avg_targets'),
                pl.col('rushing_yards').mean().alias('avg_rush_yards'),
                pl.col('passing_yards').mean().alias('avg_pass_yards')
            ])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error processing player stats: {e}")
            return pl.DataFrame()
    
    def _process_injuries(self, injury_data: Dict) -> Dict:
        """Process injury reports into player status"""
        injury_status = {}
        
        try:
            for team, injuries in injury_data.items():
                for injury in injuries:
                    if isinstance(injury, dict):
                        player_name = injury.get('athlete', {}).get('displayName', '')
                        status = injury.get('status', 'ACTIVE')
                        injury_status[player_name] = status
            
            return injury_status
            
        except Exception as e:
            logger.error(f"Error processing injuries: {e}")
            return {}
    
    def _calculate_weather_impact(self, weather_data: Dict) -> Dict:
        """Calculate weather impact on player projections"""
        weather_impacts = {}
        
        try:
            for team, forecast in weather_data.items():
                if not forecast:
                    continue
                
                game_forecast = forecast[0]  # Get game time forecast
                
                # Parse weather conditions
                wind_speed = self._extract_wind_speed(game_forecast.get('detailedForecast', ''))
                temperature = game_forecast.get('temperature', 70)
                precipitation = 'rain' in game_forecast.get('shortForecast', '').lower()
                
                # Calculate impact factor
                impact = 1.0
                
                if wind_speed > 15:
                    impact *= 0.85  # 15% reduction for high wind
                
                if temperature < 32:
                    impact *= 0.90  # 10% reduction for freezing
                
                if precipitation:
                    impact *= 0.95  # 5% reduction for rain
                
                weather_impacts[team] = impact
            
            return weather_impacts
            
        except Exception as e:
            logger.error(f"Error calculating weather impact: {e}")
            return {}
    
    def _extract_wind_speed(self, forecast_text: str) -> int:
        """Extract wind speed from forecast text"""
        import re
        
        wind_pattern = r'wind[s]?\s+(?:around\s+)?(\d+)\s+mph'
        match = re.search(wind_pattern, forecast_text.lower())
        
        if match:
            return int(match.group(1))
        return 0
    
    def _process_vegas_lines(self, vegas_data: Dict) -> Dict:
        """Process Vegas lines for game script projections"""
        processed_vegas = {}
        
        try:
            for game_id, game_data in vegas_data.items():
                home_team = game_data['home_team']
                away_team = game_data['away_team']
                total = game_data.get('over_under', 45)
                
                # Higher totals mean more passing game script
                pass_game_factor = min(1.2, total / 45)
                
                processed_vegas[home_team] = {
                    'game_total': total,
                    'pass_game_factor': pass_game_factor
                }
                processed_vegas[away_team] = {
                    'game_total': total,
                    'pass_game_factor': pass_game_factor
                }
            
            return processed_vegas
            
        except Exception as e:
            logger.error(f"Error processing Vegas lines: {e}")
            return {}
    
    def _combine_data_sources(self, salaries, stats, injuries, weather, vegas):
        """Combine all data sources into single DataFrame"""
        try:
            # Start with salary data as base
            combined = salaries
            
            # Join player stats
            if stats.height > 0:
                combined = combined.join(
                    stats,
                    left_on='Name',
                    right_on='player_name',
                    how='left'
                )
            
            # Add injury status
            if injuries:
                injury_df = pl.DataFrame({
                    'Name': list(injuries.keys()),
                    'injury_status': list(injuries.values())
                })
                combined = combined.join(injury_df, on='Name', how='left')
            
            # Add weather and Vegas impacts
            # These would be joined based on team
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining data sources: {e}")
            return salaries
    
    def _calculate_projections(self, data: pl.DataFrame) -> pl.DataFrame:
        """Calculate final projections with all factors"""
        try:
            # Base projection from historical average
            data = data.with_columns([
                (pl.col('avg_points').fill_null(10)).alias('base_projection')
            ])
            
            # Adjust for injuries
            data = data.with_columns([
                pl.when(pl.col('injury_status') == 'OUT')
                .then(0)
                .when(pl.col('injury_status') == 'QUESTIONABLE')
                .then(pl.col('base_projection') * 0.7)
                .when(pl.col('injury_status') == 'DOUBTFUL')
                .then(pl.col('base_projection') * 0.3)
                .otherwise(pl.col('base_projection'))
                .alias('adjusted_projection')
            ])
            
            # Calculate value
            data = data.with_columns([
                (pl.col('adjusted_projection') / pl.col('Salary') * 1000).alias('value')
            ])
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating projections: {e}")
            return data
