import asyncio
import aiohttp
from aiolimiter import AsyncLimiter
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
from bs4 import BeautifulSoup
from config import config
from models import Player, Game, SlateInfo
from database import db
from utils import exponential_backoff, cache_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.session = None
        self.espn_limiter = AsyncLimiter(60, 60)  # 60 requests per minute
        self.weather_limiter = AsyncLimiter(100, 3600)  # 100 per hour
        self.cache = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def collect_all_data(self, week: int = None) -> Dict:
        """Collect data from all sources"""
        logger.info("Starting comprehensive data collection...")
        
        tasks = [
            self.fetch_espn_data(week),
            self.fetch_sleeper_data(),
            self.fetch_weather_data(),
            self.fetch_injury_reports(),
            self.fetch_vegas_lines()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all data sources
        combined_data = {
            'players': [],
            'games': [],
            'injuries': {},
            'weather': {},
            'vegas': {}
        }
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                continue
            if result:
                combined_data.update(result)
        
        return combined_data
    
    @exponential_backoff(max_retries=3)
    async def fetch_espn_data(self, week: int = None) -> Dict:
        """Fetch data from ESPN's public APIs"""
        base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
        
        async with self.espn_limiter:
            # Get current week if not specified
            if not week:
                async with self.session.get(f"{base_url}/scoreboard") as resp:
                    data = await resp.json()
                    week = data.get('week', {}).get('number', 1)
            
            # Fetch player stats
            players = []
            teams_url = f"{base_url}/teams"
            async with self.session.get(teams_url) as resp:
                teams_data = await resp.json()
                
            for team in teams_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
                team_id = team['team']['id']
                roster_url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"
                
                async with self.espn_limiter:
                    async with self.session.get(roster_url) as resp:
                        if resp.status == 200:
                            roster_data = await resp.json()
                            for athlete in roster_data.get('athletes', []):
                                players.append(self._parse_espn_player(athlete, team['team']['abbreviation']))
            
            return {'players': players}
    
    def _parse_espn_player(self, athlete: dict, team: str) -> dict:
        """Parse ESPN player data"""
        return {
            'id': str(athlete.get('id')),
            'name': athlete.get('fullName', ''),
            'position': athlete.get('position', {}).get('abbreviation', ''),
            'team': team,
            'jersey': athlete.get('jersey', ''),
            'status': athlete.get('status', {}).get('type', {}).get('name', 'ACTIVE')
        }
    
    @cache_result(ttl=3600)
    async def fetch_sleeper_data(self) -> Dict:
        """Fetch trending and player data from Sleeper"""
        base_url = "https://api.sleeper.app/v1"
        
        # Get all NFL players
        async with self.session.get(f"{base_url}/players/nfl") as resp:
            all_players = await resp.json()
        
        # Get trending players (ownership insights)
        async with self.session.get(f"{base_url}/players/nfl/trending/add") as resp:
            trending = await resp.json()
        
        return {
            'sleeper_players': all_players,
            'trending': trending
        }
    
    async def fetch_weather_data(self) -> Dict:
        """Fetch weather for all outdoor stadiums"""
        weather_data = {}
        outdoor_teams = ['BUF', 'CHI', 'CIN', 'CLE', 'DEN', 'GB', 'KC', 'NE', 'NYJ', 'NYG', 'PIT', 'WAS']
        
        for team in outdoor_teams:
            if team in config.NFL_STADIUMS:
                coords = config.NFL_STADIUMS[team]
                weather = await self._fetch_single_stadium_weather(coords['lat'], coords['lon'])
                if weather:
                    weather_data[team] = weather
        
        return {'weather': weather_data}
    
    async def _fetch_single_stadium_weather(self, lat: float, lon: float) -> Optional[Dict]:
        """Fetch weather for a single stadium"""
        headers = {"User-Agent": "DFS Weather Bot (dfs@example.com)"}
        
        async with self.weather_limiter:
            try:
                # Get grid point
                points_url = f"https://api.weather.gov/points/{lat},{lon}"
                async with self.session.get(points_url, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
                
                # Get forecast
                forecast_url = data['properties']['forecast']
                async with self.session.get(forecast_url, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    forecast = await resp.json()
                
                # Parse current period
                current = forecast['properties']['periods'][0]
                return {
                    'temperature': current.get('temperature'),
                    'wind_speed': self._parse_wind_speed(current.get('windSpeed', '')),
                    'precipitation': 'rain' in current.get('shortForecast', '').lower(),
                    'description': current.get('shortForecast')
                }
            except Exception as e:
                logger.error(f"Weather fetch failed: {e}")
                return None
    
    def _parse_wind_speed(self, wind_str: str) -> int:
        """Extract wind speed from string like '10 to 15 mph'"""
        import re
        numbers = re.findall(r'\d+', wind_str)
        if numbers:
            return int(numbers[-1])  # Use higher number if range
        return 0
    
    async def fetch_injury_reports(self) -> Dict:
        """Fetch injury reports from ESPN"""
        injuries = {}
        base_url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams"
        
        for team, coords in config.NFL_STADIUMS.items():
            team_id = self._get_espn_team_id(team)
            if team_id:
                url = f"{base_url}/{team_id}/injuries"
                async with self.espn_limiter:
                    try:
                        async with self.session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                injuries[team] = data.get('items', [])
                    except Exception as e:
                        logger.error(f"Injury fetch failed for {team}: {e}")
        
        return {'injuries': injuries}
    
    def _get_espn_team_id(self, abbr: str) -> Optional[str]:
        """Map team abbreviation to ESPN ID"""
        team_map = {
            'ARI': '22', 'ATL': '1', 'BAL': '33', 'BUF': '2',
            'CAR': '29', 'CHI': '3', 'CIN': '4', 'CLE': '5',
            'DAL': '6', 'DEN': '7', 'DET': '8', 'GB': '9',
            'HOU': '34', 'IND': '11', 'JAX': '30', 'KC': '12',
            'LV': '13', 'LAC': '24', 'LAR': '14', 'MIA': '15',
            'MIN': '16', 'NE': '17', 'NO': '18', 'NYG': '19',
            'NYJ': '20', 'PHI': '21', 'PIT': '23', 'SF': '25',
            'SEA': '26', 'TB': '27', 'TEN': '10', 'WAS': '28'
        }
        return team_map.get(abbr)
    
    async def fetch_vegas_lines(self) -> Dict:
        """Fetch Vegas lines (you'd need an API key for production)"""
        # For now, return mock data
        # In production, use TheOddsAPI or similar service
        return {
            'vegas': {
                'games': [
                    {'home': 'KC', 'away': 'BUF', 'total': 54.5, 'spread': -2.5},
                    {'home': 'GB', 'away': 'CHI', 'total': 44.5, 'spread': -7.5},
                    # Add more games
                ]
            }
        }
    
    async def fetch_dfs_salaries(self) -> List[Player]:
        """Fetch current DFS salaries - implement based on your data source"""
        # This is where you'd integrate with FanDuel or DraftKings
        # For now, returning mock data
        players = []
        
        # In production, you'd scrape or API call for real salaries
        mock_players = [
            {'name': 'Patrick Mahomes', 'position': 'QB', 'team': 'KC', 'salary': 8500},
            {'name': 'Josh Allen', 'position': 'QB', 'team': 'BUF', 'salary': 8200},
            {'name': 'Christian McCaffrey', 'position': 'RB', 'team': 'SF', 'salary': 9000},
            # Add more players
        ]
        
        for p in mock_players:
            players.append(Player(
                id=p['name'].replace(' ', '_').lower(),
                name=p['name'],
                position=p['position'],
                team=p['team'],
                opponent='TBD',
                salary=p['salary'],
                projected_points=0.0  # Will be calculated
            ))
        
        return players

# Singleton collector instance
collector = DataCollector()
