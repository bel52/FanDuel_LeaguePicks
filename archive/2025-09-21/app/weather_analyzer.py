import os
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class WeatherAnalyzer:
    """Advanced weather analysis for DFS optimization."""
    
    # Stadium coordinates for all NFL teams
    STADIUM_COORDS = {
        'ARI': {'lat': 33.5276, 'lon': -112.2626, 'dome': True},
        'ATL': {'lat': 33.7553, 'lon': -84.4006, 'dome': True},
        'BAL': {'lat': 39.2780, 'lon': -76.6227, 'dome': False},
        'BUF': {'lat': 42.7738, 'lon': -78.7870, 'dome': False},
        'CAR': {'lat': 35.2271, 'lon': -80.8526, 'dome': False},
        'CHI': {'lat': 41.8623, 'lon': -87.6167, 'dome': False},
        'CIN': {'lat': 39.0955, 'lon': -84.5161, 'dome': False},
        'CLE': {'lat': 41.5061, 'lon': -81.6995, 'dome': False},
        'DAL': {'lat': 32.7473, 'lon': -97.0945, 'dome': True},
        'DEN': {'lat': 39.7439, 'lon': -105.0201, 'dome': False},
        'DET': {'lat': 42.3400, 'lon': -83.0456, 'dome': True},
        'GB': {'lat': 44.5013, 'lon': -88.0622, 'dome': False},
        'HOU': {'lat': 29.6847, 'lon': -95.4107, 'dome': True},
        'IND': {'lat': 39.7601, 'lon': -86.1639, 'dome': True},
        'JAX': {'lat': 30.3240, 'lon': -81.6373, 'dome': False},
        'KC': {'lat': 39.0489, 'lon': -94.4839, 'dome': False},
        'LAC': {'lat': 33.8644, 'lon': -118.2610, 'dome': True},
        'LAR': {'lat': 34.0141, 'lon': -118.2879, 'dome': True},
        'LV': {'lat': 36.0909, 'lon': -115.1833, 'dome': True},
        'MIA': {'lat': 25.9580, 'lon': -80.2389, 'dome': False},
        'MIN': {'lat': 44.9737, 'lon': -93.2581, 'dome': True},
        'NE': {'lat': 42.0909, 'lon': -71.2643, 'dome': False},
        'NO': {'lat': 29.9511, 'lon': -90.0812, 'dome': True},
        'NYG': {'lat': 40.8135, 'lon': -74.0745, 'dome': False},
        'NYJ': {'lat': 40.8135, 'lon': -74.0745, 'dome': False},
        'PHI': {'lat': 39.9008, 'lon': -75.1675, 'dome': False},
        'PIT': {'lat': 40.4469, 'lon': -80.0158, 'dome': False},
        'SF': {'lat': 37.4032, 'lon': -121.9698, 'dome': False},
        'SEA': {'lat': 47.5952, 'lon': -122.3316, 'dome': False},
        'TB': {'lat': 27.9759, 'lon': -82.5034, 'dome': False},
        'TEN': {'lat': 36.1665, 'lon': -86.7713, 'dome': False},
        'WAS': {'lat': 38.9077, 'lon': -76.8644, 'dome': False},
    }
    
    def __init__(self):
        self.weather_cache = {}
        self.impact_factors = {}
    
    async def fetch_weather_for_game(self, home_team: str, game_time: datetime) -> Optional[Dict]:
        """Fetch weather data for a specific game."""
        
        stadium = self.STADIUM_COORDS.get(home_team)
        if not stadium:
            logger.warning(f"No stadium coordinates for {home_team}")
            return None
        
        # Skip dome teams
        if stadium.get('dome'):
            return {
                'temperature': 72,
                'wind_speed': 0,
                'precipitation': 0,
                'conditions': 'Dome',
                'impact': 'neutral'
            }
        
        try:
            # Use weather.gov API (free, no key required)
            points_url = f"https://api.weather.gov/points/{stadium['lat']},{stadium['lon']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(points_url) as response:
                    if response.status != 200:
                        return None
                    
                    points_data = await response.json()
                    forecast_url = points_data['properties']['forecast']
                
                async with session.get(forecast_url) as response:
                    if response.status != 200:
                        return None
                    
                    forecast_data = await response.json()
                    periods = forecast_data['properties']['periods']
                    
                    # Find the period closest to game time
                    game_period = periods[0]  # Default to first period
                    
                    # Parse weather data
                    weather_data = self._parse_weather_period(game_period)
                    weather_data['team'] = home_team
                    
                    # Calculate fantasy impact
                    weather_data['impact'] = self._calculate_weather_impact(weather_data)
                    
                    return weather_data
                    
        except Exception as e:
            logger.error(f"Error fetching weather for {home_team}: {e}")
            return None
    
    def _parse_weather_period(self, period: Dict) -> Dict:
        """Parse weather period data from weather.gov."""
        
        # Extract temperature
        temp = period.get('temperature', 0)
        
        # Extract wind speed from text
        wind_speed = 0
        detailed = period.get('detailedForecast', '').lower()
        if 'wind' in detailed:
            import re
            wind_match = re.search(r'(\d+)\s*(?:to\s*(\d+))?\s*mph', detailed)
            if wind_match:
                wind_low = int(wind_match.group(1))
                wind_high = int(wind_match.group(2)) if wind_match.group(2) else wind_low
                wind_speed = (wind_low + wind_high) / 2
        
        # Check for precipitation
        precipitation = 0
        conditions = period.get('shortForecast', '').lower()
        if any(word in conditions for word in ['rain', 'snow', 'storm', 'showers']):
            if 'chance' in detailed:
                precip_match = re.search(r'(\d+)\s*percent', detailed)
                if precip_match:
                    precipitation = int(precip_match.group(1))
                else:
                    precipitation = 50  # Default if mentioned but no percentage
            else:
                precipitation = 80  # High chance if mentioned without "chance"
        
        return {
            'temperature': temp,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            'conditions': period.get('shortForecast', 'Clear'),
            'detailed': detailed
        }
    
    def _calculate_weather_impact(self, weather: Dict) -> Dict:
        """Calculate fantasy impact based on weather conditions."""
        
        impact = {
            'overall': 1.0,
            'passing': 1.0,
            'rushing': 1.0,
            'kicking': 1.0,
            'defense': 1.0,
            'severity': 'neutral'
        }
        
        temp = weather.get('temperature', 72)
        wind = weather.get('wind_speed', 0)
        precip = weather.get('precipitation', 0)
        
        # Temperature impacts
        if temp < 20:
            impact['passing'] *= 0.85
            impact['kicking'] *= 0.85
            impact['rushing'] *= 1.05
            impact['severity'] = 'severe'
        elif temp < 32:
            impact['passing'] *= 0.92
            impact['kicking'] *= 0.90
            impact['rushing'] *= 1.02
            impact['severity'] = 'moderate'
        elif temp > 90:
            impact['passing'] *= 0.95
            impact['rushing'] *= 0.95
            impact['severity'] = 'moderate'
        
        # Wind impacts
        if wind >= 25:
            impact['passing'] *= 0.75
            impact['kicking'] *= 0.70
            impact['rushing'] *= 1.10
            impact['defense'] *= 1.10
            impact['severity'] = 'severe'
        elif wind >= 15:
            impact['passing'] *= 0.88
            impact['kicking'] *= 0.85
            impact['rushing'] *= 1.05
            impact['defense'] *= 1.05
            if impact['severity'] == 'neutral':
                impact['severity'] = 'moderate'
        
        # Precipitation impacts
        if precip >= 70:
            impact['passing'] *= 0.85
            impact['kicking'] *= 0.88
            impact['rushing'] *= 1.08
            impact['defense'] *= 1.08
            if impact['severity'] != 'severe':
                impact['severity'] = 'moderate'
        elif precip >= 40:
            impact['passing'] *= 0.92
            impact['kicking'] *= 0.95
            impact['rushing'] *= 1.03
            impact['defense'] *= 1.03
        
        # Calculate overall impact
        impact['overall'] = (
            impact['passing'] * 0.4 +
            impact['rushing'] * 0.3 +
            impact['kicking'] * 0.1 +
            impact['defense'] * 0.2
        )
        
        return impact
    
    def apply_weather_to_projections(self, player_df: pd.DataFrame, weather_data: Dict[str, Dict]) -> pd.DataFrame:
        """Apply weather impacts to player projections."""
        
        adjusted_df = player_df.copy()
        adjusted_df['WEATHER_IMPACT'] = 1.0
        adjusted_df['WEATHER_NOTE'] = ''
        
        for idx, player in adjusted_df.iterrows():
            team = player['TEAM']
            pos = player['POS']
            
            # Check if this team has weather data
            if team not in weather_data:
                continue
            
            weather = weather_data[team]
            impact = weather.get('impact', {})
            
            # Apply position-specific impacts
            multiplier = 1.0
            note = ''
            
            if pos == 'QB':
                multiplier = impact.get('passing', 1.0)
                if multiplier < 0.9:
                    note = f"Weather concern: {weather.get('conditions')}"
            elif pos in ['WR', 'TE']:
                multiplier = impact.get('passing', 1.0) * 0.9  # Slightly less impact than QB
                if multiplier < 0.9:
                    note = f"Weather: {weather.get('conditions')}"
            elif pos == 'RB':
                multiplier = impact.get('rushing', 1.0)
                if multiplier > 1.05:
                    note = f"Weather boost: {weather.get('conditions')}"
            elif pos == 'DST':
                multiplier = impact.get('defense', 1.0)
                if multiplier > 1.05:
                    note = f"Weather advantage: {weather.get('conditions')}"
            
            # Apply the multiplier
            adjusted_df.loc[idx, 'PROJ PTS'] *= multiplier
            adjusted_df.loc[idx, 'WEATHER_IMPACT'] = multiplier
            adjusted_df.loc[idx, 'WEATHER_NOTE'] = note
            
            if multiplier != 1.0:
                logger.info(
                    f"Weather adjustment for {player['PLAYER NAME']}: "
                    f"{multiplier:.2f}x ({note})"
                )
        
        return adjusted_df
    
    async def get_all_game_weather(self, games: List[Dict]) -> Dict[str, Dict]:
        """Fetch weather for all games."""
        
        weather_data = {}
        tasks = []
        
        for game in games:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            game_time = game.get('game_time', datetime.now())
            
            if home_team:
                task = self.fetch_weather_for_game(home_team, game_time)
                tasks.append((home_team, away_team, task))
        
        # Fetch all weather data concurrently
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
        
        for (home_team, away_team, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Weather fetch failed for {home_team}: {result}")
                continue
            
            if result:
                # Apply weather to both teams
                weather_data[home_team] = result
                weather_data[away_team] = result  # Away team plays in same conditions
        
        return weather_data
