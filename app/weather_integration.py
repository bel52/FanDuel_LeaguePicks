"""
Weather integration for DFS optimization.
Implements multi-source weather data collection with fallback strategies.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class WeatherConditions:
    temperature: float
    wind_speed: float
    precipitation: float
    humidity: float
    visibility: float
    condition: str
    forecast_confidence: float

@dataclass
class StadiumInfo:
    name: str
    lat: float
    lon: float
    is_dome: bool
    team: str

# NFL Stadium coordinates for weather analysis
NFL_STADIUMS = {
    'ARI': StadiumInfo('State Farm Stadium', 33.5276, -112.2626, True, 'Arizona Cardinals'),
    'ATL': StadiumInfo('Mercedes-Benz Stadium', 33.7555, -84.4006, True, 'Atlanta Falcons'),
    'BAL': StadiumInfo('M&T Bank Stadium', 39.2780, -76.6227, False, 'Baltimore Ravens'),
    'BUF': StadiumInfo('Highmark Stadium', 42.7738, -78.7870, False, 'Buffalo Bills'),
    'CAR': StadiumInfo('Bank of America Stadium', 35.2258, -80.8530, False, 'Carolina Panthers'),
    'CHI': StadiumInfo('Soldier Field', 41.8623, -87.6167, False, 'Chicago Bears'),
    'CIN': StadiumInfo('Paycor Stadium', 39.0955, -84.5160, False, 'Cincinnati Bengals'),
    'CLE': StadiumInfo('Cleveland Browns Stadium', 41.5061, -81.6995, False, 'Cleveland Browns'),
    'DAL': StadiumInfo('AT&T Stadium', 32.7473, -97.0945, True, 'Dallas Cowboys'),
    'DEN': StadiumInfo('Empower Field at Mile High', 39.7439, -105.0201, False, 'Denver Broncos'),
    'DET': StadiumInfo('Ford Field', 42.3400, -83.0456, True, 'Detroit Lions'),
    'GB': StadiumInfo('Lambeau Field', 44.5013, -88.0622, False, 'Green Bay Packers'),
    'HOU': StadiumInfo('NRG Stadium', 29.6847, -95.4107, True, 'Houston Texans'),
    'IND': StadiumInfo('Lucas Oil Stadium', 39.7601, -86.1639, True, 'Indianapolis Colts'),
    'JAX': StadiumInfo('TIAA Bank Field', 30.3240, -81.6374, False, 'Jacksonville Jaguars'),
    'KC': StadiumInfo('Arrowhead Stadium', 39.0489, -94.4839, False, 'Kansas City Chiefs'),
    'LV': StadiumInfo('Allegiant Stadium', 36.0909, -115.1834, True, 'Las Vegas Raiders'),
    'LAC': StadiumInfo('SoFi Stadium', 33.9535, -118.3392, True, 'Los Angeles Chargers'),
    'LAR': StadiumInfo('SoFi Stadium', 33.9535, -118.3392, True, 'Los Angeles Rams'),
    'MIA': StadiumInfo('Hard Rock Stadium', 25.9580, -80.2389, False, 'Miami Dolphins'),
    'MIN': StadiumInfo('U.S. Bank Stadium', 44.9738, -93.2581, True, 'Minnesota Vikings'),
    'NE': StadiumInfo('Gillette Stadium', 42.0909, -71.2643, False, 'New England Patriots'),
    'NO': StadiumInfo('Caesars Superdome', 29.9511, -90.0812, True, 'New Orleans Saints'),
    'NYG': StadiumInfo('MetLife Stadium', 40.8135, -74.0745, False, 'New York Giants'),
    'NYJ': StadiumInfo('MetLife Stadium', 40.8135, -74.0745, False, 'New York Jets'),
    'PHI': StadiumInfo('Lincoln Financial Field', 39.9008, -75.1675, False, 'Philadelphia Eagles'),
    'PIT': StadiumInfo('Heinz Field', 40.4468, -80.0158, False, 'Pittsburgh Steelers'),
    'SF': StadiumInfo("Levi's Stadium", 37.4032, -121.9700, False, 'San Francisco 49ers'),
    'SEA': StadiumInfo('Lumen Field', 47.5952, -122.3316, False, 'Seattle Seahawks'),
    'TB': StadiumInfo('Raymond James Stadium', 27.9759, -82.5033, False, 'Tampa Bay Buccaneers'),
    'TEN': StadiumInfo('Nissan Stadium', 36.1665, -86.7714, False, 'Tennessee Titans'),
    'WAS': StadiumInfo('FedExField', 38.9076, -76.8645, False, 'Washington Commanders'),
}

class WeatherDataCollector:
    """Multi-source weather data collector with intelligent fallback."""
    
    def __init__(self):
        self.session = None
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "DFS Weather Bot (contact@example.com)"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_game_weather(self, home_team: str, game_time: datetime) -> Optional[WeatherConditions]:
        """Get weather conditions for a specific game."""
        if home_team not in NFL_STADIUMS:
            logger.warning(f"Unknown team: {home_team}")
            return None
            
        stadium = NFL_STADIUMS[home_team]
        
        # Dome games don't need weather analysis
        if stadium.is_dome:
            return WeatherConditions(
                temperature=72.0,
                wind_speed=0.0,
                precipitation=0.0,
                humidity=50.0,
                visibility=10.0,
                condition="Indoor",
                forecast_confidence=1.0
            )
        
        cache_key = f"{home_team}_{game_time.strftime('%Y%m%d%H')}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
        
        # Try multiple weather sources
        weather_data = await self._fetch_weather_with_fallback(stadium, game_time)
        
        if weather_data:
            self.cache[cache_key] = (weather_data, datetime.now())
            
        return weather_data
    
    async def _fetch_weather_with_fallback(self, stadium: StadiumInfo, game_time: datetime) -> Optional[WeatherConditions]:
        """Fetch weather with multiple fallback sources."""
        sources = [
            self._fetch_weather_gov,
            self._fetch_open_meteo,
            self._fetch_weatherapi_com
        ]
        
        for source_func in sources:
            try:
                weather = await source_func(stadium, game_time)
                if weather:
                    logger.info(f"Weather data from {source_func.__name__} for {stadium.team}")
                    return weather
            except Exception as e:
                logger.warning(f"Weather source {source_func.__name__} failed: {e}")
                continue
        
        logger.error(f"All weather sources failed for {stadium.team}")
        return None
    
    async def _fetch_weather_gov(self, stadium: StadiumInfo, game_time: datetime) -> Optional[WeatherConditions]:
        """Fetch from Weather.gov (National Weather Service)."""
        try:
            # Get grid info
            points_url = f"https://api.weather.gov/points/{stadium.lat},{stadium.lon}"
            async with self.session.get(points_url) as response:
                if response.status != 200:
                    return None
                points_data = await response.json()
            
            # Get forecast
            forecast_url = points_data['properties']['forecastHourly']
            async with self.session.get(forecast_url) as response:
                if response.status != 200:
                    return None
                forecast_data = await response.json()
            
            # Find closest forecast period to game time
            periods = forecast_data['properties']['periods']
            closest_period = min(periods, 
                               key=lambda p: abs(datetime.fromisoformat(p['startTime'].replace('Z', '+00:00')) - game_time))
            
            return WeatherConditions(
                temperature=closest_period['temperature'],
                wind_speed=self._parse_wind_speed(closest_period.get('windSpeed', '0 mph')),
                precipitation=0.0,  # Weather.gov doesn't provide precipitation probability in simple format
                humidity=closest_period.get('relativeHumidity', {}).get('value', 50),
                visibility=10.0,  # Default visibility
                condition=closest_period['shortForecast'],
                forecast_confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"Weather.gov API error: {e}")
            return None
    
    async def _fetch_open_meteo(self, stadium: StadiumInfo, game_time: datetime) -> Optional[WeatherConditions]:
        """Fetch from Open-Meteo (free weather API)."""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': stadium.lat,
                'longitude': stadium.lon,
                'hourly': 'temperature_2m,windspeed_10m,precipitation,relativehumidity_2m,visibility',
                'timezone': 'America/New_York',
                'forecast_days': 7
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return None
                data = await response.json()
            
            hourly = data['hourly']
            times = [datetime.fromisoformat(t) for t in hourly['time']]
            
            # Find closest time to game
            closest_idx = min(range(len(times)), 
                            key=lambda i: abs(times[i] - game_time.replace(tzinfo=None)))
            
            return WeatherConditions(
                temperature=hourly['temperature_2m'][closest_idx],
                wind_speed=hourly['windspeed_10m'][closest_idx],
                precipitation=hourly['precipitation'][closest_idx],
                humidity=hourly['relativehumidity_2m'][closest_idx],
                visibility=hourly['visibility'][closest_idx] / 1000.0,  # Convert m to km
                condition="Variable",
                forecast_confidence=0.75
            )
            
        except Exception as e:
            logger.error(f"Open-Meteo API error: {e}")
            return None
    
    async def _fetch_weatherapi_com(self, stadium: StadiumInfo, game_time: datetime) -> Optional[WeatherConditions]:
        """Fetch from WeatherAPI.com (backup source)."""
        # This would require an API key for the free tier
        # Implementation left as placeholder for premium weather data
        return None
    
    def _parse_wind_speed(self, wind_str: str) -> float:
        """Parse wind speed from various formats."""
        try:
            # Extract numbers from string like "10 mph" or "15 to 20 mph"
            import re
            numbers = re.findall(r'\d+', wind_str)
            if numbers:
                return float(numbers[0])  # Take first number
            return 0.0
        except:
            return 0.0

class WeatherImpactCalculator:
    """Calculate DFS impact of weather conditions."""
    
    @staticmethod
    def calculate_weather_adjustment(weather: WeatherConditions, position: str) -> float:
        """
        Calculate projection adjustment factor based on weather.
        Returns multiplier (1.0 = no change, 0.85 = 15% reduction)
        """
        if weather.condition == "Indoor":
            return 1.0
        
        adjustment = 1.0
        
        # Wind impact (primarily affects passing)
        if weather.wind_speed >= 15 and position in ['QB', 'WR', 'TE']:
            adjustment *= 0.85  # 15% reduction for high winds
        elif weather.wind_speed >= 20 and position in ['QB', 'WR', 'TE']:
            adjustment *= 0.75  # 25% reduction for very high winds
        elif weather.wind_speed >= 10 and position == 'K':
            adjustment *= 0.90  # 10% reduction for kickers
        
        # Temperature impact
        if weather.temperature <= 32:
            adjustment *= 0.95  # 5% reduction for freezing
        elif weather.temperature <= 20:
            adjustment *= 0.90  # 10% reduction for very cold
        
        # Precipitation impact
        if weather.precipitation > 0.1:
            if position in ['QB', 'WR', 'TE']:
                adjustment *= 0.93  # 7% reduction for passing in rain
            elif position == 'RB':
                adjustment *= 0.97  # 3% reduction for running
            elif position == 'K':
                adjustment *= 0.85  # 15% reduction for kickers
        
        # Visibility impact (fog, heavy rain/snow)
        if weather.visibility < 5.0:
            adjustment *= 0.90  # 10% reduction for poor visibility
        
        return adjustment
    
    @staticmethod
    def get_weather_narrative(weather: WeatherConditions) -> str:
        """Generate human-readable weather impact summary."""
        if weather.condition == "Indoor":
            return "Indoor game - no weather impact"
        
        impacts = []
        
        if weather.wind_speed >= 15:
            impacts.append(f"High winds ({weather.wind_speed:.0f} mph) - passing game affected")
        
        if weather.temperature <= 32:
            impacts.append(f"Freezing temperatures ({weather.temperature:.0f}°F) - offensive production may decline")
        
        if weather.precipitation > 0.1:
            impacts.append(f"Precipitation expected - ball handling and passing accuracy affected")
        
        if weather.visibility < 5.0:
            impacts.append(f"Poor visibility ({weather.visibility:.1f} km) - overall offensive impact")
        
        if not impacts:
            return f"Good weather conditions - minimal impact expected"
        
        return "; ".join(impacts)

# Example usage and testing
async def test_weather_system():
    """Test the weather data collection system."""
    async with WeatherDataCollector() as collector:
        # Test a few outdoor stadiums
        test_teams = ['GB', 'CHI', 'BUF', 'DEN']  # Cold weather teams
        game_time = datetime.now() + timedelta(days=3)  # Upcoming Sunday
        
        for team in test_teams:
            weather = await collector.get_game_weather(team, game_time)
            if weather:
                stadium = NFL_STADIUMS[team]
                print(f"\n{stadium.team} ({stadium.name}):")
                print(f"  Temperature: {weather.temperature}°F")
                print(f"  Wind Speed: {weather.wind_speed} mph")
                print(f"  Condition: {weather.condition}")
                
                # Test impact calculations
                qb_adj = WeatherImpactCalculator.calculate_weather_adjustment(weather, 'QB')
                wr_adj = WeatherImpactCalculator.calculate_weather_adjustment(weather, 'WR')
                rb_adj = WeatherImpactCalculator.calculate_weather_adjustment(weather, 'RB')
                
                print(f"  Impact - QB: {qb_adj:.3f}, WR: {wr_adj:.3f}, RB: {rb_adj:.3f}")
                print(f"  Narrative: {WeatherImpactCalculator.get_weather_narrative(weather)}")
            else:
                print(f"Failed to get weather for {team}")

if __name__ == "__main__":
    asyncio.run(test_weather_system())
