"""
Configuration settings for the DFS optimization system
"""
import os
from pathlib import Path
from typing import List, Dict

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Database configuration
DATABASE_URL = f"sqlite:///{BASE_DIR}/dfs_data.db"

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8020

# DFS Platform Settings
FANDUEL_SALARY_CAP = 60000
FANDUEL_POSITIONS = {
    'QB': 1,
    'RB': 2, 
    'WR': 3,
    'TE': 1,
    'FLEX': 1,  # RB/WR/TE
    'DST': 1
}

# H2H specific format (MVP + 5 FLEX)
H2H_POSITIONS = {
    'MVP': 1,   # Any position, gets 1.5x points
    'FLEX': 5   # Any position
}

# Total roster size for FanDuel
ROSTER_SIZE = 9

# Data update intervals (in minutes)
UPDATE_INTERVALS = {
    'player_stats': 60,      # 1 hour
    'injury_reports': 30,    # 30 minutes
    'weather': 60,           # 1 hour
    'vegas_lines': 30,       # 30 minutes
    'ownership_projections': 45  # 45 minutes
}

# ESPN API endpoints (free, no authentication required)
ESPN_ENDPOINTS = {
    'scoreboard': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
    'teams': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams',
    'news': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/news',
    'player_stats': 'https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/stats',
    'standings': 'https://site.api.espn.com/apis/v2/sports/football/nfl/standings'
}

# Weather API (free government service)
WEATHER_API = {
    'base_url': 'https://api.weather.gov',
    'user_agent': 'DFS-Optimizer/1.0 (for personal use only)'
}

# NFL Stadium coordinates for weather
NFL_STADIUMS = {
    'ARI': {'lat': 33.5276, 'lon': -112.2626, 'name': 'State Farm Stadium'},
    'ATL': {'lat': 33.7553, 'lon': -84.4006, 'name': 'Mercedes-Benz Stadium'},
    'BAL': {'lat': 39.2780, 'lon': -76.6227, 'name': 'M&T Bank Stadium'},
    'BUF': {'lat': 42.7738, 'lon': -78.7870, 'name': 'Highmark Stadium'},
    'CAR': {'lat': 35.2258, 'lon': -80.8531, 'name': 'Bank of America Stadium'},
    'CHI': {'lat': 41.8623, 'lon': -87.6167, 'name': 'Soldier Field'},
    'CIN': {'lat': 39.0955, 'lon': -84.5161, 'name': 'Paycor Stadium'},
    'CLE': {'lat': 41.5061, 'lon': -81.6995, 'name': 'Cleveland Browns Stadium'},
    'DAL': {'lat': 32.7473, 'lon': -97.0945, 'name': 'AT&T Stadium'},
    'DEN': {'lat': 39.7439, 'lon': -105.0201, 'name': 'Empower Field at Mile High'},
    'DET': {'lat': 42.3400, 'lon': -83.0456, 'name': 'Ford Field'},
    'GB': {'lat': 44.5013, 'lon': -88.0622, 'name': 'Lambeau Field'},
    'HOU': {'lat': 29.6847, 'lon': -95.4107, 'name': 'NRG Stadium'},
    'IND': {'lat': 39.7601, 'lon': -86.1639, 'name': 'Lucas Oil Stadium'},
    'JAX': {'lat': 39.9061, 'lon': -81.6995, 'name': 'TIAA Bank Field'},
    'KC': {'lat': 39.0489, 'lon': -94.4839, 'name': 'Arrowhead Stadium'},
    'LV': {'lat': 36.0909, 'lon': -115.1833, 'name': 'Allegiant Stadium'},
    'LAC': {'lat': 33.8642, 'lon': -118.2619, 'name': 'SoFi Stadium'},
    'LAR': {'lat': 33.8642, 'lon': -118.2619, 'name': 'SoFi Stadium'},
    'MIA': {'lat': 25.9580, 'lon': -80.2389, 'name': 'Hard Rock Stadium'},
    'MIN': {'lat': 44.9738, 'lon': -93.2581, 'name': 'U.S. Bank Stadium'},
    'NE': {'lat': 42.0909, 'lon': -71.2643, 'name': 'Gillette Stadium'},
    'NO': {'lat': 29.9511, 'lon': -90.0812, 'name': 'Caesars Superdome'},
    'NYG': {'lat': 40.8135, 'lon': -74.0745, 'name': 'MetLife Stadium'},
    'NYJ': {'lat': 40.8135, 'lon': -74.0745, 'name': 'MetLife Stadium'},
    'PHI': {'lat': 39.9008, 'lon': -75.1675, 'name': 'Lincoln Financial Field'},
    'PIT': {'lat': 40.4468, 'lon': -80.0158, 'name': 'Heinz Field'},
    'SF': {'lat': 37.4030, 'lon': -121.9697, 'name': "Levi's Stadium"},
    'SEA': {'lat': 47.5952, 'lon': -122.3316, 'name': 'Lumen Field'},
    'TB': {'lat': 27.9759, 'lon': -82.5034, 'name': 'Raymond James Stadium'},
    'TEN': {'lat': 36.1665, 'lon': -86.7713, 'name': 'Nissan Stadium'},
    'WAS': {'lat': 38.9076, 'lon': -76.8645, 'name': 'FedExField'}
}

# Optimization settings
OPTIMIZATION_CONFIG = {
    'max_lineups': 150,
    'correlation_threshold': 0.6,
    'ownership_threshold': 30.0,  # Avoid players with >30% ownership
    'weather_impact_threshold': 15,  # Wind speed threshold for adjustments
    'monte_carlo_simulations': 10000
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}',
    'rotation': '100 MB',
    'retention': '7 days'
}

# Rate limiting for external APIs
RATE_LIMITS = {
    'espn_api': {'calls': 100, 'period': 60},  # 100 calls per minute
    'weather_gov': {'calls': 60, 'period': 60},  # 60 calls per minute
    'nfl_data_py': {'calls': 30, 'period': 60}   # 30 calls per minute
}

# Cache expiration times (in seconds)
CACHE_TTL = {
    'player_projections': 3600,    # 1 hour
    'weather_data': 3600,          # 1 hour  
    'injury_reports': 1800,        # 30 minutes
    'vegas_lines': 1800,           # 30 minutes
    'ownership_projections': 2700,  # 45 minutes
    'optimize_lineup': 300         # 5 minutes
}

# AI Integration (optional - for advanced analysis)
AI_CONFIG = {
    'enabled': False,  # Set to True if you want AI analysis
    'model': 'gpt-4o-mini',  # Cheapest option
    'max_tokens': 1000,
    'temperature': 0.1
}

# Data validation thresholds
VALIDATION_THRESHOLDS = {
    'min_projection': 0.0,
    'max_projection': 50.0,
    'min_salary': 3000,
    'max_salary': 15000,
    'required_fields': ['player_name', 'position', 'team', 'salary', 'projection']
}

# Development vs Production settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    # Production-specific overrides
    LOGGING_CONFIG['level'] = 'WARNING'
    UPDATE_INTERVALS = {k: v * 2 for k, v in UPDATE_INTERVALS.items()}  # Less frequent updates
