"""
Enhanced configuration settings with proper current week detection
"""
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
    
# Also create subdirectories
for subdir in ["lineups", "historical", "input", "output"]:
    (DATA_DIR / subdir).mkdir(exist_ok=True)

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

# Single Game format (MVP + 5 FLEX)
SINGLE_GAME_POSITIONS = {
    'MVP': 1,   # Any position, gets 1.5x points
    'FLEX': 5   # Any position
}

# Total roster size for FanDuel
ROSTER_SIZE = 9

# Enhanced data update intervals (in minutes)
UPDATE_INTERVALS = {
    'player_stats': 60,      # 1 hour
    'injury_reports': 15,    # 15 minutes (more frequent)
    'weather': 45,           # 45 minutes
    'vegas_lines': 30,       # 30 minutes
    'ownership_projections': 30,  # 30 minutes
    'current_week_games': 60,     # 1 hour
    'espn_scoreboard': 30    # 30 minutes
}

# Game day specific intervals (more frequent updates)
GAME_DAY_INTERVALS = {
    'player_stats': 15,      # 15 minutes on game days
    'injury_reports': 5,     # 5 minutes on game days
    'weather': 15,           # 15 minutes on game days
    'vegas_lines': 10,       # 10 minutes on game days
    'ownership_projections': 15,  # 15 minutes on game days
    'current_week_games': 30,     # 30 minutes on game days
    'espn_scoreboard': 10    # 10 minutes on game days
}

# ESPN API endpoints (free, no authentication required)
ESPN_ENDPOINTS = {
    'scoreboard': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
    'teams': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams',
    'news': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/news',
    'player_stats': 'https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/stats',
    'standings': 'https://site.api.espn.com/apis/v2/sports/football/nfl/standings',
    'week_calendar': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/calendar'
}

# Weather API (free government service)
WEATHER_API = {
    'base_url': 'https://api.weather.gov',
    'user_agent': 'DFS-Optimizer/2.1 (contact@example.com)'
}

# Enhanced NFL Stadium coordinates with indoor/outdoor designation
NFL_STADIUMS = {
    'ARI': {'lat': 33.5276, 'lon': -112.2626, 'name': 'State Farm Stadium', 'type': 'retractable_roof'},
    'ATL': {'lat': 33.7553, 'lon': -84.4006, 'name': 'Mercedes-Benz Stadium', 'type': 'dome'},
    'BAL': {'lat': 39.2780, 'lon': -76.6227, 'name': 'M&T Bank Stadium', 'type': 'outdoor'},
    'BUF': {'lat': 42.7738, 'lon': -78.7870, 'name': 'Highmark Stadium', 'type': 'outdoor'},
    'CAR': {'lat': 35.2258, 'lon': -80.8531, 'name': 'Bank of America Stadium', 'type': 'outdoor'},
    'CHI': {'lat': 41.8623, 'lon': -87.6167, 'name': 'Soldier Field', 'type': 'outdoor'},
    'CIN': {'lat': 39.0955, 'lon': -84.5161, 'name': 'Paycor Stadium', 'type': 'outdoor'},
    'CLE': {'lat': 41.5061, 'lon': -81.6995, 'name': 'Cleveland Browns Stadium', 'type': 'outdoor'},
    'DAL': {'lat': 32.7473, 'lon': -97.0945, 'name': 'AT&T Stadium', 'type': 'retractable_roof'},
    'DEN': {'lat': 39.7439, 'lon': -105.0201, 'name': 'Empower Field at Mile High', 'type': 'outdoor'},
    'DET': {'lat': 42.3400, 'lon': -83.0456, 'name': 'Ford Field', 'type': 'dome'},
    'GB': {'lat': 44.5013, 'lon': -88.0622, 'name': 'Lambeau Field', 'type': 'outdoor'},
    'HOU': {'lat': 29.6847, 'lon': -95.4107, 'name': 'NRG Stadium', 'type': 'retractable_roof'},
    'IND': {'lat': 39.7601, 'lon': -86.1639, 'name': 'Lucas Oil Stadium', 'type': 'retractable_roof'},
    'JAX': {'lat': 30.3240, 'lon': -81.6373, 'name': 'TIAA Bank Field', 'type': 'outdoor'},
    'KC': {'lat': 39.0489, 'lon': -94.4839, 'name': 'Arrowhead Stadium', 'type': 'outdoor'},
    'LV': {'lat': 36.0909, 'lon': -115.1833, 'name': 'Allegiant Stadium', 'type': 'dome'},
    'LAC': {'lat': 33.8642, 'lon': -118.2619, 'name': 'SoFi Stadium', 'type': 'dome'},
    'LAR': {'lat': 33.8642, 'lon': -118.2619, 'name': 'SoFi Stadium', 'type': 'dome'},
    'MIA': {'lat': 25.9580, 'lon': -80.2389, 'name': 'Hard Rock Stadium', 'type': 'outdoor'},
    'MIN': {'lat': 44.9738, 'lon': -93.2581, 'name': 'U.S. Bank Stadium', 'type': 'dome'},
    'NE': {'lat': 42.0909, 'lon': -71.2643, 'name': 'Gillette Stadium', 'type': 'outdoor'},
    'NO': {'lat': 29.9511, 'lon': -90.0812, 'name': 'Caesars Superdome', 'type': 'dome'},
    'NYG': {'lat': 40.8135, 'lon': -74.0745, 'name': 'MetLife Stadium', 'type': 'outdoor'},
    'NYJ': {'lat': 40.8135, 'lon': -74.0745, 'name': 'MetLife Stadium', 'type': 'outdoor'},
    'PHI': {'lat': 39.9008, 'lon': -75.1675, 'name': 'Lincoln Financial Field', 'type': 'outdoor'},
    'PIT': {'lat': 40.4468, 'lon': -80.0158, 'name': 'Acrisure Stadium', 'type': 'outdoor'},
    'SF': {'lat': 37.4030, 'lon': -121.9697, 'name': "Levi's Stadium", 'type': 'outdoor'},
    'SEA': {'lat': 47.5952, 'lon': -122.3316, 'name': 'Lumen Field', 'type': 'outdoor'},
    'TB': {'lat': 27.9759, 'lon': -82.5034, 'name': 'Raymond James Stadium', 'type': 'outdoor'},
    'TEN': {'lat': 36.1665, 'lon': -86.7713, 'name': 'Nissan Stadium', 'type': 'outdoor'},
    'WAS': {'lat': 38.9076, 'lon': -76.8645, 'name': 'Northwest Stadium', 'type': 'outdoor'}
}

# Weather impact settings - only apply to outdoor/retractable roof stadiums
WEATHER_IMPACT_ENABLED = {
    stadium: info['type'] in ['outdoor', 'retractable_roof'] 
    for stadium, info in NFL_STADIUMS.items()
}

# Enhanced optimization settings with contest-specific parameters
OPTIMIZATION_CONFIG = {
    'max_lineups': 150,
    'correlation_threshold': 0.6,
    'ownership_threshold': 30.0,  # Avoid players with >30% ownership in tournaments
    'weather_impact_threshold': 15,  # Wind speed threshold for adjustments
    'monte_carlo_simulations': 10000,
    
    # Contest-specific settings
    'gpp': {
        'max_ownership': 35.0,
        'min_correlation': 0.4,
        'variance_weight': 0.3,
        'stacking_required': True,
        'bring_back_enabled': True
    },
    'cash': {
        'max_ownership': 40.0,
        'min_correlation': 0.2,
        'variance_weight': -0.1,  # Penalize variance for cash games
        'stacking_required': False,
        'bring_back_enabled': False
    },
    'contrarian': {
        'max_ownership': 15.0,
        'min_correlation': 0.5,
        'variance_weight': 0.4,
        'stacking_required': True,
        'bring_back_enabled': True
    },
    'single_game': {
        'max_ownership': 25.0,
        'min_correlation': 0.6,
        'variance_weight': 0.2,
        'stacking_required': True,
        'bring_back_enabled': False
    }
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
    'injury_reports': 900,         # 15 minutes
    'vegas_lines': 1800,           # 30 minutes
    'ownership_projections': 1800,  # 30 minutes
    'optimize_lineup': 300,         # 5 minutes
    'current_week_games': 3600,     # 1 hour
    'espn_scoreboard': 1800        # 30 minutes
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

# NFL Season Configuration
CURRENT_NFL_SEASON = 2024
NFL_SEASON_START = datetime(2024, 9, 5)  # First Thursday night game
NFL_REGULAR_SEASON_WEEKS = 18
NFL_PLAYOFF_WEEKS = 4

def get_current_nfl_week() -> int:
    """Calculate current NFL week based on date"""
    now = datetime.now()
    
    if now < NFL_SEASON_START:
        return 1
    
    days_since_start = (now - NFL_SEASON_START).days
    current_week = min(NFL_REGULAR_SEASON_WEEKS, max(1, (days_since_start // 7) + 1))
    
    return current_week

def is_game_day() -> bool:
    """Check if today is an NFL game day"""
    current_day = datetime.now().weekday()
    # NFL games typically on Thursday (3), Sunday (6), Monday (0)
    return current_day in [0, 3, 6]

def get_update_interval(data_type: str) -> int:
    """Get appropriate update interval based on whether it's a game day"""
    if is_game_day():
        return GAME_DAY_INTERVALS.get(data_type, UPDATE_INTERVALS.get(data_type, 60))
    else:
        return UPDATE_INTERVALS.get(data_type, 60)

# Team name mappings for consistency
NFL_TEAM_MAPPING = {
    # Handle common variations
    'WSH': 'WAS', 'WFT': 'WAS',  # Washington
    'JAC': 'JAX',  # Jacksonville
    'LAS': 'LV',   # Las Vegas
    'SD': 'LAC',   # Los Angeles Chargers
    'STL': 'LAR'   # Los Angeles Rams
}

def normalize_team_name(team: str) -> str:
    """Normalize team name to standard abbreviation"""
    if not team:
        return 'UNK'
    
    team = team.upper().strip()
    return NFL_TEAM_MAPPING.get(team, team)

# Contest type validation
VALID_CONTEST_TYPES = ['gpp', 'cash', 'contrarian', 'single_game']

def validate_contest_type(contest_type: str) -> bool:
    """Validate contest type"""
    return contest_type.lower() in VALID_CONTEST_TYPES

# Development vs Production settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    # Production-specific overrides
    LOGGING_CONFIG['level'] = 'WARNING'
    # Less frequent updates in production to be respectful of APIs
    for key in UPDATE_INTERVALS:
        UPDATE_INTERVALS[key] = int(UPDATE_INTERVALS[key] * 1.5)
        
    # More conservative rate limits
    for api in RATE_LIMITS:
        RATE_LIMITS[api]['calls'] = int(RATE_LIMITS[api]['calls'] * 0.8)

# Ensure critical environment variables
API_PORT = int(os.getenv('API_PORT', API_PORT))
ENVIRONMENT = os.getenv('ENVIRONMENT', ENVIRONMENT)

# Path configurations
LINEUP_EXPORT_DIR = DATA_DIR / "lineups"
HISTORICAL_DATA_DIR = DATA_DIR / "historical"
LOG_FILE_PATH = LOGS_DIR / "dfs_optimizer.log"

# Ensure export directories exist
LINEUP_EXPORT_DIR.mkdir(exist_ok=True)
HISTORICAL_DATA_DIR.mkdir(exist_ok=True)

# Backward compatibility alias
H2H_POSITIONS = SINGLE_GAME_POSITIONS
