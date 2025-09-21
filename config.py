"""
Configuration management for DFS Optimizer
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Load environment variables
load_dotenv()

class Config:
    """Central configuration management"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///dfs_optimizer.db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # FanDuel Credentials (for automated login if needed)
    FANDUEL_USERNAME = os.getenv('FANDUEL_USERNAME')
    FANDUEL_PASSWORD = os.getenv('FANDUEL_PASSWORD')
    
    # Update Intervals
    UPDATE_INTERVAL_MINUTES = int(os.getenv('UPDATE_INTERVAL_MINUTES', 30))
    WEATHER_UPDATE_HOURS = 4
    INJURY_UPDATE_MINUTES = 15
    NEWS_UPDATE_MINUTES = 10
    
    # Data Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    CACHE_DIR = DATA_DIR / 'cache'
    LOGS_DIR = BASE_DIR / 'logs'
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, CACHE_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # API Endpoints
    ESPN_BASE_URL = "https://site.api.espn.com"
    SLEEPER_BASE_URL = "https://api.sleeper.app/v1"
    WEATHER_BASE_URL = "https://api.weather.gov"
    
    # NFL Stadium Coordinates for weather
    NFL_STADIUMS = {
        'ARI': {'lat': 33.5276, 'lon': -112.2626, 'dome': True},
        'ATL': {'lat': 33.7553, 'lon': -84.4006, 'dome': True},
        'BAL': {'lat': 39.2780, 'lon': -76.6227, 'dome': False},
        'BUF': {'lat': 42.7738, 'lon': -78.7870, 'dome': False},
        'CAR': {'lat': 35.2258, 'lon': -80.8528, 'dome': False},
        'CHI': {'lat': 41.8623, 'lon': -87.6167, 'dome': False},
        'CIN': {'lat': 39.0954, 'lon': -84.5160, 'dome': False},
        'CLE': {'lat': 41.5061, 'lon': -81.6995, 'dome': False},
        'DAL': {'lat': 32.7473, 'lon': -97.0945, 'dome': True},
        'DEN': {'lat': 39.7439, 'lon': -105.0201, 'dome': False},
        'DET': {'lat': 42.3400, 'lon': -83.0456, 'dome': True},
        'GB': {'lat': 44.5013, 'lon': -88.0622, 'dome': False},
        'HOU': {'lat': 29.6847, 'lon': -95.4107, 'dome': True},
        'IND': {'lat': 39.7601, 'lon': -86.1639, 'dome': True},
        'JAX': {'lat': 30.3239, 'lon': -81.6373, 'dome': False},
        'KC': {'lat': 39.0489, 'lon': -94.4839, 'dome': False},
        'LV': {'lat': 36.0909, 'lon': -115.1833, 'dome': True},
        'LAC': {'lat': 33.9535, 'lon': -118.3390, 'dome': False},
        'LAR': {'lat': 33.9535, 'lon': -118.3390, 'dome': False},
        'MIA': {'lat': 25.9580, 'lon': -80.2389, 'dome': False},
        'MIN': {'lat': 44.9736, 'lon': -93.2575, 'dome': True},
        'NE': {'lat': 42.0909, 'lon': -71.2643, 'dome': False},
        'NO': {'lat': 29.9511, 'lon': -90.0812, 'dome': True},
        'NYG': {'lat': 40.8135, 'lon': -74.0745, 'dome': False},
        'NYJ': {'lat': 40.8135, 'lon': -74.0745, 'dome': False},
        'PHI': {'lat': 39.9008, 'lon': -75.1675, 'dome': False},
        'PIT': {'lat': 40.4468, 'lon': -80.0158, 'dome': False},
        'SF': {'lat': 37.4033, 'lon': -121.9694, 'dome': False},
        'SEA': {'lat': 47.5952, 'lon': -122.3316, 'dome': False},
        'TB': {'lat': 27.9759, 'lon': -82.5033, 'dome': False},
        'TEN': {'lat': 36.1665, 'lon': -86.7713, 'dome': False},
        'WAS': {'lat': 38.9076, 'lon': -76.8645, 'dome': False}
    }
    
    # Optimization Settings
    SALARY_CAP = 60000  # FanDuel salary cap
    MAX_LINEUPS = 150
    MIN_PROJECTION_CONFIDENCE = 0.7
    
    # Position Requirements for FanDuel
    POSITION_REQUIREMENTS = {
        'QB': 1,
        'RB': 2,
        'WR': 3,
        'TE': 1,
        'FLEX': 1,  # RB/WR/TE
        'DST': 1
    }
    
    # AI Settings
    AI_MODEL = "gpt-4o-mini"  # Cost-effective option
    MAX_TOKENS = 1500
    TEMPERATURE = 0.3  # Lower for more consistent analysis
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}'

# Initialize configuration
config = Config()

# Setup logging
from loguru import logger
logger.add(
    config.LOGS_DIR / f"dfs_optimizer_{datetime.now():%Y%m%d}.log",
    format=config.LOG_FORMAT,
    level=config.LOG_LEVEL,
    rotation="1 day",
    retention="7 days"
)
