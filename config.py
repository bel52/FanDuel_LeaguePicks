"""
Simplified configuration settings for on-demand DFS optimization
Removed all scheduling-related config
"""
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment variables only")

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
PLATFORM = "fanduel"
CONTEST_TYPES = ["gpp", "cash", "contrarian", "bestball"]

# AI Configuration
AI_ENABLED = os.getenv('AI_ENABLED', 'true').lower() == 'true'
AI_WEEKLY_BUDGET = float(os.getenv('AI_WEEKLY_BUDGET', '15.0'))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GPT_MODEL = os.getenv('GPT_MODEL', 'gpt-4o-mini')

# ESPN API Endpoints
ESPN_ENDPOINTS = {
    'scores': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
    'teams': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams',
    'schedule': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
}

# NFL Stadium Information
NFL_STADIUMS = {
    'ARI': {'name': 'State Farm Stadium', 'dome': True, 'city': 'Glendale', 'state': 'AZ'},
    'ATL': {'name': 'Mercedes-Benz Stadium', 'dome': True, 'city': 'Atlanta', 'state': 'GA'},
    'BAL': {'name': 'M&T Bank Stadium', 'dome': False, 'city': 'Baltimore', 'state': 'MD'},
    'BUF': {'name': 'Highmark Stadium', 'dome': False, 'city': 'Buffalo', 'state': 'NY'},
    'CAR': {'name': 'Bank of America Stadium', 'dome': False, 'city': 'Charlotte', 'state': 'NC'},
    'CHI': {'name': 'Soldier Field', 'dome': False, 'city': 'Chicago', 'state': 'IL'},
    'CIN': {'name': 'Paycor Stadium', 'dome': False, 'city': 'Cincinnati', 'state': 'OH'},
    'CLE': {'name': 'Cleveland Browns Stadium', 'dome': False, 'city': 'Cleveland', 'state': 'OH'},
    'DAL': {'name': 'AT&T Stadium', 'dome': True, 'city': 'Arlington', 'state': 'TX'},
    'DEN': {'name': 'Empower Field at Mile High', 'dome': False, 'city': 'Denver', 'state': 'CO'},
    'DET': {'name': 'Ford Field', 'dome': True, 'city': 'Detroit', 'state': 'MI'},
    'GB': {'name': 'Lambeau Field', 'dome': False, 'city': 'Green Bay', 'state': 'WI'},
    'HOU': {'name': 'NRG Stadium', 'dome': True, 'city': 'Houston', 'state': 'TX'},
    'IND': {'name': 'Lucas Oil Stadium', 'dome': True, 'city': 'Indianapolis', 'state': 'IN'},
    'JAX': {'name': 'TIAA Bank Field', 'dome': False, 'city': 'Jacksonville', 'state': 'FL'},
    'KC': {'name': 'Arrowhead Stadium', 'dome': False, 'city': 'Kansas City', 'state': 'MO'},
    'LV': {'name': 'Allegiant Stadium', 'dome': True, 'city': 'Las Vegas', 'state': 'NV'},
    'LAC': {'name': 'SoFi Stadium', 'dome': True, 'city': 'Los Angeles', 'state': 'CA'},
    'LAR': {'name': 'SoFi Stadium', 'dome': True, 'city': 'Los Angeles', 'state': 'CA'},
    'MIA': {'name': 'Hard Rock Stadium', 'dome': False, 'city': 'Miami Gardens', 'state': 'FL'},
    'MIN': {'name': 'U.S. Bank Stadium', 'dome': True, 'city': 'Minneapolis', 'state': 'MN'},
    'NE': {'name': 'Gillette Stadium', 'dome': False, 'city': 'Foxborough', 'state': 'MA'},
    'NO': {'name': 'Caesars Superdome', 'dome': True, 'city': 'New Orleans', 'state': 'LA'},
    'NYG': {'name': 'MetLife Stadium', 'dome': False, 'city': 'East Rutherford', 'state': 'NJ'},
    'NYJ': {'name': 'MetLife Stadium', 'dome': False, 'city': 'East Rutherford', 'state': 'NJ'},
    'PHI': {'name': 'Lincoln Financial Field', 'dome': False, 'city': 'Philadelphia', 'state': 'PA'},
    'PIT': {'name': 'Heinz Field', 'dome': False, 'city': 'Pittsburgh', 'state': 'PA'},
    'SF': {'name': "Levi's Stadium", 'dome': False, 'city': 'Santa Clara', 'state': 'CA'},
    'SEA': {'name': 'Lumen Field', 'dome': False, 'city': 'Seattle', 'state': 'WA'},
    'TB': {'name': 'Raymond James Stadium', 'dome': False, 'city': 'Tampa', 'state': 'FL'},
    'TEN': {'name': 'Nissan Stadium', 'dome': False, 'city': 'Nashville', 'state': 'TN'},
    'WAS': {'name': 'FedExField', 'dome': False, 'city': 'Landover', 'state': 'MD'}
}

# Weather API Configuration
WEATHER_API = {
    'base_url': 'https://api.weather.gov',
    'user_agent': 'DFS-Optimizer/1.0'
}

# Rate Limits
RATE_LIMITS = {
    'espn_api': 60,  # requests per minute
    'weather_api': 30,  # requests per minute
    'odds_api': 500  # requests per day
}

# Validation Thresholds
VALIDATION_THRESHOLDS = {
    'min_salary': 3000,
    'max_salary': 15000,
    'min_projection': 0,
    'max_projection': 50,
    'min_ownership': 0,
    'max_ownership': 100
}

# Data source settings
USE_NFL_DATA_PY = os.getenv('USE_NFL_DATA_PY', 'true').lower() == 'true'
USE_ESPN_HIDDEN_APIS = os.getenv('USE_ESPN_HIDDEN_APIS', 'true').lower() == 'true'
USE_WEATHER_GOV = os.getenv('USE_WEATHER_GOV', 'true').lower() == 'true'

# Contest strategy weights
GPP_VARIANCE_WEIGHT = float(os.getenv('GPP_VARIANCE_WEIGHT', '0.3'))
GPP_OWNERSHIP_THRESHOLD = float(os.getenv('GPP_OWNERSHIP_THRESHOLD', '25.0'))
CASH_VARIANCE_WEIGHT = float(os.getenv('CASH_VARIANCE_WEIGHT', '-0.1'))
CASH_OWNERSHIP_THRESHOLD = float(os.getenv('CASH_OWNERSHIP_THRESHOLD', '40.0'))
CONTRARIAN_VARIANCE_WEIGHT = float(os.getenv('CONTRARIAN_VARIANCE_WEIGHT', '0.4'))
CONTRARIAN_OWNERSHIP_THRESHOLD = float(os.getenv('CONTRARIAN_OWNERSHIP_THRESHOLD', '15.0'))

# Season long strategy
SEASON_LONG_STRATEGY = os.getenv('SEASON_LONG_STRATEGY', 'true').lower() == 'true'
WEEKLY_FLOOR_PRIORITY = float(os.getenv('WEEKLY_FLOOR_PRIORITY', '0.7'))
WEEKLY_CEILING_PRIORITY = float(os.getenv('WEEKLY_CEILING_PRIORITY', '0.3'))
CONSISTENCY_BONUS = float(os.getenv('CONSISTENCY_BONUS', '0.2'))

# Odds API
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Cache settings
CACHE_TTL = int(os.getenv('CACHE_TTL', '600'))

# FanDuel specific settings
SALARY_CAP = 60000
ROSTER_SIZE = 9
POSITION_LIMITS = {
    'QB': {'min': 1, 'max': 1},
    'RB': {'min': 2, 'max': 3},
    'WR': {'min': 3, 'max': 4},
    'TE': {'min': 1, 'max': 2},
    'FLEX': {'min': 1, 'max': 1},
    'D': {'min': 1, 'max': 1}
}

# Weather impact thresholds
WIND_SPEED_THRESHOLD = 15
PRECIPITATION_THRESHOLD = 0.1
COLD_WEATHER_THRESHOLD = 32

# Vegas odds settings
HIGH_TOTAL_THRESHOLD = 47.0
BLOWOUT_SPREAD_THRESHOLD = 14.0

# FanDuel specific constants for optimizer
FANDUEL_POSITIONS = {
    'QB': {'min': 1, 'max': 1},
    'RB': {'min': 2, 'max': 3},
    'WR': {'min': 3, 'max': 4},
    'TE': {'min': 1, 'max': 2},
    'FLEX': {'min': 1, 'max': 1},
    'D': {'min': 1, 'max': 1}
}

FANDUEL_SALARY_CAP = 60000

OPTIMIZATION_CONFIG = {
    'gpp': {
        'variance_weight': 0.3,
        'ownership_threshold': 25.0,
        'correlation_bonus': 1.2
    },
    'cash': {
        'variance_weight': -0.1,
        'ownership_threshold': 40.0,
        'correlation_bonus': 1.0
    },
    'contrarian': {
        'variance_weight': 0.4,
        'ownership_threshold': 15.0,
        'correlation_bonus': 1.3
    }
}

# Print loaded config summary on import
print(
    f"Config loaded - AI: {AI_ENABLED}, OpenAI: {'✅' if OPENAI_API_KEY else '❌'}, Anthropic: {'✅' if ANTHROPIC_API_KEY else '❌'}")
print("✅ Simplified config loaded - no scheduling, on-demand only")