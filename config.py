"""
Enhanced configuration settings for on-demand DFS optimization
FIXED: Better environment variable loading and validation
"""
import os
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# ENHANCED environment loading with validation
def load_environment_variables():
    """Load and validate environment variables with better error handling"""
    env_loaded = False

    try:
        from dotenv import load_dotenv

        # Try multiple .env file locations
        project_dir = Path(__file__).parent
        env_files = [
            project_dir / ".env",
            project_dir / ".env.local",
            Path.home() / "fanduel" / ".env"
        ]

        for env_file in env_files:
            if env_file.exists():
                load_dotenv(env_file, override=True)
                print(f"✅ Loaded environment from: {env_file}")
                env_loaded = True
                break

        if not env_loaded:
            print(f"⚠️ No .env file found in: {[str(f) for f in env_files]}")

    except ImportError:
        print("⚠️ python-dotenv not installed, using system environment variables only")

    # Validate critical API keys
    openai_key = os.getenv('OPENAI_API_KEY', '')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')

    if openai_key and len(openai_key) > 20:
        print(f"✅ OpenAI API key loaded (starts with: {openai_key[:20]}...)")
    else:
        print("❌ OpenAI API key missing or invalid")

    if anthropic_key and len(anthropic_key) > 20:
        print(f"✅ Anthropic API key loaded (starts with: {anthropic_key[:20]}...)")
    else:
        print("❌ Anthropic API key missing or invalid")

    return env_loaded

# Load environment variables immediately
load_environment_variables()

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
CONTEST_TYPES = ["gpp", "cash", "contrarian", "bestball", "friends_league", "h2h"]

# AI Configuration with validation
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
    'espn_api': 60,
    'weather_api': 30,
    'odds_api': 500
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

# Head-to-Head Single Game Settings
H2H_ROSTER_SIZE = 6
H2H_MVP_MULTIPLIER = 1.5
H2H_SALARY_CAP = 60000

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
    },
    'h2h': {
        'variance_weight': 0.35,
        'ownership_threshold': 20.0,
        'correlation_bonus': 1.5,
        'mvp_multiplier': 1.5,
        'roster_size': 6,
        'salary_cap': 60000
    }
}

# NFL Defensive Rankings (Week 6 2025 - Update weekly)
# Based on DVOA, yards/game, and points allowed
DEFENSIVE_RANKINGS = {
    'pass_defense': {
        'top_5': ['SF', 'BAL', 'PIT', 'BUF', 'NYJ'],        # Elite pass D (penalty for WR/TE/QB)
        'bottom_5': ['LAC', 'WAS', 'NO', 'NYG', 'CAR']      # Weak pass D (boost for WR/TE/QB)
    },
    'run_defense': {
        'top_5': ['BAL', 'SF', 'CLE', 'DET', 'PHI'],        # Elite run D (penalty for RB)
        'bottom_5': ['CAR', 'NYG', 'TEN', 'IND', 'DEN']     # Weak run D (boost for RB)
    }
}

# Matchup adjustment factors
MATCHUP_ADJUSTMENTS = {
    'elite_matchup': 1.10,      # +10% vs bottom-5 defense
    'good_matchup': 1.05,       # +5% vs below-average defense
    'poor_matchup': 0.95,       # -5% vs above-average defense  
    'terrible_matchup': 0.90    # -10% vs top-5 defense
}

# Print loaded config summary on import
print(f"Config loaded - AI: {AI_ENABLED}, OpenAI: {'✅' if OPENAI_API_KEY else '❌'}, Anthropic: {'✅' if ANTHROPIC_API_KEY else '❌'}")
print("✅ Enhanced config loaded with improved environment handling")
print("✅ Defensive rankings loaded for matchup adjustments")
