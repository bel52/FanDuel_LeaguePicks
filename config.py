import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///dfs_optimizer.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # FanDuel
    FANDUEL_USERNAME = os.getenv("FANDUEL_USERNAME")
    FANDUEL_PASSWORD = os.getenv("FANDUEL_PASSWORD")
    
    # Settings
    UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 300))
    OPTIMIZATION_THREADS = int(os.getenv("OPTIMIZATION_THREADS", 4))
    CACHE_TTL = int(os.getenv("CACHE_TTL", 600))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # NFL Constants
    SALARY_CAP = 60000  # FanDuel salary cap
    POSITIONS = {
        'QB': {'min': 1, 'max': 1},
        'RB': {'min': 2, 'max': 3},
        'WR': {'min': 3, 'max': 4},
        'TE': {'min': 1, 'max': 2},
        'DST': {'min': 1, 'max': 1}
    }
    
    # Correlation coefficients (from research)
    CORRELATIONS = {
        'QB-WR': 0.62,
        'QB-TE': 0.32,
        'QB-RB': 0.08,
        'QB-DST': -0.41,
        'RB-DST': 0.25
    }
    
    # Weather thresholds
    WEATHER_IMPACT = {
        'wind_threshold': 15,  # mph
        'temp_threshold': 32,   # fahrenheit
        'precip_threshold': 0.1  # inches
    }
    
    # NFL Stadium coordinates
    NFL_STADIUMS = {
        'ARI': {'lat': 33.5276, 'lon': -112.2626},
        'ATL': {'lat': 33.7553, 'lon': -84.4006},
        'BAL': {'lat': 39.2780, 'lon': -76.6227},
        'BUF': {'lat': 42.7738, 'lon': -78.7870},
        'CAR': {'lat': 35.2258, 'lon': -80.8528},
        'CHI': {'lat': 41.8623, 'lon': -87.6167},
        'CIN': {'lat': 39.0954, 'lon': -84.5160},
        'CLE': {'lat': 41.5061, 'lon': -81.6995},
        'DAL': {'lat': 32.7473, 'lon': -97.0945},
        'DEN': {'lat': 39.7439, 'lon': -105.0201},
        'DET': {'lat': 42.3400, 'lon': -83.0456},
        'GB': {'lat': 44.5013, 'lon': -88.0622},
        'HOU': {'lat': 29.6847, 'lon': -95.4107},
        'IND': {'lat': 39.7601, 'lon': -86.1639},
        'JAX': {'lat': 30.3239, 'lon': -81.6373},
        'KC': {'lat': 39.0489, 'lon': -94.4839},
        'LV': {'lat': 36.0909, 'lon': -115.1833},
        'LAC': {'lat': 33.8643, 'lon': -118.2611},
        'LAR': {'lat': 33.9535, 'lon': -118.3392},
        'MIA': {'lat': 25.9580, 'lon': -80.2389},
        'MIN': {'lat': 44.9736, 'lon': -93.2575},
        'NE': {'lat': 42.0909, 'lon': -71.2643},
        'NO': {'lat': 29.9511, 'lon': -90.0812},
        'NYG': {'lat': 40.8135, 'lon': -74.0745},
        'NYJ': {'lat': 40.8135, 'lon': -74.0745},
        'PHI': {'lat': 39.9012, 'lon': -75.1675},
        'PIT': {'lat': 40.4468, 'lon': -80.0158},
        'SF': {'lat': 37.7133, 'lon': -122.3861},
        'SEA': {'lat': 47.5952, 'lon': -122.3316},
        'TB': {'lat': 27.9759, 'lon': -82.5033},
        'TEN': {'lat': 36.1665, 'lon': -86.7713},
        'WAS': {'lat': 38.9076, 'lon': -76.8645}
    }

config = Config()
