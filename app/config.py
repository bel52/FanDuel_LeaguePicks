import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class Settings:
    """Centralized application settings"""
    
    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8010"))
    
    # Data directories
    data_dir: str = os.getenv("DATA_DIR", "data")
    input_dir: str = os.getenv("INPUT_DIR", "data/input")
    output_dir: str = os.getenv("OUTPUT_DIR", "data/output")
    targets_dir: str = os.getenv("TARGETS_DIR", "data/targets")
    logs_dir: str = os.getenv("LOGS_DIR", "logs")
    
    # API keys
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    odds_api_key: Optional[str] = os.getenv("ODDS_API_KEY")
    news_api_key: Optional[str] = os.getenv("NEWS_API_KEY")
    
    # AI settings
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    max_ai_calls_per_hour: int = int(os.getenv("MAX_AI_CALLS_PER_HOUR", "100"))
    ai_cache_ttl: int = int(os.getenv("AI_CACHE_TTL", "1800"))  # 30 minutes
    
    # Optimization settings
    salary_cap: int = int(os.getenv("SALARY_CAP", "60000"))
    min_value_threshold: float = float(os.getenv("MIN_VALUE_THRESHOLD", "2.5"))
    max_optimization_time: int = int(os.getenv("MAX_OPTIMIZATION_TIME", "60"))
    
    # Stacking rules
    enforce_qb_stack: bool = os.getenv("ENFORCE_QB_STACK", "true").lower() == "true"
    min_stack_receivers: int = int(os.getenv("MIN_STACK_RECEIVERS", "1"))
    max_team_exposure: int = int(os.getenv("MAX_TEAM_EXPOSURE", "4"))
    
    # Game type weights
    h2h_ceiling_weight: float = float(os.getenv("H2H_CEILING_WEIGHT", "0.7"))
    h2h_floor_weight: float = float(os.getenv("H2H_FLOOR_WEIGHT", "0.3"))
    league_ceiling_weight: float = float(os.getenv("LEAGUE_CEILING_WEIGHT", "0.5"))
    league_floor_weight: float = float(os.getenv("LEAGUE_FLOOR_WEIGHT", "0.5"))
    
    # Real-time monitoring
    enable_real_time_monitoring: bool = os.getenv("ENABLE_REAL_TIME_MONITORING", "true").lower() == "true"
    monitoring_interval: int = int(os.getenv("MONITORING_INTERVAL", "300"))  # 5 minutes
    weather_update_interval: int = int(os.getenv("WEATHER_UPDATE_INTERVAL", "3600"))  # 1 hour
    news_update_interval: int = int(os.getenv("NEWS_UPDATE_INTERVAL", "600"))  # 10 minutes
    
    # Auto-swap system
    enable_auto_swap: bool = os.getenv("ENABLE_AUTO_SWAP", "true").lower() == "true"
    max_swaps_per_day: int = int(os.getenv("MAX_SWAPS_PER_DAY", "3"))
    swap_severity_threshold: float = float(os.getenv("SWAP_SEVERITY_THRESHOLD", "0.6"))
    min_projection_change: float = float(os.getenv("MIN_PROJECTION_CHANGE", "2.0"))
    
    # Cache settings
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes
    enable_cache_warming: bool = os.getenv("ENABLE_CACHE_WARMING", "true").lower() == "true"
    
    # Performance settings
    max_concurrent_optimizations: int = int(os.getenv("MAX_CONCURRENT_OPTIMIZATIONS", "5"))
    optimization_timeout: int = int(os.getenv("OPTIMIZATION_TIMEOUT", "120"))  # 2 minutes
    
    # Weather settings
    weather_api_timeout: int = int(os.getenv("WEATHER_API_TIMEOUT", "10"))
    weather_impact_threshold: float = float(os.getenv("WEATHER_IMPACT_THRESHOLD", "0.1"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    enable_debug_logging: bool = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true"
    
    # Safety limits
    max_lineup_generation_attempts: int = int(os.getenv("MAX_LINEUP_GENERATION_ATTEMPTS", "1000"))
    max_players_per_position: Dict[str, int] = {
        "QB": 1,
        "RB": 3,  # 2 starters + 1 flex
        "WR": 4,  # 3 starters + 1 flex
        "TE": 2,  # 1 starter + 1 flex
        "DST": 1
    }
    
    # Position requirements (FanDuel format)
    position_requirements: Dict[str, Dict[str, int]] = {
        "QB": {"min": 1, "max": 1},
        "RB": {"min": 2, "max": 3},
        "WR": {"min": 3, "max": 4},
        "TE": {"min": 1, "max": 2},
        "DST": {"min": 1, "max": 1},
        "FLEX": {"min": 1, "max": 1}  # Can be RB, WR, or TE
    }
    
    # URLs and endpoints
    espn_api_base: str = "https://site.api.espn.com"
    weather_gov_api_base: str = "https://api.weather.gov"
    odds_api_base: str = "https://api.the-odds-api.com/v4"
    
    def __post_init__(self):
        """Validate settings after initialization"""
        # Create directories
        for directory in [self.data_dir, self.input_dir, self.output_dir, self.targets_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Validate weights
        if abs((self.h2h_ceiling_weight + self.h2h_floor_weight) - 1.0) > 0.01:
            raise ValueError("H2H weights must sum to 1.0")
        
        if abs((self.league_ceiling_weight + self.league_floor_weight) - 1.0) > 0.01:
            raise ValueError("League weights must sum to 1.0")
        
        # Validate thresholds
        if self.min_value_threshold <= 0:
            raise ValueError("Min value threshold must be positive")
        
        if self.swap_severity_threshold < 0 or self.swap_severity_threshold > 1:
            raise ValueError("Swap severity threshold must be between 0 and 1")
    
    @property
    def has_ai_enabled(self) -> bool:
        """Check if any AI service is configured"""
        return bool(self.openai_api_key or self.anthropic_api_key)
    
    @property
    def has_odds_api(self) -> bool:
        """Check if odds API is configured"""
        return bool(self.odds_api_key)
    
    @property
    def has_news_api(self) -> bool:
        """Check if news API is configured"""
        return bool(self.news_api_key)
    
    def get_strategy_weights(self, game_type: str) -> Dict[str, float]:
        """Get strategy weights for a game type"""
        if game_type.lower() == "h2h":
            return {
                "ceiling_weight": self.h2h_ceiling_weight,
                "floor_weight": self.h2h_floor_weight,
                "leverage_weight": 0.4,
                "correlation_weight": 0.6
            }
        else:  # league
            return {
                "ceiling_weight": self.league_ceiling_weight,
                "floor_weight": self.league_floor_weight,
                "leverage_weight": 0.3,
                "correlation_weight": 0.7
            }
    
    def get_required_files(self) -> list:
        """Get list of required CSV files"""
        return ["qb.csv", "rb.csv", "wr.csv", "te.csv", "dst.csv"]
    
    def get_lineup_rules(self) -> Dict[str, Any]:
        """Get lineup construction rules"""
        return {
            "salary_cap": self.salary_cap,
            "total_players": 9,
            "position_requirements": self.position_requirements,
            "enforce_stack": self.enforce_qb_stack,
            "min_stack_receivers": self.min_stack_receivers,
            "max_team_exposure": self.max_team_exposure,
            "min_value_threshold": self.min_value_threshold
        }

# Global settings instance
settings = Settings()

# Stadium coordinates for weather monitoring
NFL_STADIUMS = {
    'ARI': {'lat': 33.5276, 'lon': -112.2626, 'dome': True, 'name': 'State Farm Stadium'},
    'ATL': {'lat': 33.7553, 'lon': -84.4006, 'dome': True, 'name': 'Mercedes-Benz Stadium'},
    'BAL': {'lat': 39.2780, 'lon': -76.6227, 'dome': False, 'name': 'M&T Bank Stadium'},
    'BUF': {'lat': 42.7738, 'lon': -78.7870, 'dome': False, 'name': 'Highmark Stadium'},
    'CAR': {'lat': 35.2271, 'lon': -80.8526, 'dome': False, 'name': 'Bank of America Stadium'},
    'CHI': {'lat': 41.8623, 'lon': -87.6167, 'dome': False, 'name': 'Soldier Field'},
    'CIN': {'lat': 39.0955, 'lon': -84.5161, 'dome': False, 'name': 'Paycor Stadium'},
    'CLE': {'lat': 41.5061, 'lon': -81.6995, 'dome': False, 'name': 'Cleveland Browns Stadium'},
    'DAL': {'lat': 32.7473, 'lon': -97.0945, 'dome': True, 'name': 'AT&T Stadium'},
    'DEN': {'lat': 39.7439, 'lon': -105.0201, 'dome': False, 'name': 'Empower Field at Mile High'},
    'DET': {'lat': 42.3400, 'lon': -83.0456, 'dome': True, 'name': 'Ford Field'},
    'GB': {'lat': 44.5013, 'lon': -88.0622, 'dome': False, 'name': 'Lambeau Field'},
    'HOU': {'lat': 29.6847, 'lon': -95.4107, 'dome': True, 'name': 'NRG Stadium'},
    'IND': {'lat': 39.7601, 'lon': -86.1639, 'dome': True, 'name': 'Lucas Oil Stadium'},
    'JAX': {'lat': 30.3240, 'lon': -81.6373, 'dome': False, 'name': 'TIAA Bank Field'},
    'KC': {'lat': 39.0489, 'lon': -94.4839, 'dome': False, 'name': 'Arrowhead Stadium'},
    'LAC': {'lat': 33.8644, 'lon': -118.2610, 'dome': False, 'name': 'SoFi Stadium'},
    'LAR': {'lat': 34.0141, 'lon': -118.2879, 'dome': True, 'name': 'SoFi Stadium'},
    'LV': {'lat': 36.0909, 'lon': -115.1833, 'dome': True, 'name': 'Allegiant Stadium'},
    'MIA': {'lat': 25.9580, 'lon': -80.2389, 'dome': False, 'name': 'Hard Rock Stadium'},
    'MIN': {'lat': 44.9737, 'lon': -93.2581, 'dome': True, 'name': 'U.S. Bank Stadium'},
    'NE': {'lat': 42.0909, 'lon': -71.2643, 'dome': False, 'name': 'Gillette Stadium'},
    'NO': {'lat': 29.9511, 'lon': -90.0812, 'dome': True, 'name': 'Caesars Superdome'},
    'NYG': {'lat': 40.8135, 'lon': -74.0745, 'dome': False, 'name': 'MetLife Stadium'},
    'NYJ': {'lat': 40.8135, 'lon': -74.0745, 'dome': False, 'name': 'MetLife Stadium'},
    'PHI': {'lat': 39.9008, 'lon': -75.1675, 'dome': False, 'name': 'Lincoln Financial Field'},
    'PIT': {'lat': 40.4469, 'lon': -80.0158, 'dome': False, 'name': 'Acrisure Stadium'},
    'SF': {'lat': 37.4032, 'lon': -121.9698, 'dome': False, 'name': "Levi's Stadium"},
    'SEA': {'lat': 47.5952, 'lon': -122.3316, 'dome': False, 'name': 'Lumen Field'},
    'TB': {'lat': 27.9759, 'lon': -82.5034, 'dome': False, 'name': 'Raymond James Stadium'},
    'TEN': {'lat': 36.1665, 'lon': -86.7713, 'dome': False, 'name': 'Nissan Stadium'},
    'WAS': {'lat': 38.9077, 'lon': -76.8644, 'dome': False, 'name': 'FedExField'},
}

# Position mappings
POSITION_MAPPINGS = {
    'QUARTERBACK': 'QB',
    'RUNNING BACK': 'RB',
    'WIDE RECEIVER': 'WR',
    'TIGHT END': 'TE',
    'DEFENSE': 'DST',
    'D/ST': 'DST',
    'DEF': 'DST'
}

# Team name mappings for consistency
TEAM_MAPPINGS = {
    'JAC': 'JAX',
    'JAGS': 'JAX',
    'JAGUARS': 'JAX',
    'WFT': 'WAS',
    'WASHINGTON': 'WAS',
    'COMMANDERS': 'WAS',
    'REDSKINS': 'WAS',
    'CARDS': 'ARI',
    'CARDINALS': 'ARI',
    'NINERS': 'SF',
    '49ERS': 'SF',
    'PATS': 'NE',
    'PATRIOTS': 'NE',
    'BUCS': 'TB',
    'BUCCANEERS': 'TB',
    'PACK': 'GB',
    'PACKERS': 'GB'
}
