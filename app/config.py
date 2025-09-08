"""
Configuration management for DFS optimization system.
Provides both 'settings' and 'Config' exports for compatibility.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Settings:
    """Application settings with proper dataclass defaults."""
    
    # API Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Database Configuration  
    database_url: str = "sqlite:///./app/data/dfs.db"
    redis_url: str = "redis://localhost:6379"
    
    # DFS Configuration
    default_salary_cap: int = 50000
    optimization_timeout: int = 30
    max_lineups_per_request: int = 150
    
    # Position requirements for FanDuel NFL
    position_requirements: Dict[str, int] = field(default_factory=lambda: {
        'QB': 1,
        'RB': 2, 
        'WR': 3,
        'TE': 1,
        'K': 1,
        'DST': 1
    })
    
    # Maximum players per position for roster construction
    max_players_per_position: Dict[str, int] = field(default_factory=lambda: {
        'QB': 1,
        'RB': 3,
        'WR': 4, 
        'TE': 2,
        'K': 1,
        'DST': 1
    })
    
    # API Rate Limiting
    api_rate_limit: int = 100
    weather_rate_limit: int = 1000
    espn_rate_limit: int = 60
    
    # System Configuration
    debug: bool = True
    log_level: str = "INFO"
    max_workers: int = 4
    
    # Cache Configuration
    cache_ttl_minutes: int = 15
    player_cache_ttl: int = 300  # 5 minutes
    weather_cache_ttl: int = 3600  # 1 hour
    
    # Optimization Parameters
    correlation_threshold: float = 0.5
    diversification_factor: float = 0.1
    max_exposure_per_player: float = 0.3
    
    # Data Source URLs
    espn_base_url: str = "https://site.api.espn.com"
    sleeper_base_url: str = "https://api.sleeper.app/v1"
    weather_gov_url: str = "https://api.weather.gov"
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY', self.openai_api_key)
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', self.anthropic_api_key)
        
        # Database
        self.database_url = os.getenv('DATABASE_URL', self.database_url)
        self.redis_url = os.getenv('REDIS_URL', self.redis_url)
        
        # DFS Settings
        self.default_salary_cap = int(os.getenv('DEFAULT_SALARY_CAP', self.default_salary_cap))
        self.optimization_timeout = int(os.getenv('OPTIMIZATION_TIMEOUT', self.optimization_timeout))
        self.max_lineups_per_request = int(os.getenv('MAX_LINEUPS_PER_REQUEST', self.max_lineups_per_request))
        
        # Rate Limits
        self.api_rate_limit = int(os.getenv('API_RATE_LIMIT', self.api_rate_limit))
        self.weather_rate_limit = int(os.getenv('WEATHER_RATE_LIMIT', self.weather_rate_limit))
        
        # System
        self.debug = os.getenv('DEBUG', 'True').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', self.log_level)
        self.max_workers = int(os.getenv('MAX_WORKERS', self.max_workers))
        
        # Directory paths
        self.input_dir = os.getenv('INPUT_DIR', self.input_dir)
        self.output_dir = os.getenv('OUTPUT_DIR', self.output_dir)
        self.data_dir = os.getenv('DATA_DIR', self.data_dir)
        
        # Add missing attributes for compatibility
        self.salary_cap = self.default_salary_cap  # For optimization engine
        self.max_ai_calls_per_hour = int(os.getenv('MAX_AI_CALLS_PER_HOUR', '100'))  # For AI analyzer
        
        # Optimization engine specific settings (from grep results)
        self.cache_ttl = self.cache_ttl_minutes * 60  # Convert minutes to seconds
        self.enforce_qb_stack = os.getenv('ENFORCE_QB_STACK', 'true').lower() == 'true'
        self.min_stack_receivers = int(os.getenv('MIN_STACK_RECEIVERS', '1'))
        self.max_team_exposure = int(os.getenv('MAX_TEAM_EXPOSURE', '4'))
        self.min_value_threshold = float(os.getenv('MIN_VALUE_THRESHOLD', '2.5'))
        self.max_optimization_time = int(os.getenv('MAX_OPTIMIZATION_TIME', '60'))
        
        # Strategy weights method
    
    def get_strategy_weights(self, game_type='league'):
        """Get strategy weights for different game types."""
        if game_type == 'h2h':
            return {
                'ceiling_weight': float(os.getenv('H2H_CEILING_WEIGHT', '0.3')),
                'floor_weight': float(os.getenv('H2H_FLOOR_WEIGHT', '0.7')),
                'correlation_weight': float(os.getenv('H2H_CORRELATION_WEIGHT', '0.4'))
            }
        else:  # league/tournament
            return {
                'ceiling_weight': float(os.getenv('LEAGUE_CEILING_WEIGHT', '0.7')),
                'floor_weight': float(os.getenv('LEAGUE_FLOOR_WEIGHT', '0.3')),
                'correlation_weight': float(os.getenv('LEAGUE_CORRELATION_WEIGHT', '0.6'))
            }

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required API keys
        if not self.openai_api_key or self.openai_api_key == 'your_openai_api_key_here':
            issues.append("OpenAI API key not configured")
        
        # Validate salary cap
        if self.default_salary_cap < 10000 or self.default_salary_cap > 100000:
            issues.append(f"Invalid salary cap: {self.default_salary_cap}")
        
        # Validate timeouts
        if self.optimization_timeout < 5 or self.optimization_timeout > 300:
            issues.append(f"Invalid optimization timeout: {self.optimization_timeout}")
        
        # Validate position requirements
        required_positions = {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}
        if not all(pos in self.position_requirements for pos in required_positions):
            issues.append("Missing required positions in configuration")
        
        # Validate lineup size
        total_positions = sum(self.position_requirements.values())
        if total_positions != 9:  # FanDuel NFL requires 9 players
            issues.append(f"Invalid lineup size: {total_positions} (should be 9)")
        
        return issues

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key != 'your_openai_api_key_here')

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key and self.anthropic_api_key != 'your_anthropic_api_key_here')

# Create global instance
_settings_instance = None

def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

# Export for different import patterns
settings = get_settings()  # For: from app.config import settings
Config = get_settings       # For: from app.config import Config (function)

# Also export the settings instance directly as Config for compatibility
def Config():
    """Compatibility function that returns settings instance."""
    return get_settings()

# Position and team mappings for data processing
POSITION_MAPPINGS = {
    'QB': 'QB',
    'RB': 'RB', 
    'WR': 'WR',
    'TE': 'TE',
    'K': 'K',
    'DST': 'DST',
    'DEF': 'DST',  # Alternative defense notation
    'D/ST': 'DST'  # Yahoo format
}

TEAM_MAPPINGS = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons', 
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LV': 'Las Vegas Raiders',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders'
}

# Direct exports for compatibility with existing imports
SALARY_CAP = 50000  # Will be updated from settings

def _update_salary_cap():
    """Update SALARY_CAP from settings."""
    global SALARY_CAP
    SALARY_CAP = get_settings().default_salary_cap

# Update SALARY_CAP when module loads
_update_salary_cap()

# Export commonly used functions
def get_salary_cap() -> int:
    """Get default salary cap."""
    return get_settings().default_salary_cap

def get_position_requirements() -> Dict[str, int]:
    """Get position requirements for lineup construction."""
    return get_settings().position_requirements.copy()

def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return get_settings().debug

def validate_config() -> List[str]:
    """Validate current configuration."""
    return get_settings().validate()

# Configuration validation for startup
def ensure_valid_config():
    """Ensure configuration is valid, raise exception if not."""
    issues = validate_config()
    if issues:
        raise ValueError(f"Configuration validation failed: {'; '.join(issues)}")

if __name__ == "__main__":
    # Test configuration loading
    config = get_settings()
    print("Configuration loaded successfully!")
    print(f"Debug mode: {config.debug}")
    print(f"Salary cap: ${config.default_salary_cap:,}")
    print(f"OpenAI key configured: {config.has_openai_key}")
    
    # Test different import patterns
    print("\nTesting import compatibility:")
    print(f"settings.debug: {settings.debug}")
    print(f"Config().debug: {Config().debug}")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nConfiguration is valid!")
