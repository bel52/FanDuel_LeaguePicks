import os
from typing import Optional
from pydantic_settings import BaseSettings  # Changed this line - was "from pydantic import BaseSettings"
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    news_api_key: Optional[str] = os.getenv("NEWS_API_KEY", None)
    weather_api_key: Optional[str] = os.getenv("WEATHER_API_KEY", None)
    
    # AI Configuration
    ai_provider: str = os.getenv("AI_PROVIDER", "openai")
    gpt_model: str = os.getenv("GPT_MODEL", "gpt-4o-mini")
    use_ai_analysis: bool = os.getenv("USE_AI_ANALYSIS", "true").lower() == "true"
    ai_cache_ttl: int = int(os.getenv("AI_CACHE_TTL", "1800"))
    
    # Data Sources
    use_nfl_data_py: bool = os.getenv("USE_NFL_DATA_PY", "true").lower() == "true"
    use_espn_hidden_apis: bool = os.getenv("USE_ESPN_HIDDEN_APIS", "true").lower() == "true"
    use_weather_gov: bool = os.getenv("USE_WEATHER_GOV", "true").lower() == "true"
    use_reddit_monitoring: bool = os.getenv("USE_REDDIT_MONITORING", "true").lower() == "true"
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))
    
    # DFS Settings
    salary_cap: int = int(os.getenv("SALARY_CAP", "60000"))
    min_value_threshold: float = float(os.getenv("MIN_VALUE_THRESHOLD", "1.8"))
    min_value_threshold_te_dst: float = float(os.getenv("MIN_VALUE_THRESHOLD_TE_DST", "1.7"))
    
    # Real-time Monitoring
    news_check_interval: int = int(os.getenv("NEWS_CHECK_INTERVAL", "300"))
    weather_check_interval: int = int(os.getenv("WEATHER_CHECK_INTERVAL", "3600"))
    injury_check_interval: int = int(os.getenv("INJURY_CHECK_INTERVAL", "600"))
    
    # Auto-Swap Settings
    auto_swap_enabled: bool = os.getenv("AUTO_SWAP_ENABLED", "true").lower() == "true"
    max_swaps_per_day: int = int(os.getenv("MAX_SWAPS_PER_DAY", "3"))
    
    # Strategy Settings
    hth_strategy_weight: float = float(os.getenv("HTH_STRATEGY_WEIGHT", "0.3"))
    league_strategy_weight: float = float(os.getenv("LEAGUE_STRATEGY_WEIGHT", "0.7"))
    
    # Rate Limiting
    max_api_calls_per_minute: int = int(os.getenv("MAX_API_CALLS_PER_MINUTE", "60"))
    max_ai_calls_per_hour: int = int(os.getenv("MAX_AI_CALLS_PER_HOUR", "100"))
    
    # Paths
    data_dir: str = "data"
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    fantasypros_dir: str = "data/fantasypros"
    
    # Timezone
    timezone: str = os.getenv("TIMEZONE", "America/New_York")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_log_files: int = int(os.getenv("MAX_LOG_FILES", "7"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
