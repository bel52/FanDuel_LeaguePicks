# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional
import os

class Settings(BaseSettings):
    # App Configuration
    app_name: str = "FanDuel NFL DFS Optimizer"
    app_version: str = "3.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server Configuration  
    host: str = "0.0.0.0"
    port: int = 8000
    
    # DFS Configuration
    salary_cap: int = 60000
    min_salary: int = 59200
    max_lineups: int = 150
    default_randomness: float = 0.25
    default_salary_cap: int = 60000
    
    # AI Configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    openai_model: str = os.getenv("GPT_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
    max_ai_calls_per_hour: int = int(os.getenv("MAX_AI_CALLS_PER_HOUR", "100"))
    ai_cache_ttl: int = int(os.getenv("AI_CACHE_TTL", "1800"))
    
    # Cache Configuration
    redis_url: Optional[str] = os.getenv("REDIS_URL")
    cache_ttl: int = 3600
    
    # Data Sources
    odds_api_key: Optional[str] = os.getenv("ODDS_API_KEY")
    news_api_key: Optional[str] = os.getenv("NEWS_API_KEY")
    
    # Monitoring
    auto_monitoring: bool = os.getenv("AUTO_MONITORING", "false").lower() == "true"
    auto_swap_enabled: bool = os.getenv("AUTO_SWAP_ENABLED", "true").lower() == "true"
    max_swaps_per_day: int = int(os.getenv("MAX_SWAPS_PER_DAY", "3"))
    
    # CORS Configuration
    cors_origins: List[str] = ["*"]
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def has_openai_key(self) -> bool:
        return self.openai_api_key is not None and self.openai_api_key != "your_openai_api_key_here"
    
    @property
    def has_anthropic_key(self) -> bool:
        return self.anthropic_api_key is not None and self.anthropic_api_key != "your_anthropic_api_key_here"
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if not self.has_openai_key and not self.has_anthropic_key:
            issues.append("No AI API keys configured (OpenAI or Anthropic)")
        
        if self.salary_cap < 40000 or self.salary_cap > 100000:
            issues.append(f"Unusual salary cap: ${self.salary_cap}")
        
        if not self.odds_api_key:
            issues.append("No Odds API key configured (optional but recommended)")
        
        return issues
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Export settings instance
settings = get_settings()

# Position mappings for normalization
POSITION_MAPPINGS = {
    'QB': 'QB',
    'RB': 'RB', 
    'WR': 'WR',
    'TE': 'TE',
    'K': 'K',
    'DST': 'DST',
    'DEF': 'DST',
    'D/ST': 'DST'
}

# Team mappings for normalization
TEAM_MAPPINGS = {
    'JAC': 'JAX',
    'JAGS': 'JAX',
    'WFT': 'WAS',
    'WASHINGTON': 'WAS',
    'CARDS': 'ARI',
    'NINERS': 'SF',
    'PATS': 'NE',
    'BUCS': 'TB',
    'PACK': 'GB'
}

# DFS specific constants
SALARY_CAP = 60000
MIN_SALARY = 59200
POSITION_LIMITS = {
    'QB': (1, 1),
    'RB': (2, 3),
    'WR': (3, 4),
    'TE': (1, 2),
    'DST': (1, 1)
}

# Auto-swap thresholds
SWAP_THRESHOLDS = {
    'injury': 0.6,
    'weather': 0.4,
    'news': 0.5,
    'inactive': 1.0
}
