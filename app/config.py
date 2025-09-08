# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List, Optional

class Settings(BaseSettings):
    # App Configuration
    app_name: str = "FanDuel NFL DFS Optimizer"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # DFS Configuration
    salary_cap: int = 60000
    min_salary: int = 59200
    max_lineups: int = 150
    default_randomness: float = 0.25
    
    # Cache Configuration
    redis_url: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
