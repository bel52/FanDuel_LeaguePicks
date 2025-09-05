import os
from typing import Optional
from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # API Keys
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", None)
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # DFS Settings
    salary_cap: int = int(os.getenv("SALARY_CAP", "60000"))
    min_value_threshold: float = float(os.getenv("MIN_VALUE_THRESHOLD", "1.8"))
    min_value_threshold_te_dst: float = float(os.getenv("MIN_VALUE_THRESHOLD_TE_DST", "1.7"))
    
    # AI Settings
    use_ai_analysis: bool = os.getenv("USE_AI_ANALYSIS", "true").lower() == "true"
    ai_provider: str = os.getenv("AI_PROVIDER", "openai")
    
    # Paths
    data_dir: str = "data"
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    
    # Timezone
    timezone: str = os.getenv("TIMEZONE", "America/New_York")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
