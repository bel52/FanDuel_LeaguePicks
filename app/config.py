from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TARGETS_DIR = DATA_DIR / "targets"

class Settings(BaseSettings):
    # AI keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")

    # Models / knobs
    gpt_model: str = Field(default="gpt-4o", env="GPT_MODEL")
    claude_model: str = Field(default="claude-3-5-sonnet-20240620", env="CLAUDE_MODEL")
    max_ai_calls_per_hour: int = Field(default=120, env="MAX_AI_CALLS_PER_HOUR")
    ai_cache_ttl: int = Field(default=1800, env="AI_CACHE_TTL")

    # Cache
    redis_url: Optional[str] = Field(default="redis://redis:6379/0", env="REDIS_URL")

    # Misc
    env: str = Field(default="production", env="ENV")

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Export a singletons-style handle
settings = get_settings()

# Ensure directories exist at runtime (safe if already exist)
for p in (INPUT_DIR, OUTPUT_DIR, TARGETS_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)
class Settings:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.gpt_model = os.getenv("GPT_MODEL", "gpt-4o")
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
        self.max_ai_calls_per_hour = int(os.getenv("MAX_AI_CALLS_PER_HOUR", "100"))
        self.ai_cache_ttl = int(os.getenv("AI_CACHE_TTL", "1800"))
        # fall back to REDIS_HOST/PORT if REDIS_URL is not set
        self.redis_url = os.getenv("REDIS_URL") or f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
        self.env = os.getenv("ENV", "production")
        self.data_dir = DATA_DIR
        self.input_dir = INPUT_DIR
        self.log_dir = LOG_DIR

settings = Settings()

