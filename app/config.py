from __future__ import annotations
from functools import lru_cache
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # AI keys (read from env or .env)
    openai_api_key: str | None = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, env="ANTHROPIC_API_KEY")

    # Models / knobs
    gpt_model: str = Field(default="gpt-4o", env="GPT_MODEL")
    claude_model: str = Field(default="claude-3-5-sonnet-20240620", env="CLAUDE_MODEL")
    max_ai_calls_per_hour: int = Field(default=120, env="MAX_AI_CALLS_PER_HOUR")
    ai_cache_ttl: int = Field(default=1800, env="AI_CACHE_TTL")

    # Caching
    redis_url: str | None = Field(default=None, env="REDIS_URL")

    # Misc
    env: str = Field(default="production", env="ENV")

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()

# Singleton for easy import
settings = get_settings()
