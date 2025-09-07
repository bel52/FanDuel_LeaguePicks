# app/config.py
import os
from pathlib import Path

# Base directories
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TARGETS_DIR = DATA_DIR / "targets"
LOG_DIR = APP_ROOT / "logs"

# Create folders if they don't exist
for p in (DATA_DIR, INPUT_DIR, OUTPUT_DIR, TARGETS_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Defaults for DFS
SALARY_CAP = 60000
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

class Settings:
    """Centralised configuration using environment variables or sensible defaults."""
    def __init__(self):
        # AI keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        # Model names
        self.gpt_model = os.getenv("GPT_MODEL", "gpt-4o")
        self.claude_model = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")
        # Rate limiting and caching
        self.max_ai_calls_per_hour = int(os.getenv("MAX_AI_CALLS_PER_HOUR", "100"))
        self.ai_cache_ttl = int(os.getenv("AI_CACHE_TTL", "1800"))
        # Redis URL (override host/port if full URL provided)
        self.redis_url = os.getenv("REDIS_URL") or f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
        # Environment and paths
        self.env = os.getenv("ENV", "production")
        self.data_dir = str(DATA_DIR)
        self.input_dir = str(INPUT_DIR)
        self.log_dir = str(LOG_DIR)

settings = Settings()
