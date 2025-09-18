from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

APP_PORT = int(os.getenv("APP_PORT", "8010"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SALARY_CAP = int(os.getenv("SALARY_CAP", "60000"))

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
OPENAI_TIMEOUT_SECS = int(os.getenv("OPENAI_TIMEOUT_SECS", "40"))

ODDS_API_REGION = os.getenv("ODDS_API_REGION", "us")
ODDS_API_MARKETS = os.getenv("ODDS_API_MARKETS", "h2h,totals,spreads")
ODDS_API_SPORT = os.getenv("ODDS_API_SPORT", "americanfootball_nfl")

BASELINE_WEEK = int(os.getenv("BASELINE_WEEK", "0"))
ODDS_IMPLIED_MULTIPLIER = float(os.getenv("ODDS_IMPLIED_MULTIPLIER", "0.08"))
WEATHER_WIND_PENALTY = float(os.getenv("WEATHER_WIND_PENALTY", "0.03"))
WEATHER_RAIN_PENALTY = float(os.getenv("WEATHER_RAIN_PENALTY", "0.04"))

DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
EXPORTS_DIR = DATA_DIR / "exports"
LOG_DIR = ROOT / "logs"
