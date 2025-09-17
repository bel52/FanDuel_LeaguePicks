from pydantic import BaseModel
import os

class Settings(BaseModel):
    # External APIs (all optional; we auto-detect availability)
    ODDS_API_KEY: str | None = os.getenv("ODDS_API_KEY")

    # App/runtime
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PORT: int = int(os.getenv("PORT", "8010"))

    # Strategy knobs
    MAX_LINEUPS: int = int(os.getenv("MAX_LINEUPS", "20"))
    GAME_TYPE_DEFAULT: str = os.getenv("GAME_TYPE_DEFAULT", "league")  # 'league' or 'h2h'
    FLEX_POSITIONS: tuple[str, ...] = ("RB", "WR", "TE")

    # Locking / late-swap (foundation)
    LOCK_WINDOW_MINUTES: int = int(os.getenv("LOCK_WINDOW_MINUTES", "0"))  # 0 = lock at kickoff

settings = Settings()
