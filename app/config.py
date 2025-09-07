import os

class Settings:
    """App settings with environment variable defaults"""
    def __init__(self) -> None:
        # Data directories
        self.input_dir = os.getenv("INPUT_DIR", "data/input")
        self.output_dir = os.getenv("OUTPUT_DIR", "data/output")
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        
        # Timezone and other settings
        self.timezone = os.getenv("TIMEZONE", "America/Chicago")
        self.port = int(os.getenv("PORT", "8010"))

settings = Settings()

# Constants
SALARY_CAP = 60000
