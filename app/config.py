import os

class Settings:
    """
    Minimal app settings used by the API.
    Reads a few env vars but defaults to your repo's data/ structure.
    """
    def __init__(self) -> None:
        # Data directories
        self.input_dir = os.getenv("INPUT_DIR", "data/input")
        self.output_dir = os.getenv("OUTPUT_DIR", "data/output")

        # Timezone (for display only; logic uses UTC)
        self.timezone = os.getenv("TIMEZONE", "America/Chicago")

        # Port the FastAPI app runs on (docker-compose maps host:8010->container:8010)
        self.port = int(os.getenv("PORT", "8010"))

settings = Settings()
