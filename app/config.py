import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Base Directory ---
# This determines the root of our application inside the container, which is /app
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

# --- Core Directory Paths ---
# All other paths are built from these base paths
DATA_DIR = os.path.join(ROOT_DIR, 'data')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
INPUT_DIR = os.path.join(DATA_DIR, 'input') # This is the correct path for your CSVs

# --- API Keys & Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# --- DFS Settings ---
SALARY_CAP = 60000

# --- Ensure log directory exists on startup ---
os.makedirs(LOG_DIR, exist_ok=True)
