import redis
import json
import logging
from typing import Optional, Dict, Any

# This will correctly import from the config.py file in the same directory
from .config import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

class StateManager:
    """Handles saving and loading application state to Redis."""
    def __init__(self):
        self.redis_client = None
        try:
            self.redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
            self.redis_client.ping()
            logger.info("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to Redis: {e}")

    def save_lineup(self, lineup_data: Dict[str, Any], game_mode: str) -> bool:
        """Saves the generated lineup state to Redis."""
        if not self.redis_client:
            return False
        
        key = f"lineup:{game_mode}"
        try:
            self.redis_client.set(key, json.dumps(lineup_data))
            logger.info(f"Successfully saved '{game_mode}' lineup to state manager.")
            return True
        except Exception as e:
            logger.error(f"Failed to save lineup to Redis: {e}")
            return False

    def load_lineup(self, game_mode: str) -> Optional[Dict[str, Any]]:
        """Loads a lineup state from Redis."""
        if not self.redis_client:
            return None
            
        key = f"lineup:{game_mode}"
        try:
            lineup_json = self.redis_client.get(key)
            if lineup_json:
                return json.loads(lineup_json)
            return None
        except Exception as e:
            logger.error(f"Failed to load lineup from Redis: {e}")
            return None

# Create a global instance for easy access
state_manager = StateManager()
