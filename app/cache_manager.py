import redis
import json
from app.config import settings

class CacheManager:
    """Multi-level cache using Redis for persistent caching."""
    def __init__(self):
        self.redis = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    async def get(self, key: str):
        val = self.redis.get(key)
        if val:
            try:
                return json.loads(val)
            except:
                return val

    async def set(self, key: str, value, ttl: int = 3600):
        data = json.dumps(value)
        if ttl:
            self.redis.setex(key, ttl, data)
        else:
            self.redis.set(key, data)
