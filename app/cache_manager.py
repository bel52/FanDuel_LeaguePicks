import json
import logging
from typing import Any, Optional
import redis.asyncio as redis
from app.config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages Redis caching for API responses and data"""
    
    def __init__(self):
        self.redis = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Initialize Redis connection"""
        try:
            self.redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            self.connected = True
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.connected = False
    
    async def ping(self) -> bool:
        """Test Redis connection"""
        if not self.redis:
            return False
        try:
            await self.redis.ping()
            return True
        except:
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.connected:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value in cache with TTL"""
        if not self.connected:
            return False
        
        try:
            serialized = json.dumps(value)
            await self.redis.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.connected:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
