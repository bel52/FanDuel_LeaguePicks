import asyncio
import json
import logging
from typing import Any, Optional
import redis.asyncio as aioredis
from datetime import timedelta
import os

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages Redis caching for the DFS optimizer"""
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.default_ttl = int(os.getenv("CACHE_TTL", "300"))
        self._redis = None
        self._lock = asyncio.Lock()
    
    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection"""
        async with self._lock:
            if self._redis is None:
                try:
                    self._redis = await aioredis.from_url(
                        self.redis_url,
                        encoding="utf-8",
                        decode_responses=True
                    )
                    await self._redis.ping()
                    logger.info("Redis connection established")
                except Exception as e:
                    logger.error(f"Failed to connect to Redis: {e}")
                    # Return a dummy cache that doesn't persist
                    return DummyCache()
            return self._redis
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            redis = await self._get_redis()
            if isinstance(redis, DummyCache):
                return redis.get(key)
            
            value = await redis.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        try:
            redis = await self._get_redis()
            if isinstance(redis, DummyCache):
                return redis.set(key, value)
            
            ttl = ttl or self.default_ttl
            
            if not isinstance(value, str):
                value = json.dumps(value)
            
            await redis.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            redis = await self._get_redis()
            if isinstance(redis, DummyCache):
                return redis.delete(key)
            
            await redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis = await self._get_redis()
            if isinstance(redis, DummyCache):
                return redis.exists(key)
            
            return await redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error for {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on existing key"""
        try:
            redis = await self._get_redis()
            if isinstance(redis, DummyCache):
                return True  # Dummy cache doesn't support expiration
            
            await redis.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Cache expire error for {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        try:
            redis = await self._get_redis()
            if isinstance(redis, DummyCache):
                return 0
            
            keys = await redis.keys(pattern)
            if keys:
                return await redis.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._redis = None

class DummyCache:
    """Fallback in-memory cache when Redis is unavailable"""
    
    def __init__(self):
        self.data = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self.data.get(key)
    
    def set(self, key: str, value: Any) -> bool:
        self.data[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        if key in self.data:
            del self.data[key]
        return True
    
    def exists(self, key: str) -> bool:
        return key in self.data
