from __future__ import annotations
import asyncio, json, time, logging, os
from typing import Any, Optional

logger = logging.getLogger(__name__)

class _MemoryCache:
    def __init__(self):
        self._store: dict[str, tuple[float, Any]] = {}

    async def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at and time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        expires_at = time.time() + ttl if ttl else 0
        self._store[key] = (expires_at, value)

class CacheManager:
    """Use Redis if REDIS_URL provided, otherwise fall back to in-memory."""
    def __init__(self):
        self.backend = None
        self._init_done = False

    def _ensure_backend(self):
        if self._init_done:
            return
        redis_url = os.getenv("REDIS_URL") or "redis://redis:6379/0"
        try:
            # lazy import so we don't require redis if not used
            from redis import asyncio as aioredis
            self.backend = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            logger.info(f"CacheManager using Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"CacheManager falling back to memory cache: {e}")
            self.backend = _MemoryCache()
        self._init_done = True

    async def get(self, key: str) -> Optional[Any]:
        self._ensure_backend()
        if hasattr(self.backend, "get"):  # redis
            try:
                raw = await self.backend.get(key)
                return json.loads(raw) if raw else None
            except Exception as e:
                logger.warning(f"Redis get failed, fallback mem: {e}")
        # memory
        return await self.backend.get(key)

    async def set(self, key: str, value: Any, ttl: int = 0) -> None:
        self._ensure_backend()
        if hasattr(self.backend, "set"):  # redis
            try:
                raw = json.dumps(value)
                if ttl:
                    await self.backend.set(key, raw, ex=ttl)
                else:
                    await self.backend.set(key, raw)
                return
            except Exception as e:
                logger.warning(f"Redis set failed, fallback mem: {e}")
        # memory
        await self.backend.set(key, value, ttl=ttl)
