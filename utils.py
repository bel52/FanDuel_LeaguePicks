import time
import json
import hashlib
from functools import wraps, lru_cache
from typing import Any, Callable
import logging
import asyncio

logger = logging.getLogger(__name__)

def exponential_backoff(max_retries: int = 3):
    """Decorator for exponential backoff retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            return None
        return wrapper
    return decorator

def cache_result(ttl: int = 300):
    """Cache function results with TTL"""
    def decorator(func):
        cache = {}
        cache_time = {}
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = hashlib.md5(
                f"{func.__name__}:{str(args)}:{str(kwargs)}".encode()
            ).hexdigest()
            
            # Check cache
            if key in cache and time.time() - cache_time[key] < ttl:
                return cache[key]
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Update cache
            cache[key] = result
            cache_time[key] = time.time()
            
            return result
        return wrapper
    return decorator

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a given text (simplified without tiktoken)"""
    # Rough approximation: 1 token ≈ 4 characters for English text
    # This is less accurate but avoids the dependency
    return len(text) // 4

def estimate_cost(prompt: str, completion: str, model: str = "gpt-4o-mini") -> float:
    """Estimate API cost"""
    input_tokens = count_tokens(prompt, model)
    output_tokens = count_tokens(completion, model)
    
    pricing = {
        "gpt-4o-mini": {"input": 0.15/1000000, "output": 0.60/1000000},
        "gpt-4o": {"input": 5.00/1000000, "output": 15.00/1000000},
        "claude-3-sonnet": {"input": 3.00/1000000, "output": 15.00/1000000}
    }
    
    if model in pricing:
        cost = (input_tokens * pricing[model]["input"] + 
                output_tokens * pricing[model]["output"])
        return cost
    return 0.0

def format_currency(amount: float) -> str:
    """Format as currency"""
    return f"${amount:,.2f}"

def calculate_value(projected_points: float, salary: int) -> float:
    """Calculate player value (points per $1000)"""
    if salary <= 0:
        return 0.0
    return (projected_points / salary) * 1000

# Add a timedelta import if needed
from datetime import timedelta
