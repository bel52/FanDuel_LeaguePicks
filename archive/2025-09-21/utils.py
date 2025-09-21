import json
import time
import logging
import requests
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_backoff(tries=3, base_delay=2, factor=2):
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, tries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == tries:
                        logger.error(f"{fn.__name__} failed after {tries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= factor
        return wrapper
    return deco

@retry_with_backoff(tries=3, base_delay=2, factor=2)
def http_get(url: str, headers: dict | None = None, timeout: int = 15):
    """HTTP GET that ALWAYS returns JSON-like python objects (dict/list).
       If the body isn't JSON, returns {'_raw': <text>} to keep .get() safe."""
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()

    # Try direct JSON
    try:
        data = resp.json()
        # If JSON is a string (e.g., "\"ok\""), try to parse again if it looks like nested JSON
        if isinstance(data, str):
            s = data.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                return json.loads(s)
            # Return as dict so callers using .get() won't crash
            return {"_raw": data}
        return data
    except ValueError:
        # Not JSON — return wrapped text
        text = resp.text
        s = text.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                pass
        return {"_raw": text}

def count_tokens(prompt: str) -> int:
    # Lightweight token approximation to avoid hard deps here
    return max(1, len(prompt.split()))

def estimate_cost(prompt: str, completion: str, model: str) -> float:
    # Simple cost estimator placeholder; adjust with your real rates if needed
    in_tok = count_tokens(prompt)
    out_tok = count_tokens(completion)
    rate_per_1k = 0.005  # example rate for mini-tier model
    return ((in_tok + out_tok) / 1000.0) * rate_per_1k

def format_currency(value) -> str:
    """Format currency cleanly for CLI output."""
    try:
        v = float(value)
    except Exception:
        return str(value)
    # show no decimals for whole dollars; two decimals otherwise
    return f"${v:,.0f}" if v.is_integer() else f"${v:,.2f}"
