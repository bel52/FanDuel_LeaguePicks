import os
import json
import hashlib
import logging
import openai
import tiktoken

# Set API key if present (dotenv loaded in main)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Default low-cost model
MODEL_NAME = os.getenv('GPT_MODEL', 'gpt-4o-mini')

PRICING = {
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.60/1_000_000},
    # Optional other models can be added here
}

class TokenOptimizer:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    def estimate_cost(self, prompt: str, completion: str = ""):
        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(completion) if completion else 500
        pricing = PRICING.get(self.model, PRICING["gpt-4o-mini"])
        cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        return {"cost": cost, "input_tokens": input_tokens, "output_tokens": output_tokens}

token_optimizer = TokenOptimizer(model=MODEL_NAME)

# Redis cache (L2)
redis_client = None
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
try:
    import redis
    redis_client = redis.Redis.from_url(redis_url)
    redis_client.ping()
    logging.info(f"Connected to Redis at {redis_url}")
except Exception as e:
    redis_client = None
    logging.warning(f"Redis not available: {e}")

# In-memory cache (L1)
memory_cache = {}

def analyze_prompt_with_gpt(prompt: str) -> str:
    key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    cache_key = f"gpt:{MODEL_NAME}:{key}"

    # L1
    if cache_key in memory_cache:
        return memory_cache[cache_key]
    # L2
    if redis_client:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                text = json.loads(cached)
                memory_cache[cache_key] = text
                return text
        except Exception as re:
            logging.error(f"Redis get error: {re}")

    # Estimate/log cost
    est = token_optimizer.estimate_cost(prompt)
    logging.info(f"Estimated prompt tokens={est['input_tokens']}, approx cost=${est['cost']:.4f}")

    # API call (Chat Completions)
    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role":"user","content":prompt}],
        max_tokens=700,
        temperature=0.5
    )
    text = resp['choices'][0]['message']['content'].strip()

    # Cache
    memory_cache[cache_key] = text
    ttl = int(os.getenv('CACHE_TTL', '300'))
    if redis_client:
        try:
            redis_client.setex(cache_key, ttl, json.dumps(text))
        except Exception as re:
            logging.error(f"Redis set error: {re}")
    return text
