import os
import json
import hashlib
import logging

# Old (pre-1.0) OpenAI client
try:
    import openai
except ImportError:
    openai = None

# Anthropic (optional; we skip if OpenAI works)
try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
except Exception:
    Anthropic = None

class AIIntegration:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')

        # Support both env names: OPENAI_MODEL and GPT_MODEL
        self.openai_model = (
            os.getenv('OPENAI_MODEL')
            or os.getenv('GPT_MODEL')
            or 'gpt-4'
        )

        # Prefer OpenAI; Anthropic only as fallback (optional)
        self.anthropic_model = os.getenv('ANTHROPIC_MODEL', 'claude-2.1')

        # Feature toggle / budget
        self.use_ai = str(os.getenv('USE_AI', '1')).strip() == '1'
        self.monthly_budget = float(os.getenv('MONTHLY_BUDGET', 15.0))
        self.total_cost = 0.0

        if openai and self.openai_key:
            openai.api_key = self.openai_key

        if Anthropic and self.anthropic_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_key)
        else:
            self.anthropic_client = None

        # simple rough costs (USD/token)
        self.cost_rates = {
            'openai': {'input': 0.15/1_000_000, 'output': 0.60/1_000_000},   # ~gpt-4o-mini ballpark
            'anthropic': {'input': 3.0/1_000_000, 'output': 15.0/1_000_000}, # sonnet ballpark
        }

        self.memory_cache = {}
        try:
            import redis
            redis_url = os.getenv('REDIS_URL', '')
            self.redis_client = redis.Redis.from_url(redis_url) if redis_url else None
        except Exception as e:
            self.redis_client = None
            logging.warning(f"Redis cache not available: {e}")

    def _cache_get(self, key):
        if key in self.memory_cache:
            return self.memory_cache[key]
        if self.redis_client:
            try:
                val = self.redis_client.get(key)
                if val:
                    data = json.loads(val)
                    self.memory_cache[key] = data
                    return data
            except Exception as e:
                logging.error(f"Redis get error: {e}")
        return None

    def _cache_set(self, key, value, ttl=3600):
        self.memory_cache[key] = value
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logging.error(f"Redis set error: {e}")

    def _would_exceed_budget(self, prompt_chars: int, expected_output_tokens: int, provider='openai') -> bool:
        prompt_tokens = max(1, prompt_chars // 4)
        inp = prompt_tokens * self.cost_rates[provider]['input']
        outp = expected_output_tokens * self.cost_rates[provider]['output']
        return (self.total_cost + inp + outp) > self.monthly_budget

    def _add_cost(self, input_tokens: int, output_tokens: int, provider='openai'):
        self.total_cost += input_tokens * self.cost_rates[provider]['input']
        self.total_cost += output_tokens * self.cost_rates[provider]['output']

    def analyze_lineup(self, lineup, team_game_info=None):
        if not self.use_ai:
            return "(AI disabled.)"

        # Build prompt
        game_data = ""
        if team_game_info:
            seen = set()
            for team, info in team_game_info.items():
                opp = info.get('opponent')
                if not opp: continue
                pair = tuple(sorted([team, opp]))
                if pair in seen: continue
                seen.add(pair)
                it = info.get('implied_total')
                ot = team_game_info.get(opp, {}).get('implied_total')
                if it is not None and ot is not None:
                    game_data += f"{team} vs {opp}: implied {team}={it}, {opp}={ot}, total={round(it+ot,1)}\n"

        player_list = ", ".join([f"{p['name']} ({p['team']} {p['position']})" for p in lineup])
        prompt = (
            "You are an expert NFL DFS analyst. "
            "Given the lineup and context, provide <=6 bullets on correlation/stacking, leverage, ceiling/floor, and risks.\n\n"
            f"LINEUP: {player_list}\n"
            f"{game_data}"
        )

        cache_key = "ai:" + hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        if self.total_cost >= self.monthly_budget or self._would_exceed_budget(len(prompt), 300, 'openai'):
            return "(Analysis skipped due to budget limits.)"

        # Try OpenAI (old SDK flow)
        result_text = None
        if openai and self.openai_key:
            try:
                resp = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a concise, high-signal DFS analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.4,
                )
                result_text = resp['choices'][0]['message']['content'].strip()
                usage = resp.get('usage', {})
                in_toks = usage.get('prompt_tokens', max(1, len(prompt)//4))
                out_toks = usage.get('completion_tokens', 300)
                self._add_cost(in_toks, out_toks, 'openai')
            except Exception as e:
                logging.error(f"OpenAI API call failed: {e}")

        # Optional Anthropic fallback (only if OpenAI failed)
        if result_text is None and self.anthropic_client and self.anthropic_key:
            try:
                txt = f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"
                resp = self.anthropic_client.completions.create(
                    model=self.anthropic_model,
                    max_tokens_to_sample=300,
                    prompt=txt,
                    temperature=0.4,
                )
                result_text = (resp.completion or "").strip() if resp else None
                self._add_cost(max(1, len(prompt)//4), 300, 'anthropic')
            except Exception as e:
                logging.error(f"Anthropic API call failed: {e}")

        if not result_text:
            result_text = "(AI analysis not available.)"
        else:
            self._cache_set(cache_key, result_text, ttl=3600)

        return result_text
