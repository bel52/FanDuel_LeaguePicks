# app/ai_integration.py
import os
import json
import hashlib
import logging
# Import OpenAI and Anthropic clients if available
try:
    import openai
except ImportError:
    openai = None
try:
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
except ImportError:
    Anthropic = None

class AIIntegration:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4')
        self.anthropic_model = os.getenv('ANTHROPIC_MODEL', 'claude-4')
        self.monthly_budget = float(os.getenv('MONTHLY_BUDGET', 15.0))
        if openai and self.openai_key:
            openai.api_key = self.openai_key
        if Anthropic and self.anthropic_key:
            self.anthropic_client = Anthropic(api_key=self.anthropic_key)
        else:
            self.anthropic_client = None
        self.memory_cache = {}
        try:
            import redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.Redis.from_url(redis_url)
        except Exception as e:
            self.redis_client = None
            logging.warning(f"Redis cache not available: {e}")
        # Manage token usage and costs with budgeting and caching
        # Pricing per token (USD per token) for models
        self.cost_rates = {
            'openai': {'input': 0.15/1000000, 'output': 0.60/1000000},  # GPT-4o mini pricing
            'anthropic': {'input': 3.00/1000000, 'output': 15.00/1000000}  # Claude Sonnet 4 pricing
        }
        self.total_cost = 0.0

    def _cache_get(self, key):
        # Multi-level cache check: memory then Redis
        if key in self.memory_cache:
            return self.memory_cache[key]
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    value = json.loads(data)
                    self.memory_cache[key] = value
                    return value
            except Exception as e:
                logging.error(f"Redis get error: {e}")
        return None

    def _cache_set(self, key, value, ttl=300):
        self.memory_cache[key] = value
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logging.error(f"Redis set error: {e}")

    def analyze_lineup(self, lineup, team_game_info=None):
        """Use AI to analyze a lineup (correlations, upside, risks) with cost control."""
        # Prepare analysis prompt
        game_data = ""
        pairs = set()
        if team_game_info:
            # Summarize games (team vs opp with O/U if available)
            for team, info in team_game_info.items():
                opp = info.get('opponent')
                if not opp:
                    continue
                pair = tuple(sorted([team, opp]))
                if pair in pairs:
                    continue
                pairs.add(pair)
                imp_team = info.get('implied_total')
                imp_opp = team_game_info.get(opp, {}).get('implied_total')
                total = None
                if imp_team is not None and imp_opp is not None:
                    total = round(imp_team + imp_opp, 1)
                game_data += f"{team} vs {opp}: "
                if imp_team is not None and imp_opp is not None:
                    game_data += f"Implied {team}={imp_team} vs {opp}={imp_opp} (Total {total})\n"
                else:
                    game_data += "(no odds data)\n"
        player_list = ", ".join([f"{p['name']} ({p['team']} {p['position']})" for p in lineup])
        prompt = (f"You are an expert DFS analyst. Analyze the following DFS lineup for correlation, upside, and risk.\n"
                  f"LINEUP: {player_list}\n"
                  f"{game_data if game_data else ''}"
                  f"Provide a brief analysis of the lineup's strengths, weaknesses, and any interesting correlations.")  # Prompt structure
        cache_key = "ai:" + hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cached = self._cache_get(cache_key)
        if cached:
            logging.info("Returning cached AI analysis result.")
            return cached
        if self.total_cost >= self.monthly_budget:
            logging.warning("AI analysis budget exceeded; skipping analysis.")
            return "(Analysis skipped due to budget limits.)"
        prompt_tokens = len(prompt) // 4
        expected_completion = 300
        cost_est = prompt_tokens * self.cost_rates['openai']['input'] + expected_completion * self.cost_rates['openai']['output']
        if self.total_cost + cost_est > self.monthly_budget:
            logging.warning("Estimated AI cost would exceed budget; skipping analysis.")
            return "(Analysis skipped due to budget limits.)"
        result_text = None
        if openai and self.openai_key:
            try:
                logging.info("Calling OpenAI API for lineup analysis...")
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=expected_completion
                )
                result_text = response['choices'][0]['message']['content'].strip()
                usage = response.get('usage')
                if usage:
                    tokens_used = usage.get('total_tokens', prompt_tokens + expected_completion)
                else:
                    tokens_used = prompt_tokens + expected_completion
                cost_used = tokens_used * (0.60/1000000)
                self.total_cost += cost_used
            except Exception as e:
                logging.error(f"OpenAI API call failed: {e}")
        if result_text is None and self.anthropic_client and self.anthropic_key:
            try:
                logging.info("Calling Anthropic API for lineup analysis...")
                prompt_text = f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}"
                response = self.anthropic_client.completions.create(
                    model=self.anthropic_model,
                    max_tokens_to_sample=expected_completion,
                    prompt=prompt_text
                )
                result_text = response.completion.strip() if response else None
                tokens_used = prompt_tokens + expected_completion
                cost_used = tokens_used * (15.00/1000000)
                self.total_cost += cost_used
            except Exception as e:
                logging.error(f"Anthropic API call failed: {e}")
        if result_text is None:
            result_text = "(AI analysis not available.)"
        else:
            self._cache_set(cache_key, result_text, ttl=3600)
        return result_text
