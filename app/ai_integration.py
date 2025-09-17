import os, logging, json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from aiolimiter import AsyncLimiter
from app.config import settings
from app.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """AI-powered analysis using OpenAI and Anthropic LLMs."""
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.cache_manager = CacheManager()
        # Limit total AI calls per hour to control cost
        self.ai_limiter = AsyncLimiter(settings.MAX_AI_CALLS_PER_HOUR, time_period=3600)
        self.daily_cost = 0.0
        self.call_count = 0
        self._initialize_clients()
    
    def _initialize_clients(self):
        if settings.OPENAI_API_KEY:
            try:
                import openai
                self.openai_client = openai
                openai.api_key = settings.OPENAI_API_KEY
                logger.info("Initialized OpenAI client")
            except Exception as e:
                logger.error(f"OpenAI client init failed: {e}")
        if settings.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.anthropic_client = anthropic.Client(settings.ANTHROPIC_API_KEY)
                logger.info("Initialized Anthropic client")
            except Exception as e:
                logger.error(f"Anthropic client init failed: {e}")

    async def analyze_lineup(
        self, lineup_players: List[Dict], sim_results: Dict,
        game_type: str = "league", news_context: Optional[Dict] = None,
        weather_context: Optional[Dict] = None
    ) -> str:
        """Generate AI commentary on the final lineup and simulation."""
        cache_key = self._generate_cache_key(lineup_players, game_type)
        cached = await self.cache_manager.get(cache_key)
        if cached:
            logger.info("Using cached AI analysis")
            return cached
        prompt = self._build_analysis_prompt(lineup_players, sim_results, game_type, news_context, weather_context)
        try:
            async with self.ai_limiter:
                response = await self._call_openai(prompt, max_tokens=300)
                analysis = response.strip()
                await self.cache_manager.set(cache_key, analysis, ttl=settings.AI_CACHE_TTL)
                return analysis
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return "AI analysis unavailable"

    async def analyze_player_news_impact(
        self, player_name: str, news_items: List[Dict], current_proj: float
    ) -> Tuple[float, str]:
        """Ask AI to adjust a single player's projection based on recent news."""
        if not news_items:
            return current_proj, "No significant news"
        news_text = "\n".join([f"- {n.get('title','')}: {n.get('summary','')}" for n in news_items[:3]])
        prompt = (
            f"Analyze recent news for {player_name}:\n"
            f"Current projection: {current_proj:.1f}\nRecent news:\n{news_text}\n"
            "Provide adjusted projection and brief reasoning.\n"
            "Format: PROJECTION: <value> | REASON: <explanation>\n"
        )
        try:
            async with self.ai_limiter:
                resp = await self._call_openai(prompt, max_tokens=50)
                if "PROJECTION:" in resp and "REASON:" in resp:
                    parts = resp.split("|")
                    proj_str = parts[0].replace("PROJECTION:", "").strip()
                    reason = parts[1].replace("REASON:", "").strip()
                    try:
                        return float(proj_str), reason
                    except:
                        pass
        except Exception as e:
            logger.error(f"News AI analysis failed for {player_name}: {e}")
        return current_proj, "Analysis unavailable"

    async def suggest_optimal_swaps(
        self, current_lineup: List[Dict], available_players: List[Dict],
        salary_remaining: int, game_status: str = "EVEN"
    ) -> List[Dict]:
        """Suggest up to 3 player swaps based on game situation using AI."""
        strategy = {"BEHIND":"ceiling", "EVEN":"balanced", "AHEAD":"floor"}.get(game_status, "balanced")
        prompt = (
            f"Game status: {game_status}. Strategy: {strategy}. Salary left: ${salary_remaining}.\n"
            "Current lineup summary:\n"
            + "\n".join([f"{p['position']}-{p['name']} (${p['salary']}): {p['proj_points']} pts" for p in current_lineup]) +
            "\nSuggest up to 3 swaps for a {strategy} strategy, format: "
            "\"OUT: [Name] ($X) | IN: [Name] ($Y) | REASON: ...\""
        )
        try:
            async with self.ai_limiter:
                resp = await self._call_openai(prompt, max_tokens=200)
                return self._parse_swap_suggestions(resp)
        except Exception as e:
            logger.error(f"Swap suggestion failed: {e}")
            return []

    def _generate_cache_key(self, lineup_players: List[Dict], game_type: str) -> str:
        names = sorted(p['name'] for p in lineup_players)
        key_str = f"{names}-{game_type}-{datetime.now().strftime('%Y-%m-%d-%H')}"
        return f"ai_lineup:{hash(key_str)}"

    def _build_analysis_prompt(self, lineup, sim, game_type, news, weather):
        summary = "\n".join([f"{p['position']}-{p['name']} (${p['salary']}): {p['proj_points']} pts vs {p['opponent']}" 
                              for p in lineup])
        prompt = (
            f"Lineup analysis for {game_type} game.\nPlayers:\n{summary}\n"
            f"Projected total: {sim['mean_score']} (Std {sim['std_dev']}).\n"
            "Evaluate strengths/weaknesses and risk. Provide insights."
        )
        return prompt

    async def _call_openai(self, prompt: str, max_tokens: int = 150) -> str:
        """Call OpenAI or Anthropic with cost accounting (GPT-4o-mini)."""
        self.call_count += 1
        if self.openai_client:
            resp = await self.openai_client.ChatCompletion.acreate(
                model=settings.GPT_MODEL,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens
            )
            text = resp.choices[0].message.content
            self.daily_cost += resp.usage.total_tokens * 0.00001  # approximate cost
            return text
        if self.anthropic_client:
            resp = await self.anthropic_client.completions.create(
                model="claude-3-mini",
                prompt=prompt,
                max_tokens_to_sample=max_tokens
            )
            return resp.completion
        raise RuntimeError("No AI client available")

    def _parse_swap_suggestions(self, text: str) -> List[Dict]:
        """Parse swap suggestions from AI response."""
        suggestions = []
        lines = [line for line in text.splitlines() if line.strip()]
        for line in lines[:3]:
            try:
                parts = line.split("|")
                out = parts[0].split("OUT:")[-1].strip()
                inn = parts[1].split("IN:")[-1].strip()
                reason = parts[2].split("REASON:")[-1].strip()
                out_name, out_sal = out.rsplit("(",1)[0].strip(), out.split("$")[-1].strip(")")
                in_name, in_sal = inn.rsplit("(",1)[0].strip(), inn.split("$")[-1].strip(")")
                suggestions.append({
                    "out": out_name,
                    "in": in_name,
                    "reason": reason
                })
            except:
                continue
        return suggestions
