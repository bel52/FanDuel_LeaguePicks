import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import aiohttp

from app.config import settings
from app.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Advanced AI-powered lineup analysis with cost optimization and fallback."""

    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.cache_manager = CacheManager()
        
        # Rate limiter for AI calls (simple implementation without aiolimiter)
        self.ai_calls_made = 0
        self.ai_reset_time = datetime.now()
        self.max_calls_per_hour = int(os.getenv("MAX_AI_CALLS_PER_HOUR", "100"))
        
        # Initialize AI API clients
        self._initialize_clients()
        
        # Cost tracking for reporting
        self.daily_cost = 0.0
        self.call_count = 0

    def _initialize_clients(self):
        """Initialize OpenAI and Anthropic clients based on available API keys."""
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        if settings.anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")

    async def _check_rate_limit(self):
        """Simple rate limiting check"""
        now = datetime.now()
        if (now - self.ai_reset_time).total_seconds() > 3600:  # Reset every hour
            self.ai_calls_made = 0
            self.ai_reset_time = now
        
        if self.ai_calls_made >= self.max_calls_per_hour:
            raise Exception("AI rate limit exceeded")
        
        self.ai_calls_made += 1

    async def analyze_lineup(
        self,
        lineup_players: List[Dict],
        sim_results: Dict,
        game_type: str = "league",
        news_context: Optional[Dict] = None,
        weather_context: Optional[Dict] = None
    ) -> str:
        """Generate a comprehensive AI analysis for the lineup, with caching and cost control."""
        
        # Check cache first
        cache_key = self._generate_cache_key(lineup_players, game_type)
        cached_analysis = await self.cache_manager.get(cache_key)
        if cached_analysis:
            logger.info("Using cached AI analysis for lineup.")
            return cached_analysis

        try:
            await self._check_rate_limit()
            analysis = await self._generate_analysis(lineup_players, sim_results, game_type, news_context, weather_context)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            # Fallback to basic analysis
            analysis = self._generate_fallback_analysis(lineup_players, sim_results, game_type)

        # Cache the analysis result
        await self.cache_manager.set(cache_key, analysis, ttl=int(os.getenv("AI_CACHE_TTL", "1800")))
        return analysis

    async def analyze_player_news_impact(
        self,
        player_name: str,
        news_items: List[Dict],
        current_projection: float
    ) -> Tuple[float, str]:
        """Analyze how recent news might impact a player's projection."""
        if not news_items:
            return current_projection, "No significant news"
        
        # Build a concise news summary prompt
        news_text = "\n".join([
            f"- {item.get('title', '')}: {item.get('summary', '')}" for item in news_items[:3]
        ])
        
        prompt = f"""
        Analyze the impact of recent news on {player_name}'s DFS projection:

        Current projection: {current_projection:.1f} points

        Recent news:
        {news_text}

        Provide:
        1. Adjusted projection (number only)
        2. Brief reasoning (under 50 words)

        Format: "PROJECTION: X.X | REASONING: <short explanation>"
        """
        
        try:
            await self._check_rate_limit()
            response = await self._call_openai(prompt, max_tokens=100)
            
            # Parse the response for expected format
            if "PROJECTION:" in response and "REASONING:" in response:
                parts = response.split("|")
                proj_part = parts[0].replace("PROJECTION:", "").strip()
                reason_part = parts[1].replace("REASONING:", "").strip()
                try:
                    new_projection = float(proj_part)
                    return new_projection, reason_part
                except ValueError:
                    pass
        except Exception as e:
            logger.error(f"News impact analysis failed for {player_name}: {e}")
        
        # If parsing or API fails, return original projection
        return current_projection, "Analysis unavailable"

    async def suggest_optimal_swaps(
        self,
        current_lineup: List[Dict],
        available_players: List[Dict],
        salary_remaining: int,
        game_status: str = "EVEN"
    ) -> List[Dict]:
        """Suggest up to 3 optimal player swaps using AI given the current lineup and game situation."""
        strategy = "ceiling" if game_status.upper() == "BEHIND" else "balanced" if game_status.upper() == "EVEN" else "floor"
        
        prompt = f"""
        Current DFS situation: {game_status}
        Strategy needed: {strategy}
        Salary remaining: ${salary_remaining}

        Current lineup:
        {self._format_lineup_summary(current_lineup)}

        Suggest up to 3 player swaps optimized for a **{strategy}** strategy.
        Consider correlation, leverage, and salary constraints.

        Format each suggestion as:
        "OUT: [Player] ($X) | IN: [Player] ($Y) | REASON: <brief explanation>"
        """
        
        try:
            await self._check_rate_limit()
            response = await self._call_openai(prompt, max_tokens=200)
            return self._parse_swap_suggestions(response)
        except Exception as e:
            logger.error(f"Swap suggestion analysis failed: {e}")
            return []

    def _generate_cache_key(self, lineup_players: List[Dict], game_type: str) -> str:
        """Generate a unique cache key for a given lineup to cache AI analysis."""
        player_names = [p["name"] if "name" in p else p.get("PLAYER NAME", "") for p in lineup_players]
        key_data = {
            "players": sorted(player_names),
            "game_type": game_type,
            "hour": datetime.now().strftime("%Y-%m-%d-%H")
        }
        return f"ai_analysis:{hash(json.dumps(key_data, sort_keys=True))}"

    async def _generate_analysis(
        self,
        lineup_players: List[Dict],
        sim_results: Dict,
        game_type: str,
        news_context: Optional[Dict],
        weather_context: Optional[Dict]
    ) -> str:
        """Use OpenAI to generate the lineup analysis text."""
        prompt = self._build_analysis_prompt(lineup_players, sim_results, game_type, news_context, weather_context)
        
        # Track estimated cost for this call
        input_tokens = len(prompt) // 4  # rough estimate
        output_tokens = 300
        estimated_cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        self.daily_cost += estimated_cost
        self.call_count += 1
        
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are an expert DFS analyst specializing in NFL strategy optimization."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                logger.info(f"AI call #{self.call_count} to OpenAI, estimated cost ${estimated_cost:.6f}")
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"OpenAI call failed: {e}")
                raise
        
        # Fallback to Anthropic if OpenAI unavailable
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model=os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229"),
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}],
                    system="You are an expert DFS analyst specializing in NFL strategy optimization."
                )
                return response.content[0].text.strip()
            except Exception as e:
                logger.error(f"Anthropic call failed: {e}")
                raise
        
        raise Exception("No AI clients available")

    def _build_analysis_prompt(
        self,
        lineup_players: List[Dict],
        sim_results: Dict,
        game_type: str,
        news_context: Optional[Dict],
        weather_context: Optional[Dict]
    ) -> str:
        """Construct the prompt for AI lineup analysis."""
        # Basic lineup summary
        lineup_desc = "; ".join([f"{p['name']} ({p['position']})" for p in lineup_players])
        
        prompt = (
            f"Provide a DFS analysis for the following {game_type} lineup:\n"
            f"LINEUP: {lineup_desc}\n\n"
            f"Projected mean score: {sim_results.get('mean_score', 0):.2f}, "
            f"90th percentile: {sim_results.get('percentiles', {}).get('90th', 0):.2f}, "
            f"Sharpe ratio: {sim_results.get('sharpe_ratio', 0):.3f}.\n"
        )
        
        # Add context if available
        if news_context:
            prompt += f"Recent News: {news_context}\n"
        if weather_context:
            prompt += f"Weather Factors: {weather_context}\n"
        
        # Guidance for analysis content
        prompt += (
            "Analyze the lineup with focus on:\n"
            "1. Correlation (stack synergy and game environment)\n"
            "2. Leverage (ownership and differentiation)\n"
            "3. Risk/Upside (variance and ceiling potential)\n"
            "4. Key concerns (injuries, weather, or roster construction)\n"
            "Keep the analysis under 250 words, and provide actionable insights.\n"
        )
        
        return prompt

    def _generate_fallback_analysis(
        self,
        lineup_players: List[Dict],
        sim_results: Dict,
        game_type: str
    ) -> str:
        """Generate a basic analysis if AI calls fail."""
        # Simple stats from simulation results
        mean_score = sim_results.get('mean_score', 0)
        p90 = sim_results.get('percentiles', {}).get('90th', 0)
        sharpe = sim_results.get('sharpe_ratio', 0)
        
        # Identify primary QB and count stacks
        qb = next((p for p in lineup_players if p.get("position") == "QB"), None)
        stack_count = sum(1 for p in lineup_players if qb and p.get("team") == qb.get("team") and p.get("position") in ["WR", "TE"])
        
        high_own = [p for p in lineup_players if (p.get("own_pct") or 0) > 20]
        low_own = [p for p in lineup_players if 0 < (p.get("own_pct") or 0) < 10]
        
        strategy_note = "prioritize ceiling plays for maximum upside" if game_type == "h2h" else "balance floor and ceiling for consistency"
        
        analysis_lines = []
        analysis_lines.append(f"CORRELATION: {'Strong' if stack_count >= 2 else 'Standard'} stack around {qb['name'] if qb else 'the QB'} with {stack_count} pass-catcher(s).")
        analysis_lines.append(f"LEVERAGE: " + ("Has contrarian picks (e.g., " + ", ".join(p['name'] for p in low_own[:2]) + ")" if low_own else "Lineup is fairly chalky") + ".")
        analysis_lines.append(f"RISK/UPSIDE: 90th percentile outcome is {p90:.1f} points (Sharpe ratio {sharpe:.2f}), indicating " + ("high upside." if sharpe >= 1.5 else "some volatility."))
        
        if high_own:
            analysis_lines.append(f"KEY CONCERNS: High ownership on {high_own[0]['name']} – consider a pivot if news breaks.")
        else:
            analysis_lines.append(f"KEY CONCERNS: No major ownership risks identified; {strategy_note}.")
        
        return " ".join(analysis_lines)

    def _format_lineup_summary(self, lineup: List[Dict]) -> str:
        """Format lineup players for inclusion in AI prompts."""
        lines = []
        for player in lineup:
            name = player.get('name') or player.get('PLAYER NAME', '')
            pos = player.get('position') or player.get('POS', '')
            salary = player.get('salary') or player.get('SALARY', 0)
            lines.append(f"{pos} {name} (${salary})")
        return "; ".join(lines)

    def _parse_swap_suggestions(self, response: str) -> List[Dict]:
        """Parse AI swap suggestion text into structured list of suggestions."""
        suggestions = []
        for line in response.splitlines():
            if 'OUT:' in line and 'IN:' in line:
                try:
                    parts = line.split('|')
                    out_player = parts[0].replace('OUT:', '').strip()
                    in_player = parts[1].replace('IN:', '').strip()
                    reason = parts[2].replace('REASON:', '').strip() if len(parts) > 2 else "No reason provided"
                    suggestions.append({"out": out_player, "in": in_player, "reason": reason})
                except Exception:
                    continue
        return suggestions[:3]  # up to 3 suggestions

    async def _call_openai(self, prompt: str, max_tokens: int = 300) -> str:
        """Low-level helper to call OpenAI completion API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        response = self.openai_client.chat.completions.create(
            model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
