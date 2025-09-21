import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import aiohttp
from aiolimiter import AsyncLimiter

from app.config import settings
from app.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Advanced AI-powered lineup analysis with cost optimization"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.cache_manager = CacheManager()
        
        # Rate limiters for cost control
        self.ai_limiter = AsyncLimiter(
            max_rate=int(os.getenv("MAX_AI_CALLS_PER_HOUR", "100")), 
            time_period=3600
        )
        
        # Initialize AI clients
        self._initialize_clients()
        
        # Cost tracking
        self.daily_cost = 0.0
        self.call_count = 0
    
    def _initialize_clients(self):
        """Initialize AI clients based on configuration"""
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized (GPT-4o-mini)")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        if settings.anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
    
    async def analyze_lineup(
        self, 
        lineup_players: List[Dict], 
        sim_results: Dict,
        game_type: str = "league",  # "league" or "h2h"
        news_context: Optional[Dict] = None,
        weather_context: Optional[Dict] = None
    ) -> str:
        """Generate comprehensive AI analysis with cost optimization"""
        
        # Check cache first
        cache_key = self._generate_cache_key(lineup_players, game_type)
        cached_analysis = await self.cache_manager.get(cache_key)
        if cached_analysis:
            logger.info("Using cached AI analysis")
            return cached_analysis
        
        # Rate limit check
        async with self.ai_limiter:
            try:
                analysis = await self._generate_analysis(
                    lineup_players, sim_results, game_type, 
                    news_context, weather_context
                )
                
                # Cache the result
                await self.cache_manager.set(
                    cache_key, analysis, 
                    ttl=int(os.getenv("AI_CACHE_TTL", "1800"))
                )
                
                return analysis
                
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                return self._generate_fallback_analysis(lineup_players, sim_results, game_type)
    
    async def analyze_player_news_impact(
        self, 
        player_name: str, 
        news_items: List[Dict],
        current_projection: float
    ) -> Tuple[float, str]:
        """Analyze news impact on player projection"""
        
        if not news_items:
            return current_projection, "No significant news"
        
        # Build news analysis prompt
        news_text = "\n".join([
            f"- {item.get('title', '')}: {item.get('summary', '')}"
            for item in news_items[:3]  # Limit to 3 most recent
        ])
        
        prompt = f"""
        Analyze the impact of recent news on {player_name}'s DFS projection:
        
        Current projection: {current_projection:.1f} points
        
        Recent news:
        {news_text}
        
        Provide:
        1. Adjusted projection (number only)
        2. Brief reasoning (under 50 words)
        
        Format: "PROJECTION: X.X | REASONING: brief explanation"
        """
        
        try:
            async with self.ai_limiter:
                response = await self._call_openai(prompt, max_tokens=100)
                
                # Parse response
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
            logger.error(f"News analysis failed for {player_name}: {e}")
        
        return current_projection, "Analysis unavailable"
    
    async def suggest_optimal_swaps(
        self, 
        current_lineup: List[Dict],
        available_players: List[Dict],
        salary_remaining: int,
        game_status: str = "EVEN"
    ) -> List[Dict]:
        """AI-powered swap suggestions based on game situation"""
        
        strategy = "ceiling" if game_status == "BEHIND" else "balanced" if game_status == "EVEN" else "floor"
        
        prompt = f"""
        Current DFS situation: {game_status}
        Strategy needed: {strategy}
        Salary remaining: ${salary_remaining}
        
        Current lineup summary:
        {self._format_lineup_summary(current_lineup)}
        
        Suggest up to 3 player swaps optimized for {strategy} strategy.
        Consider correlation, leverage, and salary constraints.
        
        Format each suggestion as:
        "OUT: [Player] ($X) | IN: [Player] ($Y) | REASON: [brief explanation]"
        """
        
        try:
            async with self.ai_limiter:
                response = await self._call_openai(prompt, max_tokens=200)
                return self._parse_swap_suggestions(response)
                
        except Exception as e:
            logger.error(f"Swap analysis failed: {e}")
            return []
    
    def _generate_cache_key(self, lineup_players: List[Dict], game_type: str) -> str:
        """Generate cache key for AI analysis"""
        player_names = [p["name"] for p in lineup_players]
        key_data = {
            "players": sorted(player_names),
            "game_type": game_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d-%H")  # Cache for 1 hour
        }
        return f"ai_analysis:{hash(str(key_data))}"
    
    async def _generate_analysis(
        self, 
        lineup_players: List[Dict], 
        sim_results: Dict,
        game_type: str,
        news_context: Optional[Dict],
        weather_context: Optional[Dict]
    ) -> str:
        """Generate comprehensive AI analysis"""
        
        prompt = self._build_analysis_prompt(
            lineup_players, sim_results, game_type, 
            news_context, weather_context
        )
        
        # Use GPT-4o-mini for cost efficiency
        return await self._call_openai(prompt, max_tokens=300)
    
    def _build_analysis_prompt(
        self, 
        lineup_players: List[Dict], 
        sim_results: Dict,
        game_type: str,
        news_context: Optional[Dict],
        weather_context: Optional[Dict]
    ) -> str:
        """Build comprehensive analysis prompt"""
        
        # Extract key lineup info
        qb = next((p for p in lineup_players if p["position"] == "QB"), None)
        stack_count = sum(1 for p in lineup_players 
                         if p.get("team") == qb.get("team") and p["position"] in ["WR", "TE"])
        
        total_salary = sum(p["salary"] for p in lineup_players)
        total_proj = sum(p["proj_points"] for p in lineup_players)
        
        strategy_focus = "maximize ceiling/upside" if game_type == "h2h" else "balance ceiling and floor"
        
        prompt = f"""
        Analyze this FanDuel NFL DFS lineup for {game_type.upper()} play:
        
        LINEUP COMPOSITION:
        - Strategy: {strategy_focus}
        - Total Projection: {total_proj:.2f} points
        - Total Salary: ${total_salary:,}
        - Stack: {qb['name'] if qb else 'No QB'} with {stack_count} pass-catchers
        
        PLAYERS:
        """
        
        for p in lineup_players:
            own_pct = f"{p['own_pct']:.1f}%" if p.get('own_pct') else "Unknown"
            prompt += f"\n- {p['position']}: {p['name']} ({p['team']} vs {p['opponent']}) - ${p['salary']:,} - {p['proj_points']:.1f}pts - Own: {own_pct}"
        
        # Add news context if available
        if news_context:
            prompt += f"\n\nRECENT NEWS IMPACTS:\n{news_context.get('summary', 'No major news')}"
        
        # Add weather context if available
        if weather_context:
            prompt += f"\n\nWEATHER CONDITIONS:\n{weather_context.get('summary', 'Normal conditions')}"
        
        prompt += f"""
        
        SIMULATION RESULTS:
        - Mean Score: {sim_results.get('mean_score', 0):.2f}
        - 90th Percentile: {sim_results.get('percentiles', {}).get('90th', 0):.2f}
        - Sharpe Ratio: {sim_results.get('sharpe_ratio', 0):.3f}
        
        Provide analysis covering:
        1. CORRELATION: Stack quality and game environment synergy
        2. LEVERAGE: Tournament differentiation for {game_type} strategy
        3. RISK/UPSIDE: Variance and boom/bust potential
        4. KEY CONCERNS: Weather, news, or lineup construction issues
        
        Keep response under 250 words, focus on actionable {game_type} insights.
        """
        
        return prompt
    
    async def _call_openai(self, prompt: str, max_tokens: int = 300) -> str:
        """Make optimized OpenAI API call"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        # Track costs (GPT-4o-mini pricing)
        estimated_input_tokens = len(prompt) // 4  # Rough estimate
        estimated_cost = (estimated_input_tokens * 0.15 + max_tokens * 0.60) / 1_000_000
        self.daily_cost += estimated_cost
        self.call_count += 1
        
        response = await self.openai_client.chat.completions.create(
            model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert DFS analyst specializing in NFL strategy optimization."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        logger.info(f"AI call #{self.call_count}, estimated cost: ${estimated_cost:.6f}")
        return response.choices[0].message.content
    
    def _generate_fallback_analysis(
        self, 
        lineup_players: List[Dict], 
        sim_results: Dict,
        game_type: str
    ) -> str:
        """Generate basic analysis without AI"""
        
        qb = next((p for p in lineup_players if p["position"] == "QB"), None)
        stack_count = sum(1 for p in lineup_players 
                         if p.get("team") == qb.get("team") and p["position"] in ["WR", "TE"])
        
        high_own = [p for p in lineup_players if p.get("own_pct", 0) > 20]
        low_own = [p for p in lineup_players if 0 < p.get("own_pct", 0) < 10]
        
        strategy_note = "Prioritize ceiling plays for tournament leverage" if game_type == "h2h" else "Balance floor and ceiling for consistent scoring"
        
        analysis = f"""
CORRELATION: {'Strong' if stack_count >= 2 else 'Standard'} stack with {qb['name'] if qb else 'QB'} and {stack_count} pass-catcher(s). 
{strategy_note}.

LEVERAGE: """
        
        if low_own:
            analysis += f"Contrarian edge with {', '.join(p['name'] for p in low_own[:2])} (low ownership). "
        else:
            analysis += "Limited leverage - consider pivoting off popular plays. "
        
        analysis += f"""

RISK/UPSIDE: Sharpe ratio of {sim_results.get('sharpe_ratio', 0):.3f} indicates {'stable' if sim_results.get('sharpe_ratio', 0) > 1.5 else 'volatile'} projection profile. 
90th percentile upside of {sim_results.get('percentiles', {}).get('90th', 0):.1f} points.

KEY CONCERNS: """
        
        if high_own:
            analysis += f"Monitor {high_own[0]['name']} for potential pivot opportunities (high ownership risk)."
        else:
            analysis += "Well-balanced ownership profile - maintain lineup integrity unless breaking news emerges."
        
        return analysis.strip()
    
    def _format_lineup_summary(self, lineup: List[Dict]) -> str:
        """Format lineup for AI prompts"""
        summary = []
        for player in lineup:
            summary.append(f"{player['position']}: {player['name']} (${player['salary']})")
        return "\n".join(summary)
    
    def _parse_swap_suggestions(self, response: str) -> List[Dict]:
        """Parse AI swap suggestions"""
        suggestions = []
        lines = response.split('\n')
        
        for line in lines:
            if 'OUT:' in line and 'IN:' in line:
                try:
                    parts = line.split('|')
                    out_part = parts[0].replace('OUT:', '').strip()
                    in_part = parts[1].replace('IN:', '').strip()
                    reason_part = parts[2].replace('REASON:', '').strip() if len(parts) > 2 else "Strategic optimization"
                    
                    suggestions.append({
                        'out_player': out_part,
                        'in_player': in_part,
                        'reason': reason_part
                    })
                except Exception:
                    continue
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return {
            "daily_cost": round(self.daily_cost, 6),
            "call_count": self.call_count,
            "estimated_weekly_cost": round(self.daily_cost * 7, 4),
            "within_budget": self.daily_cost * 7 < 15.0  # $15 weekly budget
        }
