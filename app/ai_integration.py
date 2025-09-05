import os
import logging
from typing import Dict, List, Any, Optional
import asyncio
from functools import lru_cache

from app.config import settings

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """Handles AI-powered lineup analysis using OpenAI or Claude"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize OpenAI if available
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        # Initialize Anthropic if available
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
        sim_results: Dict
    ) -> str:
        """Generate AI analysis of lineup"""
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(lineup_players, sim_results)
        
        # Try OpenAI first
        if settings.ai_provider == "openai" and self.openai_client:
            try:
                return await self._analyze_with_openai(prompt)
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
        
        # Try Anthropic as fallback
        if self.anthropic_client:
            try:
                return await self._analyze_with_anthropic(prompt)
            except Exception as e:
                logger.error(f"Anthropic analysis failed: {e}")
        
        # Fallback to basic analysis
        return self._generate_basic_analysis(lineup_players, sim_results)
    
    def _build_analysis_prompt(self, lineup_players: List[Dict], sim_results: Dict) -> str:
        """Build comprehensive analysis prompt"""
        
        # Extract key lineup info
        qb = next((p for p in lineup_players if p["position"] == "QB"), None)
        stack_count = sum(1 for p in lineup_players 
                         if p.get("team") == qb.get("team") and p["position"] in ["WR", "TE"])
        
        total_salary = sum(p["salary"] for p in lineup_players)
        total_proj = sum(p["proj_points"] for p in lineup_players)
        
        prompt = f"""
        Analyze this FanDuel NFL DFS lineup for tournament play:
        
        LINEUP COMPOSITION:
        - Total Projection: {total_proj:.2f} points
        - Total Salary: ${total_salary:,}
        - Stack: {qb['name'] if qb else 'No QB'} with {stack_count} pass-catchers
        
        PLAYERS:
        """
        
        for p in lineup_players:
            own_pct = f"{p['own_pct']:.1f}%" if p.get('own_pct') else "Unknown"
            prompt += f"\n- {p['position']}: {p['name']} ({p['team']} vs {p['opponent']}) - ${p['salary']:,} - {p['proj_points']:.1f}pts - Own: {own_pct}"
        
        prompt += f"""
        
        SIMULATION RESULTS:
        - Mean Score: {sim_results.get('mean_score', 0):.2f}
        - Std Dev: {sim_results.get('std_dev', 0):.2f}
        - 90th Percentile: {sim_results.get('percentiles', {}).get('90th', 0):.2f}
        - Sharpe Ratio: {sim_results.get('sharpe_ratio', 0):.3f}
        
        Provide a concise DFS analysis covering:
        1. CORRELATION: Stack quality and game environment
        2. LEVERAGE: Tournament differentiation opportunities
        3. RISK/CEILING: Variance drivers and boom/bust potential
        4. SUGGESTED SWAPS: 1-2 specific alternative plays if needed
        
        Keep response under 200 words, focus on actionable insights.
        """
        
        return prompt
    
    async def _analyze_with_openai(self, prompt: str) -> str:
        """Generate analysis using OpenAI GPT-4"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert DFS analyst specializing in NFL tournament strategy."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    async def _analyze_with_anthropic(self, prompt: str) -> str:
        """Generate analysis using Claude"""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized")
        
        response = await self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=400,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are an expert DFS analyst specializing in NFL tournament strategy."
        )
        
        return response.content[0].text
    
    def _generate_basic_analysis(self, lineup_players: List[Dict], sim_results: Dict) -> str:
        """Generate basic analysis without AI"""
        
        qb = next((p for p in lineup_players if p["position"] == "QB"), None)
        stack_count = sum(1 for p in lineup_players 
                         if p.get("team") == qb.get("team") and p["position"] in ["WR", "TE"])
        
        high_own = [p for p in lineup_players if p.get("own_pct", 0) > 20]
        low_own = [p for p in lineup_players if 0 < p.get("own_pct", 0) < 10]
        
        analysis = f"""
CORRELATION: {'Strong' if stack_count >= 2 else 'Standard'} stack with {qb['name'] if qb else 'QB'} and {stack_count} pass-catcher(s). 
Consider adding a bring-back from the opposing team in high-total games.

LEVERAGE: """
        
        if low_own:
            analysis += f"Low-owned plays ({', '.join(p['name'] for p in low_own[:2])}) provide differentiation. "
        else:
            analysis += "Limited leverage plays - consider pivoting off chalk. "
        
        analysis += f"""

RISK/CEILING: Sharpe ratio of {sim_results.get('sharpe_ratio', 0):.3f} indicates {'stable' if sim_results.get('sharpe_ratio', 0) > 1.5 else 'volatile'} projection. 
90th percentile of {sim_results.get('percentiles', {}).get('90th', 0):.1f} points.

SUGGESTED SWAPS: """
        
        if high_own:
            analysis += f"Consider fading {high_own[0]['name']} (high ownership) for a contrarian pivot."
        else:
            analysis += "Lineup appears balanced - monitor news for late swap opportunities."
        
        return analysis.strip()
