import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import time

from app.config import settings
from app.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """AI-powered analysis for DFS optimization with cost tracking"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.openai_client = None
        self.anthropic_client = None
        
        # Cost and usage tracking
        self.daily_cost = 0.0
        self.call_count = 0
        self.last_reset = datetime.now().date()
        
        # Rate limiting
        self.calls_this_hour = 0
        self.hour_reset_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AI API clients"""
        
        if settings.openai_api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI library not available")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        if settings.anthropic_api_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic library not available")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    async def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        
        # Reset hourly counter
        if now >= self.hour_reset_time + timedelta(hours=1):
            self.calls_this_hour = 0
            self.hour_reset_time = now.replace(minute=0, second=0, microsecond=0)
        
        # Reset daily costs
        if now.date() > self.last_reset:
            self.daily_cost = 0.0
            self.call_count = 0
            self.last_reset = now.date()
        
        # Check hourly limit
        if self.calls_this_hour >= settings.max_ai_calls_per_hour:
            logger.warning("AI hourly rate limit reached")
            return False
        
        return True
    
    async def analyze_lineup(
        self,
        lineup_data: Dict[str, Any],
        game_type: str = "league",
        context: Optional[Dict] = None
    ) -> str:
        """Generate comprehensive AI analysis of a lineup"""
        
        try:
            # Check cache first
            cache_key = f"lineup_analysis:{hash(str(lineup_data))}:{game_type}"
            cached_analysis = await self.cache_manager.get(cache_key)
            if cached_analysis:
                logger.info("Using cached lineup analysis")
                return cached_analysis
            
            # Check rate limits
            if not await self._check_rate_limits():
                return self._generate_fallback_analysis(lineup_data, game_type)
            
            # Prepare analysis prompt
            prompt = self._build_lineup_analysis_prompt(lineup_data, game_type, context)
            
            # Get analysis from AI
            analysis = await self._call_ai_service(prompt, max_tokens=400)
            
            if analysis:
                # Cache the result
                await self.cache_manager.set(cache_key, analysis, ttl=settings.ai_cache_ttl)
                return analysis
            else:
                return self._generate_fallback_analysis(lineup_data, game_type)
            
        except Exception as e:
            logger.error(f"Error in lineup analysis: {e}")
            return self._generate_fallback_analysis(lineup_data, game_type)
    
    async def analyze_player_news_impact(
        self,
        player_name: str,
        news_items: List[Dict],
        current_projection: float
    ) -> Tuple[float, str]:
        """Analyze how news impacts a player's projection"""
        
        try:
            if not news_items:
                return current_projection, "No significant news"
            
            # Check rate limits
            if not await self._check_rate_limits():
                return current_projection, "Rate limit reached"
            
            # Build news analysis prompt
            news_text = "\n".join([
                f"- {item.get('title', '')}: {item.get('summary', '')}" 
                for item in news_items[:3]
            ])
            
            prompt = f"""
            Analyze the impact of recent news on {player_name}'s DFS projection:
            
            Current projection: {current_projection:.1f} points
            
            Recent news:
            {news_text}
            
            Provide:
            1. Adjusted projection (number between 0 and 50)
            2. Brief reasoning (under 50 words)
            
            Format: "PROJECTION: X.X | REASONING: <explanation>"
            """
            
            response = await self._call_ai_service(prompt, max_tokens=100)
            
            if response and "PROJECTION:" in response and "REASONING:" in response:
                try:
                    parts = response.split("|")
                    proj_part = parts[0].replace("PROJECTION:", "").strip()
                    reason_part = parts[1].replace("REASONING:", "").strip()
                    
                    new_projection = float(proj_part)
                    # Sanity check
                    if 0 <= new_projection <= 50:
                        return new_projection, reason_part
                except (ValueError, IndexError):
                    pass
            
            return current_projection, "Analysis inconclusive"
            
        except Exception as e:
            logger.error(f"Error analyzing player news: {e}")
            return current_projection, "Analysis failed"
    
    async def suggest_lineup_improvements(
        self,
        current_lineup: List[Dict],
        available_players: List[Dict],
        salary_remaining: int,
        game_status: str = "EVEN"
    ) -> List[Dict]:
        """Suggest lineup improvements using AI"""
        
        try:
            # Check rate limits
            if not await self._check_rate_limits():
                return []
            
            strategy = self._determine_strategy(game_status)
            
            prompt = f"""
            Current DFS situation: {game_status}
            Strategy needed: {strategy}
            Salary remaining: ${salary_remaining}
            
            Current lineup summary:
            {self._format_lineup_for_prompt(current_lineup)}
            
            Suggest up to 3 player swaps optimized for a {strategy} strategy.
            Consider correlation, leverage, and salary constraints.
            
            Format each suggestion as:
            "OUT: [Player] ($X) | IN: [Player] ($Y) | REASON: <brief explanation>"
            """
            
            response = await self._call_ai_service(prompt, max_tokens=200)
            
            if response:
                return self._parse_swap_suggestions(response)
            
            return []
            
        except Exception as e:
            logger.error(f"Error suggesting improvements: {e}")
            return []
    
    async def analyze_weather_impact(
        self,
        games_weather: List[Dict]
    ) -> Dict[str, Dict]:
        """Analyze weather impact on fantasy scoring"""
        
        try:
            if not games_weather or not await self._check_rate_limits():
                return {}
            
            weather_impacts = {}
            
            for game in games_weather:
                if not game.get('weather_data'):
                    continue
                
                weather = game['weather_data']
                home_team = game['home_team']
                away_team = game['away_team']
                
                prompt = f"""
                Analyze weather impact for NFL game:
                {home_team} vs {away_team}
                
                Conditions:
                - Temperature: {weather.get('temperature', 'N/A')}°F
                - Wind: {weather.get('wind_speed', 'N/A')} mph
                - Precipitation: {weather.get('precipitation', 'None')}
                - Conditions: {weather.get('conditions', 'Clear')}
                
                Provide multipliers (0.8-1.2) for fantasy scoring:
                Format: "PASSING: X.XX | RUSHING: X.XX | KICKING: X.XX | REASON: <explanation>"
                """
                
                response = await self._call_ai_service(prompt, max_tokens=150)
                
                if response:
                    impact = self._parse_weather_response(response)
                    if impact:
                        weather_impacts[home_team] = impact
                        weather_impacts[away_team] = impact
            
            return weather_impacts
            
        except Exception as e:
            logger.error(f"Error analyzing weather: {e}")
            return {}
    
    async def analyze_game_script(
        self,
        games: List[Dict]
    ) -> Dict[str, Dict]:
        """Predict game scripts and fantasy implications"""
        
        try:
            if not games or not await self._check_rate_limits():
                return {}
            
            game_scripts = {}
            
            for game in games[:5]:  # Limit to avoid token overuse
                prompt = f"""
                Predict game script for:
                {game.get('home_team')} vs {game.get('away_team')}
                
                Details:
                - Total: {game.get('total', 'N/A')}
                - Spread: {game.get('spread', 'N/A')}
                - Home Implied: {game.get('home_implied', 'N/A')}
                - Away Implied: {game.get('away_implied', 'N/A')}
                
                Predict:
                1. Script type (shootout/grind/blowout)
                2. Passing game grade (A-F)
                3. Rushing game grade (A-F)
                
                Format: "SCRIPT: X | PASSING: X | RUSHING: X | REASON: <brief explanation>"
                """
                
                response = await self._call_ai_service(prompt, max_tokens=100)
                
                if response:
                    script = self._parse_game_script_response(response)
                    if script:
                        game_key = f"{game.get('home_team')}_vs_{game.get('away_team')}"
                        game_scripts[game_key] = script
            
            return game_scripts
            
        except Exception as e:
            logger.error(f"Error analyzing game scripts: {e}")
            return {}
    
    async def _call_ai_service(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Call the available AI service (OpenAI or Anthropic)"""
        
        try:
            self.calls_this_hour += 1
            self.call_count += 1
            
            # Try OpenAI first (cheaper)
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert DFS analyst. Provide concise, actionable insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                
                # Track cost (rough estimate)
                input_tokens = len(prompt) // 4
                output_tokens = len(response.choices[0].message.content) // 4
                if settings.openai_model == "gpt-4o-mini":
                    cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
                else:
                    cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
                
                self.daily_cost += cost
                
                return response.choices[0].message.content.strip()
            
            # Fall back to Anthropic
            elif self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model=settings.anthropic_model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    system="You are an expert DFS analyst. Provide concise, actionable insights."
                )
                
                # Track cost (rough estimate)
                input_tokens = len(prompt) // 4
                output_tokens = len(response.content[0].text) // 4
                cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
                
                self.daily_cost += cost
                
                return response.content[0].text.strip()
            
            else:
                logger.warning("No AI client available")
                return None
                
        except Exception as e:
            logger.error(f"AI service call failed: {e}")
            return None
    
    def _build_lineup_analysis_prompt(
        self,
        lineup_data: Dict[str, Any],
        game_type: str,
        context: Optional[Dict]
    ) -> str:
        """Build comprehensive lineup analysis prompt"""
        
        lineup = lineup_data.get('lineup', [])
        summary = lineup_data.get('summary', {})
        stack_analysis = lineup_data.get('stack_analysis', {})
        simulation = lineup_data.get('simulation_results', {})
        
        lineup_desc = "; ".join([
            f"{p['player_name']} ({p['position']}, ${p['salary']}, {p['projection']:.1f})"
            for p in lineup[:6]  # Limit for token efficiency
        ])
        
        prompt = f"""
        Analyze this {game_type} DFS lineup:
        
        Players: {lineup_desc}
        
        Stats:
        - Total Projection: {summary.get('total_projection', 0):.1f}
        - Salary Used: ${summary.get('total_salary', 0):,}
        - Stack: {stack_analysis.get('qb_name', 'None')} + {stack_analysis.get('stack_count', 0)} receivers
        - 90th Percentile: {simulation.get('percentiles', {}).get('90th', 0):.1f}
        
        Provide analysis covering:
        1. Correlation strength and stack quality
        2. Leverage opportunities (low-owned upside)
        3. Risk/ceiling potential
        4. Key concerns or weaknesses
        
        Keep response under 250 words with actionable insights.
        """
        
        return prompt
    
    def _format_lineup_for_prompt(self, lineup: List[Dict]) -> str:
        """Format lineup for AI prompts"""
        lines = []
        for player in lineup[:6]:  # Limit for token efficiency
            name = player.get('player_name', player.get('name', ''))
            pos = player.get('position', player.get('pos', ''))
            salary = player.get('salary', 0)
            lines.append(f"{pos} {name} (${salary})")
        return "; ".join(lines)
    
    def _determine_strategy(self, game_status: str) -> str:
        """Determine strategy based on game status"""
        status_map = {
            "BEHIND": "ceiling",
            "AHEAD": "floor", 
            "EVEN": "balanced"
        }
        return status_map.get(game_status.upper(), "balanced")
    
    def _parse_swap_suggestions(self, response: str) -> List[Dict]:
        """Parse AI swap suggestions"""
        suggestions = []
        
        for line in response.split('\n'):
            if 'OUT:' in line and 'IN:' in line:
                try:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        out_part = parts[0].replace('OUT:', '').strip()
                        in_part = parts[1].replace('IN:', '').strip()
                        reason = parts[2].replace('REASON:', '').strip()
                        
                        suggestions.append({
                            "player_out": out_part,
                            "player_in": in_part,
                            "reason": reason
                        })
                except Exception:
                    continue
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _parse_weather_response(self, response: str) -> Optional[Dict]:
        """Parse weather impact response"""
        try:
            parts = response.split('|')
            if len(parts) >= 4:
                passing = float(parts[0].replace('PASSING:', '').strip())
                rushing = float(parts[1].replace('RUSHING:', '').strip())
                kicking = float(parts[2].replace('KICKING:', '').strip())
                reason = parts[3].replace('REASON:', '').strip()
                
                return {
                    'passing': max(0.8, min(1.2, passing)),
                    'rushing': max(0.8, min(1.2, rushing)),
                    'kicking': max(0.8, min(1.2, kicking)),
                    'reason': reason
                }
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _parse_game_script_response(self, response: str) -> Optional[Dict]:
        """Parse game script response"""
        try:
            parts = response.split('|')
            if len(parts) >= 4:
                script_type = parts[0].replace('SCRIPT:', '').strip()
                passing_grade = parts[1].replace('PASSING:', '').strip()
                rushing_grade = parts[2].replace('RUSHING:', '').strip()
                reason = parts[3].replace('REASON:', '').strip()
                
                return {
                    'script_type': script_type,
                    'passing_grade': passing_grade,
                    'rushing_grade': rushing_grade,
                    'reason': reason
                }
        except (ValueError, IndexError):
            pass
        
        return None
    
    def _generate_fallback_analysis(self, lineup_data: Dict[str, Any], game_type: str) -> str:
        """Generate basic analysis when AI is unavailable"""
        
        try:
            lineup = lineup_data.get('lineup', [])
            summary = lineup_data.get('summary', {})
            stack_analysis = lineup_data.get('stack_analysis', {})
            simulation = lineup_data.get('simulation_results', {})
            
            # Basic stats
            total_proj = summary.get('total_projection', 0)
            avg_own = summary.get('average_ownership', 0)
            stack_count = stack_analysis.get('stack_count', 0)
            p90 = simulation.get('percentiles', {}).get('90th', 0)
            
            # Generate analysis
            analysis_parts = []
            
            # Correlation
            if stack_count >= 2:
                analysis_parts.append(f"CORRELATION: Strong stack with {stack_analysis.get('qb_name', 'QB')} + {stack_count} receivers provides excellent ceiling potential.")
            elif stack_count == 1:
                analysis_parts.append(f"CORRELATION: Standard single stack with {stack_analysis.get('qb_name', 'QB')} offers solid correlation upside.")
            else:
                analysis_parts.append("CORRELATION: No stacking detected - missing correlation opportunities.")
            
            # Leverage
            if avg_own < 15:
                analysis_parts.append("LEVERAGE: Low-owned lineup provides excellent tournament leverage.")
            elif avg_own > 25:
                analysis_parts.append("LEVERAGE: High-owned lineup may lack differentiation in tournaments.")
            else:
                analysis_parts.append("LEVERAGE: Moderate ownership levels provide balanced exposure.")
            
            # Risk/Ceiling
            if p90 > total_proj * 1.25:
                analysis_parts.append(f"RISK/CEILING: High ceiling potential with 90th percentile at {p90:.1f} points.")
            else:
                analysis_parts.append(f"RISK/CEILING: Conservative ceiling with 90th percentile at {p90:.1f} points.")
            
            # Strategy note
            strategy_note = "Focus on ceiling plays for maximum upside" if game_type == "h2h" else "Balance floor and ceiling for consistency"
            analysis_parts.append(f"STRATEGY: {strategy_note} in {game_type} contests.")
            
            return " ".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating fallback analysis: {e}")
            return "Basic lineup analysis unavailable."
    
    def health_check(self) -> str:
        """Check AI analyzer health"""
        try:
            if self.openai_client or self.anthropic_client:
                return "healthy"
            else:
                return "no_api_keys"
        except Exception as e:
            return f"error: {e}"
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get AI usage and cost summary"""
        return {
            "daily_cost": round(self.daily_cost, 4),
            "calls_today": self.call_count,
            "calls_this_hour": self.calls_this_hour,
            "rate_limit_remaining": max(0, settings.max_ai_calls_per_hour - self.calls_this_hour),
            "last_reset": self.last_reset.isoformat(),
            "openai_available": self.openai_client is not None,
            "anthropic_available": self.anthropic_client is not None
        }
