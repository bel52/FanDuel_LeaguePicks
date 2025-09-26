"""
AI Integration for Strategic DFS Analysis - Final OpenAI Fix
Cost-efficient OpenAI integration for winning lineup strategies
"""
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

# Handle OpenAI import gracefully
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available")
    OPENAI_AVAILABLE = False

@dataclass
class AIAnalysisResult:
    """AI analysis result structure"""
    slate_overview: str
    key_leverage_spots: List[str]
    ownership_insights: Dict[str, float]
    stacking_recommendations: List[Dict]
    contrarian_targets: List[str]
    late_swap_triggers: List[str]
    confidence_score: float
    cost_estimate: float

class DFSAIAnalyzer:
    """Strategic AI analyzer for DFS optimization"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('GPT_MODEL', 'gpt-4o-mini')
        self.weekly_budget = float(os.getenv('AI_WEEKLY_BUDGET', '15.0'))
        self.weekly_spend = 0.0
        self.cost_log = []
        
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.client = OpenAI(api_key=self.openai_api_key)
                logger.info(f"AI Analyzer initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
                self.client = None
        else:
            logger.warning("OpenAI not available - AI analysis disabled")
            self.client = None
    
    def analyze_slate_for_optimization(self, player_data: List[Dict], 
                                     weather_data: Dict, 
                                     vegas_data: Dict,
                                     contest_type: str = 'gpp') -> Dict[str, Any]:
        """Main AI analysis for slate optimization"""
        
        if not self.client:
            return self._fallback_analysis(player_data, contest_type)
        
        if self.weekly_spend >= self.weekly_budget:
            logger.warning(f"AI weekly budget exceeded: ${self.weekly_spend:.2f}")
            return self._fallback_analysis(player_data, contest_type)
        
        try:
            # Prepare data for AI analysis
            slate_summary = self._prepare_slate_summary(player_data, weather_data, vegas_data)
            
            # Generate AI prompt based on contest type
            prompt = self._build_strategic_prompt(slate_summary, contest_type)
            
            # Make API call using new v1.0+ syntax
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # Cost control
                temperature=0.1  # Consistent analysis
            )
            
            # Log cost
            estimated_cost = self._estimate_cost(len(prompt), 800)
            self._log_api_cost(estimated_cost)
            
            # Parse AI response
            ai_text = response.choices[0].message.content
            analysis = self._parse_ai_response(ai_text)
            
            logger.info(f"AI Analysis complete - Cost: ${estimated_cost:.3f} - Weekly: ${self.weekly_spend:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(player_data, contest_type)
    
    def _prepare_slate_summary(self, players: List[Dict], weather: Dict, vegas: Dict) -> Dict:
        """Prepare concise data summary for AI"""
        
        # Top players by position and salary
        by_position = {}
        for player in players[:50]:  # Limit for token efficiency
            pos = player.get('position', '')
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(player)
        
        # Sort by salary and take top 3 per position
        top_players = {}
        for pos, pos_players in by_position.items():
            sorted_players = sorted(pos_players, key=lambda x: x.get('salary', 0), reverse=True)
            top_players[pos] = sorted_players[:2]  # Reduced for efficiency
        
        return {
            'top_players': top_players,
            'weather_impacts': {k: v for k, v in weather.items() if v.get('factor', 1.0) != 1.0},
            'total_players': len(players)
        }
    
    def _build_strategic_prompt(self, slate_summary: Dict, contest_type: str) -> str:
        """Build concise strategic analysis prompt"""
        
        prompt = f"NFL DFS {contest_type.upper()} Analysis:\n\nTOP PLAYERS:\n"
        
        for pos, players in slate_summary['top_players'].items():
            prompt += f"{pos}: "
            player_list = []
            for player in players:
                name = player.get('name', '')[:15]  # Truncate for efficiency
                salary = player.get('salary', 0)
                team = player.get('team', '')
                player_list.append(f"{name}({team})${salary//1000}K")
            prompt += ", ".join(player_list) + "\n"
        
        # Contest-specific request
        if contest_type == 'gpp':
            prompt += "\nProvide JSON: {\"leverage_spots\":[\"player names\"], \"stacking_recs\":[{\"qb\":\"name\",\"targets\":[\"wr1\"]}], \"strategy\":\"brief approach\"}"
        elif contest_type == 'contrarian':
            prompt += "\nProvide JSON: {\"contrarian_targets\":[\"low owned players\"], \"fade_chalk\":[\"high owned\"], \"strategy\":\"brief approach\"}"
        else:
            prompt += "\nProvide JSON: {\"safe_plays\":[\"consistent players\"], \"strategy\":\"brief approach\"}"
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Concise system prompt"""
        return "DFS expert. Focus on tournament strategy: stacking creates ceiling, ownership leverage beats projections, weather/injuries = opportunity. Respond only in requested JSON format."
    
    def _parse_ai_response(self, ai_text: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
        try:
            json_start = ai_text.find('{')
            json_end = ai_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = ai_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                return {
                    'strategy': parsed.get('strategy', 'AI analysis complete'),
                    'leverage_spots': parsed.get('leverage_spots', []),
                    'stacking_recommendations': parsed.get('stacking_recs', []),
                    'contrarian_targets': parsed.get('contrarian_targets', []),
                    'safe_plays': parsed.get('safe_plays', []),
                    'ai_confidence': 0.8
                }
            else:
                return self._fallback_analysis([], 'gpp')
                
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._fallback_analysis([], 'gpp')
    
    def _fallback_analysis(self, players: List[Dict], contest_type: str) -> Dict[str, Any]:
        """Fallback analysis when AI unavailable"""
        
        strategies = {
            'gpp': "Tournament: Target ceiling plays and unique stacks",
            'cash': "Cash: Focus on floor plays and consistency",
            'contrarian': "Contrarian: Fade chalk, find leverage"
        }
        
        return {
            'strategy': strategies.get(contest_type, 'Standard optimization'),
            'leverage_spots': [],
            'stacking_recommendations': [],
            'contrarian_targets': [],
            'safe_plays': [],
            'ai_confidence': 0.1
        }
    
    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate API cost"""
        input_cost = (prompt_tokens / 1000) * 0.00015
        output_cost = (completion_tokens / 1000) * 0.0006
        return input_cost + output_cost
    
    def _log_api_cost(self, cost: float):
        """Log API cost for budget tracking"""
        self.weekly_spend += cost
        self.cost_log.append({
            'timestamp': datetime.now().isoformat(),
            'cost': cost,
            'weekly_total': self.weekly_spend
        })
        
        if self.weekly_spend > self.weekly_budget * 0.8:
            logger.warning(f"AI spending at {self.weekly_spend/self.weekly_budget*100:.1f}% of weekly budget")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return {
            'weekly_spend': self.weekly_spend,
            'weekly_budget': self.weekly_budget,
            'remaining_budget': self.weekly_budget - self.weekly_spend,
            'calls_made': len(self.cost_log),
            'avg_cost_per_call': self.weekly_spend / len(self.cost_log) if self.cost_log else 0
        }
    
    def reset_weekly_spend(self):
        """Reset weekly spending counter"""
        self.weekly_spend = 0.0
        self.cost_log = []
        logger.info("Weekly AI spending counter reset")
