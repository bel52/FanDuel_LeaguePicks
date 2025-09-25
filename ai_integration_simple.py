"""
Simple AI Integration for DFS - Version Compatible
"""
import os
import json
from typing import List, Dict, Any
from loguru import logger

class SimpleAIAnalyzer:
    """AI analyzer with better error handling"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.enabled = os.getenv('AI_ANALYSIS_ENABLED', 'false').lower() == 'true'
        self.model = "gpt-4o-mini"
        self.client = None
        
        if self.enabled and self.api_key:
            self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI client with version handling"""
        try:
            import openai
            
            # Try new version first
            try:
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized (v1.0+)")
            except Exception:
                # Fallback to older version
                openai.api_key = self.api_key
                self.client = openai
                logger.info("OpenAI client initialized (legacy)")
                
        except ImportError:
            logger.warning("OpenAI library not installed - AI disabled")
            self.enabled = False
    
    def analyze_slate_for_optimization(self, players_data: List[Dict], 
                                     weather_data: Dict, vegas_data: Dict,
                                     contest_type: str = 'gpp') -> Dict[str, Any]:
        """AI analysis with fallback logic"""
        
        if not self.enabled or not self.client:
            logger.info("AI analysis not available - using enhanced heuristics")
            return self._enhanced_heuristic_analysis(players_data, weather_data, vegas_data, contest_type)
        
        try:
            # Prepare concise data summary
            slate_summary = self._create_slate_summary(players_data, weather_data, vegas_data)
            
            # Create analysis prompt
            prompt = f"""Analyze this NFL DFS slate for {contest_type.upper()} strategy:

{slate_summary}

Recommend in 3 lines:
1. STRATEGY: contrarian/balanced/chalky
2. TARGET: 2 specific players with reasoning
3. AVOID: 1 player to fade with reason"""

            # Call OpenAI
            if hasattr(self.client, 'chat'):
                # New version
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                ai_response = response.choices[0].message.content
            else:
                # Legacy version
                response = self.client.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.1
                )
                ai_response = response['choices'][0]['message']['content']
            
            logger.info(f"AI Analysis Complete: {ai_response[:100]}...")
            
            return {
                "strategy": self._extract_strategy(ai_response),
                "analysis": ai_response,
                "ai_enabled": True
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._enhanced_heuristic_analysis(players_data, weather_data, vegas_data, contest_type)
    
    def _create_slate_summary(self, players: List[Dict], weather: Dict, vegas: Dict) -> str:
        """Create concise slate summary"""
        
        # Top players by salary
        top_qbs = sorted([p for p in players if p.get('position') == 'QB'], 
                        key=lambda x: x.get('salary', 0), reverse=True)[:3]
        top_rbs = sorted([p for p in players if p.get('position') == 'RB'], 
                        key=lambda x: x.get('salary', 0), reverse=True)[:3]
        
        # Weather issues
        weather_teams = [t for t, w in weather.items() if w.get('factor', 1.0) < 0.95]
        
        # High totals
        high_totals = [g for g, v in vegas.items() if v.get('total_points', 0) > 47]
        
        summary = f"Top QBs: {[p['name'] for p in top_qbs]}\n"
        summary += f"Top RBs: {[p['name'] for p in top_rbs]}\n"
        summary += f"Weather concerns: {weather_teams}\n"
        summary += f"High total games: {len(high_totals)}"
        
        return summary
    
    def _enhanced_heuristic_analysis(self, players: List[Dict], weather: Dict, 
                                   vegas: Dict, contest_type: str) -> Dict[str, Any]:
        """Enhanced logic-based analysis when AI unavailable"""
        
        # Count high-salary players (likely chalk)
        expensive_players = len([p for p in players if p.get('salary', 0) > 8000])
        
        # Weather game count
        weather_affected = len([t for t, w in weather.items() if w.get('factor', 1.0) < 0.95])
        
        # High total games
        shootout_games = len([g for g, v in vegas.items() if v.get('total_points', 0) > 48])
        
        # Determine strategy
        if contest_type == 'cash':
            strategy = 'balanced'
        elif expensive_players > 15 and weather_affected < 3:
            strategy = 'contrarian'  # Lots of chalk, good weather = fade chalk
        elif shootout_games > 2:
            strategy = 'chalky'  # Multiple shootouts = play obvious
        else:
            strategy = 'balanced'
        
        logger.info(f"Enhanced heuristic analysis: {strategy} strategy")
        
        return {
            "strategy": strategy,
            "analysis": f"Heuristic analysis: {expensive_players} expensive players, {weather_affected} weather games, {shootout_games} shootouts",
            "ai_enabled": False
        }
    
    def _extract_strategy(self, ai_response: str) -> str:
        """Extract strategy from AI response"""
        response_lower = ai_response.lower()
        if 'contrarian' in response_lower:
            return 'contrarian'
        elif 'chalky' in response_lower or 'chalk' in response_lower:
            return 'chalky'
        else:
            return 'balanced'
