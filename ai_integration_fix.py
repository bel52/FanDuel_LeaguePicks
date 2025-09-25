"""
Real AI Integration for DFS Optimization
Uses OpenAI to analyze collected data and make intelligent decisions
"""
import openai
import os
import json
from typing import List, Dict, Any
from loguru import logger

class RealAIAnalyzer:
    """Actually uses OpenAI to analyze DFS data and make decisions"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"  # Cost-effective model
        self.enabled = os.getenv('AI_ANALYSIS_ENABLED', 'false').lower() == 'true'
    
    def analyze_slate_for_optimization(self, players_data: List[Dict], 
                                     weather_data: Dict, vegas_data: Dict,
                                     contest_type: str = 'gpp') -> Dict[str, Any]:
        """Use AI to analyze the slate and provide optimization guidance"""
        
        if not self.enabled:
            logger.info("AI analysis disabled - using basic optimization")
            return {"strategy": "basic", "adjustments": {}}
        
        try:
            # Prepare data summary for AI
            slate_summary = self._prepare_slate_summary(players_data, weather_data, vegas_data)
            
            # Create AI prompt for slate analysis
            prompt = self._create_slate_analysis_prompt(slate_summary, contest_type)
            
            # Get AI analysis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"AI Analysis: {ai_response[:200]}...")
            
            # Parse AI recommendations
            recommendations = self._parse_ai_recommendations(ai_response)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {"strategy": "fallback", "adjustments": {}}
    
    def _prepare_slate_summary(self, players_data: List[Dict], 
                              weather_data: Dict, vegas_data: Dict) -> str:
        """Prepare concise slate summary for AI analysis"""
        
        # Top players by position
        qbs = sorted([p for p in players_data if p.get('position') == 'QB'], 
                    key=lambda x: x.get('salary', 0), reverse=True)[:5]
        rbs = sorted([p for p in players_data if p.get('position') == 'RB'], 
                    key=lambda x: x.get('salary', 0), reverse=True)[:8]
        wrs = sorted([p for p in players_data if p.get('position') == 'WR'], 
                    key=lambda x: x.get('salary', 0), reverse=True)[:10]
        
        # Weather issues
        weather_issues = []
        for team, conditions in weather_data.items():
            if conditions.get('factor', 1.0) < 0.95:
                weather_issues.append(f"{team}: {conditions.get('conditions', 'bad weather')}")
        
        # High total games
        high_total_games = []
        for game, info in vegas_data.items():
            if info.get('total_points', 0) > 47:
                high_total_games.append(f"{game}: {info['total_points']} total")
        
        summary = f"""
SLATE ANALYSIS REQUEST:

TOP SALARIES:
QBs: {', '.join([f"{p.get('name')} ${p.get('salary'):,}" for p in qbs[:3]])}
RBs: {', '.join([f"{p.get('name')} ${p.get('salary'):,}" for p in rbs[:3]])}
WRs: {', '.join([f"{p.get('name')} ${p.get('salary'):,}" for p in wrs[:3]])}

WEATHER CONCERNS: {'; '.join(weather_issues) if weather_issues else 'None'}

HIGH TOTAL GAMES: {'; '.join(high_total_games) if high_total_games else 'None'}

TOTAL PLAYERS: {len(players_data)}
"""
        return summary.strip()
    
    def _create_slate_analysis_prompt(self, slate_summary: str, contest_type: str) -> str:
        """Create AI prompt for slate analysis"""
        
        contest_guidance = {
            'gpp': 'Focus on ceiling, leverage, and differentiation. Identify contrarian plays.',
            'cash': 'Prioritize floor, safety, and consistency. Avoid high-variance plays.',
            'contrarian': 'Find low-owned players with upside. Fade obvious chalk plays.'
        }
        
        guidance = contest_guidance.get(contest_type, contest_guidance['gpp'])
        
        prompt = f"""You are an expert DFS analyst. Analyze this NFL slate for {contest_type.upper()} contests.

{slate_summary}

CONTEST TYPE: {contest_type.upper()}
STRATEGY FOCUS: {guidance}

Provide specific recommendations in this format:
STRATEGY: [contrarian/balanced/chalky]
KEY_PLAYS: [2-3 specific player recommendations with reasoning]
AVOID: [1-2 players to fade and why]
STACK_TARGETS: [1-2 game stacks to target]
WEATHER_IMPACT: [how weather affects strategy]

Keep response under 400 words and be specific about player names and reasoning."""

        return prompt
    
    def _parse_ai_recommendations(self, ai_response: str) -> Dict[str, Any]:
        """Parse AI response into actionable recommendations"""
        
        recommendations = {
            "strategy": "balanced",
            "key_plays": [],
            "avoid_players": [],
            "stack_targets": [],
            "weather_adjustments": {},
            "raw_analysis": ai_response
        }
        
        lines = ai_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('STRATEGY:'):
                strategy_text = line.replace('STRATEGY:', '').strip().lower()
                if 'contrarian' in strategy_text:
                    recommendations['strategy'] = 'contrarian'
                elif 'chalky' in strategy_text:
                    recommendations['strategy'] = 'chalky'
                else:
                    recommendations['strategy'] = 'balanced'
            
            elif line.startswith('KEY_PLAYS:'):
                current_section = 'key_plays'
                plays_text = line.replace('KEY_PLAYS:', '').strip()
                if plays_text:
                    recommendations['key_plays'].append(plays_text)
            
            elif line.startswith('AVOID:'):
                current_section = 'avoid'
                avoid_text = line.replace('AVOID:', '').strip()
                if avoid_text:
                    recommendations['avoid_players'].append(avoid_text)
            
            elif line.startswith('STACK_TARGETS:'):
                current_section = 'stacks'
                stack_text = line.replace('STACK_TARGETS:', '').strip()
                if stack_text:
                    recommendations['stack_targets'].append(stack_text)
            
            elif current_section and line and not line.startswith(('STRATEGY', 'KEY_PLAYS', 'AVOID', 'STACK_TARGETS', 'WEATHER')):
                recommendations[current_section].append(line)
        
        return recommendations

# Create the integration function
def integrate_ai_into_optimizer():
    """Integrate AI analysis into the optimizer"""
    
    # Check if AI integration already exists in optimizer
    with open('optimizer.py', 'r') as f:
        content = f.read()
    
    if 'RealAIAnalyzer' not in content:
        # Add AI integration to optimizer
        ai_import = """
# Real AI Integration
from ai_integration_fix import RealAIAnalyzer
"""
        
        # Add to imports at top of file
        with open('optimizer.py', 'r') as f:
            lines = f.readlines()
        
        # Find import section and add AI import
        for i, line in enumerate(lines):
            if line.startswith('from config import'):
                lines.insert(i+1, ai_import)
                break
        
        # Add AI analysis to optimizer class
        optimizer_enhancement = """
    def __init__(self):
        self.ai_analyzer = RealAIAnalyzer()
"""
        
        # Find EnhancedDFSOptimizer.__init__ and replace
        content = ''.join(lines)
        content = content.replace(
            "def __init__(self):\n        pass",
            optimizer_enhancement.strip()
        )
        
        # Add AI analysis call to optimization
        ai_optimization_call = """
        # Get AI analysis of the slate
        ai_recommendations = self.ai_analyzer.analyze_slate_for_optimization(
            player_data, weather_data, contest_type
        )
        logger.info(f"AI Strategy: {ai_recommendations.get('strategy', 'basic')}")
        
        # Apply AI recommendations to optimization
        if ai_recommendations.get('strategy') == 'contrarian':
            contest_type = 'contrarian'
        elif ai_recommendations.get('strategy') == 'chalky':
            # Use more mainstream plays
            pass
"""
        
        # Add before "players = optimizer.prepare_players"
        content = content.replace(
            "optimizer = EnhancedDFSOptimizer()\n    players = optimizer.prepare_players(player_data, weather_data)",
            f"optimizer = EnhancedDFSOptimizer()\n{ai_optimization_call}\n    players = optimizer.prepare_players(player_data, weather_data)"
        )
        
        # Write updated optimizer
        with open('optimizer.py', 'w') as f:
            f.write(content)
        
        logger.info("✅ AI integration added to optimizer")
    
    return True

if __name__ == "__main__":
    integrate_ai_into_optimizer()
