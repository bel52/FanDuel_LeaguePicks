"""
Simple, working AI analyzer that focuses on basic functionality
"""
import os
import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

class SimpleAIAnalyzer:
    """Simple AI analyzer with basic OpenAI integration"""
    
    def __init__(self):
        self.openai_client = None
        self.api_available = False
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client if API key is available"""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key and api_key != 'your_openai_api_key_here':
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=api_key)
                self.api_available = True
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.api_available = False
        else:
            logger.info("No OpenAI API key found - using fallback analysis")
            self.api_available = False
    
    async def analyze_lineup(self, lineup_data: List[Dict], game_type: str = "league") -> str:
        """Generate AI analysis of the lineup"""
        
        if not self.api_available:
            return self._generate_fallback_analysis(lineup_data, game_type)
        
        try:
            # Create prompt for lineup analysis
            prompt = self._build_lineup_prompt(lineup_data, game_type)
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert DFS analyst. Provide concise, actionable insights for NFL DFS lineups."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content.strip()
            logger.info("AI analysis generated successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._generate_fallback_analysis(lineup_data, game_type)
    
    def _build_lineup_prompt(self, lineup_data: List[Dict], game_type: str) -> str:
        """Build prompt for AI lineup analysis"""
        
        # Extract key lineup info
        lineup_summary = []
        total_salary = 0
        total_projection = 0
        qb_name = ""
        stack_count = 0
        
        for player in lineup_data:
            name = player.get('player_name', '')
            pos = player.get('position', '')
            salary = player.get('salary', 0)
            projection = player.get('projection', 0)
            team = player.get('team', '')
            
            lineup_summary.append(f"{pos} {name} ({team}) - ${salary} - {projection} pts")
            total_salary += salary
            total_projection += projection
            
            if pos == 'QB':
                qb_name = name
                qb_team = team
                # Count stack mates
                stack_count = sum(1 for p in lineup_data 
                                if p.get('team') == qb_team and p.get('position') in ['WR', 'TE'])
        
        lineup_text = "\n".join(lineup_summary[:6])  # First 6 players for token efficiency
        
        prompt = f"""
Analyze this {game_type} DFS lineup for NFL:

{lineup_text}

Total Salary: ${total_salary:,}
Total Projection: {total_projection:.1f} points
Stack: {qb_name} + {stack_count} pass catchers

Provide analysis covering:
1. Correlation strength and stack quality
2. Leverage opportunities (contrarian plays)
3. Risk/ceiling potential
4. Key concerns or weaknesses

Keep response under 200 words with actionable insights.
"""
        
        return prompt
    
    def _generate_fallback_analysis(self, lineup_data: List[Dict], game_type: str) -> str:
        """Generate basic analysis when AI is unavailable"""
        
        try:
            # Basic lineup stats
            total_projection = sum(p.get('projection', 0) for p in lineup_data)
            total_salary = sum(p.get('salary', 0) for p in lineup_data)
            
            # Find QB and stack
            qb = next((p for p in lineup_data if p.get('position') == 'QB'), None)
            if qb:
                qb_team = qb.get('team', '')
                stack_count = sum(1 for p in lineup_data 
                                if p.get('team') == qb_team and p.get('position') in ['WR', 'TE'])
            else:
                qb_team = ''
                stack_count = 0
            
            # High-salary players (potential chalk)
            high_salary_players = [p for p in lineup_data if p.get('salary', 0) > 8000]
            
            # Generate analysis components
            analysis_parts = []
            
            # Correlation analysis
            if stack_count >= 2:
                analysis_parts.append(f"CORRELATION: Strong stack with {qb.get('player_name', 'QB')} + {stack_count} pass catchers provides excellent ceiling correlation.")
            elif stack_count == 1:
                analysis_parts.append(f"CORRELATION: Standard QB stack with {qb.get('player_name', 'QB')} offers solid correlation upside.")
            else:
                analysis_parts.append("CORRELATION: No QB stacking detected - missing correlation opportunities for tournament play.")
            
            # Leverage analysis
            if len(high_salary_players) <= 2:
                analysis_parts.append("LEVERAGE: Roster construction avoids excessive chalk, providing good tournament leverage.")
            else:
                analysis_parts.append("LEVERAGE: High-salary heavy lineup may lack differentiation in tournaments.")
            
            # Risk/Ceiling assessment
            if total_projection >= 140:
                analysis_parts.append(f"RISK/CEILING: High ceiling lineup ({total_projection:.1f} projected) with strong tournament upside.")
            elif total_projection >= 130:
                analysis_parts.append(f"RISK/CEILING: Solid projection ({total_projection:.1f}) balances floor and ceiling effectively.")
            else:
                analysis_parts.append(f"RISK/CEILING: Conservative projection ({total_projection:.1f}) - consider higher upside plays.")
            
            # Strategy note based on game type
            if game_type == "h2h":
                analysis_parts.append("STRATEGY: H2H lineup should prioritize ceiling over floor - look for boom/bust plays.")
            else:
                analysis_parts.append("STRATEGY: Tournament play benefits from correlation and contrarian leverage.")
            
            return " ".join(analysis_parts)
            
        except Exception as e:
            logger.error(f"Error generating fallback analysis: {e}")
            return "Basic lineup analysis: This lineup meets position requirements and salary constraints. Consider correlation and leverage opportunities for tournament play."
    
    def health_check(self) -> str:
        """Check AI analyzer health"""
        if self.api_available:
            return "healthy"
        else:
            return "no_api_key"


# Test function
async def test_ai_analyzer():
    """Test the AI analyzer"""
    
    # Sample lineup data
    sample_lineup = [
        {"player_name": "Josh Allen", "position": "QB", "team": "BUF", "salary": 8500, "projection": 22.5},
        {"player_name": "Stefon Diggs", "position": "WR", "team": "BUF", "salary": 7200, "projection": 16.8},
        {"player_name": "Derrick Henry", "position": "RB", "team": "TEN", "salary": 6800, "projection": 18.2},
        {"player_name": "Christian McCaffrey", "position": "RB", "team": "SF", "salary": 9000, "projection": 20.1},
        {"player_name": "Tyreek Hill", "position": "WR", "team": "MIA", "salary": 7800, "projection": 17.2},
        {"player_name": "Mike Evans", "position": "WR", "team": "TB", "salary": 6900, "projection": 15.9},
        {"player_name": "Travis Kelce", "position": "TE", "team": "KC", "salary": 6500, "projection": 15.3},
        {"player_name": "Buffalo", "position": "DST", "team": "BUF", "salary": 3200, "projection": 9.2}
    ]
    
    analyzer = SimpleAIAnalyzer()
    analysis = await analyzer.analyze_lineup(sample_lineup, "league")
    
    print("✅ AI Analyzer Test:")
    print(f"API Available: {analyzer.api_available}")
    print(f"Analysis: {analysis}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ai_analyzer())
