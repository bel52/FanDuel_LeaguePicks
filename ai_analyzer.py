import openai
import json
from typing import Dict, List, Optional
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from config import config
from models import Player, Lineup
from utils import count_tokens, estimate_cost

logger = logging.getLogger(__name__)

class AIAnalyzer:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.initialize_clients()
        
    def initialize_clients(self):
        """Initialize AI clients if API keys are available"""
        if config.OPENAI_API_KEY:
            openai.api_key = config.OPENAI_API_KEY
            self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            logger.info("OpenAI client initialized")
        
        if config.ANTHROPIC_API_KEY:
            try:
                from anthropic import Anthropic
                self.anthropic_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic library not installed")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_slate(self, players: List[Player], games: List, use_claude: bool = False) -> Dict:
        """Analyze entire slate for optimal strategies"""
        
        if use_claude and self.anthropic_client:
            return await self._analyze_with_claude(players, games)
        elif self.openai_client:
            return await self._analyze_with_gpt(players, games)
        else:
            logger.warning("No AI client available, using rule-based analysis")
            return self._rule_based_analysis(players, games)
    
    async def _analyze_with_gpt(self, players: List[Player], games: List) -> Dict:
        """Analyze using GPT-4"""
        # Prepare data for token efficiency
        player_data = self._prepare_player_data(players[:50])  # Limit for tokens
        game_data = self._prepare_game_data(games)
        
        prompt = f"""You are an expert DFS analyst. Analyze this NFL slate:

GAMES: {json.dumps(game_data, indent=2)}

TOP PLAYERS BY VALUE: {json.dumps(player_data, indent=2)}

Provide analysis in this exact JSON format:
{{
    "game_stacks": [
        {{"game": "TEAM1@TEAM2", "total": 48.5, "correlation_plays": ["QB", "WR1", "WR2", "OPP_WR1"], "reasoning": "..."}}
    ],
    "leverage_plays": [
        {{"player": "Name", "projected_ownership": 5, "upside_scenario": "...", "correlation": "..."}}
    ],
    "fade_candidates": [
        {{"player": "Name", "projected_ownership": 30, "concerns": "..."}}
    ],
    "weather_impacts": [
        {{"game": "TEAM1@TEAM2", "conditions": "...", "player_impacts": {{"QB": -10, "WR": -5}}}}
    ],
    "contrarian_stacks": [
        {{"primary": "QB Name", "stack": ["WR1", "WR2"], "leverage": "Low ownership with high correlation"}}
    ]
}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective model
                messages=[
                    {"role": "system", "content": "You are a DFS expert focused on finding edges in NFL contests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Log cost
            cost = estimate_cost(prompt, response.choices[0].message.content, "gpt-4o-mini")
            logger.info(f"GPT analysis cost: ${cost:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"GPT analysis failed: {e}")
            return self._rule_based_analysis(players, games)
    
    async def _analyze_with_claude(self, players: List[Player], games: List) -> Dict:
        """Analyze using Claude for more complex reasoning"""
        player_data = self._prepare_player_data(players[:100])  # Claude handles more context
        game_data = self._prepare_game_data(games)
        
        prompt = f"""Analyze this NFL DFS slate with focus on tournament-winning strategies.

Games: {json.dumps(game_data, indent=2)}
Players: {json.dumps(player_data, indent=2)}

Provide comprehensive analysis including:
1. Optimal game stacking strategies with correlation coefficients
2. Contrarian plays with <10% ownership but high upside
3. Weather and injury impacts on projections
4. Lineup construction strategies for different contest types

Format as JSON with keys: game_stacks, leverage_plays, fade_candidates, construction_rules"""
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse Claude's response
            content = response.content[0].text
            # Extract JSON from response (Claude might add explanation)
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
        
        return self._rule_based_analysis(players, games)
    
    def _rule_based_analysis(self, players: List[Player], games: List) -> Dict:
        """Fallback rule-based analysis when AI is unavailable"""
        analysis = {
            "game_stacks": [],
            "leverage_plays": [],
            "fade_candidates": [],
            "weather_impacts": [],
            "contrarian_stacks": []
        }
        
        # Identify high-total games for stacking
        for game in games:
            if hasattr(game, 'over_under') and game.over_under >= 47:
                analysis["game_stacks"].append({
                    "game": f"{game.home_team}vs{game.away_team}",
                    "total": game.over_under,
                    "correlation_plays": ["QB", "WR1", "WR2", "OPP_WR1"],
                    "reasoning": "High total game with shootout potential"
                })
        
        # Find value plays
        sorted_players = sorted(players, key=lambda p: p.value, reverse=True)
        for player in sorted_players[:10]:
            if player.ownership_projection and player.ownership_projection < 10:
                analysis["leverage_plays"].append({
                    "player": player.name,
                    "projected_ownership": player.ownership_projection,
                    "upside_scenario": f"High value at {player.value:.2f} pts/$1000"
                })
        
        # Identify chalk to fade
        for player in players:
            if player.ownership_projection and player.ownership_projection > 30:
                analysis["fade_candidates"].append({
                    "player": player.name,
                    "projected_ownership": player.ownership_projection,
                    "concerns": "High ownership reduces tournament leverage"
                })
        
        return analysis
    
    def _prepare_player_data(self, players: List[Player]) -> List[Dict]:
        """Prepare player data for AI analysis"""
        return [
            {
                'name': p.name,
                'position': p.position.value,
                'team': p.team,
                'salary': p.salary,
                'projected': round(p.projected_points, 1),
                'value': round(p.value, 2),
                'ownership': p.ownership_projection
            }
            for p in players
        ]
    
    def _prepare_game_data(self, games: List) -> List[Dict]:
        """Prepare game data for AI analysis"""
        game_list = []
        for game in games:
            if hasattr(game, 'over_under'):
                game_list.append({
                    'matchup': f"{game.away_team}@{game.home_team}",
                    'total': game.over_under,
                    'spread': game.home_spread
                })
        return game_list
    
    async def analyze_lineup(self, lineup: Lineup, contest_type: str = "gpp") -> Dict:
        """Analyze a specific lineup for strengths/weaknesses"""
        if not self.openai_client:
            return {"analysis": "AI analysis unavailable", "score": 7.0}
        
        lineup_str = self._format_lineup(lineup)
        
        prompt = f"""Analyze this DFS lineup for {contest_type} contests:

{lineup_str}

Rate the lineup (1-10) and provide:
1. Strengths
2. Weaknesses  
3. Suggested improvements
4. Correlation score
5. Ownership leverage

Format as JSON with keys: rating, strengths, weaknesses, improvements, correlation_score"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Lineup analysis failed: {e}")
            return {"rating": 7.0, "analysis": "Analysis failed"}
    
    def _format_lineup(self, lineup: Lineup) -> str:
        """Format lineup for display"""
        lines = []
        for player in lineup.players:
            lines.append(f"{player.position.value}: {player.name} ({player.team}) - ${player.salary} - {player.projected_points:.1f} pts")
        
        lines.append(f"\nTotal Salary: ${lineup.total_salary}")
        lines.append(f"Total Projected: {lineup.total_projected:.1f}")
        return "\n".join(lines)

ai_analyzer = AIAnalyzer()
