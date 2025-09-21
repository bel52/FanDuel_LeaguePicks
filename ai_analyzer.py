"""
AI-powered analysis using OpenAI GPT-4o-mini
"""
import openai
from openai import OpenAI
import json
from typing import Dict, List, Optional
import tiktoken
from functools import lru_cache
import asyncio
from loguru import logger
from config import config
import numpy as np

class DFSAIAnalyzer:
    """AI-powered DFS analysis using GPT-4o-mini"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.AI_MODEL
        self.encoding = tiktoken.encoding_for_model("gpt-4-0613")  # Use GPT-4 encoding
        self.max_tokens = config.MAX_TOKENS
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int = 500) -> float:
        """Estimate API cost in dollars"""
        # GPT-4o-mini pricing: $0.15 per 1M input, $0.60 per 1M output
        input_cost = (input_tokens / 1_000_000) * 0.15
        output_cost = (output_tokens / 1_000_000) * 0.60
        return input_cost + output_cost
    
    @lru_cache(maxsize=100)
    def analyze_slate(self, slate_data: str) -> Dict:
        """
        Analyze entire slate for key insights
        
        Args:
            slate_data: JSON string of slate information
            
        Returns:
            Dictionary with analysis results
        """
        try:
            logger.info("Running AI slate analysis")
            
            prompt = f"""You are an expert NFL DFS analyst. Analyze this slate data and provide strategic insights.

SLATE DATA:
{slate_data[:3000]}  # Limit for token efficiency

PROVIDE ANALYSIS IN THIS EXACT JSON FORMAT:
{{
    "top_plays": [
        {{"player": "Name", "position": "POS", "reasoning": "Why this player is a top play", "confidence": 0.0-1.0}}
    ],
    "contrarian_plays": [
        {{"player": "Name", "expected_ownership": "X%", "reasoning": "Why this provides leverage"}}
    ],
    "game_stacks": [
        {{"game": "TEAM @ TEAM", "stack": ["QB Name", "WR1 Name", "WR2 Name"], "correlation_score": 0.0-1.0}}
    ],
    "avoid_players": [
        {{"player": "Name", "reasoning": "Why to avoid"}}
    ],
    "weather_concerns": [
        {{"game": "TEAM @ TEAM", "impact": "Description of weather impact"}}
    ],
    "injury_pivots": [
        {{"out_player": "Name", "pivot_to": "Name", "reasoning": "Why this pivot works"}}
    ]
}}

Focus on:
1. Value plays under $6000 with high upside
2. Correlation plays for GPP tournaments
3. Contrarian angles for low ownership
4. Weather and injury impacts
"""
            
            # Check token count and cost
            input_tokens = self.count_tokens(prompt)
            estimated_cost = self.estimate_cost(input_tokens)
            logger.info(f"AI analysis cost estimate: ${estimated_cost:.4f}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional DFS analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info("AI analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._get_fallback_analysis()
    
    def analyze_lineup(self, lineup: List[Dict], context: Dict) -> Dict:
        """
        Analyze a specific lineup for strengths and weaknesses
        
        Args:
            lineup: List of players in lineup
            context: Additional context (opponents, weather, etc)
            
        Returns:
            Analysis of lineup quality
        """
        try:
            lineup_str = "\n".join([f"{p['position']}: {p['name']} (${p['salary']})" for p in lineup])
            
            prompt = f"""Analyze this DFS lineup for quality and potential issues:

LINEUP:
{lineup_str}

CONTEXT:
- Total Salary Used: ${sum(p['salary'] for p in lineup)}
- Average Ownership: {context.get('avg_ownership', 'Unknown')}
- Weather Concerns: {context.get('weather', 'None')}

Provide a brief analysis covering:
1. Lineup correlation strength (1-10)
2. GPP viability (1-10)
3. Cash game safety (1-10)
4. Key risks
5. Suggested improvements

Format as JSON with scores and text analysis."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Lineup analysis failed: {e}")
            return {"error": str(e)}
    
    def get_ownership_projections(self, players: List[Dict]) -> Dict[str, float]:
        """
        Project ownership percentages using AI analysis
        
        Args:
            players: List of player data
            
        Returns:
            Dictionary of player names to ownership projections
        """
        try:
            # Prepare concise player data
            player_data = []
            for p in players[:50]:  # Limit to top 50 for token efficiency
                player_data.append({
                    'name': p['name'],
                    'salary': p['salary'],
                    'projection': p.get('projection', 0),
                    'value': p.get('value', 0)
                })
            
            prompt = f"""Project DFS ownership percentages for these players in GPP tournaments.

PLAYERS:
{json.dumps(player_data, indent=2)}

Consider:
- Salary and value
- Recent performance
- Narrative/chalk factors
- Position scarcity

Return JSON with {{"PlayerName": ownership_percentage}} for each player.
Ownership should be between 0.5 and 50.0."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=800,
                response_format={"type": "json_object"}
            )
            
            ownership = json.loads(response.choices[0].message.content)
            
            # Validate ownership values
            validated = {}
            for player, own in ownership.items():
                if isinstance(own, (int, float)):
                    validated[player] = max(0.5, min(50.0, float(own)))
            
            return validated
            
        except Exception as e:
            logger.error(f"Ownership projection failed: {e}")
            return {}
    
    def analyze_late_swap(self, current_player: Dict, alternatives: List[Dict], 
                         game_context: Dict) -> Dict:
        """
        Analyze late swap decisions
        
        Args:
            current_player: Currently rostered player
            alternatives: List of swap alternatives
            game_context: Current game situation
            
        Returns:
            Swap recommendation
        """
        try:
            prompt = f"""Evaluate this late swap decision for DFS:

CURRENT PLAYER:
{current_player['name']} - {current_player['position']} (${current_player['salary']})
Status: {current_player.get('status', 'Active')}
Projected: {current_player.get('projection', 0)} pts

ALTERNATIVES:
{json.dumps(alternatives[:5], indent=2)}

CONTEXT:
- Minutes until lock: {game_context.get('minutes_to_lock', 0)}
- Current lineup position: {game_context.get('tournament_position', 'Unknown')}
- News: {game_context.get('latest_news', 'None')}

Should we make a swap? If yes, to whom and why?
Return JSON with {{"swap": true/false, "target": "PlayerName or null", "reasoning": "explanation"}}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for critical decisions
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Late swap analysis failed: {e}")
            return {"swap": False, "reasoning": "Analysis failed"}
    
    def _get_fallback_analysis(self) -> Dict:
        """Fallback analysis if AI fails"""
        return {
            "top_plays": [],
            "contrarian_plays": [],
            "game_stacks": [],
            "avoid_players": [],
            "weather_concerns": [],
            "injury_pivots": []
        }


class CorrelationAnalyzer:
    """Analyze player correlations for stacking"""
    
    def __init__(self):
        self.correlation_matrix = self._initialize_correlations()
    
    def _initialize_correlations(self) -> Dict:
        """Initialize base correlation values"""
        return {
            'QB-WR1': 0.62,
            'QB-WR2': 0.48,
            'QB-WR3': 0.35,
            'QB-TE': 0.32,
            'QB-RB': 0.08,
            'QB-OPP_WR1': 0.25,
            'QB-OPP_QB': 0.31,
            'QB-DST': -0.41,
            'RB-DST': 0.25,
            'WR1-WR2': 0.12,
            'WR1-TE': 0.08
        }
    
    def calculate_lineup_correlation(self, lineup: List[Dict]) -> float:
        """
        Calculate total correlation score for a lineup
        
        Args:
            lineup: List of players in lineup
            
        Returns:
            Correlation score (0-1)
        """
        total_correlation = 0.0
        correlation_pairs = 0
        
        # Find QB
        qb = next((p for p in lineup if p['position'] == 'QB'), None)
        if not qb:
            return 0.0
        
        qb_team = qb['team']
        
        # Check correlations with other players
        for player in lineup:
            if player['name'] == qb['name']:
                continue
            
            # Same team correlation
            if player['team'] == qb_team:
                if player['position'] == 'WR':
                    total_correlation += self.correlation_matrix.get('QB-WR1', 0.5)
                    correlation_pairs += 1
                elif player['position'] == 'TE':
                    total_correlation += self.correlation_matrix.get('QB-TE', 0.32)
                    correlation_pairs += 1
                elif player['position'] == 'RB':
                    total_correlation += self.correlation_matrix.get('QB-RB', 0.08)
                    correlation_pairs += 1
            
            # Opposing team correlation (game stack)
            elif player.get('opponent') == qb_team:
                if player['position'] == 'WR':
                    total_correlation += self.correlation_matrix.get('QB-OPP_WR1', 0.25)
                    correlation_pairs += 1
                elif player['position'] == 'QB':
                    total_correlation += self.correlation_matrix.get('QB-OPP_QB', 0.31)
                    correlation_pairs += 1
        
        if correlation_pairs > 0:
            return min(1.0, total_correlation / correlation_pairs)
        return 0.0
    
    def find_optimal_stacks(self, players: List[Dict], num_stacks: int = 5) -> List[Dict]:
        """
        Find optimal stacking combinations
        
        Args:
            players: List of all players
            num_stacks: Number of stacks to return
            
        Returns:
            List of optimal stack combinations
        """
        stacks = []
        
        # Group players by team
        teams = {}
        for player in players:
            team = player.get('team')
            if team not in teams:
                teams[team] = {'QB': [], 'WR': [], 'TE': [], 'RB': []}
            
            position = player.get('position', '')
            if position in teams[team]:
                teams[team][position].append(player)
        
        # Find QB-based stacks
        for team, positions in teams.items():
            if not positions['QB']:
                continue
            
            qb = positions['QB'][0]  # Take best QB
            
            # Find best stack combinations
            for wr1 in positions['WR'][:2]:  # Top 2 WRs
                for wr2 in positions['WR']:
                    if wr2['name'] == wr1['name']:
                        continue
                    
                    stack = {
                        'type': 'QB_DOUBLE_STACK',
                        'team': team,
                        'players': [qb, wr1, wr2],
                        'total_salary': qb['salary'] + wr1['salary'] + wr2['salary'],
                        'correlation_score': 0.55,  # Average of correlations
                        'projected_points': qb.get('projection', 0) + 
                                          wr1.get('projection', 0) + 
                                          wr2.get('projection', 0)
                    }
                    stacks.append(stack)
        
        # Sort by projected points per dollar
        stacks.sort(key=lambda x: x['projected_points'] / max(x['total_salary'], 1), 
                   reverse=True)
        
        return stacks[:num_stacks]
