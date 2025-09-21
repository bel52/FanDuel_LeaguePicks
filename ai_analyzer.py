"""
AI-powered analysis using OpenAI GPT-4o-mini and Claude
"""
import openai
from openai import OpenAI
import anthropic
from anthropic import Anthropic
import json
from typing import Dict, List, Optional, Literal
import tiktoken
from functools import lru_cache
import asyncio
from loguru import logger
from config import config
import numpy as np

class DFSAIAnalyzer:
    """AI-powered DFS analysis using GPT-4o-mini and Claude"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.claude_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self.model_preference = config.AI_MODEL_PREFERENCE
        self.weekly_budget = config.AI_WEEKLY_BUDGET
        self.weekly_spend = {'openai': 0.0, 'claude': 0.0}
        self.encoding = tiktoken.encoding_for_model("gpt-4-0613")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int = 500, 
                     model: str = "openai") -> float:
        """Estimate API cost in dollars"""
        if model == "openai":
            # GPT-4o-mini pricing
            input_cost = (input_tokens / 1_000_000) * 0.15
            output_cost = (output_tokens / 1_000_000) * 0.60
        else:  # claude
            # Claude Sonnet pricing
            input_cost = (input_tokens / 1_000_000) * 3.00
            output_cost = (output_tokens / 1_000_000) * 15.00
        
        return input_cost + output_cost
    
    def select_model_for_task(self, task_type: str) -> str:
        """Select best model based on task and budget"""
        remaining_budget = self.weekly_budget - sum(self.weekly_spend.values())
        
        # Task complexity mapping
        complex_tasks = ['slate_analysis', 'correlation_analysis', 'tournament_strategy']
        simple_tasks = ['player_news', 'injury_impact', 'weather_adjustment']
        
        if self.model_preference == 'both':
            # Use Claude for complex tasks if budget allows
            if task_type in complex_tasks and remaining_budget > 0.50:
                return 'claude'
            else:
                return 'openai'
        
        return self.model_preference
    
    @lru_cache(maxsize=100)
    def analyze_slate(self, slate_data: str) -> Dict:
        """
        Analyze entire slate for key insights using best AI model
        """
        try:
            model = self.select_model_for_task('slate_analysis')
            logger.info(f"Running AI slate analysis with {model}")
            
            if model == 'claude':
                return self._analyze_slate_claude(slate_data)
            else:
                return self._analyze_slate_openai(slate_data)
                
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._get_fallback_analysis()
    
    def _analyze_slate_openai(self, slate_data: str) -> Dict:
        """Analyze slate using OpenAI GPT-4o-mini"""
        prompt = f"""You are an expert NFL DFS analyst. Analyze this slate data and provide strategic insights.

SLATE DATA:
{slate_data[:3000]}

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
}}"""
        
        input_tokens = self.count_tokens(prompt)
        estimated_cost = self.estimate_cost(input_tokens, 1500, 'openai')
        self.weekly_spend['openai'] += estimated_cost
        logger.info(f"OpenAI cost: ${estimated_cost:.4f} (Week total: ${self.weekly_spend['openai']:.2f})")
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional DFS analyst. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _analyze_slate_claude(self, slate_data: str) -> Dict:
        """Analyze slate using Claude for superior analysis"""
        prompt = f"""You are an expert NFL DFS analyst with deep knowledge of game theory, correlation modeling, and tournament strategy.

Analyze this slate data with focus on:
1. Ownership leverage opportunities
2. Correlation-based stacking beyond obvious QB-WR
3. Game environment and pace factors
4. Injury/news-based pivots that create value

SLATE DATA:
{slate_data[:5000]}  # Claude handles more context

Provide a comprehensive analysis in JSON format with these sections:
- top_plays: High confidence plays with detailed reasoning
- contrarian_plays: Low ownership, high upside targets
- game_stacks: Creative stacking opportunities with correlation scores
- avoid_players: Overpriced or risky plays to fade
- weather_concerns: Specific impacts on game totals and player types
- injury_pivots: Direct replacements with similar roles
- hidden_correlations: Non-obvious correlations (RB-DEF, etc)
- tournament_construction: Optimal lineup building strategy

Be specific with percentages, point projections, and correlation coefficients."""
        
        input_tokens = self.count_tokens(prompt)
        estimated_cost = self.estimate_cost(input_tokens, 2000, 'claude')
        self.weekly_spend['claude'] += estimated_cost
        logger.info(f"Claude cost: ${estimated_cost:.4f} (Week total: ${self.weekly_spend['claude']:.2f})")
        
        response = self.claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Extract JSON from Claude's response
        content = response.content[0].text
        
        # Claude might wrap JSON in markdown, extract it
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON object from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            raise
    
    def analyze_lineup(self, lineup: List[Dict], context: Dict) -> Dict:
        """Analyze lineup using appropriate AI model"""
        model = self.select_model_for_task('lineup_analysis')
        
        lineup_str = "\n".join([f"{p['position']}: {p['name']} (${p['salary']})" for p in lineup])
        
        if model == 'claude':
            prompt = f"""Analyze this DFS lineup with advanced game theory perspective:

LINEUP:
{lineup_str}

Total Salary: ${sum(p['salary'] for p in lineup)}
Context: {json.dumps(context, indent=2)}

Provide analysis covering:
1. Correlation strength (0-10) with detailed breakdown
2. Ownership leverage score (0-10)
3. Ceiling probability for GPP
4. Floor probability for cash
5. Specific weaknesses and suggested swaps
6. Expected tournament EV

Format as JSON with numerical scores and text explanations."""
            
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            content = response.content[0].text
            try:
                return json.loads(content)
            except:
                return {"analysis": content}
        
        else:
            # Use OpenAI version (existing code)
            prompt = f"""Analyze this DFS lineup:
LINEUP:
{lineup_str}

Provide scores (1-10) for correlation, GPP viability, and cash safety.
Format as JSON."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
    
    def get_ownership_projections(self, players: List[Dict]) -> Dict[str, float]:
        """Project ownership using both AI models for consensus"""
        try:
            ownership_predictions = {}
            
            # Get predictions from both models if budget allows
            if self.model_preference == 'both' and sum(self.weekly_spend.values()) < self.weekly_budget - 0.20:
                # Get OpenAI prediction
                openai_ownership = self._get_ownership_openai(players)
                
                # Get Claude prediction
                claude_ownership = self._get_ownership_claude(players)
                
                # Average the predictions
                all_players = set(openai_ownership.keys()) | set(claude_ownership.keys())
                for player in all_players:
                    openai_val = openai_ownership.get(player, 0)
                    claude_val = claude_ownership.get(player, 0)
                    
                    if openai_val and claude_val:
                        ownership_predictions[player] = (openai_val + claude_val) / 2
                    else:
                        ownership_predictions[player] = openai_val or claude_val
            
            else:
                # Use single model
                if self.select_model_for_task('ownership_projection') == 'claude':
                    ownership_predictions = self._get_ownership_claude(players)
                else:
                    ownership_predictions = self._get_ownership_openai(players)
            
            return ownership_predictions
            
        except Exception as e:
            logger.error(f"Ownership projection failed: {e}")
            return {}
    
    def _get_ownership_openai(self, players: List[Dict]) -> Dict[str, float]:
        """Get ownership projections from OpenAI"""
        player_data = [
            {'name': p['name'], 'salary': p['salary'], 'projection': p.get('projection', 0)}
            for p in players[:40]
        ]
        
        prompt = f"""Project GPP ownership percentages for these NFL DFS players.
Consider salary, value, recent performance, and narrative.

PLAYERS:
{json.dumps(player_data, indent=2)}

Return JSON with {{"PlayerName": ownership_percentage}} for each player.
Ownership should be 0.5-50.0."""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        ownership = json.loads(response.choices[0].message.content)
        return {k: max(0.5, min(50.0, float(v))) for k, v in ownership.items() if isinstance(v, (int, float))}
    
    def _get_ownership_claude(self, players: List[Dict]) -> Dict[str, float]:
        """Get ownership projections from Claude"""
        player_data = [
            {'name': p['name'], 'salary': p['salary'], 'projection': p.get('projection', 0), 'team': p.get('team', '')}
            for p in players[:50]
        ]
        
        prompt = f"""Project tournament ownership percentages with game theory considerations.

PLAYERS:
{json.dumps(player_data, indent=2)}

Consider:
1. Recency bias impact on casual players
2. Narrative street vs sharp money divergence  
3. Salary psychology breakpoints ($8k, $10k)
4. Stacking implications on correlated ownership

Return precise ownership projections as JSON: {{"PlayerName": percentage}}"""
        
        response = self.claude_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        content = response.content[0].text
        
        # Extract JSON
        try:
            if '```json' in content:
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            
            ownership = json.loads(content)
            return {k: max(0.5, min(50.0, float(v))) for k, v in ownership.items() if isinstance(v, (int, float))}
        except:
            return {}
    
    def analyze_late_swap(self, current_player: Dict, alternatives: List[Dict], 
                         game_context: Dict) -> Dict:
"""Analyze late swap decisions using best AI model"""
        model = self.select_model_for_task('late_swap')
        
        if model == 'claude':
            prompt = f"""Expert DFS late swap decision analysis required.

CURRENT PLAYER:
{json.dumps(current_player, indent=2)}

ALTERNATIVES:
{json.dumps(alternatives[:5], indent=2)}

GAME CONTEXT:
{json.dumps(game_context, indent=2)}

Consider:
1. Injury news impact on snap count/target share
2. Game flow implications from live scores
3. Ownership leverage in tournament context
4. Correlation impacts on other lineup players

Return JSON: {{"swap": true/false, "target": "PlayerName or null", "confidence": 0.0-1.0, "reasoning": "detailed explanation", "expected_value_change": float}}"""
            
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # Very low for critical decisions
            )
            
            content = response.content[0].text
            try:
                if '```json' in content:
                    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                return json.loads(content)
            except:
                return {"swap": False, "reasoning": "Analysis failed"}
        
        else:
            # OpenAI version (more cost-effective)
            prompt = f"""Evaluate late swap: Should we replace {current_player['name']} ({current_player.get('status', 'Active')})?

Top alternatives: {[alt['name'] for alt in alternatives[:3]]}
Context: {game_context.get('latest_news', 'None')}

Return JSON: {{"swap": true/false, "target": "Name", "reasoning": "why"}}"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
    
    def get_correlation_insights(self, game_data: Dict) -> Dict:
        """Get advanced correlation insights from Claude"""
        if self.select_model_for_task('correlation_analysis') != 'claude':
            return {}
        
        try:
            prompt = f"""Analyze advanced correlations for DFS beyond standard stacking.

GAME DATA:
{json.dumps(game_data, indent=2)}

Identify:
1. Non-obvious positive correlations (example: RB1 with opposing DST in blowouts)
2. Negative correlations to avoid (example: two pass-catching RBs from same team)
3. Game script dependent correlations
4. Weather/venue specific correlations

Provide specific correlation coefficients and reasoning.
Format as JSON with correlation pairs and values."""
            
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.content[0].text
            
            # Extract JSON
            if '```json' in content:
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {}
    
    def _get_fallback_analysis(self) -> Dict:
        """Fallback analysis if AI fails"""
        return {
            "top_plays": [],
            "contrarian_plays": [],
            "game_stacks": [],
            "avoid_players": [],
            "weather_concerns": [],
            "injury_pivots": [],
            "hidden_correlations": []
        }
    
    def get_weekly_spend_report(self) -> Dict:
        """Get weekly AI spending report"""
        total_spend = sum(self.weekly_spend.values())
        remaining = self.weekly_budget - total_spend
        
        return {
            'openai_spend': self.weekly_spend['openai'],
            'claude_spend': self.weekly_spend['claude'],
            'total_spend': total_spend,
            'budget': self.weekly_budget,
            'remaining': remaining,
            'percentage_used': (total_spend / self.weekly_budget * 100) if self.weekly_budget > 0 else 0
        }


class CorrelationAnalyzer:
    """Analyze player correlations for stacking with AI enhancement"""
    
    def __init__(self):
        self.correlation_matrix = self._initialize_correlations()
        self.ai_analyzer = DFSAIAnalyzer()
    
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
            'QB-OPP_RB': 0.18,
            'QB-DST': -0.41,
            'RB-DST': 0.25,
            'WR1-WR2': 0.12,
            'WR1-TE': 0.08,
            'RB-OPP_DST': 0.22,  # Game script correlation
            'K-DST': 0.35,  # Same team kicker-defense
        }
    
    def get_enhanced_correlations(self, game_data: Dict) -> Dict:
        """Get AI-enhanced correlation insights"""
        # Get base correlations
        correlations = self.correlation_matrix.copy()
        
        # Enhance with AI insights if available
        ai_correlations = self.ai_analyzer.get_correlation_insights(game_data)
        
        if ai_correlations:
            # Merge AI insights with base correlations
            for pair, value in ai_correlations.items():
                if isinstance(value, (int, float)):
                    correlations[pair] = value
        
        return correlations
    
    def calculate_lineup_correlation(self, lineup: List[Dict]) -> float:
        """Calculate total correlation score for a lineup"""
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
                elif player['position'] == 'K':
                    total_correlation += 0.15  # Positive but weak
                    correlation_pairs += 1
            
            # Opposing team correlation (game stack)
            elif player.get('opponent') == qb_team:
                if player['position'] == 'WR':
                    total_correlation += self.correlation_matrix.get('QB-OPP_WR1', 0.25)
                    correlation_pairs += 1
                elif player['position'] == 'QB':
                    total_correlation += self.correlation_matrix.get('QB-OPP_QB', 0.31)
                    correlation_pairs += 1
                elif player['position'] == 'RB':
                    total_correlation += self.correlation_matrix.get('QB-OPP_RB', 0.18)
                    correlation_pairs += 1
        
        # Check RB-DST correlation
        rb = next((p for p in lineup if p['position'] == 'RB'), None)
        dst = next((p for p in lineup if p['position'] == 'DST'), None)
        
        if rb and dst and rb['team'] == dst['team']:
            total_correlation += self.correlation_matrix.get('RB-DST', 0.25)
            correlation_pairs += 1
        
        if correlation_pairs > 0:
            return min(1.0, total_correlation / correlation_pairs)
        return 0.0
    
    def find_optimal_stacks(self, players: List[Dict], num_stacks: int = 5) -> List[Dict]:
        """Find optimal stacking combinations with AI enhancement"""
        stacks = []
        
        # Group players by team
        teams = {}
        for player in players:
            team = player.get('team')
            if team not in teams:
                teams[team] = {'QB': [], 'WR': [], 'TE': [], 'RB': [], 'K': [], 'DST': []}
            
            position = player.get('position', '')
            if position in teams[team]:
                teams[team][position].append(player)
        
        # Find QB-based stacks
        for team, positions in teams.items():
            if not positions['QB']:
                continue
            
            qb = positions['QB'][0]
            
            # Traditional double stack
            for wr1 in positions['WR'][:2]:
                for wr2 in positions['WR']:
                    if wr2['name'] == wr1['name']:
                        continue
                    
                    stack = {
                        'type': 'QB_DOUBLE_STACK',
                        'team': team,
                        'players': [qb, wr1, wr2],
                        'total_salary': qb['salary'] + wr1['salary'] + wr2['salary'],
                        'correlation_score': 0.55,
                        'projected_points': sum(p.get('projection', 0) for p in [qb, wr1, wr2])
                    }
                    stacks.append(stack)
            
            # QB-WR-TE stack
            if positions['TE']:
                for wr in positions['WR'][:2]:
                    te = positions['TE'][0]
                    stack = {
                        'type': 'QB_WR_TE',
                        'team': team,
                        'players': [qb, wr, te],
                        'total_salary': qb['salary'] + wr['salary'] + te['salary'],
                        'correlation_score': 0.47,
                        'projected_points': sum(p.get('projection', 0) for p in [qb, wr, te])
                    }
                    stacks.append(stack)
        
        # Sort by efficiency (points per dollar with correlation boost)
        for stack in stacks:
            stack['efficiency'] = (
                stack['projected_points'] * (1 + stack['correlation_score'] * 0.2) / 
                max(stack['total_salary'], 1)
            )
        
        stacks.sort(key=lambda x: x['efficiency'], reverse=True)
        
        return stacks[:num_stacks]

