#!/usr/bin/env python3
"""
Cost-Effective AI Integration for DFS Analysis
Uses ChatGPT-4o-mini for $0.10/week vs $15 budget
"""

import asyncio
import json
import logging
import tiktoken
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import openai
from openai import AsyncOpenAI
from functools import lru_cache

logger = logging.getLogger(__name__)

class AIAnalyzer:
    """
    Cost-optimized AI analysis using ChatGPT-4o-mini
    Estimated cost: $0.10/week vs $2.09 for Claude
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        self.model = "gpt-4o-mini"
        
        # Cost tracking
        self.daily_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'calls': 0,
            'cost': 0.0
        }
        
        # Pricing (per million tokens)
        self.pricing = {
            'input': 0.15 / 1000000,   # $0.15 per 1M input tokens
            'output': 0.60 / 1000000   # $0.60 per 1M output tokens
        }
        
        # Token counter
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Cache for repeated analyses
        self.analysis_cache = {}
    
    async def enhance_projections(
        self, 
        players: List[Dict[str, Any]], 
        game_type: str = "gpp"
    ) -> List[Dict[str, Any]]:
        """
        AI-enhanced player projections with cost optimization
        """
        if not self.client:
            logger.info("🤖 AI analysis disabled - no API key")
            return players
        
        try:
            # Group players for batch analysis (cost optimization)
            enhanced_players = []
            
            # Process in batches of 10 for efficiency
            batch_size = 10
            for i in range(0, len(players), batch_size):
                batch = players[i:i + batch_size]
                enhanced_batch = await self._analyze_player_batch(batch, game_type)
                enhanced_players.extend(enhanced_batch)
            
            logger.info(f"🧠 AI enhanced {len(enhanced_players)} players")
            return enhanced_players
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return players
    
    async def _analyze_player_batch(
        self, 
        players: List[Dict[str, Any]], 
        game_type: str
    ) -> List[Dict[str, Any]]:
        """Analyze batch of players with AI"""
        
        # Create efficient prompt for batch analysis
        player_summary = []
        for p in players:
            player_summary.append({
                'name': p['name'],
                'position': p['position'], 
                'team': p['team'],
                'salary': p.get('salary', 0),
                'projection': p.get('projection', 0),
                'injury_status': p.get('injury_status', 'healthy')
            })
        
        cache_key = self._generate_cache_key(player_summary, game_type)
        
        # Check cache first
        if cache_key in self.analysis_cache:
            logger.debug("📋 Using cached AI analysis")
            cached_result = self.analysis_cache[cache_key]
            return self._apply_cached_analysis(players, cached_result)
        
        # AI analysis prompt
        prompt = self._create_analysis_prompt(player_summary, game_type)
        
        try:
            # Count tokens for cost tracking
            input_tokens = len(self.encoding.encode(prompt))
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            # Update usage tracking
            output_tokens = response.usage.completion_tokens if response.usage else 0
            await self._update_usage_stats(input_tokens, output_tokens)
            
            # Parse AI response
            ai_analysis = response.choices[0].message.content
            analysis_data = self._parse_ai_response(ai_analysis)
            
            # Cache the analysis
            self.analysis_cache[cache_key] = analysis_data
            
            # Apply enhancements
            return self._apply_ai_enhancements(players, analysis_data)
            
        except Exception as e:
            logger.error(f"AI batch analysis failed: {e}")
            return players
    
    def _create_analysis_prompt(
        self, 
        players: List[Dict[str, Any]], 
        game_type: str
    ) -> str:
        """Create optimized prompt for player analysis"""
        
        strategy_context = {
            'gpp': 'Focus on ceiling potential and contrarian plays for tournaments',
            'cash': 'Prioritize floor and consistency for cash games',
            'tournament': 'Maximum ceiling and leverage for large tournaments'
        }
        
        prompt = f"""Analyze these NFL DFS players for {game_type} strategy.
{strategy_context.get(game_type, '')}

Players: {json.dumps(players, indent=1)}

Provide analysis as JSON:
{{
  "player_name": {{
    "projection_adjustment": 0.0,  // -3.0 to +3.0 point adjustment
    "ceiling_multiplier": 1.0,     // 0.8 to 1.4 multiplier
    "floor_multiplier": 1.0,       // 0.7 to 1.2 multiplier  
    "ownership_estimate": 5.0,     // 1-50% ownership estimate
    "confidence": 7,               // 1-10 confidence score
    "notes": "key insight"
  }}
}}

Focus on:
- Injury impacts
- Weather conditions
- Matchup advantages 
- Game script potential
- Ownership leverage

Keep response under 600 tokens for cost efficiency."""

        return prompt
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured data"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
                
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse text response
        return self._parse_text_response(response)
    
    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """Fallback parser for non-JSON AI responses"""
        analysis = {}
        
        # Simple keyword-based parsing
        lines = response.split('\n')
        current_player = None
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'player' in key or any(pos in line.upper() for pos in ['QB', 'RB', 'WR', 'TE']):
                    current_player = value
                    analysis[current_player] = {}
                elif current_player and any(word in key for word in ['projection', 'ceiling', 'floor', 'ownership']):
                    try:
                        num_value = float(value.split()[0])
                        analysis[current_player][key] = num_value
                    except:
                        pass
        
        return analysis
    
    def _apply_ai_enhancements(
        self, 
        players: List[Dict[str, Any]], 
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply AI analysis to player projections"""
        
        enhanced = []
        
        for player in players:
            enhanced_player = player.copy()
            player_name = player['name']
            
            # Find matching analysis (fuzzy match)
            ai_data = None
            for analyzed_name, data in analysis.items():
                if self._names_match(player_name, analyzed_name):
                    ai_data = data
                    break
            
            if ai_data:
                # Apply AI adjustments
                base_proj = player.get('projection', 8.0)
                
                # Projection adjustment
                proj_adj = ai_data.get('projection_adjustment', 0.0)
                enhanced_player['projection'] = round(base_proj + proj_adj, 1)
                
                # Ceiling/floor multipliers
                ceiling_mult = ai_data.get('ceiling_multiplier', 1.0)
                floor_mult = ai_data.get('floor_multiplier', 1.0)
                
                enhanced_player['ceiling'] = round(base_proj * ceiling_mult, 1)
                enhanced_player['floor'] = round(base_proj * floor_mult, 1)
                
                # Ownership estimate
                enhanced_player['ownership'] = ai_data.get('ownership_estimate', 5.0)
                
                # Add AI insights
                enhanced_player['ai_confidence'] = ai_data.get('confidence', 7)
                enhanced_player['ai_notes'] = ai_data.get('notes', '')
                enhanced_player['ai_enhanced'] = True
            else:
                enhanced_player['ai_enhanced'] = False
            
            enhanced.append(enhanced_player)
        
        return enhanced
    
    def _apply_cached_analysis(
        self, 
        players: List[Dict[str, Any]], 
        cached_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Apply cached AI analysis to players"""
        return self._apply_ai_enhancements(players, cached_analysis)
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Fuzzy name matching for player identification"""
        # Simple fuzzy matching
        n1 = name1.lower().replace(' ', '').replace('.', '')
        n2 = name2.lower().replace(' ', '').replace('.', '')
        
        # Exact match
        if n1 == n2:
            return True
        
        # Substring match (last name)
        name1_parts = name1.split()
        name2_parts = name2.split()
        
        if len(name1_parts) > 0 and len(name2_parts) > 0:
            return name1_parts[-1].lower() in name2.lower()
        
        return False
    
    def _generate_cache_key(self, players: List[Dict], game_type: str) -> str:
        """Generate cache key for analysis results"""
        # Create key from player names and game type
        player_names = sorted([p['name'] for p in players])
        key_data = f"{game_type}_{hash(tuple(player_names))}"
        return key_data
    
    async def _update_usage_stats(self, input_tokens: int, output_tokens: int):
        """Track API usage and costs"""
        self.daily_usage['input_tokens'] += input_tokens
        self.daily_usage['output_tokens'] += output_tokens
        self.daily_usage['calls'] += 1
        
        # Calculate cost
        cost = (
            input_tokens * self.pricing['input'] + 
            output_tokens * self.pricing['output']
        )
        self.daily_usage['cost'] += cost
        
        # Log cost every 10 calls
        if self.daily_usage['calls'] % 10 == 0:
            logger.info(f"💰 AI usage: {self.daily_usage['calls']} calls, ${self.daily_usage['cost']:.4f}")
    
    async def analyze_news_impact(
        self, 
        player_name: str, 
        news_text: str, 
        current_projection: float
    ) -> Dict[str, Any]:
        """Analyze news impact on player projection"""
        
        if not self.client:
            return {'impact': 0, 'confidence': 0.5}
        
        prompt = f"""Analyze DFS impact of this NFL news:

Player: {player_name}
Current Projection: {current_projection} points
News: {news_text}

Provide impact analysis as JSON:
{{
  "projection_change": 0.0,  // -5.0 to +5.0 point change
  "confidence": 0.5,         // 0.0 to 1.0 confidence
  "severity": 5,            // 1-10 severity scale  
  "dfs_impact": "explanation",
  "recommended_action": "hold/swap/avoid"
}}

Consider injury severity, role changes, weather, etc."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2
            )
            
            analysis = response.choices[0].message.content
            return self._parse_ai_response(analysis)
            
        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            return {'projection_change': 0, 'confidence': 0.5}
    
    async def analyze_correlation_stacks(
        self, 
        lineup_players: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """AI analysis of lineup correlations and stacks"""
        
        if not self.client:
            return {'stack_score': 5, 'recommendations': []}
        
        teams = {}
        for player in lineup_players:
            team = player.get('team', '')
            if team not in teams:
                teams[team] = []
            teams[team].append(f"{player['name']} ({player['position']})")
        
        prompt = f"""Analyze DFS lineup correlations:

Lineup by team:
{json.dumps(teams, indent=1)}

Provide correlation analysis as JSON:
{{
  "stack_score": 7,           // 1-10 correlation quality
  "primary_stack": "team",    // Best correlated stack
  "correlation_boost": 1.2,   // Expected correlation multiplier
  "recommendations": ["specific advice"],
  "leverage_opportunities": ["contrarian plays"]
}}

Focus on QB-WR correlations, game stacks, bring-back plays."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return self._parse_ai_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {'stack_score': 5, 'recommendations': []}
    
    def get_daily_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage and cost statistics"""
        return {
            'usage': self.daily_usage.copy(),
            'weekly_projection': self.daily_usage['cost'] * 7,
            'budget_utilization': (self.daily_usage['cost'] * 7) / 15.0,  # $15 weekly budget
            'cost_per_call': self.daily_usage['cost'] / max(1, self.daily_usage['calls'])
        }
    
    def reset_daily_usage(self):
        """Reset daily usage counters"""
        self.daily_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'calls': 0,
            'cost': 0.0
        }
    
    @lru_cache(maxsize=100)
    def estimate_analysis_cost(self, num_players: int, game_type: str) -> float:
        """Estimate cost for analyzing N players"""
        # Average tokens per player analysis
        input_tokens_per_player = 50
        output_tokens_per_player = 40
        
        total_input = num_players * input_tokens_per_player
        total_output = num_players * output_tokens_per_player
        
        cost = (
            total_input * self.pricing['input'] + 
            total_output * self.pricing['output']
        )
        
        return cost
