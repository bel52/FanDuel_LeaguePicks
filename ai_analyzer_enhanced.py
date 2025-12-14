"""
Enhanced AI Analyzer for DFS Optimization
UPGRADED: Uses ALL collected data (Monte Carlo, Vegas, Weather, News) 
Provides STRATEGIC recommendations, not just player names
"""
import os
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass


@dataclass
class StackRecommendation:
    """A recommended QB + pass catcher stack"""
    qb: str
    qb_team: str
    targets: List[str]  # WRs/TEs to stack
    reasoning: str
    ceiling_score: float
    game_total: float


@dataclass 
class AIAnalysisResult:
    """Complete AI analysis output"""
    must_play: List[str]
    must_fade: List[str]
    primary_stack: Optional[StackRecommendation]
    secondary_stacks: List[StackRecommendation]
    value_plays: List[Dict[str, Any]]  # Cheap players with upside
    news_impacts: List[Dict[str, Any]]  # How news affects players
    lineup_advice: str
    confidence_score: float
    raw_notes: str


class EnhancedAIAnalyzer:
    """
    AI Analyzer that uses ALL collected data for strategic DFS decisions
    """
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.ai_enabled = bool(self.openai_api_key or self.anthropic_api_key)
        
        if self.ai_enabled:
            logger.info("🤖 Enhanced AI Analyzer initialized")
        else:
            logger.warning("⚠️ AI Analyzer disabled - no API keys found")

    def _build_comprehensive_prompt(
        self,
        players: List[Dict],
        monte_carlo_results: Dict[str, Dict],
        vegas_data: Dict,
        weather_data: Dict,
        news_items: List[Dict],
        contest_type: str = "friends_league"
    ) -> str:
        """
        Build a rich prompt with ALL available data
        """
        lines = []
        
        # System context
        lines.append("""You are an elite DFS tournament optimizer. Your goal is to identify the OPTIMAL lineup construction strategy, not just good players.

CRITICAL CONTEXT: This is a 12-person friends league. You need to WIN, not just cash. That means:
- Target players with the highest CEILINGS, not safest floors
- Stack QBs with their pass catchers from HIGH-TOTAL games
- 1st place pays, 2nd place doesn't matter

""")
        
        # Vegas Data - THE MOST IMPORTANT FACTOR
        lines.append("=" * 50)
        lines.append("VEGAS DATA (Most Important for DFS)")
        lines.append("=" * 50)
        
        high_total_games = vegas_data.get('high_total_games', [])
        if high_total_games:
            lines.append("\n🔥 HIGH-TOTAL GAMES (Target these heavily):")
            for game in high_total_games:
                teams = game.get('teams', [])
                total = game.get('total', 45)
                lines.append(f"  • {teams[0] if teams else 'TBD'} vs {teams[1] if len(teams) > 1 else 'TBD'}: {total} points")
            lines.append("")
        
        avg_total = vegas_data.get('avg_total', 45)
        lines.append(f"Slate average total: {avg_total:.1f} points")
        lines.append("")
        
        # Weather Impacts
        if weather_data:
            bad_weather_teams = [team for team, data in weather_data.items() 
                                if data.get('weather_factor', 1.0) < 0.95]
            if bad_weather_teams:
                lines.append("⛈️ WEATHER CONCERNS:")
                for team in bad_weather_teams:
                    factor = weather_data[team].get('weather_factor', 1.0)
                    lines.append(f"  • {team}: {(1-factor)*100:.0f}% projection reduction")
                lines.append("")
        
        # Breaking News
        if news_items:
            lines.append("📰 BREAKING NEWS (Last 24 hours):")
            for item in news_items[:5]:  # Top 5 news items
                lines.append(f"  • {item.get('title', item.get('headline', 'News item'))[:80]}")
            lines.append("")
        
        # Top Players by Position with Monte Carlo Data
        lines.append("=" * 50)
        lines.append("TOP PLAYERS BY POSITION (with Monte Carlo analysis)")
        lines.append("=" * 50)
        
        # Group players by position
        by_position = {}
        for p in players:
            pos = p.get('position', 'OTHER')
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(p)
        
        # Sort each position by ceiling
        for pos in ['QB', 'RB', 'WR', 'TE', 'DEF', 'D']:
            if pos not in by_position:
                continue
            
            pos_players = by_position[pos]
            
            # Get Monte Carlo data and sort by ceiling
            for p in pos_players:
                name = p.get('name', '')
                mc = monte_carlo_results.get(name, {})
                p['_ceiling_90'] = mc.get('ceiling_90', p.get('projection', 0) * 1.5)
                p['_boom_rate'] = mc.get('boom_rate', 0.15)
                p['_floor_10'] = mc.get('floor_10', p.get('projection', 0) * 0.5)
            
            pos_players.sort(key=lambda x: x.get('_ceiling_90', 0), reverse=True)
            
            display_pos = 'DEF' if pos == 'D' else pos
            lines.append(f"\n{display_pos}s (Top 8 by ceiling):")
            
            for p in pos_players[:8]:
                name = p.get('name', '')
                team = p.get('team', '')
                salary = p.get('salary', 0)
                proj = p.get('projection', 0)
                ceiling = p.get('_ceiling_90', proj * 1.5)
                boom = p.get('_boom_rate', 0.15)
                floor = p.get('_floor_10', proj * 0.5)
                game_mult = p.get('game_environment_mult', 1.0)
                
                # Highlight high-total game players
                ht_marker = "🔥" if game_mult >= 1.25 else ""
                
                lines.append(
                    f"  {ht_marker}{name} ({team}) ${salary:,} | "
                    f"Proj:{proj:.1f} Ceil:{ceiling:.1f} Floor:{floor:.1f} Boom:{boom*100:.0f}%"
                )
        
        # Value Plays (high ceiling relative to salary)
        lines.append("\n" + "=" * 50)
        lines.append("💰 VALUE PLAYS (Ceiling/Salary ratio)")
        lines.append("=" * 50)
        
        value_players = []
        for p in players:
            if p.get('salary', 10000) < 5500:  # Only cheap players
                ceiling = p.get('_ceiling_90', p.get('projection', 5) * 1.5)
                salary = p.get('salary', 5000)
                value_ratio = ceiling / (salary / 1000) if salary > 0 else 0
                if value_ratio > 3.0:  # Good value threshold
                    value_players.append({
                        'name': p.get('name'),
                        'team': p.get('team'),
                        'position': p.get('position'),
                        'salary': salary,
                        'ceiling': ceiling,
                        'value_ratio': value_ratio
                    })
        
        value_players.sort(key=lambda x: x['value_ratio'], reverse=True)
        for vp in value_players[:6]:
            lines.append(
                f"  {vp['name']} ({vp['position']}-{vp['team']}) ${vp['salary']:,} | "
                f"Ceiling:{vp['ceiling']:.1f} Value:{vp['value_ratio']:.1f}x"
            )
        
        # Request structured output
        lines.append("\n" + "=" * 50)
        lines.append("YOUR TASK")
        lines.append("=" * 50)
        lines.append("""
Based on ALL the data above, provide your analysis in this EXACT JSON format:

{
  "primary_stack": {
    "qb": "QB Name",
    "qb_team": "TEAM",
    "targets": ["WR1 Name", "WR2 Name"],
    "reasoning": "Why this stack wins the week",
    "game_total": 54.5
  },
  "secondary_stack": {
    "qb": "QB Name",
    "qb_team": "TEAM", 
    "targets": ["WR Name"],
    "reasoning": "Backup stack reasoning"
  },
  "must_play": ["Player 1", "Player 2", "Player 3", ...],
  "must_fade": ["Player 1", "Player 2", ...],
  "value_plays": [
    {"name": "Player Name", "reason": "Why they're value"}
  ],
  "bring_back": {
    "player": "Opposing player to correlate",
    "from_game": "TEAM1 vs TEAM2",
    "reasoning": "Why bring-back works"
  },
  "lineup_construction": "Specific advice on how to build the winning lineup",
  "confidence": 0.85,
  "key_insight": "The ONE thing that will decide this week"
}

Respond with ONLY valid JSON, no other text.
""")
        
        return "\n".join(lines)

    async def analyze_slate(
        self,
        players: List[Dict],
        monte_carlo_results: Dict[str, Dict],
        vegas_data: Dict,
        weather_data: Dict = None,
        news_items: List[Dict] = None,
        contest_type: str = "friends_league"
    ) -> AIAnalysisResult:
        """
        Run comprehensive AI analysis on the full slate
        """
        if not self.ai_enabled:
            logger.warning("AI disabled - returning empty analysis")
            return self._empty_result()
        
        # Build comprehensive prompt
        prompt = self._build_comprehensive_prompt(
            players=players,
            monte_carlo_results=monte_carlo_results,
            vegas_data=vegas_data,
            weather_data=weather_data or {},
            news_items=news_items or [],
            contest_type=contest_type
        )
        
        # Estimate cost
        est_tokens = len(prompt) // 4
        est_cost = est_tokens * 0.00000015  # gpt-4o-mini pricing
        logger.info(f"🤖 AI Analysis: ~{est_tokens} tokens, ~${est_cost:.3f}")
        
        # Call AI
        response_data = await self._call_ai(prompt)
        
        if not response_data:
            return self._empty_result()
        
        # Parse response
        return self._parse_response(response_data)

    async def _call_ai(self, prompt: str) -> Optional[Dict]:
        """Call OpenAI or Anthropic API"""
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.openai_api_key)
                
                logger.info("🤖 Calling OpenAI for strategic analysis...")
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert DFS NFL strategist. Respond with valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content
                
                # Clean up markdown if present
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                
                data = json.loads(content)
                logger.info("✅ OpenAI strategic analysis complete")
                return data
                
            except Exception as e:
                logger.error(f"OpenAI error: {e}")
        
        # Fallback to Anthropic
        if self.anthropic_api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                
                logger.info("🤖 Calling Anthropic for strategic analysis...")
                msg = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0.3,
                    system="You are an expert DFS NFL strategist. Respond with valid JSON only.",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                content = msg.content[0].text
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                
                data = json.loads(content)
                logger.info("✅ Anthropic strategic analysis complete")
                return data
                
            except Exception as e:
                logger.error(f"Anthropic error: {e}")
        
        return None

    def _parse_response(self, data: Dict) -> AIAnalysisResult:
        """Parse AI response into structured result"""
        
        # Extract primary stack
        primary_stack = None
        ps_data = data.get('primary_stack', {})
        if ps_data:
            primary_stack = StackRecommendation(
                qb=ps_data.get('qb', ''),
                qb_team=ps_data.get('qb_team', ''),
                targets=ps_data.get('targets', []),
                reasoning=ps_data.get('reasoning', ''),
                ceiling_score=0.0,
                game_total=ps_data.get('game_total', 45.0)
            )
        
        # Extract secondary stacks
        secondary_stacks = []
        ss_data = data.get('secondary_stack', {})
        if ss_data:
            secondary_stacks.append(StackRecommendation(
                qb=ss_data.get('qb', ''),
                qb_team=ss_data.get('qb_team', ''),
                targets=ss_data.get('targets', []),
                reasoning=ss_data.get('reasoning', ''),
                ceiling_score=0.0,
                game_total=ss_data.get('game_total', 45.0)
            ))
        
        # Extract value plays
        value_plays = data.get('value_plays', [])
        
        # Extract news impacts
        news_impacts = []
        bring_back = data.get('bring_back', {})
        if bring_back:
            news_impacts.append({
                'type': 'bring_back',
                'player': bring_back.get('player', ''),
                'game': bring_back.get('from_game', ''),
                'reasoning': bring_back.get('reasoning', '')
            })
        
        return AIAnalysisResult(
            must_play=data.get('must_play', []),
            must_fade=data.get('must_fade', []),
            primary_stack=primary_stack,
            secondary_stacks=secondary_stacks,
            value_plays=value_plays,
            news_impacts=news_impacts,
            lineup_advice=data.get('lineup_construction', ''),
            confidence_score=data.get('confidence', 0.5),
            raw_notes=data.get('key_insight', '')
        )

    def _empty_result(self) -> AIAnalysisResult:
        """Return empty result when AI is unavailable"""
        return AIAnalysisResult(
            must_play=[],
            must_fade=[],
            primary_stack=None,
            secondary_stacks=[],
            value_plays=[],
            news_impacts=[],
            lineup_advice="AI analysis unavailable",
            confidence_score=0.0,
            raw_notes=""
        )

    def apply_analysis_to_players(
        self, 
        players: List[Dict], 
        analysis: AIAnalysisResult
    ) -> List[Dict]:
        """Apply AI analysis results to player projections"""
        
        must_play_set = set(analysis.must_play)
        must_fade_set = set(analysis.must_fade)
        
        # Build stack bonus set
        stack_players = set()
        if analysis.primary_stack:
            stack_players.add(analysis.primary_stack.qb)
            stack_players.update(analysis.primary_stack.targets)
        
        value_play_set = set(vp.get('name', '') for vp in analysis.value_plays)
        
        logger.info(f"🎯 Applying AI analysis: {len(must_play_set)} must-play, {len(must_fade_set)} must-fade, {len(stack_players)} stack players")
        
        for p in players:
            name = p.get('name', '')
            
            # AI must play/fade flags
            if name in must_play_set:
                p['ai_must_play'] = True
                p['ai_boost'] = 0.25  # 25% projection boost
            elif name in must_fade_set:
                p['ai_must_fade'] = True
                p['ai_boost'] = -0.30  # 30% projection penalty
            else:
                p['ai_must_play'] = False
                p['ai_must_fade'] = False
                p['ai_boost'] = 0.0
            
            # Stack bonus
            if name in stack_players:
                p['ai_stack_target'] = True
                p['ai_boost'] = p.get('ai_boost', 0) + 0.15  # Additional 15% for stack targets
            else:
                p['ai_stack_target'] = False
            
            # Value play flag
            p['ai_value_play'] = name in value_play_set
        
        return players


# Convenience function for integration
async def run_enhanced_ai_analysis(
    players: List[Dict],
    monte_carlo_results: Dict[str, Dict],
    vegas_data: Dict,
    weather_data: Dict = None,
    news_items: List[Dict] = None,
    contest_type: str = "friends_league"
) -> AIAnalysisResult:
    """
    Main entry point for enhanced AI analysis
    """
    analyzer = EnhancedAIAnalyzer()
    return await analyzer.analyze_slate(
        players=players,
        monte_carlo_results=monte_carlo_results,
        vegas_data=vegas_data,
        weather_data=weather_data,
        news_items=news_items,
        contest_type=contest_type
    )
