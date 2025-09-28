"""
Complete Dual AI Integration for WINNING DFS Analysis
Uses both OpenAI and Claude with proper error handling
"""
import os
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime

# Import APIs with graceful fallback
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI package not available")
    OPENAI_AVAILABLE = False

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic package not available")
    CLAUDE_AVAILABLE = False

class DualAIDFSAnalyzer:
    """Dual AI system for comprehensive DFS analysis"""

    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.weekly_budget = float(os.getenv('AI_WEEKLY_BUDGET', '15.0'))
        self.weekly_spend = 0.0
        self.cost_log = []

        # Initialize OpenAI with better error handling
        self.openai_client = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
                self.openai_client = None

        # Initialize Claude with better error handling
        self.claude_client = None
        if CLAUDE_AVAILABLE and self.claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
                # Test the connection
                test_response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "test"}]
                )
                logger.info("Claude client initialized and tested successfully")
            except Exception as e:
                logger.warning(f"Claude initialization failed: {e}")
                self.claude_client = None

    def analyze_slate_for_optimization(self, player_data: List[Dict],
                                       weather_data: Dict,
                                       vegas_data: Dict,
                                       contest_type: str = 'gpp') -> Dict[str, Any]:
        """Main analysis function using available AI services"""

        if not (self.openai_client or self.claude_client):
            logger.warning("No AI services available, using fallback analysis")
            return self._fallback_analysis(player_data, contest_type)

        try:
            # Prepare comprehensive data
            analysis_data = self._prepare_slate_data(player_data, weather_data, vegas_data, contest_type)

            # Get insights from available AI services
            openai_insights = None
            claude_insights = None

            if self.openai_client:
                try:
                    openai_insights = self._get_openai_insights(analysis_data, contest_type)
                    logger.info("OpenAI analysis completed successfully")
                except Exception as e:
                    logger.error(f"OpenAI analysis failed: {e}")

            if self.claude_client:
                try:
                    claude_insights = self._get_claude_insights(analysis_data, contest_type)
                    logger.info("Claude analysis completed successfully")
                except Exception as e:
                    logger.error(f"Claude analysis failed: {e}")

            # Process and combine insights
            combined_analysis = self._synthesize_ai_insights(
                openai_insights, claude_insights, player_data, contest_type
            )

            # Track costs
            cost = 0.15 if (openai_insights and claude_insights) else 0.08
            self.weekly_spend += cost
            self._log_cost(f"ai_analysis_{contest_type}", cost)

            return combined_analysis

        except Exception as e:
            logger.error(f"AI analysis pipeline failed: {e}")
            return self._fallback_analysis(player_data, contest_type)

        async def analyze_breaking_news(self, news_events: List[Dict], current_players: List[Dict]) -> Dict[str, Any]:
            """Enhanced AI analysis of breaking news impact on lineups"""
            if not news_events:
                return {'news_impact': 'none', 'lineup_adjustments': []}

            # Extract player names for context
            player_names = [p.get('name', '') for p in current_players[:15]]

            # Format news for AI analysis
            news_summary = []
            for news in news_events:
                news_summary.append({
                    'headline': news.get('headline', ''),
                    'source': news.get('source', ''),
                    'impact_type': news.get('impact_type', ''),
                    'dfs_impact_score': news.get('dfs_impact', 0)
                })

            prompt = f"""Analyze breaking NFL news for DFS lineup impact:

    NEWS EVENTS:
    {json.dumps(news_summary, indent=2)}

    CURRENT ROSTER PLAYERS:
    {player_names}

    ANALYSIS NEEDED:
    1. Which players are DIRECTLY affected by this news?
    2. What's the impact: positive boost, negative downgrade, or neutral?
    3. Are there backup players who gain value?
    4. Should any current players be REMOVED from lineups?
    5. Should any players be ADDED to lineups?

    Provide specific player names and impact ratings (1-10 scale).
    Focus on actionable lineup changes only."""

            try:
                if self.claude_client:
                    response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=400,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    analysis_text = response.content[0].text

                    # Parse AI response for actionable recommendations
                    recommendations = self._parse_news_analysis(analysis_text, player_names)

                    # Track cost
                    self.weekly_spend += 0.05
                    self._log_cost("breaking_news_analysis", 0.05)

                    return {
                        'news_impact': 'significant' if news_events else 'none',
                        'ai_analysis': analysis_text,
                        'lineup_adjustments': recommendations,
                        'affected_players': recommendations.get('affected_players', []),
                        'remove_players': recommendations.get('remove_players', []),
                        'add_players': recommendations.get('add_players', []),
                        'confidence': recommendations.get('confidence', 0.5)
                    }

                elif self.openai_client:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=400,
                        temperature=0.1
                    )

                    analysis_text = response.choices[0].message.content
                    recommendations = self._parse_news_analysis(analysis_text, player_names)

                    # Track cost
                    self.weekly_spend += 0.03
                    self._log_cost("breaking_news_analysis", 0.03)

                    return {
                        'news_impact': 'significant' if news_events else 'none',
                        'ai_analysis': analysis_text,
                        'lineup_adjustments': recommendations,
                        'affected_players': recommendations.get('affected_players', []),
                        'remove_players': recommendations.get('remove_players', []),
                        'add_players': recommendations.get('add_players', []),
                        'confidence': recommendations.get('confidence', 0.5)
                    }
                else:
                    return {'news_impact': 'none', 'analysis': 'AI unavailable'}

            except Exception as e:
                logger.error(f"Breaking news analysis failed: {e}")
                return {'news_impact': 'error', 'analysis': f'Analysis failed: {str(e)}'}

        def _parse_news_analysis(self, analysis_text: str, current_players: List[str]) -> Dict[str, Any]:
            """Parse AI analysis text for actionable recommendations"""

            affected_players = []
            remove_players = []
            add_players = []

            analysis_lower = analysis_text.lower()

            # Look for player mentions and context
            for player_name in current_players:
                if player_name.lower() in analysis_lower:
                    # Get context around player mention
                    player_pos = analysis_lower.find(player_name.lower())
                    if player_pos >= 0:
                        context_start = max(0, player_pos - 100)
                        context_end = min(len(analysis_text), player_pos + len(player_name) + 100)
                        context = analysis_lower[context_start:context_end]

                        # Determine impact
                        remove_terms = ['remove', 'avoid', 'downgrade', 'sit', 'bench', 'out', 'injured']
                        add_terms = ['add', 'boost', 'upgrade', 'start', 'opportunity', 'value']

                        remove_score = sum(1 for term in remove_terms if term in context)
                        add_score = sum(1 for term in add_terms if term in context)

                        if remove_score > add_score:
                            remove_players.append(player_name)
                        elif add_score > remove_score:
                            add_players.append(player_name)

                        affected_players.append({
                            'name': player_name,
                            'impact': 'negative' if remove_score > add_score else 'positive' if add_score > remove_score else 'neutral',
                            'confidence': min(max(remove_score, add_score) / 3, 1.0)
                        })

            # Determine overall confidence
            confidence = 0.8 if (remove_players or add_players) else 0.3

            return {
                'affected_players': affected_players,
                'remove_players': remove_players[:3],  # Limit to 3
                'add_players': add_players[:3],  # Limit to 3
                'confidence': confidence
            }

    def _prepare_slate_data(self, player_data: List[Dict], weather_data: Dict,
                           vegas_data: Dict, contest_type: str) -> Dict:
        """Prepare comprehensive slate data for AI analysis"""

        # Extract top players by position
        positions = {'QB': 5, 'RB': 8, 'WR': 10, 'TE': 6}
        top_players_by_pos = {}

        for pos, count in positions.items():
            pos_players = [p for p in player_data if p.get('position') == pos]
            top_players_by_pos[pos] = sorted(
                pos_players,
                key=lambda x: x.get('salary', 0),
                reverse=True
            )[:count]

        # Find value plays
        value_plays = []
        for player in player_data:
            salary = player.get('salary', 5000)
            projection = player.get('projected_points', 0)
            if salary > 0 and projection > 0:
                value = projection / (salary / 1000)
                if value > 3.2 and salary < 7500:
                    value_plays.append({
                        'name': player.get('name'),
                        'position': player.get('position'),
                        'team': player.get('team'),
                        'salary': salary,
                        'projection': projection,
                        'value': value
                    })

        value_plays = sorted(value_plays, key=lambda x: x['value'], reverse=True)[:8]

        # Analyze weather impacts
        significant_weather = []
        for team, conditions in weather_data.items():
            wind_mph = 0
            precip = conditions.get('precipitation_chance', 0)

            wind_str = conditions.get('wind_speed', '0 mph')
            try:
                wind_mph = int(wind_str.split()[0])
            except:
                pass

            if wind_mph > 12 or precip > 25:
                significant_weather.append({
                    'team': team,
                    'wind_mph': wind_mph,
                    'precipitation': precip,
                    'conditions': conditions.get('conditions', '')
                })

        # Find potential stacking teams (high-priced players)
        team_salary_totals = {}
        for player in player_data:
            team = player.get('team', '')
            if team:
                if team not in team_salary_totals:
                    team_salary_totals[team] = []
                team_salary_totals[team].append(player.get('salary', 0))

        high_total_teams = []
        for team, salaries in team_salary_totals.items():
            if len(salaries) >= 3:
                avg_salary = sum(salaries) / len(salaries)
                if avg_salary > 5500:
                    high_total_teams.append(team)

        return {
            'contest_type': contest_type,
            'top_players': top_players_by_pos,
            'value_plays': value_plays,
            'weather_impacts': significant_weather,
            'stack_candidates': high_total_teams[:4],
            'slate_size': len(player_data),
            'avg_salary': sum(p.get('salary', 0) for p in player_data) / len(player_data) if player_data else 0
        }

    def _get_openai_insights(self, data: Dict, contest_type: str) -> str:
        """Get strategic insights from OpenAI"""

        prompt = f"""Analyze this NFL DFS slate for a 12-person friends league {contest_type} contest.

SLATE OVERVIEW:
- Contest: {contest_type.upper()} (12 people, need to win weekly)
- Players available: {data['slate_size']}
- Average salary: ${data['avg_salary']:.0f}

TOP PLAYERS BY POSITION:
QBs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['QB'][:4]]}
RBs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['RB'][:6]]}  
WRs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['WR'][:6]]}
TEs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['TE'][:4]]}

VALUE OPPORTUNITIES:
{[(p['name'], f"${p['salary']}", f"{p['value']:.2f}x") for p in data['value_plays'][:6]]}

WEATHER CONCERNS:
{data['weather_impacts'] if data['weather_impacts'] else 'No significant weather'}

STACKING TEAMS:
{data['stack_candidates']}

Provide specific analysis for this friends league:
1. Which 3-4 players offer best leverage against casual players?
2. What's the optimal salary allocation strategy?
3. Should we stack, and which teams/players?
4. Any weather-based pivots?
5. Contest-specific approach for {contest_type}?

Give specific player names and clear reasoning."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.1
        )

        return response.choices[0].message.content

    def _get_claude_insights(self, data: Dict, contest_type: str) -> str:
        """Get strategic insights from Claude"""

        prompt = f"""You're an expert DFS analyst for a 12-person friends league {contest_type} contest. 

SLATE DATA:
Contest Type: {contest_type.upper()}
Total Players: {data['slate_size']}
Average Salary: ${data['avg_salary']:.0f}

TOP PLAYERS:
QBs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['QB'][:4]]}
RBs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['RB'][:6]]}
WRs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['WR'][:6]]}
TEs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['TE'][:4]]}

VALUE PLAYS:
{[(p['name'], p['position'], f"${p['salary']}", f"{p['value']:.2f}x value") for p in data['value_plays'][:6]]}

WEATHER ISSUES:
{data['weather_impacts'] if data['weather_impacts'] else 'No significant weather concerns'}

HIGH-TOTAL TEAMS:
{data['stack_candidates']}

Analyze this slate for winning strategy:

1. LEVERAGE SPOTS: Which players give best edge vs casual friends?
2. SALARY STRATEGY: Where to spend up vs save money?
3. STACKING: Best correlation plays for this slate?
4. WEATHER IMPACT: How should conditions affect picks?
5. CONTEST APPROACH: Specific strategy for {contest_type} format?

Be specific with player names and actionable advice."""

        response = self.claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _synthesize_ai_insights(self, openai_insights: str, claude_insights: str,
                               player_data: List[Dict], contest_type: str) -> Dict[str, Any]:
        """Combine AI insights into actionable recommendations"""

        # Combine insights
        all_insights = ""
        sources_used = []

        if openai_insights:
            all_insights += f"=== OpenAI Analysis ===\n{openai_insights}\n\n"
            sources_used.append("OpenAI")

        if claude_insights:
            all_insights += f"=== Claude Analysis ===\n{claude_insights}\n\n"
            sources_used.append("Claude")

        if not all_insights:
            return self._fallback_analysis(player_data, contest_type)

        # Extract player recommendations
        leverage_players = []
        avoid_players = []
        stack_teams = []

        player_names = [p.get('name', '') for p in player_data if p.get('name')]

        for player_name in player_names:
            if player_name.lower() in all_insights.lower():
                # Analyze context around player mention
                insights_lower = all_insights.lower()
                player_pos = insights_lower.find(player_name.lower())

                if player_pos >= 0:
                    # Get context around the mention
                    context_start = max(0, player_pos - 100)
                    context_end = min(len(all_insights), player_pos + len(player_name) + 100)
                    context = all_insights[context_start:context_end].lower()

                    # Score the context
                    positive_terms = ['leverage', 'target', 'value', 'good', 'upside', 'like', 'stack', 'play']
                    negative_terms = ['avoid', 'fade', 'chalk', 'overpriced', 'risky', 'expensive']

                    positive_score = sum(1 for term in positive_terms if term in context)
                    negative_score = sum(1 for term in negative_terms if term in context)

                    if positive_score > negative_score and len(leverage_players) < 6:
                        leverage_players.append(player_name)
                    elif negative_score > positive_score and len(avoid_players) < 4:
                        avoid_players.append(player_name)

        # Extract team stacking recommendations
        teams = set(p.get('team', '') for p in player_data if p.get('team'))
        for team in teams:
            if team.lower() in all_insights.lower():
                team_context = all_insights.lower()
                if 'stack' in team_context and team.lower() in team_context:
                    if len(stack_teams) < 3:
                        stack_teams.append(team)

        # Generate ownership adjustments
        ownership_adjustments = {}

        # Reduce ownership for leverage players (makes optimizer more likely to select them)
        for player in leverage_players:
            if contest_type == 'gpp':
                ownership_adjustments[player] = 0.7  # 30% ownership reduction
            elif contest_type == 'contrarian':
                ownership_adjustments[player] = 0.5  # 50% reduction for contrarian
            else:
                ownership_adjustments[player] = 0.85  # 15% reduction for cash

        # Increase ownership for avoid players (discourages selection)
        for player in avoid_players:
            ownership_adjustments[player] = 1.5  # 50% ownership increase

        confidence = 0.9 if len(sources_used) == 2 else 0.7

        return {
            'type': contest_type,
            'ai_strategy': all_insights,
            'leverage_players': leverage_players,
            'avoid_players': avoid_players,
            'stack_teams': stack_teams,
            'ownership_adjustments': ownership_adjustments,
            'ai_confidence': confidence,
            'ai_enabled': True,
            'sources_used': sources_used,
            'analysis_quality': 'comprehensive' if len(sources_used) == 2 else 'single_source'
        }

    def _fallback_analysis(self, players: List[Dict], contest_type: str) -> Dict[str, Any]:
        """Intelligent fallback when AI is unavailable"""

        # Value-based analysis
        value_targets = []
        expensive_fades = []

        for player in players:
            name = player.get('name', '')
            salary = player.get('salary', 5000)
            projection = player.get('projected_points', 0)

            if salary > 0 and projection > 0 and name:
                value = projection / (salary / 1000)

                # Target high-value plays
                if value > 3.5 and salary < 7000:
                    value_targets.append(name)
                # Identify expensive potential fades
                elif salary > 9000 and value < 2.8:
                    expensive_fades.append(name)

        strategy_text = f"Fallback {contest_type} analysis: "
        if contest_type == 'gpp':
            strategy_text += "Target value plays with upside, consider stacking high-total games"
        elif contest_type == 'cash':
            strategy_text += "Focus on consistent value plays and high floors"
        else:
            strategy_text += "Fade chalk, target contrarian value with ceiling"

        return {
            'type': contest_type,
            'ai_strategy': strategy_text,
            'leverage_players': value_targets[:5],
            'avoid_players': expensive_fades[:3] if contest_type != 'cash' else [],
            'stack_teams': [],
            'ownership_adjustments': {},
            'ai_confidence': 0.4,
            'ai_enabled': False,
            'sources_used': ['fallback'],
            'analysis_quality': 'basic'
        }

    def _log_cost(self, analysis_type: str, cost: float):
        """Track AI costs"""
        self.cost_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': analysis_type,
            'cost': cost,
            'weekly_total': self.weekly_spend
        })

        logger.info(f"AI Cost: ${cost:.3f} for {analysis_type}")
        logger.info(f"Weekly total: ${self.weekly_spend:.3f} / ${self.weekly_budget:.2f}")

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return {
            'weekly_spend': self.weekly_spend,
            'weekly_budget': self.weekly_budget,
            'remaining_budget': self.weekly_budget - self.weekly_spend,
            'cost_log': self.cost_log[-10:] if self.cost_log else []
        }

# Backwards compatibility
WinningDFSAIAnalyzer = DualAIDFSAnalyzer