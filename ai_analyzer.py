"""
Complete Dual AI Integration for WINNING DFS Analysis
FIXED: Direct dotenv loading for reliable API key access
"""
# CRITICAL: Load .env before anything else
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

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

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    logger.warning("Anthropic package not available")
    CLAUDE_AVAILABLE = False

class DualAIDFSAnalyzer:
    """Dual AI system for comprehensive DFS analysis with enhanced error handling"""

    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.claude_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.weekly_budget = float(os.getenv('AI_WEEKLY_BUDGET', '15.0'))
        self.weekly_spend = 0.0
        self.cost_log = []

        # Enhanced OpenAI validation and initialization
        self.openai_client = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            try:
                # Validate API key format
                if self._validate_openai_key(self.openai_api_key):
                    self.openai_client = OpenAI(api_key=self.openai_api_key)
                    # Test with a simple request
                    test_response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    logger.info("✅ OpenAI client initialized and tested successfully")
                else:
                    logger.error("❌ OpenAI API key format validation failed")
                    self.openai_client = None
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
                self.openai_client = None

        # Enhanced Claude validation and initialization
        self.claude_client = None
        if CLAUDE_AVAILABLE and self.claude_api_key:
            try:
                # Validate API key format
                if self._validate_claude_key(self.claude_api_key):
                    self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
                    # Test the connection
                    test_response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=1,
                        messages=[{"role": "user", "content": "test"}]
                    )
                    logger.info("✅ Claude client initialized and tested successfully")
                else:
                    logger.error("❌ Claude API key format validation failed")
                    self.claude_client = None
            except Exception as e:
                logger.error(f"❌ Claude initialization failed: {e}")
                self.claude_client = None

        # Log initialization status
        openai_status = "✅" if self.openai_client else "❌"
        claude_status = "✅" if self.claude_client else "❌"
        logger.info(f"AI Services: OpenAI {openai_status} | Claude {claude_status}")

    async def analyze_edge_case_players(self, player_data: List[Dict]) -> Dict[str, Any]:
        """AI analysis of small-sample, injury-opportunity, and edge case players"""

        # Identify edge cases that need AI evaluation
        small_sample_players = []
        injury_boost_players = []
        high_variance_players = []

        for player in player_data:
            games_played = player.get('games_played', 0)
            name = player.get('name', '')
            position = player.get('position', '')
            salary = player.get('salary', 0)
            fppg = player.get('projected_points', 0)
            injury_opp = player.get('injury_opportunity', False)

            # Small sample (1-2 games, high salary, good FPPG)
            if 0 < games_played <= 2 and salary > 5000 and fppg > 10:
                small_sample_players.append({
                    'name': name,
                    'position': position,
                    'salary': salary,
                    'fppg': fppg,
                    'games': games_played,
                    'reason': f'{fppg:.1f} FPPG on only {games_played} game(s)'
                })

            # Injury opportunity players
            if injury_opp:
                injured_starter = player.get('injured_starter', '')
                injury_boost_players.append({
                    'name': name,
                    'position': position,
                    'salary': salary,
                    'fppg': fppg,
                    'injured_starter': injured_starter
                })

            # High variance plays
            value = (fppg / (salary / 1000)) if salary > 0 else 0
            if value > 3.5 and salary < 5500:
                high_variance_players.append({
                    'name': name,
                    'position': position,
                    'salary': salary,
                    'fppg': fppg,
                    'value': value
                })

        # Only call AI if there are edge cases to evaluate
        if not (small_sample_players or injury_boost_players or high_variance_players):
            return {'edge_case_recommendations': [], 'ai_analysis': 'No edge cases to evaluate'}

        # Construct AI prompt for strategic evaluation
        prompt = f"""You're evaluating edge-case NFL DFS players for a 12-person friends league tournament.
    Your goal: Determine which edge cases are REAL opportunities vs traps.

    SMALL SAMPLE PLAYERS (limited games played):
    {json.dumps(small_sample_players, indent=2) if small_sample_players else 'None'}

    INJURY OPPORTUNITY PLAYERS (backups getting starts):
    {json.dumps(injury_boost_players, indent=2) if injury_boost_players else 'None'}

    HIGH VALUE PLAYERS (3.5+ value):
    {json.dumps(high_variance_players, indent=2) if high_variance_players else 'None'}

    For each player category, provide:
    1. CONFIDENCE RATING (1-10): How confident are you they'll produce?
    2. START/FADE/MONITOR: Clear recommendation
    3. REASONING: Why this rating? (1 sentence)

    Focus on ACTIONABLE recommendations. In a 12-person league, you need to beat 11 people weekly."""

        try:
            if self.claude_client:
                response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}]
                )

                analysis_text = response.content[0].text

                # Track cost
                self.weekly_spend += 0.08
                self._log_cost("edge_case_player_analysis", 0.08)

                # Parse recommendations
                recommendations = self._parse_edge_case_recommendations(
                    analysis_text,
                    small_sample_players + injury_boost_players + high_variance_players
                )

                logger.info(f"✅ AI evaluated {len(recommendations)} edge case players")

                return {
                    'edge_case_recommendations': recommendations,
                    'ai_analysis': analysis_text,
                    'small_sample_count': len(small_sample_players),
                    'injury_opp_count': len(injury_boost_players),
                    'value_play_count': len(high_variance_players)
                }

            elif self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.2
                )

                analysis_text = response.choices[0].message.content

                # Track cost
                self.weekly_spend += 0.06
                self._log_cost("edge_case_player_analysis", 0.06)

                recommendations = self._parse_edge_case_recommendations(
                    analysis_text,
                    small_sample_players + injury_boost_players + high_variance_players
                )

                logger.info(f"✅ AI evaluated {len(recommendations)} edge case players")

                return {
                    'edge_case_recommendations': recommendations,
                    'ai_analysis': analysis_text,
                    'small_sample_count': len(small_sample_players),
                    'injury_opp_count': len(injury_boost_players),
                    'value_play_count': len(high_variance_players)
                }

            else:
                return {'edge_case_recommendations': [], 'ai_analysis': 'AI unavailable'}

        except Exception as e:
            logger.error(f"Edge case AI analysis failed: {e}")
            return {'edge_case_recommendations': [], 'ai_analysis': f'Error: {str(e)}'}

    def _parse_edge_case_recommendations(self, analysis_text: str,
                                         players: List[Dict]) -> List[Dict]:
        """Parse AI analysis into actionable recommendations - MORE AGGRESSIVE"""

        recommendations = []
        analysis_lower = analysis_text.lower()

        for player in players:
            player_name = player.get('name', '').lower()

            if player_name not in analysis_lower:
                continue

            # Extract context around player name
            player_pos = analysis_lower.find(player_name)
            context_start = max(0, player_pos - 200)
            context_end = min(len(analysis_text), player_pos + 200)
            context = analysis_lower[context_start:context_end]

            # Parse confidence (look for numbers 1-10)
            confidence = 5  # Default
            import re
            confidence_match = re.search(r'confidence[:\s]+(\d+)', context)
            if confidence_match:
                confidence = int(confidence_match.group(1))
            else:
                # ENHANCED: Infer confidence from language strength
                if any(word in context for word in ['must', 'definitely', 'excellent', 'top']):
                    confidence = 8
                elif any(word in context for word in ['should', 'good', 'solid']):
                    confidence = 7
                elif any(word in context for word in ['avoid', 'risky', 'pass']):
                    confidence = 7

            # Parse recommendation - MORE SENSITIVE to positive/negative words
            positive_words = ['start', 'play', 'roster', 'target', 'good', 'value', 'like', 'strong']
            negative_words = ['fade', 'avoid', 'skip', 'pass', 'risky', 'overpriced', 'bust']

            positive_score = sum(1 for word in positive_words if word in context)
            negative_score = sum(1 for word in negative_words if word in context)

            if positive_score > negative_score:
                recommendation = 'START'
                confidence = max(confidence, 7)  # Boost confidence for positive recs
            elif negative_score > positive_score:
                recommendation = 'FADE'
                confidence = max(confidence, 7)  # Boost confidence for negative recs
            elif 'monitor' in context or 'watch' in context:
                recommendation = 'MONITOR'
            else:
                recommendation = 'NEUTRAL'

            recommendations.append({
                'player_name': player.get('name'),
                'position': player.get('position'),
                'salary': player.get('salary'),
                'confidence': confidence,
                'recommendation': recommendation
            })

        return recommendations

        async def analyze_weekly_role_changes(self, player_data: List[Dict]) -> Dict[str, float]:
            """AI detects backup RBs becoming starters, target share shifts, TE breakouts
            Returns: {player_name: boost_factor} where boost is 1.0-1.35
            """

            # Identify role change candidates
            role_candidates = []

            for player in player_data:
                name = player.get('name', '')
                position = player.get('position', '')
                salary = player.get('salary', 0)
                fppg = player.get('projected_points', 0)
                games_played = player.get('games_played', 0)
                team = player.get('team', '')

                # CRITERIA: Recent role changes worth AI analysis
                is_candidate = False
                reason = ''

                # 1. Low games played but significant salary (new starter?)
                if 0 < games_played <= 3 and salary >= 5500 and fppg >= 8:
                    is_candidate = True
                    reason = f'New starter? {games_played} games, ${salary}, {fppg:.1f} FPPG'

                # 2. Mid-tier RB (backup getting carries)
                elif position == 'RB' and 5000 <= salary <= 7000 and fppg >= 10:
                    is_candidate = True
                    reason = f'Backup RB with volume? ${salary}, {fppg:.1f} FPPG'

                # 3. Cheap WR with targets (slot role?)
                elif position == 'WR' and salary <= 5500 and fppg >= 10:
                    is_candidate = True
                    reason = f'Value WR target share? ${salary}, {fppg:.1f} FPPG'

                # 4. TE breakout candidate
                elif position == 'TE' and 4500 <= salary <= 6000 and fppg >= 8:
                    is_candidate = True
                    reason = f'TE role expansion? ${salary}, {fppg:.1f} FPPG'

                if is_candidate:
                    role_candidates.append({
                        'name': name,
                        'position': position,
                        'team': team,
                        'salary': salary,
                        'fppg': fppg,
                        'games_played': games_played,
                        'reason': reason
                    })

            if not role_candidates:
                logger.info("No role change candidates for AI analysis")
                return {}

            logger.info(f"🔍 AI analyzing {len(role_candidates)} role change candidates")

            # AI Prompt
            prompt = f"""Analyze NFL DFS role changes for a 12-person friends league.

    ROLE CHANGE CANDIDATES:
    {json.dumps(role_candidates[:15], indent=2)}

    For each player, determine:
    1. BOOST FACTOR (1.0-1.35): How much to increase their projection
       - 1.00 = no change
       - 1.15 = moderate boost (backup getting 50% snaps)
       - 1.25 = significant boost (new starter)
       - 1.35 = major boost (featured role)

    2. CONFIDENCE (1-10): How confident in the role change

    Focus on:
    - Backup RBs getting starter snaps due to injury
    - WRs with expanded target share
    - TEs becoming primary receiving option
    - Recent coaching changes affecting usage

    Output format:
    PlayerName: boost=1.XX, confidence=X, reason

    Be conservative. Only boost if clear evidence of role expansion."""

            try:
                if self.claude_client:
                    response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=600,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    analysis_text = response.content[0].text
                    self.weekly_spend += 0.08
                    self._log_cost("role_change_analysis", 0.08)

                elif self.openai_client:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=600,
                        temperature=0.2
                    )

                    analysis_text = response.choices[0].message.content
                    self.weekly_spend += 0.05
                    self._log_cost("role_change_analysis", 0.05)

                else:
                    logger.warning("No AI available for role analysis")
                    return {}

                # Parse AI response
                boost_dict = self._parse_role_boosts(analysis_text, role_candidates)

                logger.info(f"✅ AI identified {len(boost_dict)} role changes")
                return boost_dict

            except Exception as e:
                logger.error(f"Role change AI analysis failed: {e}")
                return {}

        def _parse_role_boosts(self, analysis_text: str, candidates: List[Dict]) -> Dict[str, float]:
            """Parse AI analysis into {player_name: boost_factor}"""

            boost_dict = {}
            analysis_lower = analysis_text.lower()

            for candidate in candidates:
                player_name = candidate['name']
                name_lower = player_name.lower()

                if name_lower not in analysis_lower:
                    continue

                # Find context around player
                name_pos = analysis_lower.find(name_lower)
                context_start = max(0, name_pos - 100)
                context_end = min(len(analysis_text), name_pos + 200)
                context = analysis_lower[context_start:context_end]

                # Extract boost factor
                import re
                boost_match = re.search(r'boost[=:\s]+(1\.\d+)', context)
                if boost_match:
                    boost = float(boost_match.group(1))
                    boost = max(1.0, min(1.35, boost))  # Clamp 1.0-1.35

                    # Extract confidence
                    conf_match = re.search(r'confidence[=:\s]+(\d+)', context)
                    confidence = int(conf_match.group(1)) if conf_match else 5

                    # Only apply if confidence >= 6
                    if confidence >= 6:
                        boost_dict[player_name] = boost
                        logger.info(f"🤖 ROLE BOOST: {player_name} = {boost:.2f}x (confidence {confidence}/10)")

            return boost_dict
    def _validate_openai_key(self, key: str) -> bool:
        """Validate OpenAI API key format"""
        if not key:
            return False

        # OpenAI keys should start with 'sk-' and be substantial length
        if not key.startswith('sk-'):
            logger.error(f"OpenAI key should start with 'sk-', got: {key[:10]}...")
            return False

        # Accept both old format (~51 chars) and new project format (~164 chars)
        if len(key) < 40:
            logger.error(f"OpenAI key too short: {len(key)} chars (expected 40+)")
            return False

        if len(key) > 200:
            logger.error(f"OpenAI key too long: {len(key)} chars (expected <200)")
            return False

        # Check for truncation indicators
        if key.endswith('...') or '***' in key:
            logger.error("OpenAI key appears truncated")
            return False

        logger.info(f"✅ OpenAI key format valid: {len(key)} characters")
        return True

    def _validate_claude_key(self, key: str) -> bool:
        """Validate Claude API key format"""
        if not key:
            return False

        # Claude keys should start with 'sk-ant-'
        if not key.startswith('sk-ant-'):
            logger.error(f"Claude key should start with 'sk-ant-', got: {key[:10]}...")
            return False

        # Should be substantial length
        if len(key) < 40:
            logger.error(f"Claude key too short: {len(key)} chars (expected 40+)")
            return False

        return True

    def analyze_slate_for_optimization(self, player_data: List[Dict],
                                       weather_data: Dict,
                                       vegas_data: Dict,
                                       contest_type: str = 'gpp') -> Dict[str, Any]:
        """Main analysis function using available AI services with enhanced error handling"""

        if not (self.openai_client or self.claude_client):
            logger.warning("No AI services available, using enhanced fallback analysis")
            return self._enhanced_fallback_analysis(player_data, contest_type)

        try:
            # Prepare comprehensive data
            analysis_data = self._prepare_slate_data(player_data, weather_data, vegas_data, contest_type)

            # Get insights from available AI services
            openai_insights = None
            claude_insights = None

            if self.openai_client:
                try:
                    openai_insights = self._get_openai_insights(analysis_data, contest_type)
                    logger.info("✅ OpenAI analysis completed successfully")
                except Exception as e:
                    logger.error(f"❌ OpenAI analysis failed: {e}")

            if self.claude_client:
                try:
                    claude_insights = self._get_claude_insights(analysis_data, contest_type)
                    logger.info("✅ Claude analysis completed successfully")
                except Exception as e:
                    logger.error(f"❌ Claude analysis failed: {e}")

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
            return self._enhanced_fallback_analysis(player_data, contest_type)

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

    async def analyze_live_game_impact(self, locked_players: List[Dict],
                                       early_game_results: Dict,
                                       late_slate_players: List[Dict]) -> Dict[str, Any]:
        """Analyze how locked players' live performance affects late slate strategy"""

        analysis = {
            'locked_performance': {},
            'late_slate_adjustments': [],
            'strategy_changes': [],
            'confidence': 0.0
        }

        # Analyze each locked player's live performance
        for player in locked_players:
            player_name = player.get('name', '')
            team = player.get('team', '')
            position = player.get('position', '')

            # Get live game data for this player's team
            team_performance = early_game_results.get(team, {})

            if team_performance:
                game_script = self._analyze_game_script(team_performance)
                player_impact = self._calculate_player_live_impact(player, game_script)

                analysis['locked_performance'][player_name] = {
                    'live_points': player_impact.get('current_points', 0),
                    'projected_final': player_impact.get('projected_final', 0),
                    'vs_expectation': player_impact.get('vs_expectation', 0),
                    'game_script': game_script
                }

        # Calculate late slate strategy adjustments
        total_locked_performance = sum(
            p.get('projected_final', 0) for p in analysis['locked_performance'].values()
        )

        expected_locked_performance = sum(p.get('projected_points', 0) for p in locked_players)
        performance_delta = total_locked_performance - expected_locked_performance

        # Adjust late slate strategy based on locked player performance
        if performance_delta > 5:  # Locked players exceeding expectations
            analysis['strategy_changes'].append(
                "SAFE APPROACH: Locked players performing well, target floor plays in late slate")
            late_slate_focus = "floor"
        elif performance_delta < -5:  # Locked players underperforming
            analysis['strategy_changes'].append(
                "CEILING APPROACH: Locked players struggling, need high upside late slate")
            late_slate_focus = "ceiling"
        else:
            analysis['strategy_changes'].append(
                "BALANCED APPROACH: Locked players on track, maintain original strategy")
            late_slate_focus = "balanced"

        # Specific late slate player adjustments
        for late_player in late_slate_players:
            adjustment = self._calculate_late_slate_adjustment(late_player, late_slate_focus, performance_delta)
            if adjustment:
                analysis['late_slate_adjustments'].append(adjustment)

        analysis['confidence'] = 0.8 if early_game_results else 0.3
        return analysis

    def _analyze_game_script(self, team_performance: Dict) -> str:
        """Determine game script from live performance"""
        score_diff = team_performance.get('score_differential', 0)
        time_remaining = team_performance.get('time_remaining_pct', 100)

        if score_diff > 14 and time_remaining < 50:
            return "blowout_winning"
        elif score_diff < -14 and time_remaining < 50:
            return "blowout_losing"
        elif abs(score_diff) <= 7:
            return "close_game"
        else:
            return "neutral"

    def _calculate_player_live_impact(self, player: Dict, game_script: str) -> Dict:
        """Calculate live player impact based on game script"""
        base_projection = player.get('projected_points', 0)

        # Adjust based on game script
        if game_script == "blowout_winning":
            if player.get('position') == 'RB':
                projected_final = base_projection * 1.2  # More carries
            else:
                projected_final = base_projection * 0.9  # Less passing
        elif game_script == "blowout_losing":
            if player.get('position') in ['QB', 'WR', 'TE']:
                projected_final = base_projection * 1.15  # Garbage time
            else:
                projected_final = base_projection * 0.8  # Less runs
        else:
            projected_final = base_projection

        return {
            'current_points': 0,  # Would need live API
            'projected_final': projected_final,
            'vs_expectation': projected_final - base_projection
        }

    def _calculate_late_slate_adjustment(self, player: Dict, focus: str, performance_delta: float) -> Dict:
        """Calculate specific adjustments for late slate players"""

        adjustment = {
            'player': player.get('name'),
            'original_value': player.get('value', 0),
            'adjustment_factor': 1.0,
            'reasoning': ''
        }

        if focus == "ceiling" and player.get('boom_rate', 0) > 0.2:
            adjustment['adjustment_factor'] = 1.15
            adjustment['reasoning'] = "High ceiling needed due to locked player underperformance"
        elif focus == "floor" and player.get('bust_rate', 0) < 0.15:
            adjustment['adjustment_factor'] = 1.10
            adjustment['reasoning'] = "Safe floor play to protect locked player advantage"

        return adjustment if adjustment['adjustment_factor'] != 1.0 else None

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
        """Prepare COMPREHENSIVE slate data for AI analysis with all analytics"""

        # Build rich player profiles with ALL available data
        enriched_players = []

        for player in player_data:
            name = player.get('name', '')
            position = player.get('position', '')
            team = player.get('team', '')
            salary = player.get('salary', 0)

            # Skip invalid players
            if not name or salary < 1000:
                continue

            # Core projections
            base_projection = player.get('projected_points', 0)

            # Monte Carlo analytics (if available)
            ceiling = player.get('ceiling_90', base_projection * 1.3)
            floor = player.get('floor_10', base_projection * 0.7)
            median = player.get('median_50', base_projection)
            boom_rate = player.get('boom_rate', 0.3)
            bust_rate = player.get('bust_rate', 0.2)

            # Value metrics
            value_ratio = base_projection / (salary / 1000) if salary > 0 else 0

            # Weather impact (if available)
            weather_factor = player.get('weather_factor', 1.0)

            # Vegas context (if available)
            vegas_mult = player.get('vegas_multiplier', 1.0)

            # Injury status
            injury_status = player.get('injury_status', 'healthy')

            enriched_players.append({
                'name': name,
                'position': position,
                'team': team,
                'salary': salary,
                'projection': round(base_projection, 1),
                'ceiling': round(ceiling, 1),
                'floor': round(floor, 1),
                'median': round(median, 1),
                'value_ratio': round(value_ratio, 2),
                'boom_rate': round(boom_rate, 2),
                'bust_rate': round(bust_rate, 2),
                'weather_factor': round(weather_factor, 2),
                'vegas_mult': round(vegas_mult, 2),
                'injury_status': injury_status
            })

        # Sort by projection for top players
        enriched_players = sorted(enriched_players, key=lambda x: x['projection'], reverse=True)

        # Organize by position with FULL analytics
        positions = {'QB': 5, 'RB': 8, 'WR': 10, 'TE': 6}
        top_players_by_pos = {}

        for pos, count in positions.items():
            pos_players = [p for p in enriched_players if p['position'] == pos]
            top_players_by_pos[pos] = pos_players[:count]

        # Identify true value plays (high ceiling + good value)
        value_plays = [
            p for p in enriched_players
            if p['value_ratio'] > 3.2
               and p['salary'] < 7500
               and p['boom_rate'] > 0.25
        ]
        value_plays = sorted(value_plays, key=lambda x: x['value_ratio'], reverse=True)[:8]

        # Extract Vegas game context
        high_total_games = vegas_data.get('high_total_games', [])
        vegas_summary = []
        for game in high_total_games[:3]:
            vegas_summary.append({
                'game': game.get('game_id', ''),
                'total': game.get('total', 0),
                'spread': game.get('spread', 0),
                'home_implied': game.get('home_implied', 0),
                'away_implied': game.get('away_implied', 0)
            })

        # Weather impacts
        weather_summary = []
        for team, conditions in weather_data.items():
            if conditions.get('wind_speed', 0) > 10 or conditions.get('precipitation_chance', 0) > 20:
                weather_summary.append({
                    'team': team,
                    'conditions': conditions.get('conditions', ''),
                    'wind': conditions.get('wind_speed', 0),
                    'precip': conditions.get('precipitation_chance', 0),
                    'temp': conditions.get('temperature', 0)
                })

        return {
            'contest_type': contest_type,
            'top_players': top_players_by_pos,
            'value_plays': value_plays,
            'vegas_games': vegas_summary,
            'weather_impacts': weather_summary,
            'slate_size': len(enriched_players),
            'avg_salary': sum(p['salary'] for p in enriched_players) / len(enriched_players) if enriched_players else 0,
            # Include ALL players for H2H (small slate)
            'all_players': enriched_players if contest_type == 'h2h' else []
        }

    def _get_openai_insights(self, data: Dict, contest_type: str) -> str:
            """Get strategic insights from OpenAI with FULL analytics"""

            if contest_type == 'h2h':
                # Format rich player data for AI
                qb_details = []
                for qb in data['top_players']['QB'][:3]:
                    qb_details.append(
                        f"{qb['name']} (${qb['salary']}) - "
                        f"Proj: {qb['projection']}pts, "
                        f"Ceiling: {qb['ceiling']}pts, "
                        f"Floor: {qb['floor']}pts, "
                        f"Boom: {qb['boom_rate']:.0%}"
                    )

                te_details = []
                for te in data['top_players']['TE'][:4]:
                    te_details.append(
                        f"{te['name']} (${te['salary']}) - "
                        f"Proj: {te['projection']}pts, "
                        f"Ceiling: {te['ceiling']}pts, "
                        f"Value: {te['value_ratio']:.2f}x"
                    )

                wr_details = []
                for wr in data['top_players']['WR'][:6]:
                    wr_details.append(
                        f"{wr['name']} (${wr['salary']}) - "
                        f"Proj: {wr['projection']}pts, "
                        f"Ceiling: {wr['ceiling']}pts, "
                        f"Boom: {wr['boom_rate']:.0%}"
                    )

                rb_details = []
                for rb in data['top_players']['RB'][:6]:
                    rb_details.append(
                        f"{rb['name']} (${rb['salary']}) - "
                        f"Proj: {rb['projection']}pts, "
                        f"Ceiling: {rb['ceiling']}pts, "
                        f"Boom: {rb['boom_rate']:.0%}"
                    )

                vegas_details = "\n".join([
                    f"{g['game']}: {g['total']}pt total, {g['home_implied']:.1f} vs {g['away_implied']:.1f} implied"
                    for g in data.get('vegas_games', [])
                ])

                weather_str = "Indoor/No impact" if not data.get('weather_impacts') else "\n".join([
                    f"{w['team']}: {w['conditions']}, {w['wind']}mph wind, {w['precip']}% precip"
                    for w in data['weather_impacts']
                ])

                prompt = f"""You are analyzing NFL DFS HEAD-TO-HEAD (1v1) with COMPLETE analytics.

    🎯 H2H MISSION: Beat ONE person. Maximum ceiling wins.

    GAME CONTEXT:
    {vegas_details}

    WEATHER:
    {weather_str}

    QUARTERBACKS (with Monte Carlo ceiling analysis):
    {chr(10).join(qb_details)}

    RUNNING BACKS (ceiling matters more than floor in H2H):
    {chr(10).join(rb_details)}

    WIDE RECEIVERS (boom rate = upside potential):
    {chr(10).join(wr_details)}

    TIGHT ENDS (stack partners for QBs):
    {chr(10).join(te_details)}

    VALUE PLAYS (high ceiling + good value):
    {[(p['name'], f"${p['salary']}", f"{p['ceiling']}pt ceiling", f"{p['value_ratio']:.2f}x value") for p in data['value_plays'][:5]]}

    ANALYZE & RECOMMEND:
    1. MUST-PLAY QB: Which QB has highest CEILING (not projection)? Consider boom rate.
    2. STACK-WITH: Which 2 teammates give best correlation? Must be TE/WR $6K+.
    3. BRING-BACK: Best opposing player for game script hedge?
    4. AVOID: Any player with ceiling <12pts or bust rate >30% or salary <$3K?

    Be SPECIFIC with names. Focus on CEILING not floor. This is 1v1 ceiling game."""

            else:
                # GPP/Cash prompts...
                prompt = f"""... standard prompts for other contests ..."""

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

HIGH-TOTAL GAMES (47+ points - DFS GOLD):
{[(g.get('game_id'), f"{g.get('total')}pts") for g in data.get('vegas_high_total_games', [])[:6]]}

CRITICAL: Players from these high-scoring games should be prioritized heavily in friends league format.
In a 12-person league, you need ceiling plays from games expected to produce 24+ points per team.

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
        # Log the raw AI insights for debugging
        logger.info(f"🤖 RAW AI INSIGHTS (first 800 chars):\n{all_insights[:800]}")
        if not all_insights:
            return self._enhanced_fallback_analysis(player_data, contest_type)

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

    def _enhanced_fallback_analysis(self, players: List[Dict], contest_type: str) -> Dict[str, Any]:
        """Enhanced fallback when AI is unavailable"""

        # Value-based analysis with smarter logic
        value_targets = []
        expensive_fades = []
        stack_candidates = []

        # Group by team for stacking analysis
        teams = {}
        for player in players:
            team = player.get('team', '')
            if team:
                if team not in teams:
                    teams[team] = []
                teams[team].append(player)

        for player in players:
            name = player.get('name', '')
            position = player.get('position', '')
            salary = player.get('salary', 5000)
            projection = player.get('projected_points', 0)
            team = player.get('team', '')

            if salary > 0 and projection > 0 and name:
                value = projection / (salary / 1000)

                # Enhanced value targeting
                if contest_type == 'gpp':
                    # GPP: Target ceiling + value
                    if value > 3.2 and salary < 8000 and position in ['WR', 'TE']:
                        value_targets.append(name)
                    elif salary > 9500 and value < 2.5:
                        expensive_fades.append(name)
                elif contest_type == 'cash':
                    # Cash: Target consistent value
                    if value > 3.8 and salary < 7000:
                        value_targets.append(name)
                elif contest_type == 'contrarian':
                    # Contrarian: Target low ownership value
                    if value > 3.0 and salary < 6000:
                        value_targets.append(name)
                    elif salary > 9000:
                        expensive_fades.append(name)

        # Identify potential stacking teams
        for team, team_players in teams.items():
            qbs = [p for p in team_players if p.get('position') == 'QB']
            wrs = [p for p in team_players if p.get('position') == 'WR']

            if qbs and len(wrs) >= 2:
                avg_salary = sum(p.get('salary', 0) for p in qbs + wrs[:2]) / 3
                if avg_salary > 6500:  # High-priced stack
                    stack_candidates.append(team)

        # Contest-specific strategy
        strategy_text = f"Enhanced {contest_type} analysis: "
        if contest_type == 'gpp':
            strategy_text += "Target ceiling plays with correlation stacking, leverage low-owned value"
        elif contest_type == 'cash':
            strategy_text += "Focus on high-floor value plays, avoid volatile expensive options"
        elif contest_type == 'contrarian':
            strategy_text += "Fade obvious chalk, target contrarian value with upside"
        else:
            strategy_text += "Balanced approach with salary efficiency"

        return {
            'type': contest_type,
            'ai_strategy': strategy_text,
            'leverage_players': value_targets[:6],
            'avoid_players': expensive_fades[:4],
            'stack_teams': stack_candidates[:3],
            'ownership_adjustments': {},
            'ai_confidence': 0.6,  # Higher than basic fallback
            'ai_enabled': False,
            'sources_used': ['enhanced_fallback'],
            'analysis_quality': 'enhanced_logic'
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

        async def analyze_edge_case_players(self, player_data: List[Dict]) -> Dict[str, Any]:
            """AI analysis of small-sample, injury-opportunity, and edge case players"""

            # Identify edge cases needing AI evaluation
            small_sample_players = []
            injury_boost_players = []
            high_variance_players = []

            for player in player_data:
                games_played = player.get('games_played', 0)
                name = player.get('name', '')
                position = player.get('position', '')
                salary = player.get('salary', 0)
                fppg = player.get('projected_points', 0)
                injury_opp = player.get('injury_opportunity', False)

                # Small sample (1-2 games, significant salary, good FPPG)
                if 0 < games_played <= 2 and salary > 5000 and fppg > 10:
                    small_sample_players.append({
                        'name': name,
                        'position': position,
                        'salary': salary,
                        'fppg': fppg,
                        'games': games_played,
                        'reason': f'{fppg:.1f} FPPG on only {games_played} game(s)'
                    })

                # Injury opportunity players
                if injury_opp:
                    injured_starter = player.get('injured_starter', '')
                    injury_boost_players.append({
                        'name': name,
                        'position': position,
                        'salary': salary,
                        'fppg': fppg,
                        'injured_starter': injured_starter
                    })

                # High variance value plays
                value = (fppg / (salary / 1000)) if salary > 0 else 0
                if value > 3.5 and salary < 5500:
                    high_variance_players.append({
                        'name': name,
                        'position': position,
                        'salary': salary,
                        'fppg': fppg,
                        'value': value
                    })

            # Only call AI if there are edge cases
            if not (small_sample_players or injury_boost_players or high_variance_players):
                return {'edge_case_recommendations': [], 'ai_analysis': 'No edge cases to evaluate'}

            # Construct AI prompt
            prompt = f"""You're evaluating edge-case NFL DFS players for a 12-person friends league tournament.
    Goal: Determine which edge cases are REAL opportunities vs traps.

    SMALL SAMPLE PLAYERS (limited games):
    {json.dumps(small_sample_players[:10], indent=2) if small_sample_players else 'None'}

    INJURY OPPORTUNITY PLAYERS (backups getting starts):
    {json.dumps(injury_boost_players[:10], indent=2) if injury_boost_players else 'None'}

    HIGH VALUE PLAYERS (3.5+ value):
    {json.dumps(high_variance_players[:10], indent=2) if high_variance_players else 'None'}

    For each player, provide:
    1. CONFIDENCE (1-10): How confident they'll produce?
    2. RECOMMENDATION: START/FADE/MONITOR
    3. REASONING: Why? (1 sentence)

    Focus on actionable advice for beating 11 other people weekly."""

            try:
                if self.claude_client:
                    response = self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=800,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    analysis_text = response.content[0].text
                    self.weekly_spend += 0.08
                    self._log_cost("edge_case_player_analysis", 0.08)

                elif self.openai_client:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800,
                        temperature=0.2
                    )

                    analysis_text = response.choices[0].message.content
                    self.weekly_spend += 0.06
                    self._log_cost("edge_case_player_analysis", 0.06)

                else:
                    return {'edge_case_recommendations': [], 'ai_analysis': 'AI unavailable'}

                # Parse recommendations
                recommendations = self._parse_edge_case_recommendations(
                    analysis_text,
                    small_sample_players + injury_boost_players + high_variance_players
                )

                logger.info(f"AI evaluated {len(recommendations)} edge case players")

                return {
                    'edge_case_recommendations': recommendations,
                    'ai_analysis': analysis_text,
                    'small_sample_count': len(small_sample_players),
                    'injury_opp_count': len(injury_boost_players),
                    'value_play_count': len(high_variance_players)
                }

            except Exception as e:
                logger.error(f"Edge case AI analysis failed: {e}")
                return {'edge_case_recommendations': [], 'ai_analysis': f'Error: {str(e)}'}

        def _parse_edge_case_recommendations(self, analysis_text: str,
                                             players: List[Dict]) -> List[Dict]:
            """Parse AI analysis into actionable recommendations"""

            recommendations = []
            analysis_lower = analysis_text.lower()

            for player in players:
                player_name = player.get('name', '').lower()

                if player_name not in analysis_lower:
                    continue

                # Extract context around player name
                player_pos = analysis_lower.find(player_name)
                context_start = max(0, player_pos - 200)
                context_end = min(len(analysis_text), player_pos + 200)
                context = analysis_lower[context_start:context_end]

                # Parse confidence (look for numbers 1-10)
                confidence = 5  # Default
                import re
                confidence_match = re.search(r'confidence[:\s]+(\d+)', context)
                if confidence_match:
                    confidence = int(confidence_match.group(1))

                # Parse recommendation
                if 'start' in context or 'play' in context:
                    recommendation = 'START'
                elif 'fade' in context or 'avoid' in context:
                    recommendation = 'FADE'
                elif 'monitor' in context:
                    recommendation = 'MONITOR'
                else:
                    recommendation = 'NEUTRAL'

                recommendations.append({
                    'player_name': player.get('name'),
                    'position': player.get('position'),
                    'salary': player.get('salary'),
                    'confidence': confidence,
                    'recommendation': recommendation
                })

            return recommendations
# Backwards compatibility
WinningDFSAIAnalyzer = DualAIDFSAnalyzer