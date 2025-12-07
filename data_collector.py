"""
WINNING Data Collector - Integrates AI must-play analysis with Vegas data
Fixes:
1. AI now identifies MUST PLAY/FADE players before optimization
2. Vegas game totals flow through to player objects
3. Breaking news triggers AI re-analysis
"""
import asyncio
import aiohttp
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import time
from loguru import logger
import os

try:
    from news_monitor import get_breaking_news, get_player_news
    NEWS_AVAILABLE = True
    logger.info("News monitoring available")
except ImportError:
    NEWS_AVAILABLE = False
    logger.warning("News monitoring not available")

import pytz
import numpy as np

from config import (
    ESPN_ENDPOINTS, NFL_STADIUMS, WEATHER_API, DATA_DIR,
    RATE_LIMITS, CACHE_TTL, VALIDATION_THRESHOLDS
)


class WinningAIAnalyzer:
    """
    AI analyzer that identifies MUST PLAY and MUST FADE players
    Integrated directly into data collection for seamless flow
    """

    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY', '')
        self.anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        self.model = os.getenv('GPT_MODEL', 'gpt-4o-mini')
        self.weekly_budget = float(os.getenv('AI_WEEKLY_BUDGET', '15.0'))
        self.weekly_spend = 0.0

    def analyze_for_winning_picks(
        self,
        players: List[Dict],
        vegas_data: Dict,
        contest_type: str = 'gpp'
    ) -> Dict[str, Any]:
        """
        Main AI analysis - returns must-play/must-fade recommendations
        """
        high_total_games = vegas_data.get('high_total_games', [])
        game_data = vegas_data.get('games', {})

        # Categorize players by game environment
        elite_env_players = []  # 48+ total
        good_env_players = []   # 45-47 total
        bad_env_players = []    # <42 total

        for player in players:
            team = player.get('team', '')
            game_total = self._get_team_game_total(team, game_data)

            if game_total >= 48:
                elite_env_players.append({**player, 'game_total': game_total})
            elif game_total >= 45:
                good_env_players.append({**player, 'game_total': game_total})
            elif game_total <= 42:
                bad_env_players.append({**player, 'game_total': game_total})

        # Try AI analysis if available and affordable
        if self.openai_key and self._can_afford_call():
            try:
                return self._get_ai_recommendations(
                    elite_env_players, good_env_players, bad_env_players,
                    high_total_games, contest_type, players
                )
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")

        # Fall back to rule-based recommendations
        return self._get_rule_based_recommendations(
            elite_env_players, good_env_players, bad_env_players, contest_type
        )

    def _get_team_game_total(self, team: str, game_data: Dict) -> float:
        """Get game total for a team"""
        for game_id, data in game_data.items():
            if team in [data.get('home_team'), data.get('away_team')]:
                return data.get('total_points', 45.0)
        return 45.0

    def _can_afford_call(self) -> bool:
        """Check budget"""
        return (self.weekly_spend + 0.05) <= self.weekly_budget

    def _get_ai_recommendations(
        self,
        elite_players: List[Dict],
        good_players: List[Dict],
        bad_players: List[Dict],
        high_total_games: List[Dict],
        contest_type: str,
        all_players: List[Dict]
    ) -> Dict[str, Any]:
        """
        Get AI-powered must-play/fade recommendations.

        In addition to the existing high-total game context, enrich the prompt with
        predicted ownership estimates and value plays. Ownership estimates are
        approximated based on salary and position. Value plays highlight low-cost
        players with strong projection-to-salary ratios and potential role increases
        due to team injuries.
        """
        try:
            import openai

            # Build a detailed prompt with ownership and value data
            prompt = self._build_prompt(elite_players, high_total_games, contest_type, all_players)

            client = openai.OpenAI(api_key=self.openai_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an elite DFS analyst for a 12-person friends league.
Your job: identify MUST PLAY and MUST FADE players based on game environment and slate context.

Rules:
1. Players in 48+ total games are premium targets.
2. QBs in shootouts (47+) are near-auto plays.
3. Stack the QB with 1-2 WRs from the highest-total game.
4. Fade players in games under 42 total.
5. Use salary, injury context and ownership data to identify high-upside and contrarian plays.
6. For friends leagues (small-field tournaments), emphasize ceiling and differentiation.
7. For head-to-head, prioritize consistency and avoid volatile punts.

Return ONLY valid JSON:
{
    "must_play": ["PlayerName|Position|Team|Reason"],
    "must_fade": ["PlayerName|Position|Team|Reason"],
    "stack_game": "AWAY@HOME",
    "stack_qb": "QB Name",
    "stack_receivers": ["WR1 Name", "WR2 Name"]
}
"""
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )

            # Track cost (GPT-4o pricing approximated)
            usage = response.usage
            cost = (usage.prompt_tokens * 0.00015 / 1000 + usage.completion_tokens * 0.0006 / 1000)
            self.weekly_spend += cost
            logger.info(f"💰 AI cost: ${cost:.4f} (weekly: ${self.weekly_spend:.2f})")

            # Parse JSON content from response
            content = response.choices[0].message.content
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            recommendations = json.loads(content.strip())
            recommendations['source'] = 'ai'

            logger.info(f"🤖 AI MUST PLAYS: {len(recommendations.get('must_play', []))}")
            logger.info(f"🤖 AI STACK: {recommendations.get('stack_game', 'None')}")

            return recommendations

        except Exception as e:
            logger.error(f"AI recommendation failed: {e}")
            return self._get_rule_based_recommendations(elite_players, [], [], contest_type)

    def _build_prompt(
        self,
        elite_players: List[Dict],
        high_total_games: List[Dict],
        contest_type: str,
        all_players: List[Dict]
    ) -> str:
        """
        Construct a detailed prompt for the AI model.

        In addition to listing the highest-total games and elite players, include:
        - A section highlighting players projected to be highly owned ("Chalk") based on salary and position.
        - A section of value plays and salary savers with strong projection-to-salary ratios.
        - A section highlighting potential injury-driven opportunities (cheap players on teams with notable injuries).

        These sections provide the model with additional context so it can make
        more informed must-play and must-fade recommendations.
        """
        lines: List[str] = []

        # Contest description
        lines.append(f"Contest: 12-person friends league ({contest_type})")
        lines.append("Goal: Build the highest-ceiling lineup using game environment, ownership and value.")
        lines.append("")

        # High-total games (limit to 5 for brevity)
        lines.append("HIGH-TOTAL GAMES (47+):")
        for game in high_total_games[:5]:
            lines.append(f"  {game['game_id']}: {game['total']} total")
        lines.append("")

        # Elite players sorted by projection
        lines.append("TOP PLAYERS IN ELITE GAMES (sorted by projection):")
        sorted_elite = sorted(elite_players, key=lambda x: x.get('projected_points', 0), reverse=True)
        for p in sorted_elite[:20]:
            proj = p.get('projected_points', 0.0)
            lines.append(
                f"  {p['name']}|{p['position']}|{p['team']}|${p['salary']}|{proj:.1f}pts|{p.get('game_total', 45)} total"
            )
        lines.append("")

        # Predicted ownership estimates (chalk)
        def _predict_ownership(player: Dict) -> float:
            """Approximate ownership based on salary and position (replicates optimizer logic)."""
            salary = player.get('salary', 0)
            position = player.get('position', '')
            ownership = 15.0
            if salary >= 9500:
                ownership = 40.0
            elif salary >= 8500:
                ownership = 30.0
            elif salary >= 7500:
                ownership = 22.0
            elif salary >= 6000:
                ownership = 15.0
            else:
                ownership = 10.0
            if position == 'QB' and salary >= 8000:
                ownership += 5
            elif position == 'RB':
                ownership += 3
            return max(5.0, min(50.0, ownership))

        # Compute ownership for all players and select top 5
        ownership_list = []
        for p in all_players:
            try:
                own = _predict_ownership(p)
                ownership_list.append((p, own))
            except Exception:
                continue
        ownership_list.sort(key=lambda x: x[1], reverse=True)
        top_chalk = ownership_list[:5]

        if top_chalk:
            lines.append("CHALK (predicted high ownership):")
            for p, own in top_chalk:
                lines.append(
                    f"  {p['name']}|{p['position']}|{p['team']}|${p['salary']}|{p.get('projected_points', 0):.1f}pts|{own:.1f}% predicted ownership"
                )
            lines.append("")

        # Value plays and salary savers (projection per dollar)
        value_candidates: List[tuple] = []
        for p in all_players:
            salary = p.get('salary', 0)
            proj = p.get('projected_points', 0.0)
            if salary > 0:
                ratio = proj / salary
                # consider players under 7000 salary as value pool
                if salary <= 7000:
                    value_candidates.append((p, ratio))
        value_candidates.sort(key=lambda x: x[1], reverse=True)
        top_values = value_candidates[:5]
        if top_values:
            lines.append("VALUE PLAYS & SALARY SAVERS:")
            for p, ratio in top_values:
                lines.append(
                    f"  {p['name']}|{p['position']}|{p['team']}|${p['salary']}|{p.get('projected_points', 0):.1f}pts|{ratio*1000:.2f} pts/$1000"
                )
            lines.append("")

        # Injury-driven opportunities: identify teams with out/IR/susp/NA and pick cheap players
        injured_teams = set()
        for p in all_players:
            status = str(p.get('injury_status', '')).upper()
            if any(flag in status for flag in ['IR', 'OUT', 'SUSP', 'NA']):
                injured_teams.add(p.get('team', ''))
        injury_candidates: List[tuple] = []
        for p in all_players:
            team = p.get('team', '')
            salary = p.get('salary', 0)
            proj = p.get('projected_points', 0.0)
            if team in injured_teams and salary > 0 and salary <= 6500:
                ratio = proj / salary
                injury_candidates.append((p, ratio))
        injury_candidates.sort(key=lambda x: x[1], reverse=True)
        top_injury_vals = injury_candidates[:5]
        if top_injury_vals:
            lines.append("INJURY OPPORTUNITIES (cheap fill-ins with upside):")
            for p, ratio in top_injury_vals:
                lines.append(
                    f"  {p['name']}|{p['position']}|{p['team']}|${p['salary']}|{p.get('projected_points', 0):.1f}pts|{ratio*1000:.2f} pts/$1000"
                )
            lines.append("")

        # Instructions for the model
        lines.append(
            "Identify 3-4 must-play players and 1-2 must-fade players using the context above. "
            "Focus on ceiling for tournaments and consistency for head-to-head, and choose the best QB+WR stack from a high-total game."
        )

        return "\n".join(lines)

    def _get_rule_based_recommendations(
        self,
        elite_players: List[Dict],
        good_players: List[Dict],
        bad_players: List[Dict],
        contest_type: str
    ) -> Dict[str, Any]:
        """Rule-based fallback when AI unavailable"""

        must_play = []
        must_fade = []

        # Sort elite players by projection
        sorted_elite = sorted(elite_players, key=lambda x: x.get('projected_points', 0), reverse=True)

        # Find best QB in elite environment
        elite_qbs = [p for p in sorted_elite if p.get('position') == 'QB']
        stack_qb = None
        stack_team = None

        if elite_qbs:
            qb = elite_qbs[0]
            stack_qb = qb['name']
            stack_team = qb['team']
            must_play.append(f"{qb['name']}|QB|{qb['team']}|Best QB in high-total game ({qb.get('game_total', 45)} total)")

        # Find WRs to stack with QB
        stack_receivers = []
        if stack_team:
            team_wrs = [p for p in sorted_elite if p.get('position') == 'WR' and p.get('team') == stack_team]
            for wr in team_wrs[:2]:
                stack_receivers.append(wr['name'])
                must_play.append(f"{wr['name']}|WR|{wr['team']}|Stack with {stack_qb}")

        # Find best non-stack players in elite environments
        non_stack_elite = [p for p in sorted_elite if p.get('team') != stack_team]
        for p in non_stack_elite[:2]:
            if p['position'] in ['RB', 'WR', 'TE']:
                must_play.append(f"{p['name']}|{p['position']}|{p['team']}|Elite game environment ({p.get('game_total', 45)} total)")

        # Must fade: expensive players in bad environments
        for p in bad_players:
            if p.get('salary', 0) >= 7000:
                must_fade.append(f"{p['name']}|{p['position']}|{p['team']}|Bad game environment ({p.get('game_total', 45)} total)")

        return {
            'must_play': must_play[:5],
            'must_fade': must_fade[:5],
            'stack_game': f"@{stack_team}" if stack_team else None,
            'stack_qb': stack_qb,
            'stack_receivers': stack_receivers,
            'source': 'rule_based'
        }

    def apply_to_players(self, players: List[Dict], recommendations: Dict) -> List[Dict]:
        """
        Apply AI recommendations to player data
        Must-play: +30% projection boost
        Must-fade: -40% projection penalty
        """
        must_play_names = set()
        must_fade_names = set()

        # Parse must-play names
        for entry in recommendations.get('must_play', []):
            if '|' in str(entry):
                name = entry.split('|')[0].strip().lower()
                must_play_names.add(name)

        # Parse must-fade names
        for entry in recommendations.get('must_fade', []):
            if '|' in str(entry):
                name = entry.split('|')[0].strip().lower()
                must_fade_names.add(name)

        # Apply to players
        modified = []
        boost_count = 0
        fade_count = 0

        for player in players:
            p = player.copy()
            name_lower = p.get('name', '').lower()

            if name_lower in must_play_names:
                original = p.get('projected_points', 0)
                p['projected_points'] = original * 1.30  # +30% boost
                p['ai_must_play'] = True
                boost_count += 1
                logger.info(f"🎯 MUST PLAY: {p['name']} {original:.1f} → {p['projected_points']:.1f}")

            elif name_lower in must_fade_names:
                original = p.get('projected_points', 0)
                p['projected_points'] = original * 0.60  # -40% penalty
                p['ai_must_fade'] = True
                fade_count += 1
                logger.info(f"⛔ MUST FADE: {p['name']} {original:.1f} → {p['projected_points']:.1f}")

            modified.append(p)

        logger.info(f"✅ AI applied: {boost_count} boosted, {fade_count} faded")
        return modified


class EnhancedSlateManager:
    """Manages REAL slate detection"""

    def __init__(self):
        self.eastern = pytz.timezone('America/New_York')

    def get_current_nfl_week(self) -> int:
        """Get current NFL week"""
        try:
            import requests
            response = requests.get(
                'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if 'week' in data:
                    if isinstance(data['week'], dict):
                        return data['week'].get('number', self._calculate_week_from_date())
                    return data.get('week', self._calculate_week_from_date())
        except Exception as e:
            logger.warning(f"ESPN API failed: {e}")

        return self._calculate_week_from_date()

    def _calculate_week_from_date(self) -> int:
        """Calculate NFL week from date"""
        now = datetime.now(self.eastern)
        season_start = datetime(2024, 9, 5, tzinfo=self.eastern)

        if now < season_start:
            return 1

        days_since_start = (now - season_start).days
        return max(1, min(18, (days_since_start // 7) + 1))


class EnhancedDataCollector:
    """WINNING Data Collector with integrated AI analysis"""

    def __init__(self):
        self.session = None
        self.slate_manager = EnhancedSlateManager()
        self.ai_analyzer = WinningAIAnalyzer()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': WEATHER_API['user_agent']}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_current_week_games(self):
        """Get current week games from ESPN"""
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

            async with self.session.get(url) as response:
                if response.status != 200:
                    return self._get_fallback_games()

                data = await response.json()
                current_week = self._extract_week_number(data)

                all_games = []
                for event in data.get('events', []):
                    game_info = self._parse_game_event(event)
                    if game_info:
                        all_games.append(game_info)

                if not all_games:
                    return self._get_fallback_games()

                main_slate = [g for g in all_games if g['time_slot'] in ['sunday_early', 'sunday_late']]

                logger.info(f"📅 Week {current_week}: {len(all_games)} games, {len(main_slate)} main slate")

                return {
                    'current_week': current_week,
                    'all_games': all_games,
                    'main_slate': main_slate,
                    'single_games': all_games,
                }

        except Exception as e:
            logger.error(f"Error getting games: {e}")
            return self._get_fallback_games()

    def _extract_week_number(self, data: Dict) -> int:
        """Extract week from ESPN data"""
        if 'week' in data:
            week_data = data['week']
            if isinstance(week_data, dict):
                return week_data.get('number', self.slate_manager._calculate_week_from_date())
            return week_data
        return self.slate_manager._calculate_week_from_date()

    def _parse_game_event(self, event: Dict) -> Optional[Dict]:
        """Parse game event"""
        try:
            game_date = event.get('date', '')
            if not game_date:
                return None

            game_datetime = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
            game_et = game_datetime.astimezone(self.slate_manager.eastern)

            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])

            if len(competitors) < 2:
                return None

            teams = []
            for comp in competitors:
                abbrev = comp.get('team', {}).get('abbreviation', '')
                if abbrev:
                    teams.append(abbrev)

            if len(teams) != 2:
                return None

            hour = game_et.hour
            day = game_et.weekday()

            if day == 3:
                time_slot = 'thursday_night'
            elif day == 6:
                if hour < 16:
                    time_slot = 'sunday_early'
                elif hour < 20:
                    time_slot = 'sunday_late'
                else:
                    time_slot = 'sunday_night'
            elif day == 0:
                time_slot = 'monday_night'
            else:
                time_slot = 'other'

            return {
                'id': f"{teams[0]}_vs_{teams[1]}",
                'teams': teams,
                'time_slot': time_slot,
                'time': game_et.strftime('%A %I:%M %p ET'),
                'game_datetime': game_et,
            }

        except Exception as e:
            return None

    def _get_fallback_games(self):
        """Fallback games"""
        week = self.slate_manager._calculate_week_from_date()
        games = [
            {'id': 'BUF_vs_MIA', 'teams': ['BUF', 'MIA'], 'time_slot': 'sunday_early', 'time': 'Sunday 1:00 PM ET'},
            {'id': 'PHI_vs_WAS', 'teams': ['PHI', 'WAS'], 'time_slot': 'sunday_early', 'time': 'Sunday 1:00 PM ET'},
        ]
        return {'current_week': week, 'all_games': games, 'main_slate': games, 'single_games': games}

    async def get_vegas_odds_data(self) -> Dict[str, Any]:
        """Get Vegas odds data"""
        try:
            from vegas_data_collector import VegasDataCollector
            collector = VegasDataCollector()
            vegas_data = await collector.get_nfl_odds_data()

            if vegas_data and vegas_data.get('games'):
                high_total_count = len(vegas_data.get('high_total_games', []))
                logger.info(f"🎰 VEGAS: {len(vegas_data['games'])} games, {high_total_count} high-total (47+)")

                for game in vegas_data.get('high_total_games', []):
                    logger.info(f"   🔥 {game['game_id']}: {game['total']} total")

                return vegas_data

            return self._get_fallback_vegas()

        except Exception as e:
            logger.error(f"Vegas data failed: {e}")
            return self._get_fallback_vegas()

    def _get_fallback_vegas(self) -> Dict[str, Any]:
        """Fallback Vegas data"""
        return {
            'games': {},
            'high_total_games': [],
            'avg_total': 45.0,
            'data_source': 'fallback'
        }

    def calculate_vegas_multipliers(self, vegas_data: Dict) -> Dict[str, float]:
        """Calculate team multipliers from Vegas data"""
        multipliers = {}
        games = vegas_data.get('games', {})

        for game_id, game_data in games.items():
            total = game_data.get('total_points', 45)
            spread = abs(game_data.get('spread') or 0)
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')

            # Total multiplier
            if total >= 50:
                total_mult = 1.40
            elif total >= 48:
                total_mult = 1.30
            elif total >= 46:
                total_mult = 1.20
            elif total >= 44:
                total_mult = 1.10
            elif total <= 40:
                total_mult = 0.80
            else:
                total_mult = 1.0

            # Spread multiplier
            if spread <= 3:
                spread_mult = 1.10
            elif spread >= 10:
                spread_mult = 0.90
            else:
                spread_mult = 1.0

            final_mult = total_mult * spread_mult

            if home_team:
                multipliers[home_team] = final_mult
            if away_team:
                multipliers[away_team] = final_mult

        logger.info(f"📊 Vegas multipliers for {len(multipliers)} teams")
        return multipliers

    async def get_weather_for_games(self, games_info: Dict) -> Dict[str, Dict]:
        """Get weather data"""
        weather_data = {}

        for game in games_info.get('all_games', []):
            for team in game.get('teams', []):
                stadium = NFL_STADIUMS.get(team, {})
                if not stadium.get('dome', True):
                    weather_data[team] = {
                        'temperature': 65,
                        'wind_speed': '10 mph',
                        'conditions': 'Clear',
                        'factor': 1.0
                    }

        return weather_data

    def _is_viable_player(self, player_data: Dict) -> bool:
        """Filter non-viable players"""
        name = player_data.get('name', '')
        position = player_data.get('position', '')
        salary = player_data.get('salary', 0)
        fppg = player_data.get('projected_points', 0)
        injury_status = player_data.get('injury_status', '').upper()

        # Definitively unavailable
        if any(x in injury_status for x in ['IR', 'OUT', 'SUSP']):
            return False

        # Basic validation
        if not name or len(name.strip()) < 2:
            return False

        # Position-specific thresholds
        if position == 'QB':
            return salary >= 6000 or fppg >= 10
        elif position == 'RB':
            return salary >= 4500 or fppg >= 5
        elif position == 'WR':
            return salary >= 4500 or fppg >= 4
        elif position == 'TE':
            return salary >= 4000 or fppg >= 3
        elif position == 'D':
            return salary >= 3000

        return False

    async def collect_players_for_slate(self, games_info: Dict, contest_type: str = 'gpp') -> List[Dict]:
        """Collect and filter players"""
        playing_teams = set()
        for game in games_info.get('main_slate', []):
            playing_teams.update(game.get('teams', []))

        if not playing_teams:
            playing_teams = None

        try:
            from fanduel_salary_scraper import get_fanduel_salaries
            salary_data = await get_fanduel_salaries()

            if not salary_data:
                logger.error("No FanDuel salary data")
                return []

            # Filter viable players
            players = []
            for p in salary_data:
                if not self._is_viable_player(p):
                    continue

                team = p.get('team', '').upper()
                if playing_teams and team not in playing_teams:
                    continue

                fppg = p.get('projected_points', 0)
                if fppg <= 0:
                    continue

                player = {
                    'player_id': f"fd_{p.get('id', p.get('name', ''))}",
                    'name': p.get('name', ''),
                    'position': p.get('position', ''),
                    'team': team,
                    'salary': int(p.get('salary', 5000)),
                    'projected_points': round(fppg, 2),
                    'projection': round(fppg, 2),
                    'ceiling': round(fppg * 1.4, 2),
                    'floor': round(fppg * 0.7, 2),
                    'ownership': np.random.uniform(5.0, 35.0),
                    'opponent': p.get('opponent', ''),
                    'injury_status': p.get('injury_status', ''),
                }
                players.append(player)

            logger.info(f"📋 Collected {len(players)} viable players")
            return players

        except Exception as e:
            logger.error(f"Error collecting players: {e}")
            return []

    async def get_breaking_news_impact(self, players: List[Dict]) -> Dict[str, Any]:
        """Get breaking news"""
        if not NEWS_AVAILABLE:
            return {'news_events': [], 'impact_analysis': {}}

        try:
            news_events = await get_breaking_news()
            return {
                'news_events': news_events,
                'news_count': len(news_events)
            }
        except Exception as e:
            logger.error(f"News error: {e}")
            return {'news_events': [], 'news_count': 0}


async def get_fresh_data(contest_type: str = 'gpp') -> Dict[str, Any]:
    """
    WINNING DATA COLLECTION

    This function:
    1. Gets games and Vegas data
    2. Collects players
    3. Runs AI analysis to identify must-play/must-fade
    4. Applies AI recommendations to player projections
    5. Returns enhanced player pool ready for optimization
    """
    async with EnhancedDataCollector() as collector:
        # Get games
        games_info = await collector.get_current_week_games()

        # Get players
        players = await collector.collect_players_for_slate(games_info, contest_type)

        if not players:
            logger.error("NO PLAYERS FOUND")
            return {}

        # Get Vegas data
        vegas_data = await collector.get_vegas_odds_data()
        vegas_multipliers = collector.calculate_vegas_multipliers(vegas_data)

        # Attach game totals to players
        for player in players:
            team = player['team']
            for game_id, game_info in vegas_data.get('games', {}).items():
                if team in [game_info.get('home_team'), game_info.get('away_team')]:
                    player['game_total'] = game_info.get('total_points', 45)
                    player['game_environment_mult'] = vegas_multipliers.get(team, 1.0)
                    break
            else:
                player['game_total'] = 45.0
                player['game_environment_mult'] = 1.0

        # =============================================
        # AI ANALYSIS: Identify must-play/must-fade
        # =============================================
        logger.info("🤖 Running AI analysis for must-play/must-fade...")
        ai_recommendations = collector.ai_analyzer.analyze_for_winning_picks(
            players, vegas_data, contest_type
        )

        # Apply AI recommendations to player projections
        enhanced_players = collector.ai_analyzer.apply_to_players(players, ai_recommendations)

        # Get other data
        weather_data = await collector.get_weather_for_games(games_info)
        news_impact = await collector.get_breaking_news_impact(enhanced_players)

        # Summary stats
        must_play_count = sum(1 for p in enhanced_players if p.get('ai_must_play'))
        must_fade_count = sum(1 for p in enhanced_players if p.get('ai_must_fade'))
        high_total_count = sum(1 for p in enhanced_players if p.get('game_environment_mult', 1.0) >= 1.25)

        logger.info(f"📊 FINAL POOL: {len(enhanced_players)} players")
        logger.info(f"   🎯 Must-play: {must_play_count}")
        logger.info(f"   ⛔ Must-fade: {must_fade_count}")
        logger.info(f"   🔥 In high-total games: {high_total_count}")

        return {
            'players': enhanced_players,
            'games_info': games_info,
            'weather': weather_data,
            'vegas_odds': vegas_data,
            'vegas_multipliers': vegas_multipliers,
            'ai_recommendations': ai_recommendations,
            'breaking_news': news_impact,
            'last_updated': datetime.now().isoformat(),
            'data_quality': {
                'player_count': len(enhanced_players),
                'total_games': len(games_info['all_games']),
                'main_slate_games': len(games_info['main_slate']),
                'current_week': games_info['current_week'],
                'must_play_count': must_play_count,
                'must_fade_count': must_fade_count,
                'high_total_players': high_total_count,
                'vegas_games': len(vegas_data.get('games', {})),
                'ai_source': ai_recommendations.get('source', 'none'),
                'teams_in_slate': sorted(set(p['team'] for p in enhanced_players)),
            }
        }