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
import math

from config import (
    ESPN_ENDPOINTS, NFL_STADIUMS, WEATHER_API, DATA_DIR,
    RATE_LIMITS, CACHE_TTL, VALIDATION_THRESHOLDS
)


def safe_float(val, default=0.0):
    """Sanitize float values - replace NaN/inf with default"""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


class WinningAIAnalyzer:
    """
    AI analyzer that identifies:
    - MUST PLAY players based on projection, game environment, role, injuries
    - MUST FADE players based on risk, ownership, injuries, role uncertainty
    - Provides adjusted projections to feed optimizer
    """
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.ai_enabled = bool(self.openai_api_key or self.anthropic_api_key)
        self.last_prompt_cost = 0.0

        if self.ai_enabled:
            logger.info("AI Analyzer initialized - projections will be enhanced")
        else:
            logger.warning("AI Analyzer disabled - missing API keys")

    def _estimate_cost(self, prompt_tokens: int, model: str = "gpt-4o-mini") -> float:
        """
        Rough cost estimate so we don't burn your $15/week
        """
        # Very rough estimate in dollars
        if "gpt-4o" in model and "mini" not in model:
            return prompt_tokens * 0.000005  # $5 / 1M tokens
        elif "gpt-4o-mini" in model:
            return prompt_tokens * 0.00000015  # $0.15 / 1M tokens
        elif "claude" in model:
            return prompt_tokens * 0.000003  # $3 / 1M tokens
        else:
            return prompt_tokens * 0.000002

    def should_use_ai(self, num_players: int, contest_type: str) -> bool:
        """
        Decide whether to use AI given cost constraints and slate size
        """
        if not self.ai_enabled:
            return False

        # Conservative: avoid huge prompts for large slates
        estimated_tokens = num_players * 80  # rough guess: 80 tokens per player
        est_cost = self._estimate_cost(estimated_tokens)

        # If we're over $0.50 for a single run, that's too rich
        if est_cost > 0.50:
            logger.warning(f"Skipping AI (estimated single-prompt cost ${est_cost:.2f})")
            return False

        logger.info(f"AI analysis approved (estimated cost ${est_cost:.2f})")
        self.last_prompt_cost = est_cost
        return True

    def _build_ai_prompt(self, players: List[Dict], vegas_data: Dict, contest_type: str) -> str:
        """
        Build a rich prompt that gives the model everything it needs to make
        genuinely useful decisions.
        """
        lines = []
        lines.append(
            "You are an expert DFS tournament strategist for FanDuel NFL contests.\n"
            "Your job is to identify MUST PLAY and MUST FADE players for a given slate.\n"
            "You are optimizing for TOURNAMENT WINNING upside, not cash-game safety.\n"
        )

        # Explain contest type
        if contest_type == "friends_league":
            lines.append(
                "Contest context: 12-person private friends league with a single optimal lineup.\n"
                "You should focus on high-upside, correlated plays, but you do not need extreme contrarianism.\n"
            )
        elif contest_type == "h2h":
            lines.append(
                "Contest context: single-game head-to-head FanDuel contest.\n"
                "You care more about median and ceiling combined for a single lineup, and correlation between QB and pass-catchers.\n"
            )
        else:
            lines.append(
                f"Contest context: {contest_type} tournament.\n"
                "Assume large-field GPP dynamics: leverage, correlation, and ceiling matter more than raw median.\n"
            )

        # Vegas overview
        avg_total = vegas_data.get("avg_total", 45.0)
        lines.append(f"\nSlate average game total: {avg_total:.1f} points.\n")
        high_total_games = vegas_data.get("high_total_games", [])
        if high_total_games:
            lines.append("High-total games (better environments):\n")
            for g in high_total_games:
                lines.append(
                    f"- {g.get('away_team')} @ {g.get('home_team')}, "
                    f"total={g.get('total_points')}, spread={g.get('spread')}\n"
                )

        lines.append(
            "\nEach player below has: name, team, position, salary, projection, "
            "ceiling, floor, game_total, game_environment_mult, injury_status, "
            "and projected ownership.\n"
        )
        lines.append(
            "Your task:\n"
            "1. Identify ~8-15 MUST PLAY players who are high-ceiling, viable, and fit well in tournament lineups.\n"
            "2. Identify ~8-15 MUST FADE players who are overpriced, fragile, or bad leverage.\n"
            "3. Focus heavily on QBs and their WR/TE stacks in high-total games.\n"
            "4. Look for injury situations where a cheap player gets a big role bump.\n"
        )

        # Player table
        lines.append("\nPlayer data:\n")
        for p in players:
            lines.append(
                f"- {p.get('name')} | {p.get('team')} | {p.get('position')} | "
                f"salary={p.get('salary')} | proj={p.get('projection')} | "
                f"ceil={p.get('ceiling')} | floor={p.get('floor')} | "
                f"game_total={p.get('game_total')} | env_mult={p.get('game_environment_mult')} | "
                f"injury={p.get('injury_status', 'healthy')} | own={p.get('ownership', 0)}\n"
            )

        lines.append(
            "\nRespond ONLY in JSON with this structure:\n"
            "{\n"
            '  "must_play": ["Player Name 1", "Player Name 2", ...],\n'
            '  "must_fade": ["Player Name 3", "Player Name 4", ...],\n'
            '  "notes": "Any brief strategic notes about stacks, leverage, or injuries."\n'
            "}\n"
        )

        return "".join(lines)

    async def analyze_players(
        self,
        players: List[Dict],
        vegas_data: Dict,
        contest_type: str = "gpp"
    ) -> Dict[str, Any]:
        """
        Run AI analysis on the player pool:
        - Returns {must_play: [...], must_fade: [...], notes: "..."}
        """
        if not players:
            return {"must_play": [], "must_fade": [], "notes": ""}

        if not self.should_use_ai(len(players), contest_type):
            return {"must_play": [], "must_fade": [], "notes": "AI disabled or too expensive for this slate."}

        prompt = self._build_ai_prompt(players, vegas_data, contest_type)

        # Try OpenAI first (new SDK syntax for openai>=1.0.0)
        if self.openai_api_key:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=self.openai_api_key)

                logger.info("Calling OpenAI for MUST PLAY / MUST FADE analysis...")
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert DFS NFL tournament optimizer. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                )
                content = response.choices[0].message.content
                # Strip markdown code blocks if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                data = json.loads(content)
                logger.info(f"✅ OpenAI analysis complete: {len(data.get('must_play', []))} must-play, {len(data.get('must_fade', []))} must-fade")
                return {
                    "must_play": data.get("must_play", []),
                    "must_fade": data.get("must_fade", []),
                    "notes": data.get("notes", ""),
                }

            except Exception as e:
                logger.error(f"Error during OpenAI analysis: {e}")

        # Anthropic fallback (synchronous client - no await needed)
        if self.anthropic_api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)

                logger.info("Calling Anthropic for MUST PLAY / MUST FADE analysis (fallback)...")
                msg = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    temperature=0.4,
                    system="You are an expert DFS NFL tournament optimizer. Always respond with valid JSON only.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                content = msg.content[0].text
                # Strip markdown code blocks if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                content = content.strip()
                data = json.loads(content)
                logger.info(f"✅ Anthropic analysis complete: {len(data.get('must_play', []))} must-play, {len(data.get('must_fade', []))} must-fade")
                return {
                    "must_play": data.get("must_play", []),
                    "must_fade": data.get("must_fade", []),
                    "notes": data.get("notes", ""),
                }
            except Exception as e:
                logger.error(f"Error during Anthropic analysis: {e}")

        return {"must_play": [], "must_fade": [], "notes": "AI call failed; using base projections only."}

    def apply_to_players(self, players: List[Dict], ai_result: Dict[str, Any]) -> List[Dict]:
        """
        Adjust projections based on AI MUST PLAY / MUST FADE recommendations.
        """
        if not ai_result:
            return players

        must_play = set(ai_result.get("must_play", []))
        must_fade = set(ai_result.get("must_fade", []))

        if not must_play and not must_fade:
            return players

        logger.info(
            f"AI flags: {len(must_play)} MUST PLAY, {len(must_fade)} MUST FADE. "
            f"Notes: {ai_result.get('notes', '')[:200]}..."
        )

        adjusted_players = []
        for p in players:
            name = p.get("name")
            # Sanitize projection - this is critical to prevent NaN propagation
            proj = safe_float(p.get("projection", p.get("projected_points", 0)), 5.0)
            base_proj = proj

            # Sanitize environment multiplier
            env_mult = safe_float(p.get("game_environment_mult", 1.0), 1.0)

            if name in must_play:
                # Big boost for must-plays
                proj *= 1.20
                p["ai_must_play"] = True
            elif name in must_fade:
                # Cut projection for must-fades
                proj *= 0.80
                p["ai_must_fade"] = True
            else:
                p["ai_must_play"] = False
                p["ai_must_fade"] = False

            # Final projection with sanitization
            final_proj = safe_float(proj * env_mult, base_proj)
            p["projection"] = round(final_proj, 2)
            p["projected_points"] = p["projection"]
            p["projection_source"] = "ai_enhanced" if (name in must_play or name in must_fade) else "base+env"
            p["base_projection"] = base_proj

            # Sanitize other float fields that will be used in optimizer
            p["ceiling"] = safe_float(p.get("ceiling", final_proj * 1.5), final_proj * 1.5)
            p["floor"] = safe_float(p.get("floor", final_proj * 0.5), final_proj * 0.5)
            p["ownership"] = safe_float(p.get("ownership", 10.0), 10.0)
            p["game_environment_mult"] = env_mult

            adjusted_players.append(p)

        return adjusted_players


class EnhancedSlateManager:
    """
    Handles ESPN game data and time slots
    """
    def __init__(self):
        self.tz = pytz.timezone("US/Eastern")

    def get_current_time_et(self) -> datetime:
        return datetime.now(self.tz)

    def _calculate_week_from_date(self, ref_date: Optional[datetime] = None) -> int:
        """
        Very rough NFL week calculation if ESPN data fails
        """
        if ref_date is None:
            ref_date = self.get_current_time_et()

        # NFL regular season typically starts around early September
        season_start = datetime(ref_date.year, 9, 1, tzinfo=self.tz)
        delta_days = (ref_date - season_start).days
        if delta_days < 0:
            return 1

        week = 1 + (delta_days // 7)
        return max(1, min(18, week))

    def bucket_games_by_time(self, games: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group games into main slate vs single games
        """
        buckets = {"main_slate": [], "single_games": [], "all_games": games}

        for g in games:
            slot = g.get("time_slot")
            if slot in ("sunday_early", "sunday_late"):
                buckets["main_slate"].append(g)
            else:
                buckets["single_games"].append(g)

        return buckets


class EnhancedDataCollector:
    """WINNING Data Collector with integrated AI analysis"""

    def __init__(self):
        self.session = None
        self.slate_manager = EnhancedSlateManager()
        self.ai_analyzer = WinningAIAnalyzer()

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'FanDuelWinningBot/1.0'}
        )
        return self

    async def __aexit__(self, exc_type, exc, tb):
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

                buckets = self.slate_manager.bucket_games_by_time(all_games)
                buckets['current_week'] = current_week
                return buckets

        except Exception as e:
            logger.error(f"Error fetching ESPN games: {e}")
            return self._get_fallback_games()

    def _extract_week_number(self, data: Dict) -> int:
        """Extract week number from scoreboard data"""
        try:
            week_info = data.get('week', {})
            num = week_info.get('number')
            if num:
                return int(num)
        except Exception:
            pass
        return self.slate_manager._calculate_week_from_date()

    def _parse_game_event(self, event: Dict) -> Optional[Dict]:
        """Parse a single ESPN game event"""
        try:
            competitions = event.get('competitions', [])
            if not competitions:
                return None

            competition = competitions[0]
            status = competition.get('status', {})
            game_clock = status.get('displayClock')
            game_status = status.get('type', {}).get('name')

            start_time_str = competition.get('date')
            game_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            game_et = game_time.astimezone(self.slate_manager.tz)

            competitors = competition.get('competitors', [])

            if len(competitors) < 2:
                return None

            teams = []
            home_team = None
            away_team = None
            for comp in competitors:
                abbrev = comp.get('team', {}).get('abbreviation', '')
                if abbrev:
                    teams.append(abbrev)
                    if comp.get('homeAway') == 'home':
                        home_team = abbrev
                    else:
                        away_team = abbrev

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
                'id': event.get('id'),
                'teams': teams,
                'home_team': home_team,
                'away_team': away_team,
                'time_slot': time_slot,
                'time': game_et.strftime("%A %I:%M %p ET"),
                'game_clock': game_clock,
                'game_status': game_status,
            }
        except Exception:
            return None

    def _get_fallback_games(self):
        """Fallback games"""
        week = self.slate_manager._calculate_week_from_date()
        games = [
            {'id': 'BUF_vs_MIA', 'teams': ['BUF', 'MIA'], 'home_team': 'MIA', 'away_team': 'BUF', 'time_slot': 'sunday_early', 'time': 'Sunday 1:00 PM ET'},
            {'id': 'PHI_vs_WAS', 'teams': ['PHI', 'WAS'], 'home_team': 'WAS', 'away_team': 'PHI', 'time_slot': 'sunday_early', 'time': 'Sunday 1:00 PM ET'},
        ]
        return {'current_week': week, 'all_games': games, 'main_slate': games, 'single_games': []}

    async def get_vegas_odds_data(self) -> Dict[str, Any]:
        """Get Vegas odds data"""
        try:
            from vegas_data_collector import VegasDataCollector
            collector = VegasDataCollector()
            vegas_data = await collector.get_nfl_odds_data()

            if not vegas_data or not vegas_data.get('games'):
                logger.warning("Vegas data empty, falling back")
                return self._get_fallback_vegas()

            return vegas_data
        except Exception as e:
            logger.error(f"Error fetching Vegas odds: {e}")
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
            elif total >= 47:
                total_mult = 1.25
            elif total >= 44:
                total_mult = 1.10
            else:
                total_mult = 1.0

            # Spread multiplier (very modest, we care more about totals)
            if spread <= 3:
                spread_mult = 1.05
            elif spread <= 6:
                spread_mult = 1.02
            else:
                spread_mult = 0.98

            env_mult = total_mult * spread_mult

            if home_team:
                multipliers[home_team] = env_mult
            if away_team:
                multipliers[away_team] = env_mult

        return multipliers

    def _validate_player(self, player: Dict, contest_type: str) -> bool:
        """
        Basic guardrails to avoid garbage in optimization
        """
        salary = player.get('salary', 0)
        position = player.get('position')
        fppg = player.get('projection', player.get('projected_points', 0))

        if not position or not player.get('name') or not player.get('team'):
            return False

        if salary < 3000:
            return False

        if position == 'QB':
            return salary >= 6000 or fppg >= 10
        elif position == 'RB':
            return salary >= 5000 or fppg >= 7
        elif position == 'WR':
            return salary >= 4500 or fppg >= 4
        elif position == 'TE':
            return salary >= 4000 or fppg >= 3
        elif position in ('D', 'DEF'):
            return salary >= 3000

        return False

    def _estimate_ownership(self, salary: int, projection: float, position: str) -> float:
        """Estimate ownership based on salary and value"""
        if salary <= 0:
            return 5.0

        projection = safe_float(projection, 5.0)
        value = safe_float(projection / (salary / 1000), 1.0) if salary > 0 else 0

        # Higher salary + good value = higher ownership
        base_own = 5.0
        if salary >= 9000:
            base_own = 15.0
        elif salary >= 7500:
            base_own = 10.0
        elif salary >= 6000:
            base_own = 7.0

        # Adjust for value
        if value >= 3.0:
            base_own *= 1.5
        elif value >= 2.5:
            base_own *= 1.2
        elif value < 2.0:
            base_own *= 0.8

        return safe_float(min(base_own, 40.0), 10.0)

    async def collect_players_for_slate(self, games_info: Dict, contest_type: str = 'gpp') -> List[Dict]:
        """Collect and filter players from FanDuel CSV"""

        playing_teams = set()

        # Use different game buckets depending on contest type
        if contest_type == 'h2h':
            for game in games_info.get('single_games', []):
                playing_teams.update(game.get('teams', []))
        else:
            for game in games_info.get('main_slate', []):
                playing_teams.update(game.get('teams', []))

        if not playing_teams:
            playing_teams = None

        try:
            # Determine which CSV to use
            manual_csv = DATA_DIR / "fanduel_salaries_manual.csv"
            h2h_csv = DATA_DIR / "fanduel_h2h_salaries.csv"

            csv_path = None
            if contest_type == 'h2h' and h2h_csv.exists():
                csv_path = h2h_csv
                logger.info(f"Using H2H salary CSV: {h2h_csv}")
            elif manual_csv.exists():
                csv_path = manual_csv
                logger.info(f"Using manual salary CSV: {manual_csv}")

            if not csv_path or not csv_path.exists():
                logger.error("No FanDuel salary CSV found")
                return []

            # Read the CSV
            salary_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(salary_df)} rows from CSV")

            # Convert FanDuel CSV format to player dicts
            players = []
            for _, row in salary_df.iterrows():
                try:
                    # Extract position (normalize D to DEF)
                    position = str(row.get('Position', '')).strip().upper()
                    if position == 'D':
                        position = 'DEF'

                    # Build player name
                    nickname = row.get('Nickname', '')
                    first_name = row.get('First Name', '')
                    last_name = row.get('Last Name', '')

                    if nickname and str(nickname).strip():
                        name = str(nickname).strip()
                    else:
                        name = f"{first_name} {last_name}".strip()

                    if not name or name == ' ':
                        continue

                    # Get salary
                    salary = int(row.get('Salary', 0))
                    if salary < 3000:
                        continue

                    # Get team
                    team = str(row.get('Team', '')).strip().upper()
                    if not team:
                        continue

                    # Filter by playing teams if we have that info
                    if playing_teams and team not in playing_teams:
                        continue

                    # Get opponent
                    opponent = str(row.get('Opponent', '')).strip().upper()

                    # Get projection (FPPG from FanDuel) - sanitize to prevent NaN
                    fppg = safe_float(row.get('FPPG', 0), 0.0)

                    # Estimate projection if missing
                    if fppg <= 0:
                        if position == 'QB':
                            fppg = salary / 550
                        elif position == 'RB':
                            fppg = salary / 650
                        elif position == 'WR':
                            fppg = salary / 700
                        elif position == 'TE':
                            fppg = salary / 750
                        elif position == 'DEF':
                            fppg = salary / 600

                    # Ensure fppg is valid
                    fppg = safe_float(fppg, 5.0)

                    # Get injury info
                    injury_indicator = str(row.get('Injury Indicator', '') or '').strip()
                    injury_details = str(row.get('Injury Details', '') or '').strip()

                    # Skip IR players
                    if 'IR' in injury_indicator.upper() or 'OUT' in injury_indicator.upper():
                        continue

                    # Get game info
                    game_str = str(row.get('Game', '')).strip()

                    # Build player dict with sanitized float values
                    player = {
                        'id': str(row.get('Id', '')),
                        'name': name,
                        'position': position,
                        'team': team,
                        'opponent': opponent,
                        'salary': salary,
                        'projection': safe_float(fppg, 5.0),
                        'projected_points': safe_float(fppg, 5.0),
                        'fppg': safe_float(fppg, 5.0),
                        'ceiling': safe_float(fppg * 1.5, 7.5),
                        'floor': safe_float(fppg * 0.5, 2.5),
                        'game': game_str,
                        'injury_status': injury_indicator if injury_indicator else 'healthy',
                        'injury_details': injury_details,
                        'ownership': safe_float(self._estimate_ownership(salary, fppg, position), 10.0),
                        'is_confirmed_starter': True,
                        'snap_percentage': 80.0,
                    }

                    # Validate player meets minimum thresholds
                    if self._validate_player(player, contest_type):
                        players.append(player)

                except Exception as e:
                    logger.debug(f"Skipping row due to error: {e}")
                    continue

            logger.info(f"✅ Converted {len(players)} valid players for {contest_type}")

            # Log position breakdown
            from collections import Counter
            pos_counts = Counter(p['position'] for p in players)
            logger.info(f"📊 Position breakdown: {dict(pos_counts)}")

            return players

        except Exception as e:
            logger.error(f"Error collecting players: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    async def get_weather_for_games(self, games_info: Dict) -> Dict[str, Any]:
        """Get weather data for outdoor stadium games"""
        weather_data = {}

        # Outdoor stadiums (dome teams excluded)
        dome_teams = {'ARI', 'ATL', 'DAL', 'DET', 'HOU', 'IND', 'LAC', 'LAR', 'LV', 'MIN', 'NO', 'NYJ'}

        try:
            all_games = games_info.get('main_slate', []) + games_info.get('single_games', [])

            for game in all_games:
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')

                # Skip dome games
                if home_team in dome_teams:
                    weather_data[home_team] = {'is_dome': True, 'weather_factor': 1.0}
                    if away_team:
                        weather_data[away_team] = {'is_dome': True, 'weather_factor': 1.0}
                    continue

                # Default outdoor weather (neutral)
                if home_team:
                    weather_data[home_team] = {
                        'is_dome': False,
                        'weather_factor': 1.0,
                        'conditions': 'normal'
                    }
                if away_team:
                    weather_data[away_team] = {
                        'is_dome': False,
                        'weather_factor': 1.0,
                        'conditions': 'normal'
                    }

        except Exception as e:
            logger.error(f"Error getting weather data: {e}")

        return weather_data

    async def get_breaking_news_impact(self, players: List[Dict]) -> Dict[str, Any]:
        """Get breaking news"""
        if not NEWS_AVAILABLE:
            return {'news_events': [], 'impact_analysis': {}}

        try:
            news_events = await get_breaking_news()
            return {
                'news_events': news_events,
                'impact_analysis': {}
            }
        except Exception as e:
            logger.error(f"Error fetching breaking news: {e}")
            return {'news_events': [], 'impact_analysis': {}}


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

        # Attach game environment multipliers
        for p in players:
            team = p.get('team')
            p['game_environment_mult'] = vegas_multipliers.get(team, 1.0)
            p['game_total'] = vegas_data.get('games', {}).get(
                f"{team}_game_total", vegas_data.get('avg_total', 45.0)
            )

        # Get weather data
        weather_data = await collector.get_weather_for_games(games_info)

        # Run AI analysis
        ai_result = await collector.ai_analyzer.analyze_players(players, vegas_data, contest_type)

        # Apply AI recommendations
        enhanced_players = collector.ai_analyzer.apply_to_players(players, ai_result)

        # Get news impact
        news_impact = await collector.get_breaking_news_impact(enhanced_players)

        # Summary stats
        must_play_count = sum(1 for p in enhanced_players if p.get('ai_must_play'))
        must_fade_count = sum(1 for p in enhanced_players if p.get('ai_must_fade'))
        high_total_count = sum(1 for p in enhanced_players if p.get('game_environment_mult', 1.0) >= 1.25)

        logger.info(f"📊 FINAL POOL: {len(enhanced_players)} players")
        logger.info(f"   🎯 Must-play: {must_play_count}")
        logger.info(f"   ⛔ Must-fade: {must_fade_count}")
        logger.info(f"   🔥 In high-total games: {high_total_count}")

        # Build data quality summary
        from collections import Counter
        pos_counts = Counter(p['position'] for p in enhanced_players)
        team_set = set(p['team'] for p in enhanced_players)
        real_proj_count = sum(1 for p in enhanced_players if p.get('fppg', 0) > 0)

        data_quality = {
            'current_week': games_info.get('current_week', 'Unknown'),
            'main_slate_games': len(games_info.get('main_slate', [])),
            'real_projections': real_proj_count,
            'teams_in_slate': list(team_set),
            'position_counts': dict(pos_counts),
        }

        return {
            'players': enhanced_players,
            'games_info': games_info,
            'vegas_data': vegas_data,
            'vegas_odds': vegas_data,  # Alias for compatibility
            'vegas_multipliers': vegas_multipliers,
            'weather': weather_data,
            'news': news_impact,
            'data_quality': data_quality,
        }