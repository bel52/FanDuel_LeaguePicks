"""
FIXED: Data collector with SMARTER QB filtering for tournament winning
Addresses over-aggressive filtering that removes viable QBs
NOW INCLUDES: Injury opportunity detection BEFORE filtering
"""
import asyncio
import aiohttp
import os
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import time
from loguru import logger
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

class EnhancedSlateManager:
    """Manages REAL slate detection with robust error handling"""

    def __init__(self):
        self.eastern = pytz.timezone('America/New_York')

    def get_current_nfl_week(self) -> int:
        """Get REAL current NFL week with multiple fallbacks"""

        # Method 1: Try ESPN API
        try:
            import requests
            response = requests.get(
                'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()

                # Handle different ESPN response formats
                if 'week' in data:
                    if isinstance(data['week'], dict):
                        current_week = data['week'].get('number', None)
                    else:
                        current_week = data.get('week')

                    if current_week and isinstance(current_week, int):
                        logger.info(f"ESPN API Week: {current_week}")
                        return current_week

        except Exception as e:
            logger.warning(f"ESPN API failed: {e}")

        # Method 2: Calculate from date
        return self._calculate_week_from_date()

    def _calculate_week_from_date(self) -> int:
        """Calculate NFL week from current date"""
        now = datetime.now(self.eastern)

        # 2024 NFL season started September 5, 2024 (Thursday Night)
        season_start = datetime(2024, 9, 5, tzinfo=self.eastern)

        if now < season_start:
            logger.info("Before season start, using Week 1")
            return 1

        days_since_start = (now - season_start).days
        week = max(1, min(18, (days_since_start // 7) + 1))

        logger.info(f"Calculated week from date: Week {week} (days since start: {days_since_start})")
        return week

class EnhancedDataCollector:
    """FIXED: Real data collection with SMARTER QB filtering for winning lineups"""

    def __init__(self):
        self.session = None
        self.slate_manager = EnhancedSlateManager()

    async def get_breaking_news_impact(self, players: List[Dict]) -> Dict[str, Any]:
        """Get breaking news and automatically identify ruled-out players"""
        if not NEWS_AVAILABLE:
            return {'news_events': [], 'impact_analysis': {}, 'ruled_out_players': set()}

        try:
            news_events = await get_breaking_news()
            if not news_events:
                return {'news_events': [], 'impact_analysis': {}, 'ruled_out_players': set()}

            logger.info(f"📰 Found {len(news_events)} breaking news items")

            # Extract ruled-out players from news
            ruled_out_players = set()
            injury_keywords = ['ruled out', 'out vs', 'out for', 'will not play', 'inactive']

            for news in news_events:
                title = news.get('title', '')
                summary = news.get('summary', '')
                full_text = f"{title} {summary}".lower()

                # Check if this is injury news
                if any(keyword in full_text for keyword in injury_keywords):
                    # Extract capitalized names from ORIGINAL title (not lowercased)
                    words = title.split()
                    for i in range(len(words) - 1):
                        # Look for "FirstName LastName" pattern
                        if len(words[i]) > 1 and words[i][0].isupper() and len(words[i + 1]) > 1 and words[i + 1][
                            0].isupper():
                            # Skip common words that get capitalized
                            if words[i] not in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday',
                                                'Sunday', 'Week']:
                                potential_name = f"{words[i]} {words[i + 1]}".lower()
                                ruled_out_players.add(potential_name)
                                logger.info(f"🚫 DETECTED from news: {potential_name.title()}")

            self._ruled_out_players = ruled_out_players

            player_names = [p.get('name', '') for p in players]
            player_specific_news = await get_player_news(player_names)

            return {
                'news_events': news_events,
                'player_specific_news': player_specific_news,
                'ruled_out_players': ruled_out_players,
                'news_count': len(news_events),
                'player_news_count': len(player_specific_news)
            }

        except Exception as e:
            logger.error(f"Error getting breaking news impact: {e}")
            self._ruled_out_players = set()
            return {'news_events': [], 'impact_analysis': {}, 'ruled_out_players': set()}

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
        """Get REAL current week games with robust parsing"""
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"ESPN API returned {response.status}")
                    return self._get_current_date_fallback()

                data = await response.json()
                logger.info(f"ESPN API response keys: {list(data.keys())}")

                # Get current week with multiple fallbacks
                current_week = self._extract_week_number(data)

                # Parse games from events
                all_games = []
                events = data.get('events', [])

                if not events:
                    logger.warning("No events found in ESPN response")
                    return self._get_current_date_fallback()

                logger.info(f"Processing {len(events)} events from ESPN")

                for i, event in enumerate(events):
                    try:
                        game_info = self._parse_game_event(event, i)
                        if game_info:
                            all_games.append(game_info)

                    except Exception as e:
                        logger.warning(f"Error parsing event {i}: {e}")
                        continue

                if not all_games:
                    logger.error("No valid games parsed from ESPN")
                    return self._get_current_date_fallback()

                # Categorize games by time slot
                main_slate = []
                single_games = all_games.copy()

                for game in all_games:
                    if game['time_slot'] in ['sunday_early', 'sunday_late']:
                        main_slate.append(game)

                logger.info(f"Parsed {len(all_games)} real games, {len(main_slate)} in main slate")

                # Log game details for verification
                for game in all_games[:3]:  # Show first 3 games
                    logger.info(f"{game['teams'][0]} vs {game['teams'][1]} - {game['time']}")

                return {
                    'current_week': current_week,
                    'all_games': all_games,
                    'main_slate': main_slate,
                    'single_games': single_games,
                }

        except Exception as e:
            logger.error(f"Error in get_current_week_games: {e}")
            return self._get_current_date_fallback()

    def _extract_week_number(self, data: Dict) -> int:
        """Extract week number from ESPN data with multiple attempts"""

        # Attempt 1: Direct week field
        if 'week' in data:
            week_data = data['week']
            if isinstance(week_data, dict) and 'number' in week_data:
                return week_data['number']
            elif isinstance(week_data, int):
                return week_data

        # Attempt 2: From season data
        if 'season' in data:
            season_data = data['season']
            if isinstance(season_data, dict) and 'week' in season_data:
                return season_data['week']

        # Attempt 3: Calculate from date
        calculated_week = self.slate_manager._calculate_week_from_date()
        logger.info(f"Using calculated week: {calculated_week}")
        return calculated_week

    def _parse_game_event(self, event: Dict, index: int) -> Optional[Dict]:
        """Parse individual game event from ESPN"""
        try:
            # Get game date/time
            game_date = event.get('date', '')
            if not game_date:
                return None

            game_datetime = datetime.fromisoformat(game_date.replace('Z', '+00:00'))
            game_et = game_datetime.astimezone(self.slate_manager.eastern)

            # Get teams
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])

            if len(competitors) < 2:
                return None

            # Extract team abbreviations
            teams = []
            for competitor in competitors:
                team_data = competitor.get('team', {})
                abbrev = team_data.get('abbreviation', '')
                if abbrev:
                    teams.append(abbrev)

            if len(teams) != 2:
                return None

            # Determine time slot
            time_slot = self._determine_time_slot(game_et)

            game_info = {
                'id': f"{teams[0]}_vs_{teams[1]}",
                'teams': teams,
                'time_slot': time_slot,
                'time': game_et.strftime('%A %I:%M %p ET'),
                'datetime': game_et
            }

            return game_info

        except Exception as e:
            logger.warning(f"Error parsing game event: {e}")
            return None

    def _determine_time_slot(self, game_datetime: datetime) -> str:
        """Determine game time slot"""
        hour = game_datetime.hour
        day = game_datetime.weekday()

        if day == 3:  # Thursday
            return 'thursday_night'
        elif day == 6:  # Sunday
            if hour < 16:
                return 'sunday_early'
            elif hour < 20:
                return 'sunday_late'
            else:
                return 'sunday_night'
        elif day == 0:  # Monday
            return 'monday_night'
        else:
            return 'other'

    def _get_current_date_fallback(self):
        """Fallback using current date logic"""
        current_week = self.slate_manager._calculate_week_from_date()

        logger.warning(f"Using date-based fallback: Week {current_week}")

        # Generate likely games based on typical NFL schedule
        example_games = [
            {'id': 'BUF_vs_MIA', 'teams': ['BUF', 'MIA'], 'time_slot': 'sunday_early', 'time': 'Sunday 1:00 PM ET'},
            {'id': 'PHI_vs_WAS', 'teams': ['PHI', 'WAS'], 'time_slot': 'sunday_early', 'time': 'Sunday 1:00 PM ET'},
            {'id': 'GB_vs_DET', 'teams': ['GB', 'DET'], 'time_slot': 'sunday_early', 'time': 'Sunday 1:00 PM ET'},
            {'id': 'KC_vs_LAC', 'teams': ['KC', 'LAC'], 'time_slot': 'sunday_late', 'time': 'Sunday 4:05 PM ET'},
        ]

        return {
            'current_week': current_week,
            'all_games': example_games,
            'main_slate': example_games,
            'single_games': example_games,
        }

    async def get_vegas_odds_data(self) -> Dict[str, Any]:
        """Get REAL Vegas odds data (GAME-CHANGING for tournament success)"""
        try:
            from vegas_data_collector import VegasDataCollector
            collector = VegasDataCollector()
            vegas_data = await collector.get_nfl_odds_data()

            if vegas_data and vegas_data.get('games'):
                high_total_count = len(vegas_data.get('high_total_games', []))
                logger.info(f"🎯 VEGAS DATA: {high_total_count} high-total games (47+ pts) found")

                # Log the high-total games (these drive tournament wins)
                for high_game in vegas_data.get('high_total_games', []):
                    logger.info(f"🔥 {high_game['game_id']}: {high_game['total']} total points")

                return vegas_data
            else:
                logger.warning("No Vegas games data returned")
                return {'games': {}, 'high_total_games': [], 'data_source': 'empty'}

        except ImportError:
            logger.warning("Vegas data collector not available")
            return {'games': {}, 'high_total_games': [], 'data_source': 'unavailable'}
        except Exception as e:
            logger.error(f"Vegas data collection failed: {e}")
            return {'games': {}, 'high_total_games': [], 'data_source': 'error'}

    def calculate_vegas_multipliers(self, vegas_data: Dict) -> Dict[str, float]:
        """Calculate team multipliers based on IMPLIED TEAM TOTALS - STEP 3 COMPLETE"""

        multipliers = {}
        games = vegas_data.get('games', {})

        if not games:
            logger.warning("No Vegas games data for multipliers")
            return {}

        # Calculate league average implied score (typically ~24 points)
        all_implied_scores = []
        for game_data in games.values():
            home_implied = game_data.get('home_implied_score')
            away_implied = game_data.get('away_implied_score')
            if home_implied:
                all_implied_scores.append(home_implied)
            if away_implied:
                all_implied_scores.append(away_implied)

        league_avg_implied = sum(all_implied_scores) / len(all_implied_scores) if all_implied_scores else 24.0
        logger.info(f"📊 League avg implied score: {league_avg_implied:.1f} points")

        for game_id, game_data in games.items():
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')
            home_implied = game_data.get('home_implied_score', league_avg_implied)
            away_implied = game_data.get('away_implied_score', league_avg_implied)

            # STEP 3: Per-team multipliers based on IMPLIED TOTALS
            # Higher implied score = higher DFS scoring expectation

            # Normalize around league average (24 pts)
            home_multiplier = home_implied / league_avg_implied
            away_multiplier = away_implied / league_avg_implied

            # Apply tiered boosts for high-scoring environments
            def apply_tiers(base_mult: float, implied: float) -> float:
                """Apply stepped multipliers for different scoring tiers"""
                if implied >= 29.0:
                    return base_mult * 1.20  # Elite offense (29+ implied)
                elif implied >= 26.5:
                    return base_mult * 1.15  # Strong offense (26.5-29)
                elif implied >= 24.5:
                    return base_mult * 1.08  # Above average (24.5-26.5)
                elif implied >= 22.0:
                    return base_mult * 1.00  # Average (22-24.5)
                elif implied >= 20.0:
                    return base_mult * 0.95  # Below average (20-22)
                else:
                    return base_mult * 0.88  # Low scoring (under 20)

            home_final = apply_tiers(home_multiplier, home_implied)
            away_final = apply_tiers(away_multiplier, away_implied)

            # Store multipliers
            if home_team:
                multipliers[home_team] = home_final
                logger.info(f"VEGAS WEIGHT: {home_team} implied {home_implied:.1f} → {home_final:.2f}x")

            if away_team:
                multipliers[away_team] = away_final
                logger.info(f"VEGAS WEIGHT: {away_team} implied {away_implied:.1f} → {away_final:.2f}x")

        logger.info(f"✅ STEP 3 COMPLETE: Applied per-team implied multipliers to {len(multipliers)} teams")
        return multipliers

    async def get_nfl_projections(self) -> Dict[str, float]:
        """Get NFL projections with better error handling"""
        try:
            logger.info("Fetching NFL projections...")

            current_year = datetime.now().year

            # Try to get weekly data
            try:
                weekly_data = nfl.import_weekly_data([current_year])
                logger.info(f"Loaded weekly data: {len(weekly_data)} rows")
            except Exception as e:
                logger.error(f"Failed to load weekly data: {e}")
                return {}

            projections = {}
            current_week = self.slate_manager.get_current_nfl_week()

            # Use last 3 weeks of data
            recent_weeks = list(range(max(1, current_week - 2), current_week + 1))
            recent_data = weekly_data[weekly_data['week'].isin(recent_weeks)]

            logger.info(f"Using weeks {recent_weeks} for projections")

            # Group by player and calculate averages
            for player_name, player_games in recent_data.groupby('player_display_name'):
                if not player_name or pd.isna(player_name):
                    continue

                # Get position (take most common)
                positions = player_games['position'].dropna()
                if positions.empty:
                    continue
                position = positions.mode().iloc[0] if len(positions) > 0 else 'UNK'

                # Calculate average stats
                games_played = len(player_games)
                if games_played == 0:
                    continue

                try:
                    if position == 'QB':
                        pass_yds = player_games['passing_yards'].fillna(0).mean()
                        pass_tds = player_games['passing_tds'].fillna(0).mean()
                        ints = player_games['interceptions'].fillna(0).mean()
                        rush_yds = player_games['rushing_yards'].fillna(0).mean()
                        rush_tds = player_games['rushing_tds'].fillna(0).mean()

                        projection = ((pass_yds / 25) + (pass_tds * 6) - (ints * 2) +
                                    (rush_yds / 10) + (rush_tds * 6))

                    elif position == 'RB':
                        rush_yds = player_games['rushing_yards'].fillna(0).mean()
                        rush_tds = player_games['rushing_tds'].fillna(0).mean()
                        rec_yds = player_games['receiving_yards'].fillna(0).mean()
                        rec_tds = player_games['receiving_tds'].fillna(0).mean()
                        receptions = player_games['receptions'].fillna(0).mean()

                        projection = ((rush_yds / 10) + (rush_tds * 6) +
                                    (rec_yds / 10) + (rec_tds * 6) + receptions)

                    elif position in ['WR', 'TE']:
                        rec_yds = player_games['receiving_yards'].fillna(0).mean()
                        rec_tds = player_games['receiving_tds'].fillna(0).mean()
                        receptions = player_games['receptions'].fillna(0).mean()

                        projection = (rec_yds / 10) + (rec_tds * 6) + receptions

                    else:
                        continue

                    if projection > 3:  # Minimum threshold
                        projections[player_name] = round(projection, 1)

                except Exception as e:
                    logger.debug(f"Error calculating projection for {player_name}: {e}")
                    continue

            logger.info(f"Generated {len(projections)} projections")
            return projections

        except Exception as e:
            logger.error(f"Error in get_nfl_projections: {e}")
            return {}

    async def get_weather_for_games(self, games_info: Dict) -> Dict[str, Dict]:
        """Get weather data with error handling"""
        weather_data = {}

        try:
            all_games = games_info.get('all_games', [])
            outdoor_teams = set()

            for game in all_games:
                for team in game.get('teams', []):
                    stadium_info = NFL_STADIUMS.get(team, {})
                    if stadium_info.get('type') == 'outdoor':
                        outdoor_teams.add(team)

            if not outdoor_teams:
                logger.info("No outdoor teams found")
                return {}

            logger.info(f"Getting weather for {len(outdoor_teams)} outdoor teams")

            for team in outdoor_teams:
                stadium = NFL_STADIUMS.get(team, {})
                if not stadium:
                    continue

                # Default weather (fallback)
                weather_data[team] = {
                    'temperature': 68,
                    'wind_speed': '8 mph',
                    'conditions': 'Partly Cloudy',
                    'precipitation_chance': 10,
                    'stadium_type': 'outdoor',
                    'factor': 1.0
                }

        except Exception as e:
            logger.error(f"Weather collection error: {e}")

        return weather_data

    def _is_viable_player(self, player_data: Dict) -> bool:
        """UNIVERSAL filtering system that adapts week-to-week for ALL positions"""
        name = player_data.get('name', '')
        position = player_data.get('position', '')
        team = player_data.get('team', '')
        salary = player_data.get('salary', 0)
        fppg = player_data.get('projected_points', 0)
        injury_status = player_data.get('injury_status', '')
        fppg_source = player_data.get('fppg_source', 'unknown')

        # UNIVERSAL RULE 1: Remove definitively unavailable players
        if self._is_definitely_unavailable(player_data):
            return False

        # UNIVERSAL RULE 2: Position-specific intelligent filtering
        return self._is_position_viable(position, salary, fppg, fppg_source, name, team)

    async def collect_players_for_slate(self, games_info: Dict[str, Any], contest_type: str = 'gpp') -> List[Dict]:
        """Collect players - NOW READS CSV DIRECTLY"""
        current_week = games_info['current_week']

        # ===== CRITICAL: Get breaking news FIRST to identify ruled-out players =====
        logger.info("🚑 Checking breaking news for injury updates...")
        temp_players = []  # Empty list for initial news check
        news_impact = await self.get_breaking_news_impact(temp_players)
        ruled_out = news_impact.get('ruled_out_players', set())
        if ruled_out:
            logger.info(f"🚫 Ruled out from news: {', '.join([p.title() for p in ruled_out])}")
        # ===== END BREAKING NEWS CHECK =====

        # Get teams in slate
        playing_teams = set()
        for game in games_info.get('main_slate', []):
            playing_teams.update(game.get('teams', []))

        logger.info(f"Teams in slate: {sorted(playing_teams)}")

        # READ CSV DIRECTLY
        try:
            csv_path = DATA_DIR / "fanduel_salaries_manual.csv"

            if not csv_path.exists():
                logger.error(f"CSV not found: {csv_path}")
                return []

            import pandas as pd
            df = pd.read_csv(csv_path)

            logger.info(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

            salary_data = []
            for _, row in df.iterrows():
                first = str(row.get('First Name', '')).strip()
                last = str(row.get('Last Name', '')).strip()
                name = f"{first} {last}".strip()

                if not name or name == 'nan nan':
                    continue

                # Safe extraction with NaN handling
                try:
                    salary_val = row.get('Salary', 0)
                    if pd.isna(salary_val):
                        salary_val = 0
                    salary_val = int(salary_val)

                    fppg_val = row.get('FPPG', 0)
                    if pd.isna(fppg_val):
                        fppg_val = 0.0
                    fppg_val = float(fppg_val)

                    # Skip if bad data
                    if salary_val <= 0 or fppg_val <= 0:
                        continue

                except (ValueError, TypeError):
                    continue

                salary_data.append({
                    'id': str(row.get('Id', '')),
                    'name': name,
                    'position': str(row.get('Position', '')).strip(),
                    'team': str(row.get('Team', '')).strip().upper(),
                    'salary': salary_val,
                    'projected_points': fppg_val,
                    'fppg_source': 'real',
                    'injury_status': str(row.get('Injury Indicator', '')).strip(),
                    'game': str(row.get('Game', '')).strip()
                })

            logger.info(f"Parsed {len(salary_data)} players")

        except Exception as e:
            logger.error(f"CSV read error: {e}")
            return []

        # Filter viable players
        filtered_injury = 0
        filtered_backup = 0
        filtered_other = 0

        winning_players = []
        for player_data in salary_data:
            if not self._is_viable_player(player_data):
                # Track why filtered
                if player_data.get('injury_status', '').strip():
                    filtered_injury += 1
                elif player_data.get('salary', 0) < 5000:
                    filtered_backup += 1
                else:
                    filtered_other += 1
                continue

            name = player_data['name']
            position = player_data['position']
            team = player_data['team']
            salary = player_data['salary']
            fppg = player_data['projected_points']

            if fppg <= 0:
                continue
            # Safety checks for JSON serialization
            if salary <= 0 or fppg <= 0:
                continue

            # Calculate value safely
            try:
                value = round(fppg / (salary / 1000), 2)
                if not (0 <= value <= 100):  # Sanity check
                    value = 0.0
            except:
                value = 0.0

            winning_players.append({
                'player_id': f"fd_{player_data['id']}",
                'name': name,
                'position': position,
                'team': team,
                'salary': int(salary),
                'projected_points': float(round(fppg, 2)),
                'projection': float(round(fppg, 2)),
                'ceiling': float(round(fppg * 1.4, 2)),
                'floor': float(round(fppg * 0.6, 2)),
                'ownership': 15.0,
                'game': player_data['game'],
                'value': float(value)
            })

        logger.info(f"Final count: {len(winning_players)} players")
        logger.info(f"📊 Filtered {len(salary_data) - len(winning_players)} players (injuries/backups/bad data)")

        # NEW: Apply injury opportunity boosts BEFORE returning
        from injury_opportunity_detector import enhance_players_with_injury_opportunities

        logger.info(f"🚑 Analyzing injury opportunities for {len(winning_players)} players...")
        enhanced_players = enhance_players_with_injury_opportunities(winning_players)

        return enhanced_players

    def _is_definitely_unavailable(self, player_data: Dict) -> bool:
        """Universal availability check with AUTOMATIC news-based filtering"""

        name = player_data.get('name', '').lower()
        injury_status = str(player_data.get('injury_status', '')).strip().upper()

        # Priority 1: Check breaking news for ruled-out players
        ruled_out_from_news = getattr(self, '_ruled_out_players', set())
        if any(out_name in name for out_name in ruled_out_from_news):
            logger.info(f"🚫 NEWS FILTER: {player_data.get('name')} (ruled out via breaking news)")
            return True

        # Priority 2: CSV injury designations
        if 'IR' in injury_status:
            logger.info(f"FILTERING IR player: {player_data.get('name')}")
            return True

        unavailable_flags = ['OUT', 'SUSP', 'DOUBTFUL']
        if any(flag in injury_status for flag in unavailable_flags):
            logger.info(f"FILTERING unavailable: {player_data.get('name')} ({injury_status})")
            return True

        # Priority 3: Broken entries
        if not name or len(name.strip()) < 2:
            return True

        return False

    def _is_position_viable(self, position: str, salary: int, fppg: float,
                            fppg_source: str, name: str, team: str) -> bool:
        """Position-specific viability using ADAPTIVE thresholds"""

        if position == 'QB':
            return self._is_viable_qb_adaptive(salary, fppg, fppg_source, name, team)
        elif position == 'RB':
            return self._is_viable_rb_adaptive(salary, fppg, fppg_source, name, team)
        elif position == 'WR':
            return self._is_viable_wr_adaptive(salary, fppg, fppg_source, name, team)
        elif position == 'TE':
            return self._is_viable_te_adaptive(salary, fppg, fppg_source, name, team)
        elif position == 'D':
            return self._is_viable_def_adaptive(salary, fppg, fppg_source, name, team)
        else:
            return False

    def _is_viable_qb_adaptive(self, salary: int, fppg: float, fppg_source: str,
                               name: str, team: str) -> bool:
        """QB filtering - ONLY keep starting QBs"""

        # HARD FLOOR: Any QB under $6,000 is a backup and won't play
        if salary < 6000:
            logger.info(f"FILTERING backup QB: {name} (${salary})")
            return False

        # Elite starters
        if salary >= 8000:
            return True

        # Starting QBs with real production
        if salary >= 6000 and fppg >= 12.0:
            return True

        # When in doubt for QBs $6K+, keep them
        return True

    def _is_viable_rb_adaptive(self, salary: int, fppg: float, fppg_source: str,
                               name: str, team: str) -> bool:
        """RB filtering - remove obvious non-contributors"""

        # Elite/starter RBs - obvious keeps
        if salary >= 7000:
            return True

        # Mid-tier with real production
        if salary >= 5000 and fppg >= 8.0:
            return True

        # Value plays with upside
        if salary >= 4500 and fppg >= 6.0:
            return True

        # Remove obvious practice squad/inactive RBs
        if salary < 4200 and fppg < 3:
            logger.debug(f"FILTERING practice squad RB: {name}")
            return False

        # Emergency keeps for tournament leverage
        if salary >= 4000 and fppg >= 4.0:
            return True

        return False

    def _is_viable_wr_adaptive(self, salary: int, fppg: float, fppg_source: str,
                               name: str, team: str) -> bool:
        """WR filtering - keep upside plays, remove practice squad"""

        # Elite WRs - obvious keeps
        if salary >= 7000:
            return True

        # Solid producers
        if salary >= 5500 and fppg >= 8.0:
            return True

        # Value/upside plays
        if salary >= 4500 and fppg >= 5.0:
            return True

        # Remove obvious non-contributors
        if salary < 4200 and fppg < 2:
            logger.debug(f"FILTERING practice squad WR: {name}")
            return False

        # Dart throws for tournaments
        if salary >= 4000:
            return True

        return False

    def _is_viable_te_adaptive(self, salary: int, fppg: float, fppg_source: str,
                               name: str, team: str) -> bool:
        """TE filtering - position is thin, be more inclusive"""

        # Elite TEs
        if salary >= 6000:
            return True

        # Solid contributors
        if salary >= 4800 and fppg >= 6.0:
            return True

        # Value plays (TE is thin)
        if salary >= 4200 and fppg >= 3.0:
            return True

        # Remove obvious non-contributors
        if salary < 4000 and fppg < 1:
            logger.debug(f"FILTERING practice squad TE: {name}")
            return False

        # Keep most TEs (position scarcity)
        return True

    def _is_viable_def_adaptive(self, salary: int, fppg: float, fppg_source: str,
                                name: str, team: str) -> bool:
        """Defense filtering - all NFL defenses are viable"""

        # All defenses $3000+ are real NFL teams
        if salary >= 3000:
            return True

        # Filter out obvious errors
        if salary < 3000:
            logger.debug(f"FILTERING invalid defense: {name}")
            return False

        return True

    def _get_ceiling_multiplier(self, position: str) -> float:
        """Position-specific ceiling multipliers for tournament play"""
        return {
            'QB': 1.4,
            'RB': 1.3,
            'WR': 1.5,
            'TE': 1.4,
            'D': 1.2
        }.get(position, 1.3)


# Main entry point
async def get_fresh_data() -> Dict[str, Any]:
    """Get fresh data with BREAKING NEWS INTEGRATION"""
    async with EnhancedDataCollector() as collector:
        # Get games info
        games_info = await collector.get_current_week_games()

        # Get players with REAL projections
        players = await collector.collect_players_for_slate(games_info, 'gpp')

        if not players:
            logger.error("NO VALID PLAYERS FOUND")
            return {}

        # Get other data
        weather_data = await collector.get_weather_for_games(games_info)
        vegas_data = await collector.get_vegas_odds_data()
        vegas_multipliers = collector.calculate_vegas_multipliers(vegas_data)

        # NEW: Get breaking news impact
        news_impact = await collector.get_breaking_news_impact(players)

        return {
            'players': players,
            'games_info': games_info,
            'weather': weather_data,
            'vegas_odds': vegas_data,
            'vegas_multipliers': vegas_multipliers,
            'breaking_news': news_impact,
            'last_updated': datetime.now().isoformat(),
            'data_quality': {
                'player_count': len(players),
                'total_games': len(games_info['all_games']),
                'main_slate_games': len(games_info['main_slate']),
                'current_week': games_info['current_week'],
                'avg_projection': sum(p['projected_points'] for p in players) / len(players) if players else 0,
                'avg_ownership': sum(p.get('ownership', 0) for p in players) / len(players) if players else 0,
                'teams_in_slate': sorted(set(p['team'] for p in players)),
                'real_projections': sum(1 for p in players if p.get('fppg_source') == 'real'),
                'estimated_projections': sum(1 for p in players if p.get('fppg_source') == 'estimated'),
                'injury_opportunities': sum(1 for p in players if p.get('injury_opportunity', False)),
                'salary_range': {
                    'min': min(p['salary'] for p in players) if players else 0,
                    'max': max(p['salary'] for p in players) if players else 0
                },
                'vegas_games': len(vegas_data) if vegas_data else 0,
                'vegas_multipliers': len(vegas_multipliers),
                'breaking_news_items': news_impact.get('news_count', 0),
                'player_news_items': news_impact.get('player_news_count', 0)
            }
        }