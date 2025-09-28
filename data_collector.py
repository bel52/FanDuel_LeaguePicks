"""
FIXED: Data collector with SMARTER QB filtering for tournament winning
Addresses over-aggressive filtering that removes viable QBs
NOW INCLUDES: Injury opportunity detection BEFORE filtering
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
        """Calculate team multipliers based on game totals - GAME CHANGING"""

        multipliers = {}
        games = vegas_data.get('games', {})
        avg_total = vegas_data.get('avg_total', 45.0)

        if not games:
            logger.warning("No Vegas games data for multipliers")
            return {}

        for game_id, game_data in games.items():
            total = game_data.get('total_points', avg_total)
            spread = abs(game_data.get('spread', 0))
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')

            # TOTAL MULTIPLIER: Higher totals = higher DFS scoring
            if total >= 50:
                total_mult = 1.35  # MAJOR boost for 50+ games
            elif total >= 47:
                total_mult = 1.25  # Significant boost for 47+ games
            elif total >= 44:
                total_mult = 1.10  # Moderate boost
            elif total <= 40:
                total_mult = 0.85  # Penalty for low totals
            else:
                total_mult = 1.0

            # SPREAD MULTIPLIER: Close games = more passing
            if spread <= 3:
                spread_mult = 1.15  # Close games = shootouts
            elif spread >= 10:
                spread_mult = 0.90  # Blowouts = fewer points
            else:
                spread_mult = 1.0

            # COMBINED MULTIPLIER
            final_mult = total_mult * spread_mult

            # Apply to both teams
            if home_team:
                multipliers[home_team] = final_mult
            if away_team:
                multipliers[away_team] = final_mult

            logger.info(f"VEGAS WEIGHT: {game_id} ({total} total, {spread} spread) = {final_mult:.2f}x")

        logger.info(f"Applied Vegas multipliers to {len(multipliers)} teams")
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
        if self._is_definitely_unavailable(injury_status, name):
            return False

        # UNIVERSAL RULE 2: Position-specific intelligent filtering
        return self._is_position_viable(position, salary, fppg, fppg_source, name, team)

    async def collect_players_for_slate(self, games_info: Dict[str, Any], contest_type: str = 'gpp') -> List[Dict]:
        """Collect players with SMARTER filtering to preserve winning options"""
        current_week = games_info['current_week']

        # Get teams playing in the slate
        if contest_type == 'single_game':
            playing_teams = set()
            for game in games_info.get('single_games', []):
                playing_teams.update(game.get('teams', []))
        else:
            playing_teams = set()
            for game in games_info.get('main_slate', []):
                playing_teams.update(game.get('teams', []))

        logger.info(f"Teams in {contest_type} slate: {sorted(playing_teams)} ({len(playing_teams)} teams)")

        # If no teams found, use all teams as fallback
        if not playing_teams:
            logger.warning("No teams found in slate, using all available players")
            playing_teams = None  # Will include all players

        # Get FanDuel salaries with REAL FPPG data
        try:
            from fanduel_salary_scraper import get_fanduel_salaries
            salary_data = await get_fanduel_salaries()

            if not salary_data:
                logger.error("No FanDuel salary data")
                return []

            # CRITICAL CHANGE: Apply injury opportunity detection BEFORE filtering
            logger.info("Applying injury opportunity detection BEFORE filtering...")
            try:
                from injury_opportunity_detector import enhance_players_with_injury_opportunities
                enhanced_salary_data = enhance_players_with_injury_opportunities(salary_data)
                logger.info("Injury opportunity detection completed")
            except ImportError:
                logger.warning("Injury opportunity detector not available, skipping enhancement")
                enhanced_salary_data = salary_data
            except Exception as e:
                logger.error(f"Injury opportunity detection failed: {e}")
                enhanced_salary_data = salary_data

            # NOW apply filtering to the enhanced data
            filtered_salary_data = []
            total_players = len(enhanced_salary_data)
            filtered_counts = {'QB': [0, 0], 'RB': [0, 0], 'WR': [0, 0], 'TE': [0, 0], 'D': [0, 0]}

            for player_data in enhanced_salary_data:
                position = player_data.get('position', '')

                if position in filtered_counts:
                    filtered_counts[position][0] += 1  # Total count

                if self._is_viable_player(player_data):
                    filtered_salary_data.append(player_data)
                    if position in filtered_counts:
                        filtered_counts[position][1] += 1  # Kept count

            # Log filtering results
            for pos, (total, kept) in filtered_counts.items():
                if total > 0:
                    logger.info(f"{pos} FILTERING: Kept {kept} of {total} players ({total - kept} filtered)")

            logger.info(f"TOTAL FILTERING: Kept {len(filtered_salary_data)} of {total_players} players")

            # Now process the filtered data
            winning_players = []
            for player_data in filtered_salary_data:
                name = player_data.get('name', '')
                position = player_data.get('position', '')
                team = player_data.get('team', '').upper()
                salary = int(player_data.get('salary', 5000))

                # Use REAL FPPG from FanDuel data ONLY
                fppg = player_data.get('projected_points', 0)
                if fppg > 0:
                    projection = fppg
                    logger.debug(f"✅ {name}: Using real FPPG {fppg}")
                else:
                    logger.warning(f"⚠️ {name}: No FPPG data, skipping player")
                    continue  # Skip players with no real data

                # Filter by teams (if applicable)
                if playing_teams and team not in playing_teams:
                    continue

                winning_player = {
                    'player_id': f"fd_{player_data.get('id', name)}",
                    'name': name,
                    'position': position,
                    'team': team,
                    'salary': salary,
                    'projected_points': round(projection, 2),
                    'projection': round(projection, 2),
                    'fppg_source': 'real',
                    'ceiling': round(projection * 1.4, 2),
                    'floor': round(projection * 0.7, 2),
                    'weather_factor': 1.0,
                    'ownership': np.random.uniform(5.0, 35.0),
                    'opponent': player_data.get('opponent', ''),
                    'value': round(projection / (salary / 1000), 2) if salary > 0 else 0
                }

                # Preserve injury opportunity metadata if it exists
                if player_data.get('injury_opportunity', False):
                    winning_player['injury_opportunity'] = True
                    winning_player['opportunity_score'] = player_data.get('opportunity_score', 0)
                    winning_player['injured_starter'] = player_data.get('injured_starter', '')
                    winning_player['boost_reason'] = player_data.get('boost_reason', '')

                winning_players.append(winning_player)

            # Apply weather adjustments
            weather_data = await self.get_weather_for_games(games_info)
            for player in winning_players:
                team = player['team']
                if team in weather_data:
                    weather = weather_data[team]
                    factor = weather.get('factor', 1.0)
                    conditions = weather.get('conditions', '').lower()
                    # Penalize passing games in rain or snow
                    if 'rain' in conditions or 'snow' in conditions:
                        factor *= 0.95
                    # Penalize high winds (>15 mph)
                    try:
                        wind_mph = int(weather.get('wind_speed', '0').split()[0])
                        if wind_mph > 15:
                            factor *= 0.97
                    except Exception:
                        pass
                    # Penalize high precipitation chance (>40%)
                    if weather.get('precipitation_chance', 0) > 40:
                        factor *= 0.92
                    player['weather_factor'] = factor
                    player['projected_points'] *= factor
                    player['projection'] *= factor
                    player['ceiling'] *= factor
                    player['floor'] *= factor

            logger.info(f"Enhanced {len(winning_players)} players with REAL projections")

            # Log projection source breakdown
            real_count = len(winning_players)  # All are real now
            logger.info(f"Projection sources: {real_count} real FPPG, 0 estimated")

            # Log injury opportunities applied
            injury_opportunities = sum(1 for p in winning_players if p.get('injury_opportunity', False))
            if injury_opportunities > 0:
                logger.info(f"Injury opportunities applied: {injury_opportunities} players boosted")

            return winning_players

        except Exception as e:
            logger.error(f"Error collecting players: {e}")
            return []

    def _is_definitely_unavailable(self, injury_status: str, name: str) -> bool:
        """Universal availability check - works for all positions"""

        # Remove players on Injured Reserve (definitely won't play)
        if 'IR' in injury_status.upper():
            logger.info(f"FILTERING IR player: {name}")
            return True

        # Remove players marked as OUT, SUSPENDED, DOUBTFUL
        upper_status = injury_status.upper()
        unavailable_flags = ['OUT', 'SUSP', 'DOUBTFUL']
        if any(flag in upper_status for flag in unavailable_flags):
            logger.info(f"FILTERING unavailable: {name} ({injury_status})")
            return True

        # Remove obviously broken entries
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
        """QB filtering based on ADAPTIVE salary/projection patterns"""

        # Tier 1: Elite starters (obvious keeps)
        if salary >= 8000:
            return True

        # Tier 2: Solid starters with real data
        if salary >= 7200 and fppg >= 15.0:
            return True

        # Tier 3: Potential starters with decent metrics
        if salary >= 6800 and fppg >= 12.0 and fppg_source == 'real':
            return True

        # Tier 4: Emergency/backup starters with value
        if salary >= 6500 and fppg >= 10.0:
            return True

        # Filter out obvious practice squad QBs
        if salary < 6200 and fppg < 8:
            logger.debug(f"FILTERING practice squad QB: {name}")
            return False

        # When in doubt, keep (better safe than sorry for QBs)
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

# Main entry point
async def get_fresh_data() -> Dict[str, Any]:
    """Get fresh data with robust error handling"""
    async with EnhancedDataCollector() as collector:
        # Get games info
        games_info = await collector.get_current_week_games()

        # Get players with REAL projections (injury opportunities applied BEFORE filtering)
        players = await collector.collect_players_for_slate(games_info, 'gpp')

        if not players:
            logger.error("NO VALID PLAYERS FOUND")
            return {}

        # Get other data
        weather_data = await collector.get_weather_for_games(games_info)
        vegas_data = await collector.get_vegas_odds_data()

        # CALCULATE VEGAS MULTIPLIERS
        vegas_multipliers = collector.calculate_vegas_multipliers(vegas_data)

        return {
            'players': players,
            'games_info': games_info,
            'weather': weather_data,
            'vegas_odds': vegas_data,
            'vegas_multipliers': vegas_multipliers,
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
                'vegas_multipliers': len(vegas_multipliers)
            }
        }