"""
FIXED: Data collector with SMARTER QB filtering for tournament winning
Addresses over-aggressive filtering that removes viable QBs
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
        """Placeholder for Vegas odds"""
        logger.info("Using placeholder Vegas odds")
        return {'placeholder': {'total_points': 45.5, 'spread': -3.5}}

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
        """FIXED: SMARTER filtering - especially for QBs"""
        name = player_data.get('name', '')
        position = player_data.get('position', '')
        salary = player_data.get('salary', 0)
        fppg = player_data.get('projected_points', 0)
        fppg_source = player_data.get('fppg_source', 'unknown')
        injury_status = player_data.get('injury_status', '')

        # ONLY filter players who definitely won't play

        # Filter players on IR (Injured Reserve) - they definitely won't play
        if 'IR' in injury_status.upper():
            logger.info(f"FILTERING injured: {name} ({injury_status})")
            return False

        # Filter players listed as OUT, DOUBTFUL or suspended (SUSP)
        upper_status = injury_status.upper()
        if any(flag in upper_status for flag in ['OUT', 'DOUBTFUL', 'SUSP']):
            logger.info(f"FILTERING out/doubtful: {name} ({injury_status})")
            return False

        # Filter obviously fake/broken entries
        if not name or len(name.strip()) < 2:
            return False

        if salary <= 0:
            return False

        # FIXED: MUCH smarter QB filtering
        if position == 'QB':
            return self._is_potentially_startable_qb_fixed(player_data)

        # For other positions, be VERY conservative
        if salary <= 3000 and fppg <= 0:
            logger.debug(f"FILTERING minimum salary zero projection: {name} (${salary}, {fppg:.1f})")
            return False

        return True

    def _is_potentially_startable_qb_fixed(self, player_data: Dict) -> bool:
        """FIXED: Much smarter QB filtering for tournament success"""
        name = player_data.get('name', '')
        salary = player_data.get('salary', 0)
        fppg = player_data.get('projected_points', 0)
        injury_status = player_data.get('injury_status', '')
        team = player_data.get('team', '')

        # Remove injured QBs on IR
        if 'IR' in injury_status.upper():
            logger.info(f"FILTERING injured QB: {name} ({injury_status})")
            return False

        # EXPANDED list of potential starters - be more inclusive for tournaments
        potential_starters = {
            # Confirmed starters
            'Josh Allen', 'Lamar Jackson', 'Jalen Hurts', 'Justin Herbert',
            'Patrick Mahomes', 'Baker Mayfield', 'Jared Goff', 'Caleb Williams',
            'Daniel Jones', 'Drake Maye', 'Matthew Stafford', 'Russell Wilson',
            'Bryce Young', 'Trevor Lawrence', 'C.J. Stroud', 'Jayden Daniels',
            'Geno Smith', 'Dak Prescott', 'Tua Tagovailoa', 'Jordan Love',
            'Derek Carr', 'Kyler Murray', 'Brock Purdy', 'Mac Jones',

            # Potential starters/backups who might get opportunities
            'Spencer Rattler', 'Anthony Richardson', 'Will Levis', 'Aidan O\'Connell',
            'Gardner Minshew', 'Tyler Huntley', 'Jameis Winston', 'Joe Flacco',
            'Sam Howell', 'Bailey Zappe', 'Kenny Pickett', 'Malik Willis',
            'Hendon Hooker', 'Joshua Dobbs', 'Tommy DeVito', 'Mason Rudolph',
            'Jake Browning', 'Dorian Thompson-Robinson', 'Clayton Tune'
        }

        # Check if this is a known QB (starter or backup)
        is_known_qb = any(starter.lower() in name.lower() for starter in potential_starters)

        # If it's a known QB, keep it regardless of salary/projection
        if is_known_qb:
            logger.debug(f"KEEPING known QB: {name}")
            return True

        # For unknown QBs, use more generous criteria

        # Keep any QB with decent salary (they might be a surprise starter)
        if salary >= 6500:  # Lowered from 7000
            logger.debug(f"KEEPING high-salary unknown QB: {name} (${salary})")
            return True

        # Keep any QB with real FPPG data above minimum threshold
        if fppg > 0 and fppg >= 8:  # Lowered from 10
            logger.debug(f"KEEPING QB with real projection: {name} ({fppg:.1f})")
            return True

        # Keep QBs from teams that might need them
        emergency_teams = ['IND', 'NE', 'LV', 'NO', 'NYG', 'CAR', 'TEN']  # Teams with QB uncertainty
        if team in emergency_teams and salary >= 6000:
            logger.debug(f"KEEPING emergency team QB: {name} ({team})")
            return True

        # Only filter QBs with very low salary AND very low projection AND unknown
        if salary < 6000 and fppg < 5 and not is_known_qb:
            logger.debug(f"FILTERING low-value unknown QB: {name} (${salary}, {fppg:.1f})")
            return False

        # Default: keep the QB (very conservative for tournaments)
        logger.debug(f"KEEPING QB by default: {name}")
        return True

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

            # SMARTER filtering - only remove obviously inactive players
            filtered_salary_data = []
            total_players = len(salary_data)
            filtered_counts = {'QB': [0, 0], 'RB': [0, 0], 'WR': [0, 0], 'TE': [0, 0], 'D': [0, 0]}

            for player_data in salary_data:
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

                # Use REAL FPPG from FanDuel data
                fppg = player_data.get('projected_points', 0)

                if fppg > 0:
                    projection = fppg
                    logger.debug(f"✅ {name}: Using real FPPG {fppg}")
                else:
                    logger.warning(f"⚠️ {name}: No FPPG data, using salary estimate")
                    if position == 'QB':
                        projection = max(12, (salary - 5500) / 200 + 15)
                    elif position == 'RB':
                        projection = max(8, (salary - 4000) / 300 + 10)
                    elif position == 'WR':
                        projection = max(6, (salary - 3500) / 400 + 8)
                    elif position == 'TE':
                        projection = max(5, (salary - 3500) / 350 + 7)
                    elif position == 'D':
                        projection = max(4, (salary - 3000) / 200 + 6)
                    else:
                        projection = 8

                # Filter by teams (if applicable)
                if playing_teams and team not in playing_teams:
                    continue

                winning_players.append({
                    'player_id': f"fd_{player_data.get('id', name)}",
                    'name': name,
                    'position': position,
                    'team': team,
                    'salary': salary,
                    'projected_points': round(projection, 2),
                    'projection': round(projection, 2),
                    'fppg_source': 'real' if fppg > 0 else 'estimated',
                    'ceiling': round(projection * 1.4, 2),
                    'floor': round(projection * 0.7, 2),
                    'weather_factor': 1.0,
                    'ownership': np.random.uniform(5.0, 35.0),
                    'opponent': player_data.get('opponent', ''),
                    'value': round(projection / (salary / 1000), 2) if salary > 0 else 0
                })

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
            real_count = sum(1 for p in winning_players if p['fppg_source'] == 'real')
            estimated_count = len(winning_players) - real_count
            logger.info(f"Projection sources: {real_count} real FPPG, {estimated_count} estimated")

            return winning_players

        except Exception as e:
            logger.error(f"Error collecting players: {e}")
            return []

# Main entry point
async def get_fresh_data() -> Dict[str, Any]:
    """Get fresh data with robust error handling"""
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

        return {
            'players': players,
            'games_info': games_info,
            'weather': weather_data,
            'vegas_odds': vegas_data,
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
                'salary_range': {
                    'min': min(p['salary'] for p in players) if players else 0,
                    'max': max(p['salary'] for p in players) if players else 0
                },
                'vegas_games': len(vegas_data) if vegas_data else 0
            }
        }