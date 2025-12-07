# late_swap.py
"""
Late-Swap Engine for DFS Optimization
Handles:
1. Game start time tracking
2. Auto-locking players in started games
3. Filtering available player pool to unstarted games only
4. Optimizing remaining roster slots
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import pytz
from loguru import logger


class LateSwapEngine:
    """
    Manages late-swap logic for mid-slate lineup adjustments
    """

    def __init__(self):
        self.eastern = pytz.timezone('US/Eastern')
        # Buffer before game start (minutes) - can't swap within this window
        self.lockout_buffer_minutes = 5

    def get_current_time_et(self) -> datetime:
        """Get current time in Eastern timezone"""
        return datetime.now(self.eastern)

    def parse_game_datetime(self, game_info: Dict) -> Optional[datetime]:
        """
        Parse game datetime from various formats and return
        a timezone-aware datetime in Eastern time.

        This method supports both actual ``datetime`` objects as well as ISO
        formatted strings. ESPN's API typically returns a native datetime
        object in ``game_datetime`` but may occasionally return an ISO
        formatted string if serialization happens elsewhere. If no explicit
        datetime is provided the method falls back to parsing the human
        readable ``time`` string (e.g., "Sunday 1:00 PM ET").
        """
        # Prefer an explicit datetime value when available. ESPN's API uses
        # ``game_datetime`` to hold a timezone‐aware ``datetime``. However,
        # callers may serialize this to a string; handle that scenario as well.
        dt = game_info.get('game_datetime')
        if dt:
            # If we received a datetime instance, normalise it to Eastern.
            if isinstance(dt, datetime):
                # Naive datetimes are assumed to be Eastern.
                if dt.tzinfo is None:
                    return self.eastern.localize(dt)
                return dt.astimezone(self.eastern)

            # Support ISO formatted strings (e.g., "2025-12-07T18:00:00Z").
            if isinstance(dt, str):
                try:
                    # Replace 'Z' suffix with '+00:00' for UTC aware parsing
                    iso_str = dt.replace('Z', '+00:00')
                    parsed = datetime.fromisoformat(iso_str)
                    # Treat naive results as UTC then convert to Eastern
                    if parsed.tzinfo is None:
                        parsed = pytz.utc.localize(parsed)
                    return parsed.astimezone(self.eastern)
                except Exception:
                    # Fall back to time string parsing
                    pass

        # If no explicit datetime is available, parse the human readable
        # ``time`` string like "Sunday 1:00 PM ET". This assumes the game
        # occurs within the current NFL week (Sunday to Monday). If the
        # ``time`` field is malformed this function will log a warning and
        # return ``None`` which results in the game being treated as
        # unstarted.
        time_str = game_info.get('time')
        if time_str:
            return self._parse_time_string(time_str)

        # Without a datetime or time string we cannot determine the game time.
        logger.warning(f"Missing time information for game: {game_info}")
        return None

    def _parse_time_string(self, time_str: str) -> Optional[datetime]:
        """
        Parse time string like 'Sunday 1:00 PM ET' into datetime
        Assumes current week's games
        """
        try:
            # Get current date info
            now = self.get_current_time_et()

            # Map day names to weekday numbers
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }

            parts = time_str.lower().split()
            if not parts:
                return None

            day_name = parts[0]
            target_weekday = day_map.get(day_name)
            if target_weekday is None:
                return None

            # Parse time portion (e.g., "1:00 PM")
            time_part = ' '.join(parts[1:3]) if len(parts) >= 3 else '1:00 PM'
            time_part = time_part.upper().replace('ET', '').strip()

            # Parse hour and minute
            try:
                time_obj = datetime.strptime(time_part, '%I:%M %p')
            except ValueError:
                try:
                    time_obj = datetime.strptime(time_part, '%I:%M%p')
                except ValueError:
                    # Default to 1 PM
                    time_obj = datetime.strptime('1:00 PM', '%I:%M %p')

            # Calculate the target date
            current_weekday = now.weekday()
            days_ahead = target_weekday - current_weekday

            # If day already passed this week, it might be from last week or next week
            # For NFL, assume within -1 to +6 days
            if days_ahead < -1:
                days_ahead += 7

            target_date = now.date() + timedelta(days=days_ahead)

            # Combine date and time
            game_dt = datetime.combine(
                target_date,
                time_obj.time()
            )

            return self.eastern.localize(game_dt)

        except Exception as e:
            logger.warning(f"Failed to parse time string '{time_str}': {e}")
            return None

    def has_game_started(self, game_info: Dict, buffer_minutes: int = None) -> bool:
        """
        Check if a game has started (or is about to start within buffer)
        """
        if buffer_minutes is None:
            buffer_minutes = self.lockout_buffer_minutes

        game_time = self.parse_game_datetime(game_info)
        if game_time is None:
            # If we can't determine time, assume NOT started (safer for optimization)
            logger.warning(f"Could not determine game time for {game_info.get('id', 'unknown')}")
            return False

        now = self.get_current_time_et()
        lockout_time = game_time - timedelta(minutes=buffer_minutes)

        return now >= lockout_time

    def get_started_teams(self, games_info: Dict) -> Set[str]:
        """
        Get set of team abbreviations whose games have started
        """
        started_teams = set()

        all_games = games_info.get('all_games', [])
        if not all_games:
            all_games = games_info.get('main_slate', [])

        for game in all_games:
            if self.has_game_started(game):
                teams = game.get('teams', [])
                started_teams.update(teams)
                logger.info(f"🔒 Game started: {' vs '.join(teams)}")

        return started_teams

    def get_available_teams(self, games_info: Dict) -> Set[str]:
        """
        Get set of team abbreviations whose games have NOT started
        """
        available_teams = set()

        all_games = games_info.get('all_games', [])
        if not all_games:
            all_games = games_info.get('main_slate', [])

        for game in all_games:
            if not self.has_game_started(game):
                teams = game.get('teams', [])
                available_teams.update(teams)

        return available_teams

    def filter_players_for_late_swap(
            self,
            players: List[Dict],
            games_info: Dict,
            locked_players: List[str] = None
    ) -> Tuple[List[Dict], List[Dict], Set[str]]:
        """
        Filter players for late-swap optimization

        Returns:
            - available_players: Players in unstarted games (can be selected)
            - locked_players_data: Player data for locked selections
            - started_teams: Teams whose games have started
        """
        locked_players = locked_players or []
        locked_set = set(locked_players)

        started_teams = self.get_started_teams(games_info)
        available_teams = self.get_available_teams(games_info)

        logger.info(f"⏰ Late-swap status:")
        logger.info(f"   🔒 Started teams: {sorted(started_teams) if started_teams else 'None'}")
        logger.info(f"   ✅ Available teams: {sorted(available_teams) if available_teams else 'None'}")

        available_players = []
        locked_players_data = []

        for player in players:
            player_name = player.get('name', player.get('player_id', ''))
            player_team = player.get('team', '')

            # Check if player is explicitly locked
            is_locked = player_name in locked_set or player.get('locked', False)

            if is_locked:
                # Keep locked player data regardless of game status
                player['locked'] = True
                locked_players_data.append(player)
                logger.debug(f"🔒 Locked: {player_name} ({player_team})")
            elif player_team in started_teams:
                # Player's game started but wasn't locked - skip them
                logger.debug(f"⏭️ Skipping {player_name} - game started")
                continue
            elif player_team in available_teams:
                # Player available for selection
                available_players.append(player)
            else:
                # Team not in any game list - skip
                continue

        logger.info(f"📊 Late-swap pool: {len(available_players)} available, {len(locked_players_data)} locked")

        return available_players, locked_players_data, started_teams

    def get_remaining_positions(self, locked_players_data: List[Dict]) -> Dict[str, int]:
        """
        Calculate remaining position slots after accounting for locked players

        FanDuel format: QB(1) + RB(2) + WR(3) + TE(1) + FLEX(1) + DEF(1) = 9
        """
        # Full roster requirements
        full_roster = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'FLEX': 1,  # RB/WR/TE
            'D': 1
        }

        # Count locked by position
        locked_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'D': 0}
        for player in locked_players_data:
            pos = player.get('position', '')
            if pos in locked_counts:
                locked_counts[pos] += 1

        # Calculate remaining
        remaining = {}

        # Handle core positions
        remaining['QB'] = max(0, full_roster['QB'] - locked_counts['QB'])
        remaining['D'] = max(0, full_roster['D'] - locked_counts['D'])

        # RB: Need 2 minimum, can have 3 with FLEX
        rb_locked = locked_counts['RB']
        if rb_locked >= 2:
            remaining['RB'] = 0  # Have minimum, could still use for FLEX
        else:
            remaining['RB'] = 2 - rb_locked

        # WR: Need 3 minimum, can have 4 with FLEX
        wr_locked = locked_counts['WR']
        if wr_locked >= 3:
            remaining['WR'] = 0
        else:
            remaining['WR'] = 3 - wr_locked

        # TE: Need 1 minimum, can have 2 with FLEX
        te_locked = locked_counts['TE']
        if te_locked >= 1:
            remaining['TE'] = 0
        else:
            remaining['TE'] = 1 - te_locked

        # FLEX: Calculate if there's room
        total_locked = sum(locked_counts.values())
        if total_locked >= 9:
            remaining['FLEX'] = 0
        else:
            # Check if FLEX is still available
            flex_eligible_locked = locked_counts['RB'] + locked_counts['WR'] + locked_counts['TE']
            min_flex_eligible_needed = 2 + 3 + 1  # Minimum RB + WR + TE
            if flex_eligible_locked > min_flex_eligible_needed:
                remaining['FLEX'] = 0  # FLEX already filled
            else:
                remaining['FLEX'] = 1

        logger.info(f"📋 Remaining slots: {remaining}")
        return remaining

    def get_game_status_summary(self, games_info: Dict) -> List[Dict]:
        """
        Get summary of all games with their status for UI display
        """
        summary = []
        now = self.get_current_time_et()

        all_games = games_info.get('all_games', [])
        if not all_games:
            all_games = games_info.get('main_slate', [])

        for game in all_games:
            game_time = self.parse_game_datetime(game)
            started = self.has_game_started(game)

            if game_time:
                if started:
                    status = "IN PROGRESS"
                    time_display = "LOCKED"
                else:
                    time_until = game_time - now
                    minutes = int(time_until.total_seconds() / 60)
                    if minutes < 60:
                        time_display = f"{minutes}m until lock"
                    else:
                        hours = minutes // 60
                        mins = minutes % 60
                        time_display = f"{hours}h {mins}m until lock"
                    status = "UPCOMING"
            else:
                status = "UNKNOWN"
                time_display = game.get('time', 'TBD')

            summary.append({
                'game_id': game.get('id', 'Unknown'),
                'teams': game.get('teams', []),
                'time_slot': game.get('time_slot', 'unknown'),
                'time_display': time_display,
                'status': status,
                'started': started
            })

        # Sort: started games first, then by time
        summary.sort(key=lambda x: (not x['started'], x['time_display']))

        return summary


# Convenience function for use in data_collector
def filter_for_late_swap(
        players: List[Dict],
        games_info: Dict,
        locked_players: List[str] = None
) -> Dict[str, any]:
    """
    Main entry point for late-swap filtering

    Returns dict with:
        - available_players: Players that can still be selected
        - locked_players_data: Data for locked players
        - started_teams: Teams whose games started
        - remaining_positions: Position slots still needed
        - game_status: Summary of all game statuses
    """
    engine = LateSwapEngine()

    available, locked_data, started = engine.filter_players_for_late_swap(
        players, games_info, locked_players
    )

    remaining = engine.get_remaining_positions(locked_data)
    game_status = engine.get_game_status_summary(games_info)

    return {
        'available_players': available,
        'locked_players_data': locked_data,
        'started_teams': started,
        'remaining_positions': remaining,
        'game_status': game_status,
        'total_locked': len(locked_data),
        'total_available': len(available)
    }