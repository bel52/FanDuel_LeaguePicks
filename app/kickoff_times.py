import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

# Default NFL kickoff times (ET)
DEFAULT_KICKOFFS = {
    'SUNDAY_EARLY': '13:00',
    'SUNDAY_LATE': '16:05', 
    'SUNDAY_NIGHT': '20:20',
    'MONDAY_NIGHT': '20:15',
    'THURSDAY_NIGHT': '20:15'
}

def build_kickoff_map(df: pd.DataFrame) -> Dict[str, datetime]:
    """Build a mapping of teams to kickoff times"""
    
    kickoff_map = {}
    eastern = pytz.timezone('America/New_York')
    now = datetime.now(eastern)
    
    # Parse game info if available
    if 'GAME INFO' in df.columns:
        for _, row in df.iterrows():
            team = row['TEAM']
            if team in kickoff_map:
                continue
            
            game_info = str(row.get('GAME INFO', ''))
            
            # Try to extract time from game info
            kickoff_time = extract_kickoff_time(game_info, now)
            if kickoff_time:
                kickoff_map[team] = kickoff_time
    
    # Use default times for any missing teams
    for _, row in df.iterrows():
        team = row['TEAM']
        if team not in kickoff_map:
            # Default to Sunday early game
            kickoff_map[team] = get_next_sunday_kickoff(now, 'SUNDAY_EARLY')
    
    return kickoff_map

def extract_kickoff_time(game_info: str, base_date: datetime) -> Optional[datetime]:
    """Extract kickoff time from game info string"""
    
    import re
    
    # Look for time patterns like "1:00PM" or "13:00"
    time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM)?'
    match = re.search(time_pattern, game_info)
    
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        am_pm = match.group(3)
        
        if am_pm:
            if am_pm.upper() == 'PM' and hour != 12:
                hour += 12
            elif am_pm.upper() == 'AM' and hour == 12:
                hour = 0
        
        # Create datetime for kickoff
        kickoff = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Check day of week in game info
        if any(day in game_info.upper() for day in ['MON', 'MONDAY']):
            # Adjust to Monday
            days_ahead = 0 if base_date.weekday() == 0 else (7 - base_date.weekday())
            kickoff += timedelta(days=days_ahead)
        elif any(day in game_info.upper() for day in ['THU', 'THURSDAY']):
            # Adjust to Thursday
            days_ahead = 3 - base_date.weekday()
            if days_ahead < 0:
                days_ahead += 7
            kickoff += timedelta(days=days_ahead)
        else:
            # Default to Sunday
            days_ahead = 6 - base_date.weekday()
            if days_ahead < 0:
                days_ahead += 7
            kickoff += timedelta(days=days_ahead)
        
        return kickoff
    
    return None

def get_next_sunday_kickoff(base_date: datetime, slot: str = 'SUNDAY_EARLY') -> datetime:
    """Get the next Sunday kickoff time"""
    
    # Find next Sunday
    days_ahead = 6 - base_date.weekday()
    if days_ahead < 0:
        days_ahead += 7
    
    next_sunday = base_date + timedelta(days=days_ahead)
    
    # Parse kickoff time
    time_str = DEFAULT_KICKOFFS.get(slot, '13:00')
    hour, minute = map(int, time_str.split(':'))
    
    return next_sunday.replace(hour=hour, minute=minute, second=0, microsecond=0)

def auto_lock_started_players(
    last_lineup: List[Dict],
    kickoff_map: Dict[str, datetime]
) -> List[str]:
    """Auto-lock players whose games have started"""
    
    if not last_lineup:
        return []
    
    eastern = pytz.timezone('America/New_York')
    current_time = datetime.now(eastern)
    locked_players = []
    
    for player in last_lineup:
        team = player.get('team', '')
        kickoff = kickoff_map.get(team)
        
        if kickoff and current_time > kickoff:
            locked_players.append(player['name'])
            logger.info(f"Auto-locking {player['name']} - game started at {kickoff}")
    
    return locked_players

def save_last_lineup(lineup_players: List[Dict], file_path: str = "data/output/last_lineup.json"):
    """Save the last generated lineup for auto-lock reference"""
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'players': lineup_players
            }, f, indent=2)
        
        logger.info(f"Saved lineup to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving lineup: {e}")

def load_last_lineup(file_path: str = "data/output/last_lineup.json") -> List[Dict]:
    """Load the last saved lineup"""
    
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('players', [])
    except Exception as e:
        logger.error(f"Error loading last lineup: {e}")
        return []

def get_late_swap_window(kickoff_map: Dict[str, datetime]) -> Dict[str, str]:
    """Determine late swap window status for each game"""
    
    eastern = pytz.timezone('America/New_York')
    current_time = datetime.now(eastern)
    swap_status = {}
    
    # Group games by kickoff time
    kickoff_groups = {}
    for team, kickoff in kickoff_map.items():
        kickoff_str = kickoff.strftime('%Y-%m-%d %H:%M')
        if kickoff_str not in kickoff_groups:
            kickoff_groups[kickoff_str] = []
        kickoff_groups[kickoff_str].append(team)
    
    # Determine swap status
    for kickoff_str, teams in kickoff_groups.items():
        kickoff = datetime.strptime(kickoff_str, '%Y-%m-%d %H:%M')
        kickoff = eastern.localize(kickoff)
        
        if current_time < kickoff:
            status = 'OPEN'
        elif current_time < kickoff + timedelta(hours=3):
            status = 'IN_PROGRESS'
        else:
            status = 'CLOSED'
        
        for team in teams:
            swap_status[team] = status
    
    return swap_status

def filter_swappable_players(
    df: pd.DataFrame,
    kickoff_map: Dict[str, datetime]
) -> pd.DataFrame:
    """Filter to only players whose games haven't started"""
    
    swap_status = get_late_swap_window(kickoff_map)
    
    # Only include players from teams with OPEN status
    swappable = df[df['TEAM'].isin([
        team for team, status in swap_status.items() if status == 'OPEN'
    ])]
    
    return swappable

def get_game_schedule(week: int = None, season: int = None) -> Dict[str, Any]:
    """Get NFL game schedule for the week"""
    
    # This would integrate with an API for real schedule data
    # For now, return mock schedule
    
    if season is None:
        season = datetime.now().year
    
    schedule = {
        'week': week or get_current_nfl_week(),
        'season': season,
        'games': [
            {
                'home': 'BUF',
                'away': 'MIA',
                'kickoff': '2024-12-08 13:00:00',
                'tv': 'CBS'
            },
            {
                'home': 'KC',
                'away': 'LV', 
                'kickoff': '2024-12-08 16:25:00',
                'tv': 'CBS'
            }
            # Add more games...
        ]
    }
    
    return schedule

def get_current_nfl_week() -> int:
    """Calculate current NFL week number"""
    
    # NFL season typically starts first Thursday in September
    # This is simplified - you'd want more precise logic
    
    now = datetime.now()
    
    # 2024 season starts September 5
    season_start = datetime(2024, 9, 5)
    
    if now < season_start:
        return 1
    
    weeks_elapsed = (now - season_start).days // 7
    
    # NFL regular season is 18 weeks
    return min(weeks_elapsed + 1, 18)
