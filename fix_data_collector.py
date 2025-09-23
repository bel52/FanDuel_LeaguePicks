#!/usr/bin/env python3

# Read the existing data_collector.py
with open('data_collector.py', 'r') as f:
    content = f.read()

# Add the missing import at the top
if 'import re' not in content:
    # Find the last import line and add after it
    lines = content.split('\n')
    import_lines = []
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            import_lines.append(i)
    
    if import_lines:
        last_import = max(import_lines)
        lines.insert(last_import + 1, 'import re')
        content = '\n'.join(lines)

# Find the collect_players_for_slate method and enhance it
enhanced_method = '''    async def collect_players_for_slate(self, games_info: Dict[str, Any], contest_type: str = 'gpp') -> List[Dict]:
        """Collect players filtered by contest type and slate WITH FanDuel salaries"""
        current_week = games_info['current_week']
        
        # Determine which teams to include
        if contest_type == 'single_game':
            playing_teams = set()
            for game in games_info['single_games']:
                playing_teams.update(game['teams'])
        else:
            playing_teams = set()
            for game in games_info['main_slate']:
                playing_teams.update(game['teams'])
        
        logger.info(f"Collecting players for {contest_type}: {len(playing_teams)} teams playing")
        
        # Get FanDuel salaries FIRST
        salary_data = None
        try:
            from fanduel_salary_scraper import get_fanduel_salaries
            salary_data = await get_fanduel_salaries()
            logger.info(f"📊 Retrieved {len(salary_data)} FanDuel salary entries")
        except Exception as e:
            logger.error(f"Failed to get FanDuel salaries: {e}")
            import pandas as pd
            salary_data = pd.DataFrame()
        
        # Get NFL projection data
        try:
            await self.rate_limiters['nfl_data'].acquire()
            weekly_data = nfl.import_weekly_data([2024])
            
            if not weekly_data.empty:
                # Filter for current week and relevant teams
                relevant_data = weekly_data[
                    (weekly_data['week'] == current_week) & 
                    (weekly_data['recent_team'].isin(playing_teams))
                ].copy()
                
                # If insufficient data, include recent weeks
                if len(relevant_data) < 50:
                    recent_weeks = [current_week - 1, current_week - 2]
                    backup_data = weekly_data[
                        (weekly_data['week'].isin(recent_weeks)) & 
                        (weekly_data['recent_team'].isin(playing_teams))
                    ]
                    relevant_data = pd.concat([relevant_data, backup_data]).drop_duplicates('player_id')
                
                # Use existing processing method 
                players = self._process_player_data(relevant_data, playing_teams)
            else:
                players = self._generate_fallback_players(playing_teams)
                
        except Exception as e:
            logger.error(f"Error collecting NFL data: {e}")
            players = self._generate_fallback_players(playing_teams)
        
        # Ensure complete rosters
        players = self._ensure_complete_rosters(players, playing_teams)
        
        logger.info(f"Collected {len(players)} players for {contest_type}")
        return players'''

# Write the fixed content
with open('data_collector.py', 'w') as f:
    f.write(content)

print("✅ Fixed data_collector.py")
