"""
Fix for single game lineup generation and default lineup counts
"""

# First, let's update the API to fix the team mapping and single game logic
def fix_api_single_game():
    api_fixes = '''
# Add this function to api.py after line 350 (in the get_teams_from_game_id function)

def get_teams_from_game_id(game_id: str) -> List[str]:
    """Get team codes from game ID - enhanced with current week logic"""
    logger.info(f"Looking up teams for game ID: {game_id}")
    
    # Try to get teams from current ESPN data first
    try:
        scheduler = get_scheduler()
        if scheduler and scheduler.current_data:
            espn_data = scheduler.current_data.get('espn_data', {})
            if 'scoreboard' in espn_data:
                for event in espn_data['scoreboard'].get('events', []):
                    event_id = f"game_{event.get('id')}"
                    if event_id == game_id:
                        teams = []
                        if 'competitions' in event:
                            for comp in event['competitions']:
                                if 'competitors' in comp:
                                    for competitor in comp['competitors']:
                                        team_abbr = competitor.get('team', {}).get('abbreviation', '')
                                        if team_abbr:
                                            teams.append(team_abbr.upper())
                        if len(teams) >= 2:
                            logger.info(f"Found teams from ESPN data: {teams}")
                            return teams
    except Exception as e:
        logger.warning(f"Could not get teams from ESPN data: {e}")
    
    # Fallback to static mapping - UPDATE THIS WEEKLY
    game_team_mapping = {
        "game_1": ["PHI", "WAS"],
        "game_2": ["BAL", "BUF"], 
        "game_3": ["DET", "GB"],
        "game_4": ["KC", "LAC"],
        "game_5": ["SF", "DAL"],
        "game_6": ["TEN", "MIA"],
        "game_7": ["NYG", "MIN"],
        "game_8": ["CIN", "PIT"],
        "game_9": ["HOU", "JAX"],
        "game_10": ["ATL", "CAR"],
        "game_11": ["LAR", "ARI"],
        "game_12": ["TB", "NO"],
        "game_13": ["DEN", "NYJ"],
        "game_14": ["CLE", "LV"],
        "game_15": ["NE", "SEA"],
        "game_16": ["CHI", "IND"]
    }
    
    teams = game_team_mapping.get(game_id, [])
    logger.info(f"Game ID {game_id} maps to teams: {teams}")
    
    # If no mapping found, try to extract from game_id pattern
    if not teams and "_" in game_id:
        try:
            # Look for common team patterns in fallback games
            fallback_games = get_fallback_current_week_games()
            for game in fallback_games:
                if game['id'] == game_id:
                    teams = [game['away_team'], game['home_team']]
                    logger.info(f"Found teams from fallback: {teams}")
                    break
        except Exception as e:
            logger.error(f"Error extracting teams from fallback: {e}")
    
    return teams

# Update the fallback games function to have more current week games
def get_fallback_current_week_games() -> List[Dict]:
    """Enhanced fallback games for current week"""
    return [
        {
            "id": "game_1", "away_team": "PHI", "home_team": "WAS",
            "time": "Sunday 1:00 PM ET", "entry_range": "$1-$25", "total_points": 47.5, "week": 3
        },
        {
            "id": "game_2", "away_team": "BAL", "home_team": "BUF", 
            "time": "Sunday 1:00 PM ET", "entry_range": "$1-$25", "total_points": 51.0, "week": 3
        },
        {
            "id": "game_3", "away_team": "DET", "home_team": "GB",
            "time": "Sunday 1:00 PM ET", "entry_range": "$1-$25", "total_points": 49.5, "week": 3
        },
        {
            "id": "game_4", "away_team": "KC", "home_team": "LAC",
            "time": "Sunday 4:25 PM ET", "entry_range": "$1-$25", "total_points": 53.0, "week": 3
        },
        {
            "id": "game_5", "away_team": "SF", "home_team": "DAL",
            "time": "Sunday 8:20 PM ET", "entry_range": "$1-$25", "total_points": 46.0, "week": 3
        },
        {
            "id": "game_6", "away_team": "TEN", "home_team": "MIA",
            "time": "Monday 8:15 PM ET", "entry_range": "$1-$25", "total_points": 44.5, "week": 3
        },
        {
            "id": "game_7", "away_team": "NYG", "home_team": "MIN",
            "time": "Sunday 1:00 PM ET", "entry_range": "$1-$25", "total_points": 48.0, "week": 3
        },
        {
            "id": "game_8", "away_team": "CIN", "home_team": "PIT", 
            "time": "Sunday 1:00 PM ET", "entry_range": "$1-$25", "total_points": 45.5, "week": 3
        },
        {
            "id": "game_9", "away_team": "HOU", "home_team": "JAX",
            "time": "Sunday 1:00 PM ET", "entry_range": "$1-$25", "total_points": 43.0, "week": 3
        },
        {
            "id": "game_10", "away_team": "ATL", "home_team": "CAR",
            "time": "Sunday 1:00 PM ET", "entry_range": "$1-$25", "total_points": 50.5, "week": 3
        }
    ]
'''
    return api_fixes

# Fix the optimizer to handle single game better
def fix_optimizer_single_game():
    optimizer_fixes = '''
# Add these methods to the EnhancedDFSOptimizer class:

def _filter_players_for_single_game(self, players: List[Player], teams: List[str]) -> List[Player]:
    """Filter and enhance players for single game contests"""
    filtered_players = []
    
    logger.info(f"Filtering {len(players)} players for single game teams: {teams}")
    
    for player in players:
        if player.team.upper() in [t.upper() for t in teams]:
            # Create a copy with single game adjustments
            sg_player = Player(
                id=player.id, name=player.name, position=player.position,
                team=player.team, salary=player.salary, projection=player.projection,
                ownership=player.ownership, weather_factor=player.weather_factor,
                injury_risk=player.injury_risk, value=player.value, variance=player.variance
            )
            
            # Boost projections for single game (higher scoring)
            sg_player.projection *= 1.1
            
            # Recalculate value
            sg_player.value = sg_player.projection / (sg_player.salary / 1000) if sg_player.salary > 0 else 0
            
            filtered_players.append(sg_player)
    
    logger.info(f"Filtered to {len(filtered_players)} players for single game")
    
    # Ensure we have enough players per position
    positions_count = {}
    for player in filtered_players:
        positions_count[player.position] = positions_count.get(player.position, 0) + 1
    
    logger.info(f"Single game position counts: {positions_count}")
    
    return filtered_players

def _single_game_constraints(self, prob, players: List[Player], player_vars: Dict):
    """Add single game specific constraints"""
    
    # Salary cap
    prob += pulp.lpSum([players[i].salary * player_vars[i] for i in range(len(players))]) <= FANDUEL_SALARY_CAP
    
    # Exactly 6 players for single game
    prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 6
    
    # At least 1 QB (for MVP consideration)
    qb_players = [i for i, p in enumerate(players) if p.position == 'QB']
    if qb_players:
        prob += pulp.lpSum([player_vars[i] for i in qb_players]) >= 1
    
    # At least 2 different positions
    position_groups = {}
    for i, player in enumerate(players):
        pos = player.position
        if pos not in position_groups:
            position_groups[pos] = []
        position_groups[pos].append(i)
    
    # Ensure position diversity
    if len(position_groups) >= 2:
        for pos, indices in position_groups.items():
            if len(indices) > 0:
                # At least 1 from each major position group if available
                if pos in ['QB', 'RB', 'WR', 'TE'] and len(indices) >= 1:
                    prob += pulp.lpSum([player_vars[i] for i in indices]) >= 1
'''
    return optimizer_fixes

if __name__ == "__main__":
    print("Single Game Fix Instructions:")
    print("1. Update api.py with the enhanced team mapping")
    print("2. Update optimizer.py with single game filtering")
    print("3. Fix default lineup counts in the HTML")
    print("4. Test single game generation")
