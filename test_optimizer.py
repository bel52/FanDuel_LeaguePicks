import asyncio
from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups

async def test_optimizer():
    print('Getting fresh data...')
    data = await get_fresh_data()
    
    print('REAL ESPN GAMES (15 games):')
    for game in data['games_info']['all_games'][:5]:
        teams = ' vs '.join(game['teams'])
        print(f'  {teams}')
    
    print(f'Players available: {len(data["players"])}')
    
    # Convert data format for optimizer
    player_data = []
    for p in data['players']:
        player_data.append({
            'player_id': p.get('player_id'),
            'name': p.get('name'),
            'position': p.get('position'),
            'team': p.get('team'),
            'salary': p.get('salary'),
            'projected_points': p.get('projected_points'),
            'ownership': p.get('ownership', 15.0)
        })
    
    print(f'Building lineup with {len(player_data)} players...')
    
    # Test GPP optimization
    lineups = optimize_dfs_lineups(
        player_data=player_data,
        num_lineups=1,
        contest_type='gpp',
        use_monte_carlo=False
    )
    
    print(f'Generated {len(lineups)} lineup(s)')
    
    if lineups:
        lineup = lineups[0]
        print(f'Lineup: ${lineup.total_salary:,} salary, {lineup.projected_points:.1f} projected')
        
        for i, player in enumerate(lineup.players):
            print(f'  {i+1}. {player.name} ({player.position}) ${player.salary} - {player.projection:.1f} pts')

if __name__ == "__main__":
    asyncio.run(test_optimizer())
