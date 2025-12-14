import asyncio
from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups
import os

async def test_strategies():
    os.environ['AI_ENABLED'] = 'false'  # Speed up testing
    
    data = await get_fresh_data()
    player_data = []
    for p in data['players']:
        player_data.append({
            'player_id': p.get('player_id'),
            'name': p.get('name'),
            'position': p.get('position'),
            'team': p.get('team'),
            'salary': p.get('salary'),
            'projected_points': p.get('projected_points')
        })
    
    strategies = ['gpp', 'cash', 'friends_league']
    
    for strategy in strategies:
        print(f'\n=== {strategy.upper()} STRATEGY ===')
        lineups = optimize_dfs_lineups(
            player_data=player_data,
            num_lineups=1,
            contest_type=strategy,
            use_monte_carlo=False
        )
        
        if lineups:
            lineup = lineups[0]
            print(f'${lineup.total_salary:,} salary, {lineup.projected_points:.1f} pts')
            print('Top 3 plays:')
            for i, player in enumerate(lineup.players[:3]):
                print(f'  {player.name} ({player.position}) ${player.salary}')

asyncio.run(test_strategies())
