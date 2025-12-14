import asyncio
from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups

async def quick_test():
    print('Getting data...')
    data = await get_fresh_data()
    
    # Count positions in ALL 260 players
    positions = {}
    for p in data['players']:
        pos = p.get('position', 'UNKNOWN')
        positions[pos] = positions.get(pos, 0) + 1
    
    print('ALL 260 PLAYERS BY POSITION:')
    for pos, count in sorted(positions.items()):
        print(f'  {pos}: {count} players')
    
    # Quick optimization test - DISABLE AI for speed
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
    
    print('\nTesting optimizer (AI DISABLED for speed)...')
    lineups = optimize_dfs_lineups(
        player_data=player_data,
        num_lineups=1,
        contest_type='gpp',
        use_monte_carlo=False
    )
    
    print(f'Result: {len(lineups)} lineup(s) generated')
    if lineups:
        lineup = lineups[0]
        print(f'Success: ${lineup.total_salary:,}, {lineup.projected_points:.1f} pts')

if __name__ == "__main__":
    # Disable AI for this test
    import os
    os.environ['AI_ENABLED'] = 'false'
    asyncio.run(quick_test())
