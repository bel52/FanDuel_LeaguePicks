#!/usr/bin/env python3
"""
Quick test script for single game functionality
"""
import sys
sys.path.append('.')
import asyncio
from optimizer import optimize_dfs_lineups
from data_collector import get_fresh_data

async def test_single_game():
    print("🧪 Testing Single Game Functionality")
    print("=" * 40)
    
    try:
        # Get real data
        print("📊 Getting fresh data...")
        data = await get_fresh_data()
        
        if not data or not data.get('players'):
            print("❌ No player data available")
            return False
        
        players = data['players']
        print(f"✅ Got {len(players)} players")
        
        # Test single game with PHI vs WAS
        print("\n🏈 Testing PHI vs WAS single game...")
        lineups = optimize_dfs_lineups(
            player_data=players,
            num_lineups=3,
            contest_type='single_game',
            single_game_teams=['PHI', 'WAS']
        )
        
        if lineups:
            print(f"✅ Generated {len(lineups)} single game lineups")
            for i, lineup in enumerate(lineups, 1):
                print(f"\nLineup {i}:")
                print(f"  Points: {lineup.projected_points:.1f} (with MVP 1.5x)")
                print(f"  Salary: ${lineup.total_salary:,}")
                print(f"  Players: {len(lineup.players)}")
                for j, player in enumerate(lineup.players):
                    mvp_text = " (MVP 1.5x)" if j == 0 else ""
                    print(f"    {player.position}: {player.name} ({player.team}){mvp_text}")
            return True
        else:
            print("❌ No single game lineups generated")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_game())
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
