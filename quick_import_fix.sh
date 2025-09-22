#!/bin/bash

echo "🔧 Fixing DFSOptimizer import error..."

cd /home/brett/fanduel
source venv/bin/activate

# Fix the import in scheduler.py
sed -i 's/from optimizer import optimize_dfs_lineups, DFSOptimizer/from optimizer import optimize_dfs_lineups, EnhancedDFSOptimizer as DFSOptimizer/g' scheduler.py

# Also add the missing export to optimizer.py if needed
echo "" >> optimizer.py
echo "# Backward compatibility alias" >> optimizer.py
echo "DFSOptimizer = EnhancedDFSOptimizer" >> optimizer.py

echo "✅ Import fixed"

# Test the single game directly
echo "🧪 Testing single game with current data..."

python3 -c "
import sys
sys.path.append('.')
import requests
import json

# Test the single game endpoint
try:
    response = requests.get('http://localhost:8020/games/current-week', timeout=5)
    if response.status_code == 200:
        games = response.json()
        print(f'✅ Games endpoint working: {len(games)} games available')
        if games:
            game = games[0]
            print(f'   Sample game: {game[\"away_team\"]} @ {game[\"home_team\"]} ({game[\"id\"]})')
            
            # Test single game optimization
            print('🎯 Testing single game optimization...')
            payload = {
                'contest_type': 'single_game',
                'num_lineups': 3,
                'single_game_id': game['id'],
                'avoid_high_ownership': False,
                'force_stacks': False
            }
            
            opt_response = requests.post(
                'http://localhost:8020/optimize',
                json=payload,
                timeout=30
            )
            
            if opt_response.status_code == 200:
                lineups = opt_response.json()
                print(f'✅ Single game optimization worked: {len(lineups)} lineups')
                if lineups:
                    lineup = lineups[0]
                    print(f'   Sample lineup: {lineup[\"projected_points\"]:.1f} pts, \${lineup[\"total_salary\"]:,}')
                    print(f'   Players: {len(lineup[\"players\"])}')
            else:
                print(f'❌ Single game optimization failed: {opt_response.status_code}')
                print(f'   Error: {opt_response.text[:200]}')
        else:
            print('❌ No games returned')
    else:
        print(f'❌ Games endpoint failed: {response.status_code}')
        
except Exception as e:
    print(f'❌ Test failed: {e}')
"

echo ""
echo "🎉 Quick fix complete!"
echo "Your system should now work properly at http://localhost:8020"
