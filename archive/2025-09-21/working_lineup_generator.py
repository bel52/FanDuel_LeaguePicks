#!/usr/bin/env python3
"""
Working lineup generator with correct column names for your optimization engine
"""

import sys
import asyncio
import pandas as pd
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

def print_success(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_info(msg):
    print(f"ℹ️  {msg}")

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

async def generate_working_lineup():
    """Generate a lineup using the correct DataFrame format."""
    print_header("DFS Lineup Generator")
    
    try:
        from app.cache_manager import CacheManager
        from app.optimization_engine import OptimizationEngine
        
        # Initialize components
        cache_manager = CacheManager()
        optimizer = OptimizationEngine(cache_manager)
        
        print_success("Optimization engine initialized")
        
        # Create DataFrame with the EXACT column names your optimizer expects
        players_data = {
            'PLAYER NAME': [
                'Josh Allen', 'Lamar Jackson',  # QBs
                'Derrick Henry', 'Christian McCaffrey', 'Austin Ekeler',  # RBs
                'Davante Adams', 'Stefon Diggs', 'Tyreek Hill', 'Mike Evans',  # WRs
                'Travis Kelce', 'Mark Andrews',  # TEs
                'Justin Tucker', 'Harrison Butker',  # Ks
                'Buffalo', 'San Francisco'  # DSTs
            ],
            'POS': [
                'QB', 'QB',
                'RB', 'RB', 'RB', 
                'WR', 'WR', 'WR', 'WR',
                'TE', 'TE',
                'K', 'K',
                'DST', 'DST'
            ],
            'SALARY': [
                8500, 8200,  # QBs
                6800, 9000, 7400,  # RBs
                8000, 7200, 7800, 6900,  # WRs
                6500, 5800,  # TEs
                4800, 4600,  # Ks
                3200, 3400  # DSTs
            ],
            'PROJ PTS': [
                22.5, 21.8,  # QBs
                18.2, 20.1, 16.5,  # RBs
                17.5, 16.8, 17.2, 15.9,  # WRs
                15.3, 13.8,  # TEs
                8.5, 8.2,  # Ks
                9.2, 8.8  # DSTs
            ],
            'TEAM': [
                'BUF', 'BAL',
                'TEN', 'SF', 'LAC',
                'LV', 'BUF', 'MIA', 'TB', 
                'KC', 'BAL',
                'BAL', 'KC',
                'BUF', 'SF'
            ]
        }
        
        # Add optional columns that your optimizer might use
        players_data['CEILING'] = [pts * 1.4 for pts in players_data['PROJ PTS']]
        players_data['FLOOR'] = [pts * 0.6 for pts in players_data['PROJ PTS']]
        players_data['OWN_PCT'] = [15.0] * len(players_data['PLAYER NAME'])  # Default ownership
        
        # Create DataFrame
        df = pd.DataFrame(players_data)
        
        print_info(f"Created DataFrame with {len(df)} players")
        print_info(f"Columns: {list(df.columns)}")
        print_info(f"Total salary available: ${sum(df['SALARY']):,}")
        
        print("\nPlayer Pool Preview:")
        print(df[['PLAYER NAME', 'POS', 'SALARY', 'PROJ PTS', 'TEAM']].to_string(index=False))
        
        # Generate lineup
        print_info("\nGenerating optimal lineup...")
        
        try:
            result = await optimizer.optimize_lineup(
                player_data=df,
                game_type='league',
                salary_cap=50000,
                enforce_stack=True,
                use_ai=False  # Disable AI to avoid API calls for testing
            )
            
            if result and isinstance(result, dict):
                print_success("Lineup optimization successful!")
                
                # Extract lineup from result
                if 'lineup' in result:
                    lineup = result['lineup']
                    print_header("OPTIMAL LINEUP")
                    
                    total_salary = 0
                    total_projection = 0
                    
                    for i, player in enumerate(lineup):
                        player_name = player.get('PLAYER NAME', f'Player {i}')
                        position = player.get('POS', 'N/A')
                        salary = player.get('SALARY', 0)
                        projection = player.get('PROJ PTS', 0)
                        team = player.get('TEAM', 'N/A')
                        
                        print(f"{position:3} | {player_name:20} | {team:3} | ${salary:5,} | {projection:5.1f} pts")
                        total_salary += salary
                        total_projection += projection
                    
                    print("-" * 65)
                    print(f"{'TOT':3} | {'':20} | {'':3} | ${total_salary:5,} | {total_projection:5.1f} pts")
                    print(f"Remaining salary: ${50000 - total_salary:,}")
                    
                    # Show additional result info
                    if 'total_projected_points' in result:
                        print(f"Expected points: {result['total_projected_points']:.1f}")
                    if 'optimization_time' in result:
                        print(f"Optimization time: {result['optimization_time']:.3f}s")
                    
                    print_header("SUCCESS!")
                    print("🎉 Your DFS optimization system is working!")
                
                elif 'error' in result:
                    print_error(f"Optimization error: {result['error']}")
                    return False
                else:
                    print_info("Result format:")
                    for key, value in result.items():
                        print(f"  {key}: {value}")
            
            elif result is None:
                print_error("Optimization returned None - check constraints or player pool")
                return False
            
            else:
                print_error(f"Unexpected result type: {type(result)}")
                print_info(f"Result: {result}")
                return False
            
            return True
            
        except Exception as e:
            print_error(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print_error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_sample_data():
    """Test creating sample data from DataIngestion."""
    print_header("Sample Data Test")
    
    try:
        from app.data_ingestion import DataIngestion
        
        di = DataIngestion()
        print_success("DataIngestion initialized")
        
        # Create data directory if it doesn't exist
        import os
        os.makedirs('data/input', exist_ok=True)
        print_info("Created data/input directory")
        
        sample_data = di.create_sample_data()
        
        if sample_data is not None:
            print_success(f"Sample data created: {type(sample_data)}")
            
            if hasattr(sample_data, 'columns'):
                print_info(f"Columns: {list(sample_data.columns)}")
                print_info(f"Shape: {sample_data.shape}")
                
                if len(sample_data) > 0:
                    print("\nSample data preview:")
                    print(sample_data.head().to_string())
                    
                    # Try optimization with real sample data
                    print_info("\nTesting optimization with sample data...")
                    
                    from app.cache_manager import CacheManager
                    from app.optimization_engine import OptimizationEngine
                    
                    cache_manager = CacheManager()
                    optimizer = OptimizationEngine(cache_manager)
                    
                    try:
                        result = await optimizer.optimize_lineup(
                            player_data=sample_data,
                            game_type='league',
                            use_ai=False
                        )
                        
                        if result:
                            print_success("Sample data optimization successful!")
                        else:
                            print_error("Sample data optimization returned None")
                            
                    except Exception as e:
                        print_error(f"Sample data optimization failed: {e}")
            else:
                print_error("Sample data is not a DataFrame")
        else:
            print_error("Sample data creation returned None")
        
    except Exception as e:
        print_error(f"Sample data test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Working DFS Lineup Generator')
    parser.add_argument('action', nargs='?', default='lineup', 
                       choices=['lineup', 'sample'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    async def main():
        if args.action == 'lineup':
            success = await generate_working_lineup()
            if success:
                print("\n🏈 Ready to optimize real lineups!")
                print("📁 Place your CSV files in data/input/ to process real data")
            else:
                print("\n🔧 Debug the issues above to get optimization working")
        elif args.action == 'sample':
            await test_sample_data()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
