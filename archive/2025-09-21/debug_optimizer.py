#!/usr/bin/env python3
"""
Debug script to understand what format the optimization engine expects
"""

import sys
import asyncio
import pandas as pd
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

async def debug_optimization():
    """Debug the optimization engine to understand expected data format."""
    
    try:
        from app.cache_manager import CacheManager
        from app.optimization_engine import OptimizationEngine
        
        # Initialize
        cache_manager = CacheManager()
        optimizer = OptimizationEngine(cache_manager)
        
        print("✅ Optimization engine initialized")
        
        # Try different data formats to see what works
        
        # Format 1: List of dictionaries (what we tried)
        print("\n🔍 Testing Format 1: List of dictionaries")
        players_list = [
            {'id': 1, 'name': 'Josh Allen', 'position': 'QB', 'salary': 8500, 'projected_points': 22.5, 'team': 'BUF'},
            {'id': 2, 'name': 'Derrick Henry', 'position': 'RB', 'salary': 6800, 'projected_points': 18.2, 'team': 'TEN'},
        ]
        
        constraints_list = {
            'salary_cap': 50000,
            'positions': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1}
        }
        
        try:
            result = await optimizer.optimize_lineup(players_list, constraints_list)
            print(f"✅ List format works! Result: {result}")
        except Exception as e:
            print(f"❌ List format failed: {e}")
        
        # Format 2: Pandas DataFrame (likely what it expects)
        print("\n🔍 Testing Format 2: Pandas DataFrame")
        players_df = pd.DataFrame([
            {'id': 1, 'name': 'Josh Allen', 'position': 'QB', 'salary': 8500, 'projected_points': 22.5, 'team': 'BUF'},
            {'id': 2, 'name': 'Derrick Henry', 'position': 'RB', 'salary': 6800, 'projected_points': 18.2, 'team': 'TEN'},
            {'id': 3, 'name': 'Christian McCaffrey', 'position': 'RB', 'salary': 9000, 'projected_points': 20.1, 'team': 'SF'},
            {'id': 4, 'name': 'Davante Adams', 'position': 'WR', 'salary': 8000, 'projected_points': 17.5, 'team': 'LV'},
            {'id': 5, 'name': 'Stefon Diggs', 'position': 'WR', 'salary': 7200, 'projected_points': 16.8, 'team': 'BUF'},
            {'id': 6, 'name': 'Tyreek Hill', 'position': 'WR', 'salary': 7800, 'projected_points': 17.2, 'team': 'MIA'},
            {'id': 7, 'name': 'Travis Kelce', 'position': 'TE', 'salary': 6500, 'projected_points': 15.3, 'team': 'KC'},
            {'id': 8, 'name': 'Justin Tucker', 'position': 'K', 'salary': 4800, 'projected_points': 8.5, 'team': 'BAL'},
            {'id': 9, 'name': 'Buffalo DST', 'position': 'DST', 'salary': 3200, 'projected_points': 9.2, 'team': 'BUF'},
        ])
        
        print(f"DataFrame shape: {players_df.shape}")
        print(f"DataFrame columns: {list(players_df.columns)}")
        
        try:
            result = await optimizer.optimize_lineup(players_df, constraints_list)
            print(f"✅ DataFrame format works! Result type: {type(result)}")
            if result is not None:
                print(f"Result length: {len(result)}")
                print(f"First result item: {result[0] if len(result) > 0 else 'Empty'}")
        except Exception as e:
            print(f"❌ DataFrame format failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Format 3: Check what the data ingestion class produces
        print("\n🔍 Testing Format 3: Check DataIngestion output format")
        try:
            from app.data_ingestion import DataIngestion
            
            data_ingestion = DataIngestion()
            
            # Check if we have sample data
            try:
                sample_data = data_ingestion.create_sample_data()
                if sample_data is not None:
                    print(f"✅ Sample data created, type: {type(sample_data)}")
                    
                    if hasattr(sample_data, 'columns'):
                        print(f"Sample data columns: {list(sample_data.columns)}")
                        print(f"Sample data shape: {sample_data.shape}")
                        print(f"First few rows:\n{sample_data.head()}")
                    
                    # Try optimization with real data format
                    try:
                        result = await optimizer.optimize_lineup(sample_data, constraints_list)
                        print(f"✅ Real data format works! Result: {result}")
                    except Exception as e:
                        print(f"❌ Real data format failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("⚠️  Sample data creation returned None")
            except Exception as e:
                print(f"❌ Sample data creation failed: {e}")
                
        except Exception as e:
            print(f"❌ DataIngestion test failed: {e}")
        
        # Format 4: Check the optimization engine source to see expected format
        print("\n🔍 Inspecting optimize_lineup method signature")
        import inspect
        sig = inspect.signature(optimizer.optimize_lineup)
        print(f"Method signature: {sig}")
        
        # Check the docstring if available
        if optimizer.optimize_lineup.__doc__:
            print(f"Docstring: {optimizer.optimize_lineup.__doc__}")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_optimization())
