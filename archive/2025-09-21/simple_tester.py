#!/usr/bin/env python3
"""
Simple DFS System Tester - Works with your existing code structure
"""

import sys
import asyncio
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

def print_success(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_warning(msg):
    print(f"⚠️  {msg}")

def print_info(msg):
    print(f"ℹ️  {msg}")

def print_header(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

async def test_config():
    """Test configuration loading."""
    print_header("Configuration Test")
    
    try:
        from app.config import settings, POSITION_MAPPINGS, TEAM_MAPPINGS, SALARY_CAP
        
        print_success("Configuration imports successful")
        print_info(f"Salary cap: ${settings.default_salary_cap:,}")
        print_info(f"Debug mode: {settings.debug}")
        print_info(f"OpenAI key configured: {settings.has_openai_key}")
        print_info(f"Anthropic key configured: {settings.has_anthropic_key}")
        
        # Validate config
        issues = settings.validate()
        if issues:
            print_warning("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print_success("Configuration validation passed")
        
        return True
        
    except Exception as e:
        print_error(f"Configuration test failed: {e}")
        return False

async def test_data_ingestion():
    """Test data ingestion component."""
    print_header("Data Ingestion Test")
    
    try:
        from app.data_ingestion import DataIngestion
        
        # Initialize data ingestion
        data_ingestion = DataIngestion()
        print_success("DataIngestion initialized successfully")
        
        # Test available methods
        methods = ['check_data_availability', 'load_weekly_data', 'refresh_data', 'create_sample_data']
        for method in methods:
            if hasattr(data_ingestion, method):
                print_success(f"Method available: {method}")
            else:
                print_warning(f"Method missing: {method}")
        
        # Test data availability check
        try:
            status = await data_ingestion.check_data_availability()
            print_info(f"Data availability status: {status}")
        except Exception as e:
            print_warning(f"Data availability check failed: {e}")
        
        # Test detailed status
        try:
            if hasattr(data_ingestion, 'get_detailed_status'):
                detailed_status = await data_ingestion.get_detailed_status()
                print_info(f"Detailed status available")
        except Exception as e:
            print_warning(f"Detailed status failed: {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Data ingestion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_cache_manager():
    """Test cache manager (required for other components)."""
    print_header("Cache Manager Test")
    
    try:
        # Try to import and initialize cache manager
        try:
            from app.cache_manager import CacheManager
            cache_manager = CacheManager()
            print_success("CacheManager imported and initialized")
            return cache_manager
        except ImportError:
            print_warning("CacheManager not found - creating mock")
            # Create a simple mock cache manager
            class MockCacheManager:
                def __init__(self):
                    self.cache = {}
                
                async def get(self, key):
                    return self.cache.get(key)
                
                async def set(self, key, value, ttl=None):
                    self.cache[key] = value
                
                async def clear(self):
                    self.cache.clear()
            
            return MockCacheManager()
            
    except Exception as e:
        print_error(f"Cache manager test failed: {e}")
        return None

async def test_optimization_engine():
    """Test optimization engine with cache manager."""
    print_header("Optimization Engine Test")
    
    try:
        # Get cache manager first
        cache_manager = await test_cache_manager()
        if not cache_manager:
            print_error("Cache manager required for optimization engine")
            return False
        
        from app.optimization_engine import OptimizationEngine
        
        # Initialize with cache manager
        optimizer = OptimizationEngine(cache_manager)
        print_success("OptimizationEngine initialized successfully")
        
        # Check available methods
        methods = [m for m in dir(optimizer) if not m.startswith('_') and callable(getattr(optimizer, m))]
        print_info(f"Available methods: {methods}")
        
        # Test basic functionality if optimize method exists
        if hasattr(optimizer, 'optimize_lineup'):
            print_info("Testing basic optimization...")
            
            # Create sample player data
            sample_players = [
                {'id': 1, 'name': 'Test QB', 'position': 'QB', 'salary': 8000, 'projected_points': 20.0},
                {'id': 2, 'name': 'Test RB1', 'position': 'RB', 'salary': 7000, 'projected_points': 15.0},
                {'id': 3, 'name': 'Test RB2', 'position': 'RB', 'salary': 6000, 'projected_points': 12.0},
                {'id': 4, 'name': 'Test WR1', 'position': 'WR', 'salary': 6500, 'projected_points': 14.0},
                {'id': 5, 'name': 'Test WR2', 'position': 'WR', 'salary': 5500, 'projected_points': 11.0},
                {'id': 6, 'name': 'Test WR3', 'position': 'WR', 'salary': 4500, 'projected_points': 9.0},
                {'id': 7, 'name': 'Test TE', 'position': 'TE', 'salary': 5000, 'projected_points': 10.0},
                {'id': 8, 'name': 'Test K', 'position': 'K', 'salary': 4000, 'projected_points': 7.0},
                {'id': 9, 'name': 'Test DST', 'position': 'DST', 'salary': 3000, 'projected_points': 8.0},
            ]
            
            constraints = {
                'salary_cap': 50000,
                'positions': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1}
            }
            
            try:
                start_time = time.time()
                result = await optimizer.optimize_lineup(sample_players, constraints)
                end_time = time.time()
                
                if result:
                    print_success(f"Optimization successful in {end_time - start_time:.3f}s")
                    print_info(f"Lineup generated with {len(result)} players")
                else:
                    print_warning("Optimization returned no result")
                    
            except Exception as e:
                print_warning(f"Optimization test failed: {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Optimization engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ai_analyzer():
    """Test AI analyzer with cache manager."""
    print_header("AI Analyzer Test")
    
    try:
        # Get cache manager first
        cache_manager = await test_cache_manager()
        if not cache_manager:
            print_error("Cache manager required for AI analyzer")
            return False
        
        from app.ai_analyzer import AIAnalyzer
        
        # Initialize with cache manager
        ai_analyzer = AIAnalyzer(cache_manager)
        print_success("AIAnalyzer initialized successfully")
        
        # Check available methods
        methods = [m for m in dir(ai_analyzer) if not m.startswith('_') and callable(getattr(ai_analyzer, m))]
        print_info(f"Available methods: {methods}")
        
        # Test basic functionality if analyze method exists
        sample_data = {
            'player': 'Test Player',
            'position': 'QB',
            'salary': 8000,
            'projected_points': 20.0
        }
        
        for method_name in ['analyze_player', 'analyze_correlations', 'analyze_lineup']:
            if hasattr(ai_analyzer, method_name):
                print_info(f"Method available: {method_name}")
                try:
                    # Test the method with sample data
                    if method_name == 'analyze_player':
                        result = await ai_analyzer.analyze_player(sample_data)
                    elif method_name == 'analyze_correlations':
                        result = await ai_analyzer.analyze_correlations([sample_data])
                    elif method_name == 'analyze_lineup':
                        result = await ai_analyzer.analyze_lineup([sample_data])
                    
                    if result:
                        print_success(f"{method_name} test successful")
                    else:
                        print_warning(f"{method_name} returned no result")
                        
                except Exception as e:
                    print_warning(f"{method_name} test failed: {e}")
            else:
                print_warning(f"Method missing: {method_name}")
        
        return True
        
    except Exception as e:
        print_error(f"AI analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_weather():
    """Test weather integration."""
    print_header("Weather Integration Test")
    
    try:
        from weather_integration import WeatherDataCollector, NFL_STADIUMS
        from datetime import datetime, timedelta
        
        print_success("Weather integration imports successful")
        print_info(f"NFL stadiums configured: {len(NFL_STADIUMS)}")
        
        # Test weather for a few teams
        test_teams = ['GB', 'BUF']  # Outdoor stadiums
        game_time = datetime.now() + timedelta(days=3)
        
        async with WeatherDataCollector() as collector:
            for team in test_teams:
                try:
                    weather = await collector.get_game_weather(team, game_time)
                    if weather:
                        print_success(f"Weather data for {team}: {weather.condition}")
                    else:
                        print_warning(f"No weather data for {team}")
                except Exception as e:
                    print_warning(f"Weather failed for {team}: {e}")
        
        return True
        
    except Exception as e:
        print_error(f"Weather integration test failed: {e}")
        return False

async def run_quick_test():
    """Run quick system test."""
    print("🏈 DFS System Quick Test")
    print("=" * 50)
    
    results = {}
    
    # Test configuration
    results['config'] = await test_config()
    
    # Test data ingestion
    results['data'] = await test_data_ingestion()
    
    # Test optimization (with cache manager)
    results['optimization'] = await test_optimization_engine()
    
    # Test AI analyzer (with cache manager)
    results['ai'] = await test_ai_analyzer()
    
    # Test weather (optional)
    results['weather'] = await test_weather()
    
    # Summary
    print_header("Test Summary")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL" 
        print(f"{test_name.upper()}: {status}")
    
    if passed == total:
        print_success(f"All tests passed! ({passed}/{total})")
    else:
        print_warning(f"Some tests failed: {passed}/{total} passed")
    
    return passed == total

async def generate_test_lineup():
    """Generate a test lineup to verify optimization works."""
    print_header("Test Lineup Generation")
    
    try:
        # Get cache manager
        cache_manager = await test_cache_manager()
        if not cache_manager:
            print_error("Cache manager required")
            return False
        
        from app.optimization_engine import OptimizationEngine
        
        optimizer = OptimizationEngine(cache_manager)
        
        # Sample player pool
        players = [
            {'id': 1, 'name': 'Josh Allen', 'position': 'QB', 'salary': 8500, 'projected_points': 22.5, 'team': 'BUF'},
            {'id': 2, 'name': 'Derrick Henry', 'position': 'RB', 'salary': 6800, 'projected_points': 18.2, 'team': 'TEN'},
            {'id': 3, 'name': 'Christian McCaffrey', 'position': 'RB', 'salary': 9000, 'projected_points': 20.1, 'team': 'SF'},
            {'id': 4, 'name': 'Davante Adams', 'position': 'WR', 'salary': 8000, 'projected_points': 17.5, 'team': 'LV'},
            {'id': 5, 'name': 'Stefon Diggs', 'position': 'WR', 'salary': 7200, 'projected_points': 16.8, 'team': 'BUF'},
            {'id': 6, 'name': 'Tyreek Hill', 'position': 'WR', 'salary': 7800, 'projected_points': 17.2, 'team': 'MIA'},
            {'id': 7, 'name': 'Travis Kelce', 'position': 'TE', 'salary': 6500, 'projected_points': 15.3, 'team': 'KC'},
            {'id': 8, 'name': 'Mark Andrews', 'position': 'TE', 'salary': 5800, 'projected_points': 13.8, 'team': 'BAL'},
            {'id': 9, 'name': 'Justin Tucker', 'position': 'K', 'salary': 4800, 'projected_points': 8.5, 'team': 'BAL'},
            {'id': 10, 'name': 'Harrison Butker', 'position': 'K', 'salary': 4600, 'projected_points': 8.2, 'team': 'KC'},
            {'id': 11, 'name': 'Buffalo DST', 'position': 'DST', 'salary': 3200, 'projected_points': 9.2, 'team': 'BUF'},
            {'id': 12, 'name': 'San Francisco DST', 'position': 'DST', 'salary': 3400, 'projected_points': 8.8, 'team': 'SF'},
        ]
        
        constraints = {
            'salary_cap': 50000,
            'positions': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1}
        }
        
        print_info("Generating optimal lineup...")
        start_time = time.time()
        
        lineup = await optimizer.optimize_lineup(players, constraints)
        
        end_time = time.time()
        
        if lineup:
            print_success(f"Lineup generated in {end_time - start_time:.3f}s")
            print("\n" + "=" * 60)
            print(" OPTIMAL LINEUP")
            print("=" * 60)
            
            total_salary = 0
            total_projection = 0
            
            for player in lineup:
                print(f"{player['position']:3} | {player['name']:20} | ${player['salary']:5,} | {player['projected_points']:5.1f} pts")
                total_salary += player['salary']
                total_projection += player['projected_points']
            
            print("-" * 60)
            print(f"{'TOTAL':3} | {'':20} | ${total_salary:5,} | {total_projection:5.1f} pts")
            print(f"Remaining salary: ${50000 - total_salary:,}")
            print("=" * 60)
            
            return True
        else:
            print_error("Failed to generate lineup")
            return False
        
    except Exception as e:
        print_error(f"Lineup generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple DFS System Tester')
    parser.add_argument('action', nargs='?', default='test', 
                       choices=['test', 'lineup', 'config', 'data', 'optimization', 'ai', 'weather'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    async def main():
        if args.action == 'test':
            await run_quick_test()
        elif args.action == 'lineup':
            await generate_test_lineup()
        elif args.action == 'config':
            await test_config()
        elif args.action == 'data':
            await test_data_ingestion()
        elif args.action == 'optimization':
            await test_optimization_engine()
        elif args.action == 'ai':
            await test_ai_analyzer()
        elif args.action == 'weather':
            await test_weather()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
