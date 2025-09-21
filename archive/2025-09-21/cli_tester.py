#!/usr/bin/env python3
"""
DFS CLI Testing Interface
Comprehensive testing suite for FanDuel NFL DFS optimization system.
"""

import asyncio
import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import traceback

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'app'))

try:
    from app.config import Config
    from app.data_ingestion import DataIngestionPipeline
    from app.ai_analyzer import AIAnalyzer
    from app.optimization_engine import OptimizationEngine
    from weather_integration import WeatherDataCollector, NFL_STADIUMS
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root and all modules exist.")
    sys.exit(1)

class Colors:
    """ANSI color codes for pretty CLI output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colored_print(message: str, color: str = Colors.ENDC):
    """Print colored message."""
    print(f"{color}{message}{Colors.ENDC}")

def print_header(title: str):
    """Print section header."""
    colored_print(f"\n{'='*60}", Colors.HEADER)
    colored_print(f" {title.upper()}", Colors.HEADER + Colors.BOLD)
    colored_print(f"{'='*60}", Colors.HEADER)

def print_success(message: str):
    """Print success message."""
    colored_print(f"✅ {message}", Colors.OKGREEN)

def print_error(message: str):
    """Print error message."""
    colored_print(f"❌ {message}", Colors.FAIL)

def print_warning(message: str):
    """Print warning message."""
    colored_print(f"⚠️  {message}", Colors.WARNING)

def print_info(message: str):
    """Print info message."""
    colored_print(f"ℹ️  {message}", Colors.OKBLUE)

class DFSSystemTester:
    """Comprehensive testing suite for DFS optimization system."""
    
    def __init__(self):
        self.config = Config()
        self.data_pipeline = None
        self.ai_analyzer = None
        self.optimizer = None
        self.weather_collector = None
        
    async def initialize_components(self):
        """Initialize all system components."""
        print_info("Initializing DFS system components...")
        
        try:
            # Initialize data pipeline
            self.data_pipeline = DataIngestionPipeline()
            print_success("Data ingestion pipeline initialized")
            
            # Initialize AI analyzer
            self.ai_analyzer = AIAnalyzer()
            print_success("AI analyzer initialized")
            
            # Initialize optimization engine
            self.optimizer = OptimizationEngine()
            print_success("Optimization engine initialized")
            
            # Initialize weather collector
            self.weather_collector = WeatherDataCollector()
            print_success("Weather data collector initialized")
            
            return True
            
        except Exception as e:
            print_error(f"Component initialization failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_configuration(self):
        """Test system configuration."""
        print_header("Configuration Test")
        
        try:
            # Test API keys
            api_keys = {
                'OpenAI': self.config.OPENAI_API_KEY is not None,
                'Anthropic': getattr(self.config, 'ANTHROPIC_API_KEY', None) is not None
            }
            
            for service, has_key in api_keys.items():
                if has_key:
                    print_success(f"{service} API key configured")
                else:
                    print_warning(f"{service} API key missing")
            
            # Test database configuration
            if hasattr(self.config, 'DATABASE_URL'):
                print_success("Database configuration found")
            else:
                print_warning("Database configuration missing")
            
            # Test Redis configuration  
            if hasattr(self.config, 'REDIS_URL'):
                print_success("Redis configuration found")
            else:
                print_warning("Redis configuration missing (caching disabled)")
            
            return True
            
        except Exception as e:
            print_error(f"Configuration test failed: {e}")
            return False
    
    async def test_data_ingestion(self):
        """Test data ingestion pipeline."""
        print_header("Data Ingestion Test")
        
        try:
            if not self.data_pipeline:
                await self.initialize_components()
            
            print_info("Testing ESPN API connection...")
            
            # Test ESPN scoreboard
            espn_data = await self.data_pipeline.fetch_espn_scores()
            if espn_data:
                print_success(f"ESPN API working - got {len(espn_data.get('events', []))} games")
            else:
                print_error("ESPN API failed")
                return False
            
            print_info("Testing Sleeper API connection...")
            
            # Test Sleeper trending players
            sleeper_data = await self.data_pipeline.fetch_sleeper_trends()
            if sleeper_data:
                print_success(f"Sleeper API working - got {len(sleeper_data)} trending players")
            else:
                print_warning("Sleeper API failed (non-critical)")
            
            # Test player data aggregation
            print_info("Testing player data aggregation...")
            all_players = await self.data_pipeline.get_all_players()
            
            if all_players and len(all_players) > 0:
                print_success(f"Player data aggregated - {len(all_players)} players found")
                
                # Show sample player data
                sample_player = next(iter(all_players.values()))
                print_info(f"Sample player: {sample_player.get('name', 'Unknown')} - {sample_player.get('position', 'N/A')}")
                
            else:
                print_error("Player data aggregation failed")
                return False
            
            return True
            
        except Exception as e:
            print_error(f"Data ingestion test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_weather_integration(self):
        """Test weather data collection."""
        print_header("Weather Integration Test")
        
        try:
            if not self.weather_collector:
                await self.initialize_components()
            
            # Test weather for a few outdoor stadiums
            test_teams = ['GB', 'CHI', 'BUF']  # Cold weather outdoor stadiums
            game_time = datetime.now() + timedelta(days=3)  # Next Sunday
            
            async with self.weather_collector:
                for team in test_teams:
                    print_info(f"Testing weather for {NFL_STADIUMS[team].team}...")
                    
                    weather = await self.weather_collector.get_game_weather(team, game_time)
                    
                    if weather:
                        print_success(f"Weather data obtained for {team}")
                        print(f"   Temperature: {weather.temperature}°F")
                        print(f"   Wind Speed: {weather.wind_speed} mph")
                        print(f"   Condition: {weather.condition}")
                    else:
                        print_warning(f"Weather data failed for {team}")
            
            # Test dome handling
            dome_team = 'DET'  # Ford Field is a dome
            async with WeatherDataCollector() as collector:
                dome_weather = await collector.get_game_weather(dome_team, game_time)
                if dome_weather and dome_weather.condition == "Indoor":
                    print_success("Dome stadium handling works correctly")
                else:
                    print_error("Dome stadium handling failed")
            
            return True
            
        except Exception as e:
            print_error(f"Weather integration test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_ai_analysis(self):
        """Test AI-powered analysis components."""
        print_header("AI Analysis Test")
        
        try:
            if not self.ai_analyzer:
                await self.initialize_components()
            
            # Create sample player data for testing
            sample_players = [
                {
                    'name': 'Josh Allen',
                    'position': 'QB',
                    'team': 'BUF',
                    'salary': 8500,
                    'projected_points': 22.5,
                    'ownership_projection': 15.2
                },
                {
                    'name': 'Stefon Diggs',
                    'position': 'WR', 
                    'team': 'BUF',
                    'salary': 7200,
                    'projected_points': 16.8,
                    'ownership_projection': 18.5
                },
                {
                    'name': 'Derrick Henry',
                    'position': 'RB',
                    'team': 'TEN',
                    'salary': 6800,
                    'projected_points': 18.2,
                    'ownership_projection': 22.1
                }
            ]
            
            print_info("Testing player analysis...")
            
            # Test individual player analysis
            analysis = await self.ai_analyzer.analyze_player(sample_players[0])
            if analysis:
                print_success("Player analysis completed")
                print(f"   Analysis preview: {analysis[:100]}...")
            else:
                print_error("Player analysis failed")
                return False
            
            print_info("Testing correlation analysis...")
            
            # Test correlation analysis
            correlations = await self.ai_analyzer.analyze_correlations(sample_players)
            if correlations:
                print_success("Correlation analysis completed")
                print(f"   Found {len(correlations)} correlation insights")
            else:
                print_warning("Correlation analysis returned no results")
            
            print_info("Testing ownership analysis...")
            
            # Test ownership analysis
            ownership = await self.ai_analyzer.analyze_ownership(sample_players)
            if ownership:
                print_success("Ownership analysis completed")
            else:
                print_warning("Ownership analysis failed")
            
            return True
            
        except Exception as e:
            print_error(f"AI analysis test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_optimization_engine(self):
        """Test lineup optimization."""
        print_header("Optimization Engine Test")
        
        try:
            if not self.optimizer:
                await self.initialize_components()
            
            # Create sample player pool for optimization
            player_pool = [
                {'id': 1, 'name': 'Josh Allen', 'position': 'QB', 'team': 'BUF', 'salary': 8500, 'projected_points': 22.5},
                {'id': 2, 'name': 'Lamar Jackson', 'position': 'QB', 'team': 'BAL', 'salary': 8200, 'projected_points': 21.8},
                {'id': 3, 'name': 'Derrick Henry', 'position': 'RB', 'team': 'TEN', 'salary': 6800, 'projected_points': 18.2},
                {'id': 4, 'name': 'Christian McCaffrey', 'position': 'RB', 'team': 'SF', 'salary': 9000, 'projected_points': 20.1},
                {'id': 5, 'name': 'Davante Adams', 'position': 'WR', 'team': 'LV', 'salary': 8000, 'projected_points': 17.5},
                {'id': 6, 'name': 'Stefon Diggs', 'position': 'WR', 'team': 'BUF', 'salary': 7200, 'projected_points': 16.8},
                {'id': 7, 'name': 'Tyreek Hill', 'position': 'WR', 'team': 'MIA', 'salary': 7800, 'projected_points': 17.2},
                {'id': 8, 'name': 'Travis Kelce', 'position': 'TE', 'team': 'KC', 'salary': 6500, 'projected_points': 15.3},
                {'id': 9, 'name': 'Mark Andrews', 'position': 'TE', 'team': 'BAL', 'salary': 5800, 'projected_points': 13.8},
                {'id': 10, 'name': 'Justin Tucker', 'position': 'K', 'team': 'BAL', 'salary': 4800, 'projected_points': 8.5},
                {'id': 11, 'name': 'Buffalo', 'position': 'DST', 'team': 'BUF', 'salary': 3200, 'projected_points': 9.2},
            ]
            
            print_info("Testing basic lineup optimization...")
            
            # Test basic optimization
            constraints = {
                'salary_cap': 50000,
                'positions': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1}
            }
            
            start_time = time.time()
            lineup = await self.optimizer.optimize_lineup(player_pool, constraints)
            optimization_time = time.time() - start_time
            
            if lineup:
                print_success(f"Basic optimization completed in {optimization_time:.3f}s")
                total_salary = sum(player['salary'] for player in lineup)
                total_projection = sum(player['projected_points'] for player in lineup)
                print(f"   Total salary: ${total_salary:,}")
                print(f"   Total projection: {total_projection:.1f} points")
                
                # Validate lineup
                if total_salary <= constraints['salary_cap']:
                    print_success("Salary constraint satisfied")
                else:
                    print_error(f"Salary constraint violated: ${total_salary} > ${constraints['salary_cap']}")
                    return False
                    
            else:
                print_error("Basic optimization failed")
                return False
            
            print_info("Testing multi-lineup generation...")
            
            # Test multiple lineup generation
            num_lineups = 5
            start_time = time.time()
            lineups = await self.optimizer.generate_multiple_lineups(player_pool, constraints, num_lineups)
            multi_time = time.time() - start_time
            
            if lineups and len(lineups) == num_lineups:
                print_success(f"Generated {len(lineups)} lineups in {multi_time:.3f}s")
                
                # Check for diversity
                unique_lineups = set()
                for lineup in lineups:
                    lineup_key = tuple(sorted(player['id'] for player in lineup))
                    unique_lineups.add(lineup_key)
                
                if len(unique_lineups) == len(lineups):
                    print_success("All lineups are unique")
                else:
                    print_warning(f"Only {len(unique_lineups)}/{len(lineups)} lineups are unique")
                    
            else:
                print_error("Multi-lineup generation failed")
                return False
            
            return True
            
        except Exception as e:
            print_error(f"Optimization engine test failed: {e}")
            traceback.print_exc()
            return False
    
    async def test_full_pipeline(self):
        """Test complete end-to-end pipeline."""
        print_header("Full Pipeline Test")
        
        try:
            print_info("Running complete DFS optimization pipeline...")
            
            # Step 1: Data collection
            print_info("Step 1: Collecting player data...")
            if not await self.test_data_ingestion():
                print_error("Data ingestion failed - aborting pipeline test")
                return False
            
            # Step 2: Weather analysis
            print_info("Step 2: Analyzing weather conditions...")
            await self.test_weather_integration()  # Non-critical for pipeline
            
            # Step 3: AI analysis
            print_info("Step 3: Running AI analysis...")
            if not await self.test_ai_analysis():
                print_warning("AI analysis had issues - continuing with basic optimization")
            
            # Step 4: Optimization
            print_info("Step 4: Optimizing lineups...")
            if not await self.test_optimization_engine():
                print_error("Optimization failed - pipeline incomplete")
                return False
            
            print_success("Complete pipeline test successful!")
            return True
            
        except Exception as e:
            print_error(f"Full pipeline test failed: {e}")
            traceback.print_exc()
            return False
    
    async def run_performance_test(self):
        """Run performance benchmarks."""
        print_header("Performance Test")
        
        try:
            print_info("Running performance benchmarks...")
            
            # Test component initialization speed
            start_time = time.time()
            await self.initialize_components()
            init_time = time.time() - start_time
            print_info(f"Component initialization: {init_time:.3f}s")
            
            # Test optimization speed with varying player pool sizes
            for pool_size in [50, 100, 200]:
                # Generate sample players
                sample_pool = []
                for i in range(pool_size):
                    sample_pool.append({
                        'id': i,
                        'name': f'Player {i}',
                        'position': ['QB', 'RB', 'WR', 'TE', 'K', 'DST'][i % 6],
                        'team': f'T{i % 32}',
                        'salary': 3000 + (i * 100) % 8000,
                        'projected_points': 5.0 + (i * 0.2) % 20.0
                    })
                
                constraints = {
                    'salary_cap': 50000,
                    'positions': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1}
                }
                
                start_time = time.time()
                if self.optimizer and hasattr(self.optimizer, 'optimize_lineup'):
                    try:
                        await self.optimizer.optimize_lineup(sample_pool, constraints)
                        opt_time = time.time() - start_time
                        print_info(f"Optimization ({pool_size} players): {opt_time:.3f}s")
                    except Exception as e:
                        print_warning(f"Optimization test failed for {pool_size} players: {e}")
                else:
                    print_warning("Optimization method not available for performance test")
            
            return True
            
        except Exception as e:
            print_error(f"Performance test failed: {e}")
            traceback.print_exc()
            return False

async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='DFS System Testing CLI')
    parser.add_argument('test_type', nargs='?', default='all', 
                       choices=['config', 'data', 'weather', 'ai', 'optimization', 'pipeline', 'performance', 'all'],
                       help='Type of test to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    colored_print("""
██████╗ ███████╗███████╗    ████████╗███████╗███████╗████████╗███████╗██████╗ 
██╔══██╗██╔════╝██╔════╝    ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔════╝██╔══██╗
██║  ██║█████╗  ███████╗       ██║   █████╗  ███████╗   ██║   █████╗  ██████╔╝
██║  ██║██╔══╝  ╚════██║       ██║   ██╔══╝  ╚════██║   ██║   ██╔══╝  ██╔══██╗
██████╔╝██║     ███████║       ██║   ███████╗███████║   ██║   ███████╗██║  ██║
╚═════╝ ╚═╝     ╚══════╝       ╚═╝   ╚══════╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝
    """, Colors.HEADER + Colors.BOLD)
    
    colored_print("FanDuel NFL DFS Optimization System - Testing Suite", Colors.HEADER + Colors.BOLD)
    
    tester = DFSSystemTester()
    
    # Initialize components
    if not await tester.initialize_components():
        print_error("Failed to initialize system components")
        sys.exit(1)
    
    # Run tests based on arguments
    test_results = {}
    
    if args.test_type in ['config', 'all']:
        test_results['config'] = await tester.test_configuration()
    
    if args.test_type in ['data', 'all']:
        test_results['data'] = await tester.test_data_ingestion()
    
    if args.test_type in ['weather', 'all']:
        test_results['weather'] = await tester.test_weather_integration()
    
    if args.test_type in ['ai', 'all']:
        test_results['ai'] = await tester.test_ai_analysis()
    
    if args.test_type in ['optimization', 'all']:
        test_results['optimization'] = await tester.test_optimization_engine()
    
    if args.test_type in ['pipeline', 'all']:
        test_results['pipeline'] = await tester.test_full_pipeline()
    
    if args.test_type in ['performance', 'all']:
        test_results['performance'] = await tester.run_performance_test()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.OKGREEN if result else Colors.FAIL
        colored_print(f"{test_name.upper()}: {status}", color)
    
    print()
    if passed == total:
        print_success(f"All tests passed! ({passed}/{total})")
    else:
        print_warning(f"Some tests failed: {passed}/{total} passed")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_error("\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
