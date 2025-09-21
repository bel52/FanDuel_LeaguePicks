#!/usr/bin/env python3
"""
Quick test script to verify the DFS system works
"""
import sys
import asyncio
import os
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

async def test_optimizer():
    """Test the simple optimizer"""
    print_header("Testing Simple Optimizer")
    
    try:
        from app.simple_optimizer import SimpleDFSOptimizer
        
        optimizer = SimpleDFSOptimizer()
        print_success("SimpleDFSOptimizer imported and initialized")
        
        # Run the built-in test
        await optimizer.test_optimizer() if hasattr(optimizer, 'test_optimizer') else None
        
        return True
        
    except Exception as e:
        print_error(f"Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ai_analyzer():
    """Test the AI analyzer"""
    print_header("Testing AI Analyzer")
    
    try:
        from app.working_ai_analyzer import SimpleAIAnalyzer
        
        analyzer = SimpleAIAnalyzer()
        print_success("SimpleAIAnalyzer imported and initialized")
        print_info(f"API Available: {analyzer.api_available}")
        
        return True
        
    except Exception as e:
        print_error(f"AI analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_data_loading():
    """Test data loading"""
    print_header("Testing Data Loading")
    
    try:
        from app.data_ingestion import load_data_from_input_dir, create_sample_data
        
        # Test data loading
        df, warnings = load_data_from_input_dir()
        
        if df is not None and not df.empty:
            print_success(f"Data loaded successfully: {len(df)} players")
            
            # Show data structure
            print_info(f"Columns: {list(df.columns)}")
            print_info(f"Positions: {df['POS'].value_counts().to_dict()}")
            
            if warnings:
                print_info("Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            
            return True
        else:
            print_error("No data loaded")
            return False
        
    except Exception as e:
        print_error(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_system():
    """Test the complete system"""
    print_header("Testing Complete DFS System")
    
    try:
        from app.simple_optimizer import SimpleDFSOptimizer
        from app.working_ai_analyzer import SimpleAIAnalyzer
        from app.data_ingestion import load_data_from_input_dir
        
        # Load data
        print_info("Loading player data...")
        df, warnings = load_data_from_input_dir()
        
        if df is None or df.empty:
            print_error("No data available for testing")
            return False
        
        print_success(f"Loaded {len(df)} players")
        
        # Initialize components
        optimizer = SimpleDFSOptimizer()
        ai_analyzer = SimpleAIAnalyzer()
        
        print_success("Components initialized")
        
        # Run optimization
        print_info("Running optimization...")
        result = await optimizer.optimize_lineup(df, game_type="league", enforce_stack=True)
        
        if result.get('success'):
            print_success("Optimization successful!")
            print_info(f"Generated lineup with {result['total_projection']:.1f} projected points")
            print_info(f"Total salary: ${result['total_salary']:,}")
            
            # Test AI analysis
            if ai_analyzer.api_available:
                print_info("Running AI analysis...")
                analysis = await ai_analyzer.analyze_lineup(result['lineup'], "league")
                print_success("AI analysis completed")
                print(f"Analysis: {analysis[:100]}...")
            else:
                print_info("AI analysis skipped (no API key)")
            
            # Display lineup
            print_header("GENERATED LINEUP")
            for i, player in enumerate(result['lineup']):
                print(f"{i+1}. {player['position']} {player['player_name']} - ${player['salary']} - {player['projection']:.1f} pts")
            
            return True
        else:
            print_error(f"Optimization failed: {result.get('error')}")
            return False
        
    except Exception as e:
        print_error(f"Full system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test FastAPI endpoints"""
    print_header("Testing API Endpoints")
    
    try:
        import aiohttp
        
        base_url = "http://localhost:8010"
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            try:
                async with session.get(f"{base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        print_success("Health endpoint working")
                        print_info(f"Status: {data.get('status')}")
                    else:
                        print_error(f"Health endpoint failed: {response.status}")
            except Exception as e:
                print_error(f"Could not connect to API (is it running?): {e}")
                return False
            
            # Test optimize endpoint
            try:
                async with session.get(f"{base_url}/optimize?game_type=league") as response:
                    if response.status == 200:
                        data = await response.json()
                        print_success("Optimize endpoint working")
                        if data.get('success'):
                            print_info(f"Generated lineup with {data.get('total_projection', 0):.1f} points")
                        else:
                            print_error(f"Optimization failed: {data.get('error')}")
                    else:
                        print_error(f"Optimize endpoint failed: {response.status}")
            except Exception as e:
                print_error(f"Optimize endpoint error: {e}")
                return False
        
        return True
        
    except ImportError:
        print_info("aiohttp not available - skipping API tests")
        return True
    except Exception as e:
        print_error(f"API test failed: {e}")
        return False

def check_file_structure():
    """Check if required files exist"""
    print_header("Checking File Structure")
    
    required_files = [
        "app/main.py",
        "app/simple_optimizer.py", 
        "app/working_ai_analyzer.py",
        "app/data_ingestion.py",
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"{file_path} exists")
        else:
            print_error(f"{file_path} missing")
            all_good = False
    
    # Check data directory
    if os.path.exists("data/input"):
        print_success("data/input directory exists")
        
        # Check for CSV files
        csv_files = [f for f in os.listdir("data/input") if f.endswith('.csv')]
        if csv_files:
            print_info(f"Found CSV files: {csv_files}")
        else:
            print_info("No CSV files found - will use sample data")
    else:
        print_info("data/input directory missing - will create with sample data")
    
    return all_good

async def run_all_tests():
    """Run all tests"""
    print("🏈 DFS System Complete Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Check file structure first
    results['files'] = check_file_structure()
    
    # Test individual components
    results['data'] = await test_data_loading()
    results['optimizer'] = await test_optimizer()
    results['ai'] = await test_ai_analyzer()
    
    # Test complete system
    results['system'] = await test_full_system()
    
    # Test API if running
    results['api'] = await test_api_endpoints()
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.upper()}: {status}")
    
    print()
    if passed == total:
        print_success(f"All tests passed! ({passed}/{total})")
        print()
        print("🎉 Your DFS optimization system is working!")
        print()
        print("Next steps:")
        print("1. Add your OpenAI API key to .env file for AI analysis")
        print("2. Place FantasyPros CSV files in data/input/ for real data")
        print("3. Start the server: docker compose up -d")
        print("4. Test: curl http://localhost:8010/optimize_text")
    else:
        print_error(f"Some tests failed: {passed}/{total} passed")
        print()
        print("Please fix the failed tests above.")
    
    return passed == total

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DFS System Test Suite')
    parser.add_argument('test', nargs='?', default='all',
                       choices=['all', 'data', 'optimizer', 'ai', 'system', 'api', 'files'],
                       help='Which test to run')
    
    args = parser.parse_args()
    
    async def main():
        if args.test == 'all':
            await run_all_tests()
        elif args.test == 'data':
            await test_data_loading()
        elif args.test == 'optimizer':
            await test_optimizer()
        elif args.test == 'ai':
            await test_ai_analyzer()
        elif args.test == 'system':
            await test_full_system()
        elif args.test == 'api':
            await test_api_endpoints()
        elif args.test == 'files':
            check_file_structure()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
