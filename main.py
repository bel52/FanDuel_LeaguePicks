#!/usr/bin/env python3
"""
Enhanced main entry point for the NFL DFS Optimization System
"""
import asyncio
import sys
import signal
from pathlib import Path
import argparse
from loguru import logger

# Add the project directory to the Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

try:
    from config import LOGS_DIR, LOGGING_CONFIG, API_PORT
    from scheduler import start_background_scheduler, stop_background_scheduler, get_scheduler
    from data_collector import get_fresh_data
    from optimizer import optimize_dfs_lineups  # This should exist
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔧 Installing missing dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "loguru>=0.7.0"], check=True)
    
    # Try importing again
    try:
        from loguru import logger
        from config import LOGS_DIR, LOGGING_CONFIG, API_PORT
        from scheduler import start_background_scheduler, stop_background_scheduler, get_scheduler
        from data_collector import get_fresh_data
        from optimizer import optimize_dfs_lineups
        print("✅ Dependencies installed successfully")
    except ImportError as e2:
        print(f"❌ Still missing dependencies: {e2}")
        print("🔧 Run: pip install -r requirements.txt")
        sys.exit(1)

def setup_logging():
    """Configure logging for the application"""
    logger.remove()
    
    # Add file logging
    logger.add(
        LOGS_DIR / "dfs_optimizer_{time:YYYY-MM-DD}.log",
        rotation=LOGGING_CONFIG['rotation'],
        retention=LOGGING_CONFIG['retention'],
        format=LOGGING_CONFIG['format'],
        level=LOGGING_CONFIG['level']
    )
    
    # Add console logging
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
        level=LOGGING_CONFIG['level']
    )

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    stop_background_scheduler()
    sys.exit(0)

async def run_data_collection_only():
    """Run data collection once and exit"""
    logger.info("Running data collection...")
    try:
        data = await get_fresh_data()
        if data and 'players' in data:
            logger.info(f"✅ Successfully collected data for {len(data['players'])} players")
            logger.info(f"📊 Data quality: {data.get('data_quality', {})}")
            return True
        else:
            logger.error("❌ Data collection failed or returned empty data")
            return False
    except Exception as e:
        logger.error(f"❌ Error in data collection: {e}")
        return False

async def run_optimization_only():
    """Run optimization once with current data and exit"""
    logger.info("Running lineup optimization...")
    try:
        # Get fresh data first
        data = await get_fresh_data()
        if not data or not data.get('players'):
            logger.error("❌ No player data available for optimization")
            return False
        
        # Test different contest types
        contest_types = [
            ('gpp', 10, 'Tournament'),
            ('cash', 5, 'Cash Game'),
            ('contrarian', 8, 'Contrarian')
        ]
        
        all_lineups = {}
        
        for contest_type, num_lineups, display_name in contest_types:
            logger.info(f"Generating {num_lineups} {display_name} lineups...")
            
            lineups = optimize_dfs_lineups(
                player_data=data['players'],
                weather_data=data.get('weather', {}),
                num_lineups=num_lineups,
                contest_type=contest_type
            )
            
            if lineups:
                all_lineups[contest_type] = lineups
                logger.info(f"✅ Generated {len(lineups)} {display_name} lineups")
                
                # Show sample lineup
                sample = lineups[0]
                logger.info(f"Sample {display_name} lineup:")
                logger.info(f"  Salary: ${sample.total_salary:,} | Projected: {sample.projected_points:.1f}")
                logger.info(f"  Ownership: {sample.ownership_total:.1f}% | Correlation: {sample.correlation_score:.3f}")
                for player in sample.players[:3]:  # Show first 3 players
                    logger.info(f"    {player.position}: {player.name} (${player.salary:,})")
            else:
                logger.warning(f"❌ Failed to generate {display_name} lineups")
        
        return len(all_lineups) > 0
            
    except Exception as e:
        logger.error(f"❌ Error in optimization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_web_api():
    """Run the web API server"""
    logger.info(f"🌐 Starting Enhanced DFS Optimizer Web API on port {API_PORT}...")
    
    try:
        import uvicorn
        
        # Import the API
        try:
            from app import app
        except Exception as e:
            logger.error(f"❌ Error importing API: {e}")
            return False
        
        # Start the background scheduler
        start_background_scheduler()
        
        # Run the API server
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=API_PORT,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Web API failed to start: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        stop_background_scheduler()
    
    return True

def test_system():
    """Run system tests"""
    logger.info("🧪 Running system tests...")
    
    try:
        # Test imports
        import pandas as pd
        from data_collector import EnhancedDataCollector
        from optimizer import EnhancedDFSOptimizer
        from config import get_current_nfl_week, is_game_day
        logger.info("✅ All imports successful")
        
        # Test current week detection
        current_week = get_current_nfl_week()
        game_day = is_game_day()
        logger.info(f"✅ Current NFL week: {current_week}, Game day: {game_day}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False

def display_welcome():
    """Display welcome message and system info"""
    welcome_message = """
🏈 NFL DFS OPTIMIZER v2.1 - ENHANCED
====================================================

🎯 Enhanced Features:
- Proper contest type differentiation (GPP vs Cash vs Contrarian)
- Current week game detection and filtering
- Advanced correlation modeling with weather integration
- Real-time ownership projection with contest-specific adjustments
- Single game format support (MVP + 5 FLEX)
- Enhanced weather impact for outdoor stadiums only

📊 Data Sources:
- NFL-data-py (comprehensive player stats with week filtering)
- ESPN API (real-time scores and current week games)
- Weather.gov (stadium weather conditions for outdoor venues)
- Enhanced injury report monitoring

⚡ Optimization Engine:
- Contest-specific strategies that actually differ
- Proper single game team filtering
- Enhanced correlation matrices with game context
- Monte Carlo simulation for ceiling/floor projections
- Advanced diversification algorithms

🎮 Contest Types:
- Tournament/GPP: High-ceiling, correlation stacking, ownership leverage
- Cash Game: High-floor, consistent plays, minimal stacking
- Contrarian: Low-ownership fades, unconventional stacks
- Single Game: MVP selection + game-specific correlation plays

🔧 Technical Improvements:
- Fixed syntax errors and import issues
- Enhanced error handling and logging
- Proper async/await patterns
- Multi-level caching with intelligent invalidation
- Real-time current week detection
"""
    print(welcome_message)

def main():
    """Enhanced main entry point with better error handling"""
    parser = argparse.ArgumentParser(description="Enhanced NFL DFS Optimization System")
    parser.add_argument(
        'mode',
        choices=['scheduler', 'web', 'collect', 'optimize', 'test'],
        help='Operation mode: scheduler (automated), web (API server), collect (data only), optimize (lineups only), test (system test)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        LOGGING_CONFIG['level'] = 'DEBUG'
    setup_logging()
    
    # Display welcome message
    display_welcome()
    
    # Ensure required directories exist
    for directory in [LOGS_DIR, Path('data'), Path('cache')]:
        directory.mkdir(exist_ok=True)
    
    logger.info(f"🔧 Running in {args.mode} mode")
    
    success = False
    
    try:
        if args.mode == 'scheduler':
            success = run_scheduler_mode()
        elif args.mode == 'web':
            success = run_web_api()
        elif args.mode == 'collect':
            success = asyncio.run(run_data_collection_only())
        elif args.mode == 'optimize':
            success = asyncio.run(run_optimization_only())
        elif args.mode == 'test':
            success = test_system()
            
    except KeyboardInterrupt:
        logger.info("👋 Interrupted by user")
        success = True
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        success = False
    
    if success:
        logger.info("✅ Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Operation failed")
        sys.exit(1)

def run_scheduler_mode():
    """Run the automated scheduler"""
    logger.info("🚀 Starting DFS Optimizer in scheduler mode...")
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the scheduler
        scheduler = start_background_scheduler()
        
        logger.info("📅 Automated scheduler started successfully!")
        logger.info("🌐 Web interface available at: http://localhost:8020")
        logger.info("⏹️  Press Ctrl+C to stop")
        
        # Keep the main thread alive
        while True:
            try:
                import time
                time.sleep(1)
                    
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        logger.error(f"❌ Error in scheduler mode: {e}")
        return False
    finally:
        stop_background_scheduler()
        
    return True

if __name__ == "__main__":
    main()
