#!/usr/bin/env python3
"""
Main entry point for the NFL DFS Optimization System
Run this script to start the automated DFS optimizer
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

from config import LOGS_DIR, LOGGING_CONFIG, API_PORT
from scheduler import start_background_scheduler, stop_background_scheduler, get_scheduler
from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups

def setup_logging():
    """Configure logging for the application"""
    # Remove default handler
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
        
        # Run optimization
        lineups = optimize_dfs_lineups(
            player_data=data['players'],
            weather_data=data.get('weather', {}),
            num_lineups=10,
            contest_type='gpp'
        )
        
        if lineups:
            logger.info(f"✅ Successfully generated {len(lineups)} optimized lineups")
            
            # Export to CSV
            from optimizer import DFSOptimizer
            optimizer = DFSOptimizer()
            csv_file = optimizer.export_lineups_to_csv(lineups)
            logger.info(f"📄 Lineups exported to {csv_file}")
            
            # Display top 3 lineups
            logger.info("\n🏆 Top 3 Lineups:")
            for i, lineup in enumerate(lineups[:3], 1):
                logger.info(f"\nLineup {i} (${lineup.total_salary:,} | {lineup.projected_points:.1f} pts):")
                for player in lineup.players:
                    logger.info(f"  {player.position}: {player.name} ({player.team}) - ${player.salary:,}")
            
            return True
        else:
            logger.error("❌ Optimization failed to generate lineups")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in optimization: {e}")
        return False

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
        logger.info("🌐 Web interface available at: http://localhost:8000")
        logger.info("📊 API documentation at: http://localhost:8000/docs")
        logger.info("⏹️  Press Ctrl+C to stop")
        
        # Keep the main thread alive
        while True:
            try:
                import time
                time.sleep(1)
                
                # Periodically log status
                if hasattr(scheduler, 'last_update') and scheduler.last_update:
                    pass  # Status updates handled by web interface
                    
            except KeyboardInterrupt:
                break
                
    except Exception as e:
        logger.error(f"❌ Error in scheduler mode: {e}")
        return False
    finally:
        stop_background_scheduler()
        
    return True

def run_web_api():
    """Run the web API server"""
    logger.info(f"🌐 Starting DFS Optimizer Web API on port {API_PORT}...")
    
    try:
        import uvicorn
        from api import app
        
        # Start the background scheduler
        start_background_scheduler()
        
        # Run the API server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8020,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting web API: {e}")
        return False
    finally:
        stop_background_scheduler()
    
    return True

def display_welcome():
    """Display welcome message and system info"""
    welcome_message = """
🏈 NFL DFS OPTIMIZER v2.0
==========================

🎯 Features:
• Automated data collection from multiple free sources
• Advanced optimization with correlation modeling
• Weather impact analysis
• Injury report monitoring
• Real-time lineup generation
• Web-based dashboard

📊 Data Sources:
• NFL-data-py (comprehensive player stats)
• ESPN API (real-time scores and news)
• Weather.gov (stadium weather conditions)
• Sleeper API (trending players)

⚡ Optimization Engine:
• Integer Linear Programming (ILP)
• Correlation-aware stacking strategies
• Ownership projection and contrarian plays
• Weather adjustments
• Multi-objective optimization

🎮 Contest Types Supported:
• Tournament/GPP lineups
• Cash game lineups
• Contrarian strategies
• High-stakes optimization
"""
    print(welcome_message)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NFL DFS Optimization System")
    parser.add_argument(
        'mode',
        choices=['scheduler', 'web', 'collect', 'optimize'],
        help='Operation mode: scheduler (automated), web (API server), collect (data only), optimize (lineups only)'
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
    
    if args.mode == 'scheduler':
        success = run_scheduler_mode()
    elif args.mode == 'web':
        success = run_web_api()
    elif args.mode == 'collect':
        success = asyncio.run(run_data_collection_only())
    elif args.mode == 'optimize':
        success = asyncio.run(run_optimization_only())
    
    if success:
        logger.info("✅ Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Operation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
