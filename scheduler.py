"""
Automated data collection and optimization scheduler with Weekly NFL Cadence
"""
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, Any, List
from loguru import logger
import threading

from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups
from config import UPDATE_INTERVALS, DATA_DIR, OPTIMIZATION_CONFIG

class DFSScheduler:
    """Manages automated data collection and lineup optimization with NFL cadence"""
    
    def __init__(self):
        self.is_running = False
        self.last_update = {}
        self.current_data = {}
        self.latest_lineups = []
        self.scheduler_thread = None
        self.weekly_state = "midweek"
        
        # Initialize directories
        self.ensure_directories()
        
        # Setup logging for scheduler
        logger.add(
            "logs/scheduler_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        )
    
    def ensure_directories(self):
        """Create necessary directories"""
        for directory in [DATA_DIR, DATA_DIR / "lineups", DATA_DIR / "historical"]:
            directory.mkdir(exist_ok=True)

    async def collect_and_update_data(self):
        """Main data collection task"""
        try:
            logger.info("Starting scheduled data collection")
            
            # Collect fresh data
            fresh_data = await get_fresh_data()
            
            if fresh_data and 'players' in fresh_data:
                self.current_data = fresh_data
                self.last_update['data_collection'] = datetime.now()
                
                # Save timestamped data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                data_file = DATA_DIR / f"scheduled_data_{timestamp}.json"
                
                with open(data_file, 'w') as f:
                    json.dump(fresh_data, f, indent=2, default=str)
                
                logger.info(f"Data collection completed. {len(fresh_data.get('players', []))} players updated")
                
                # Trigger lineup optimization if we have good data
                if len(fresh_data.get('players', [])) >= 100:
                    await self.optimize_lineups()
                
            else:
                logger.warning("Data collection returned empty or invalid data")
                
        except Exception as e:
            logger.error(f"Error in scheduled data collection: {e}")
    
    async def optimize_lineups(self):
        """Generate optimized lineups with FIXED default counts"""
        try:
            if not self.current_data or 'players' not in self.current_data:
                logger.warning("No current data available for optimization")
                return
            
            logger.info("Starting lineup optimization")
            
            # FIXED lineup configs - changed from 20 to 5 for GPP
            lineup_configs = [
                {'contest_type': 'gpp', 'num_lineups': 5, 'name': 'tournament'},     # FIXED: was 20
                {'contest_type': 'cash', 'num_lineups': 5, 'name': 'cash_game'},
                {'contest_type': 'gpp', 'num_lineups': 5, 'name': 'contrarian'}      # FIXED: was 10
            ]
            
            all_lineups = {}
            
            for config in lineup_configs:
                lineups = optimize_dfs_lineups(
                    player_data=self.current_data['players'],
                    weather_data=self.current_data.get('weather', {}),
                    num_lineups=config['num_lineups'],
                    contest_type=config['contest_type']
                )
                
                if lineups:
                    all_lineups[config['name']] = lineups
                    logger.info(f"Generated {len(lineups)} {config['name']} lineups")
            
            # Save lineups
            if all_lineups:
                self.latest_lineups = all_lineups
                self.last_update['optimization'] = datetime.now()
                
                # Export to files
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                for lineup_type, lineups in all_lineups.items():
                    # Export to CSV for upload
                    from optimizer import EnhancedDFSOptimizer
                    optimizer = EnhancedDFSOptimizer()
                    csv_file = DATA_DIR / "lineups" / f"{lineup_type}_{timestamp}.csv"
                    optimizer.export_lineups_to_csv(lineups, str(csv_file))
                
                # Save summary data
                summary_file = DATA_DIR / "lineups" / f"lineup_summary_{timestamp}.json"
                summary_data = {
                    'timestamp': datetime.now().isoformat(),
                    'lineup_counts': {k: len(v) for k, v in all_lineups.items()},
                    'data_quality': self.current_data.get('data_quality', {}),
                    'weather_conditions': len(self.current_data.get('weather', {})),
                    'injury_reports': len(self.current_data.get('injuries', []))
                }
                
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                
                logger.info(f"Lineup optimization completed. Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error in lineup optimization: {e}")
    
    def get_data_freshness(self) -> Dict[str, str]:
        """Check how fresh our current data is"""
        freshness = {}
        
        if 'data_collection' in self.last_update:
            time_diff = datetime.now() - self.last_update['data_collection']
            freshness['data_age'] = str(time_diff)
            freshness['data_fresh'] = time_diff.total_seconds() < 3600
        else:
            freshness['data_age'] = "No data collected yet"
            freshness['data_fresh'] = False
        
        if 'optimization' in self.last_update:
            time_diff = datetime.now() - self.last_update['optimization']
            freshness['lineups_age'] = str(time_diff)
            freshness['lineups_fresh'] = time_diff.total_seconds() < 1800
        else:
            freshness['lineups_age'] = "No lineups generated yet"
            freshness['lineups_fresh'] = False
        
        return freshness
    
    def should_update_data(self) -> bool:
        """Determine if data needs updating based on schedule and game times"""
        current_time = datetime.now()
        
        # Always update if no data exists
        if 'data_collection' not in self.last_update:
            return True
        
        # Check if enough time has passed since last update
        time_since_update = current_time - self.last_update['data_collection']
        if time_since_update.total_seconds() > UPDATE_INTERVALS['player_stats'] * 60:
            return True
        
        # More frequent updates on game days
        if self.is_game_day():
            if time_since_update.total_seconds() > 900:  # 15 minutes on game days
                return True
        
        return False
    
    def is_game_day(self) -> bool:
        """Check if today is an NFL game day"""
        current_day = datetime.now().weekday()
        # NFL games typically on Thursday (3), Sunday (6), Monday (0)
        return current_day in [0, 3, 6]
    
    def setup_scheduler(self):
        """Setup the automated scheduling"""
        # Data collection every hour
        schedule.every().hour.do(lambda: asyncio.run(self.collect_and_update_data()))
        
        # More frequent updates on game days
        if self.is_game_day():
            schedule.every(15).minutes.do(lambda: asyncio.run(self.collect_and_update_data()))
        
        # Lineup optimization every 30 minutes if we have fresh data
        schedule.every(30).minutes.do(self.check_and_optimize)
        
        # Daily cleanup at 3 AM
        schedule.every().day.at("03:00").do(self.daily_cleanup)
        
        logger.info("🗓️ Enhanced scheduler setup completed with NFL weekly cadence")
    
    def check_and_optimize(self):
        """Check if optimization is needed and run it"""
        if self.current_data and 'players' in self.current_data:
            if 'optimization' not in self.last_update or \
               (datetime.now() - self.last_update['optimization']).total_seconds() > 1800:
                asyncio.run(self.optimize_lineups())
    
    def daily_cleanup(self):
        """Clean up old files and maintain storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            
            # Clean old data files
            for file_path in DATA_DIR.glob("scheduled_data_*.json"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
            
            # Clean old lineup files
            lineup_dir = DATA_DIR / "lineups"
            for file_path in lineup_dir.glob("*"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
            
            logger.info("Daily cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in daily cleanup: {e}")
    
    def start_scheduler(self):
        """Start the scheduler in a background thread"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.setup_scheduler()
        self.is_running = True
        
        def run_scheduler():
            logger.info("🚀 DFS Scheduler started with NFL weekly cadence")
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            logger.info("DFS Scheduler stopped")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        # Run initial data collection
        asyncio.run(self.collect_and_update_data())
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the scheduler and data"""
        return {
            'is_running': self.is_running,
            'weekly_state': self.weekly_state,
            'last_updates': self.last_update,
            'data_freshness': self.get_data_freshness(),
            'current_data_summary': {
                'player_count': len(self.current_data.get('players', [])),
                'weather_locations': len(self.current_data.get('weather', {})),
                'injury_reports': len(self.current_data.get('injuries', []))
            } if self.current_data else {},
            'lineup_summary': {
                lineup_type: len(lineups) for lineup_type, lineups in self.latest_lineups.items()
            } if isinstance(self.latest_lineups, dict) else {},
            'is_game_day': self.is_game_day()
        }
    
    def force_update(self) -> Dict[str, str]:
        """Force an immediate data update and optimization"""
        try:
            logger.info("Force update requested")
            asyncio.run(self.collect_and_update_data())
            return {"status": "success", "message": "Data updated successfully"}
        except Exception as e:
            logger.error(f"Force update failed: {e}")
            return {"status": "error", "message": str(e)}

# Global scheduler instance
scheduler_instance = None

def get_scheduler() -> DFSScheduler:
    """Get or create the global scheduler instance"""
    global scheduler_instance
    if scheduler_instance is None:
        scheduler_instance = DFSScheduler()
    return scheduler_instance

def start_background_scheduler():
    """Start the background scheduler - call this once on startup"""
    scheduler = get_scheduler()
    scheduler.start_scheduler()
    return scheduler

def stop_background_scheduler():
    """Stop the background scheduler"""
    scheduler = get_scheduler()
    scheduler.stop_scheduler()
