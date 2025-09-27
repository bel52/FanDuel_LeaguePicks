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
                #if len(fresh_data.get('players', [])) >= 100:
                #    await self.optimize_lineups()
                
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

class NFLWeeklyCadence:
    """Implements proper NFL weekly cadence scheduling"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.locked_players = set()
        self.early_game_results = {}
        
    def setup_nfl_schedule(self):
        """Setup NFL-specific weekly cadence"""
        import schedule
        
        # Wednesday 9:00 AM - Deep build with baseline lineups
        schedule.every().wednesday.at("09:00").do(self.wednesday_baseline_build)
        
        # Thursday-Saturday: Daily refreshes
        schedule.every().thursday.at("10:00").do(self.daily_refresh)
        schedule.every().friday.at("10:00").do(self.daily_refresh)
        schedule.every().saturday.at("10:00").do(self.daily_refresh)
        
        # Sunday Morning 11:30 AM - Final early slate
        schedule.every().sunday.at("11:30").do(self.sunday_am_finalization)
        
        # Sunday 2:15 PM - Mid-slate analysis (lock early games)
        schedule.every().sunday.at("14:15").do(self.mid_slate_lock_and_analyze)
        
        # Sunday 3:55 PM - Final late swap opportunities
        schedule.every().sunday.at("15:55").do(self.final_late_swap)
        
        logger.info("🏈 NFL Weekly Cadence Schedule Configured")
    
    def wednesday_baseline_build(self):
        """Wednesday: Deep build with AI analysis"""
        logger.info("📅 WEDNESDAY: Building baseline lineups + exposure plan")
        try:
            # Force fresh data collection
            asyncio.run(self.scheduler.collect_and_update_data())
            
            # Generate baseline lineups for all contest types
            asyncio.run(self.scheduler.optimize_lineups())
            
            # Save as baseline for the week
            self._save_weekly_baseline()
            
        except Exception as e:
            logger.error(f"Wednesday baseline build failed: {e}")
    
    def daily_refresh(self):
        """Thu-Sat: Refresh data and refine exposures"""
        day = datetime.now().strftime('%A')
        logger.info(f"📅 {day.upper()}: Daily refresh + exposure refinement")
        
        try:
            # Refresh data
            asyncio.run(self.scheduler.collect_and_update_data())
            
            # Refine lineups based on new data
            asyncio.run(self.scheduler.optimize_lineups())
            
        except Exception as e:
            logger.error(f"{day} daily refresh failed: {e}")
    
    def sunday_am_finalization(self):
        """Sunday AM: Final early slate preparation"""
        logger.info("📅 SUNDAY AM: Finalizing early-slate lineups")
        
        try:
            # Full data refresh
            asyncio.run(self.scheduler.collect_and_update_data())
            
            # Process inactives
            self._process_sunday_inactives()
            
            # Finalize early slate lineups
            asyncio.run(self.scheduler.optimize_lineups())
            
        except Exception as e:
            logger.error(f"Sunday AM finalization failed: {e}")
    
    def mid_slate_lock_and_analyze(self):
        """Sunday Mid-Early: Lock started players + analyze early results"""
        logger.info("📅 SUNDAY MID-SLATE: Locking started players + ingesting early results")
        
        try:
            # Lock players from started games
            self._lock_started_games()
            
            # Ingest early game results vs projections
            self._analyze_early_game_results()
            
            # Refactor late slate based on early results
            self._refactor_late_slate()
            
        except Exception as e:
            logger.error(f"Mid-slate analysis failed: {e}")
    
    def final_late_swap(self):
        """Sunday Final: Last chance swaps with leverage logic"""
        logger.info("📅 SUNDAY FINAL: Final swap opportunities with leverage")
        
        try:
            # Final data check
            asyncio.run(self.scheduler.collect_and_update_data())
            
            # Generate final late slate pivots
            self._generate_late_slate_pivots()
            
        except Exception as e:
            logger.error(f"Final late swap failed: {e}")
    
    def _lock_started_games(self):
        """Lock players from games that have started"""
        # Get current game times and lock started players
        # Implementation needed
        pass
    
    def _analyze_early_game_results(self):
        """Analyze early game performance vs projections"""
        # Compare actual vs projected for chalk plays
        # Implementation needed
        pass
    
    def _refactor_late_slate(self):
        """Refactor late slate lineups based on early results"""
        # Use early game data to pivot late slate strategy
        # Implementation needed
        pass
    
    def _process_sunday_inactives(self):
        """Process Sunday inactive players"""
        # Handle late scratches and replacements
        # Implementation needed
        pass
    
    def _save_weekly_baseline(self):
        """Save Wednesday baseline for comparison"""
        # Implementation needed
        pass
    
    def _generate_late_slate_pivots(self):
        """Generate final pivot opportunities"""
        # Implementation needed
        pass

import requests
from datetime import datetime, timedelta
import pytz

class GameLockingEngine:
    """Handles game start times and player locking"""
    
    def __init__(self):
        self.eastern = pytz.timezone('America/New_York')
        self.locked_players = set()
        self.game_times = {}
        self.early_results = {}
    
    def get_current_game_times(self):
        """Get real-time game start times from ESPN"""
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                game_times = {}
                
                for event in data.get('events', []):
                    date_str = event.get('date')
                    if date_str:
                        game_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        game_time = game_time.astimezone(self.eastern)
                        
                        # Get team abbreviations
                        competitors = event.get('competitions', [{}])[0].get('competitors', [])
                        teams = [comp.get('team', {}).get('abbreviation') for comp in competitors]
                        
                        if len(teams) == 2:
                            game_id = f"{teams[0]}_vs_{teams[1]}"
                            game_times[game_id] = {
                                'start_time': game_time,
                                'teams': teams,
                                'status': event.get('status', {}).get('type', {}).get('name', 'scheduled')
                            }
                
                self.game_times = game_times
                logger.info(f"🕐 Updated game times for {len(game_times)} games")
                return game_times
                
        except Exception as e:
            logger.error(f"Failed to get game times: {e}")
            return {}
    
    def get_started_games(self):
        """Get games that have already started"""
        current_time = datetime.now(self.eastern)
        started_games = []
        
        for game_id, game_info in self.game_times.items():
            start_time = game_info['start_time']
            status = game_info['status']
            
            # Game has started if current time > start time OR status indicates in progress
            if (current_time > start_time or 
                status.lower() in ['in progress', 'halftime', 'final']):
                started_games.append(game_id)
        
        return started_games
    
    def lock_players_from_started_games(self, all_players):
        """Lock players from games that have started"""
        started_games = self.get_started_games()
        newly_locked = set()
        
        for player in all_players:
            player_team = player.get('team', '')
            
            # Check if player's team is in a started game
            for game_id in started_games:
                teams = self.game_times[game_id]['teams']
                if player_team in teams:
                    player_id = player.get('player_id', player.get('name'))
                    if player_id not in self.locked_players:
                        newly_locked.add(player_id)
                        self.locked_players.add(player_id)
        
        if newly_locked:
            logger.info(f"🔒 Locked {len(newly_locked)} players from {len(started_games)} started games")
        
        return list(newly_locked)
    
    def get_early_game_results(self):
        """Get real-time results from early games"""
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = {}
                
                for event in data.get('events', []):
                    competitors = event.get('competitions', [{}])[0].get('competitors', [])
                    
                    for competitor in competitors:
                        team_abbr = competitor.get('team', {}).get('abbreviation')
                        team_stats = competitor.get('statistics', [])
                        
                        if team_abbr and team_stats:
                            results[team_abbr] = {
                                'score': competitor.get('score', 0),
                                'statistics': team_stats
                            }
                
                self.early_results = results
                return results
                
        except Exception as e:
            logger.error(f"Failed to get early game results: {e}")
            return {}


class LateSwapEngine:
    """Handles late swap decisions and leverage pivots"""
    
    def __init__(self, game_locker):
        self.game_locker = game_locker
        self.swap_history = []
    
    def analyze_chalk_performance(self, early_results):
        """Analyze how chalk plays performed in early games"""
        chalk_analysis = {}
        
        # This would analyze high-ownership players vs their actual performance
        # For now, we'll create a framework
        
        for team, stats in early_results.items():
            score = stats.get('score', 0)
            
            # Simple logic: if team scored >21 points, their players likely hit
            if score > 21:
                chalk_analysis[team] = 'outperformed'
            elif score < 14:
                chalk_analysis[team] = 'underperformed'
            else:
                chalk_analysis[team] = 'neutral'
        
        logger.info(f"📊 Chalk analysis: {chalk_analysis}")
        return chalk_analysis
    
    def generate_late_slate_pivots(self, available_players, chalk_analysis):
        """Generate pivot recommendations for late slate"""
        pivot_recommendations = []
        
        # If early chalk failed, pivot to late slate contrarian plays
        underperforming_early = [team for team, result in chalk_analysis.items() 
                               if result == 'underperformed']
        
        if len(underperforming_early) > 1:
            strategy = 'contrarian'
            logger.info("🔄 Early chalk failed - recommending contrarian late slate")
        else:
            strategy = 'balanced'
            logger.info("📈 Early chalk performed - maintaining balanced approach")
        
        return {
            'strategy': strategy,
            'pivot_count': len(underperforming_early),
            'recommended_approach': self._get_strategy_approach(strategy)
        }
    
    def _get_strategy_approach(self, strategy):
        """Get specific approach for strategy"""
        approaches = {
            'contrarian': {
                'ownership_threshold': 10.0,
                'variance_weight': 0.5,
                'stacking_preference': 'unconventional'
            },
            'balanced': {
                'ownership_threshold': 25.0,
                'variance_weight': 0.3,
                'stacking_preference': 'traditional'
            }
        }
        return approaches.get(strategy, approaches['balanced'])

# Update the NFLWeeklyCadence class methods
def _lock_started_games(self):
    """Lock players from games that have started"""
    if not hasattr(self, 'game_locker'):
        self.game_locker = GameLockingEngine()
    
    # Get current game times
    self.game_locker.get_current_game_times()
    
    # Lock players from started games
    if self.scheduler.current_data and 'players' in self.scheduler.current_data:
        locked = self.game_locker.lock_players_from_started_games(
            self.scheduler.current_data['players']
        )
        logger.info(f"🔒 Game locking complete: {len(locked)} players locked")

def _analyze_early_game_results(self):
    """Analyze early game performance vs projections"""
    if not hasattr(self, 'game_locker'):
        self.game_locker = GameLockingEngine()
    
    # Get early game results
    early_results = self.game_locker.get_early_game_results()
    
    if early_results:
        # Initialize late swap engine
        if not hasattr(self, 'swap_engine'):
            self.swap_engine = LateSwapEngine(self.game_locker)
        
        # Analyze chalk performance
        chalk_analysis = self.swap_engine.analyze_chalk_performance(early_results)
        
        # Store for late slate decisions
        self.early_game_analysis = chalk_analysis
        logger.info("📊 Early game analysis complete")

def _refactor_late_slate(self):
    """Refactor late slate lineups based on early results"""
    if hasattr(self, 'early_game_analysis') and hasattr(self, 'swap_engine'):
        # Generate pivot recommendations
        pivots = self.swap_engine.generate_late_slate_pivots(
            self.scheduler.current_data.get('players', []),
            self.early_game_analysis
        )
        
        # Apply pivot strategy to late slate optimization
        late_slate_params = {
            'contest_type': pivots['strategy'],
            'num_lineups': 10,
            'ownership_threshold': pivots['recommended_approach']['ownership_threshold']
        }
        
        logger.info(f"🔄 Refactoring late slate with {pivots['strategy']} strategy")
        
        # Re-optimize with new strategy
        asyncio.run(self.scheduler.optimize_lineups())
