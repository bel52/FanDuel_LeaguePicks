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
        self.weekly_state = "midweek"  # midweek, wednesday, thursday-saturday, sunday-am, sunday-live
        
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

    # ============================================================================
    # WEEKLY NFL CADENCE METHODS (NEW)
    # ============================================================================
    
    async def wednesday_baseline_build(self):
        """Wednesday: Pre-build baseline lineups + exposure plan for main slate"""
        logger.info("🗓️ WEDNESDAY: Starting baseline build for main slate")
        self.weekly_state = "wednesday"
        
        try:
            # Collect fresh data with emphasis on projections
            fresh_data = await get_fresh_data()
            if not fresh_data or not fresh_data.get('players'):
                logger.warning("No data available for Wednesday baseline build")
                return
            
            self.current_data = fresh_data
            
            # Generate baseline lineups for each contest type
            baseline_configs = [
                {'contest_type': 'gpp', 'num_lineups': 30, 'name': 'tournament_baseline'},
                {'contest_type': 'cash', 'num_lineups': 10, 'name': 'cash_baseline'},
                {'contest_type': 'contrarian', 'num_lineups': 15, 'name': 'contrarian_baseline'}
            ]
            
            baseline_lineups = {}
            exposure_plan = {}
            
            for config in baseline_configs:
                lineups = optimize_dfs_lineups(
                    player_data=fresh_data['players'],
                    weather_data=fresh_data.get('weather', {}),
                    num_lineups=config['num_lineups'],
                    contest_type=config['contest_type']
                )
                
                if lineups:
                    baseline_lineups[config['name']] = lineups
                    exposure_plan[config['name']] = self._calculate_exposure_plan(lineups)
                    logger.info(f"✅ Generated {len(lineups)} {config['name']} baseline lineups")
            
            # Save baseline data
            baseline_file = DATA_DIR / "lineups" / f"wednesday_baseline_{datetime.now().strftime('%Y%m%d')}.json"
            baseline_data = {
                'timestamp': datetime.now().isoformat(),
                'weekly_state': self.weekly_state,
                'lineup_counts': {k: len(v) for k, v in baseline_lineups.items()},
                'exposure_plan': exposure_plan,
                'data_quality': fresh_data.get('data_quality', {})
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            self.latest_lineups = baseline_lineups
            self.last_update['wednesday_baseline'] = datetime.now()
            
            logger.info(f"🎯 Wednesday baseline build completed. Saved to {baseline_file}")
            
        except Exception as e:
            logger.error(f"Error in Wednesday baseline build: {e}")
    
    async def thursday_saturday_refresh(self):
        """Thu-Sat (daily): Refresh data, refine exposures, rebuild lineups"""
        logger.info("🔄 THU-SAT: Refreshing data and refining lineups")
        self.weekly_state = "thursday-saturday"
        
        try:
            # Get current day for context
            current_day = datetime.now().strftime('%A')
            logger.info(f"Running {current_day} refresh cycle")
            
            # Collect latest data
            fresh_data = await get_fresh_data()
            if not fresh_data or not fresh_data.get('players'):
                logger.warning(f"No data available for {current_day} refresh")
                return
            
            # Compare with baseline data to identify significant changes
            data_changes = self._analyze_data_changes(fresh_data)
            
            if data_changes['significant_changes']:
                logger.info(f"📊 Significant changes detected: {data_changes['summary']}")
                
                # Rebuild lineups with updated data
                refined_configs = [
                    {'contest_type': 'gpp', 'num_lineups': 25, 'name': 'tournament_refined'},
                    {'contest_type': 'cash', 'num_lineups': 8, 'name': 'cash_refined'},
                    {'contest_type': 'contrarian', 'num_lineups': 12, 'name': 'contrarian_refined'}
                ]
                
                refined_lineups = {}
                
                for config in refined_configs:
                    lineups = optimize_dfs_lineups(
                        player_data=fresh_data['players'],
                        weather_data=fresh_data.get('weather', {}),
                        num_lineups=config['num_lineups'],
                        contest_type=config['contest_type']
                    )
                    
                    if lineups:
                        refined_lineups[config['name']] = lineups
                        logger.info(f"✅ Refined {len(lineups)} {config['name']} lineups")
                
                self.latest_lineups = refined_lineups
                self.current_data = fresh_data
            else:
                logger.info("📊 No significant changes detected, keeping current lineups")
            
            self.last_update['thursday_saturday_refresh'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in Thursday-Saturday refresh: {e}")
    
    async def sunday_am_finalize(self):
        """Sunday AM: Full refresh; finalize early-slate lineups"""
        logger.info("🌅 SUNDAY AM: Finalizing lineups for early slate")
        self.weekly_state = "sunday-am"
        
        try:
            # Get absolute latest data before games start
            fresh_data = await get_fresh_data()
            if not fresh_data or not fresh_data.get('players'):
                logger.warning("No data available for Sunday AM finalization")
                return
            
            # Focus on early games (1PM ET slate)
            early_games = self._identify_early_slate_games(fresh_data)
            logger.info(f"🕐 Identified {len(early_games)} early slate games")
            
            # Generate final lineups with latest injury/inactive reports
            final_configs = [
                {'contest_type': 'gpp', 'num_lineups': 20, 'name': 'tournament_final'},
                {'contest_type': 'cash', 'num_lineups': 5, 'name': 'cash_final'},
                {'contest_type': 'contrarian', 'num_lineups': 10, 'name': 'contrarian_final'}
            ]
            
            final_lineups = {}
            
            for config in final_configs:
                lineups = optimize_dfs_lineups(
                    player_data=fresh_data['players'],
                    weather_data=fresh_data.get('weather', {}),
                    num_lineups=config['num_lineups'],
                    contest_type=config['contest_type']
                )
                
                if lineups:
                    final_lineups[config['name']] = lineups
                    logger.info(f"✅ Finalized {len(lineups)} {config['name']} lineups")
                    
                    # Export final lineups to CSV for upload
                    from optimizer import EnhancedDFSOptimizer
                    optimizer = EnhancedDFSOptimizer()
                    csv_file = DATA_DIR / "lineups" / f"FINAL_{config['name']}_week_{datetime.now().strftime('%Y%m%d')}.csv"
                    optimizer.export_lineups_to_csv(lineups, str(csv_file))
                    logger.info(f"📄 Exported {config['name']} to {csv_file}")
            
            self.latest_lineups = final_lineups
            self.current_data = fresh_data
            self.last_update['sunday_am_finalize'] = datetime.now()
            
            logger.info("🎯 Sunday AM finalization completed - lineups ready for upload!")
            
        except Exception as e:
            logger.error(f"Error in Sunday AM finalization: {e}")
    
    async def sunday_mid_early_late_swap(self):
        """Sunday Mid-Early-Slate: Lock started players, ingest early results, refactor late slate"""
        logger.info("⚡ SUNDAY MID-SLATE: Processing early results and optimizing late slate")
        self.weekly_state = "sunday-live"
        
        try:
            # Get current game states
            game_states = await self._get_current_game_states()
            started_games = [g for g in game_states if g['status'] in ['live', 'final']]
            upcoming_games = [g for g in game_states if g['status'] == 'scheduled']
            
            logger.info(f"🎮 Game states: {len(started_games)} started, {len(upcoming_games)} upcoming")
            
            if not started_games:
                logger.info("No games started yet, skipping late swap")
                return
            
            # Lock players from started games (this is the key late-swap feature)
            locked_players = self._identify_locked_players(started_games)
            logger.info(f"🔒 Locked {len(locked_players)} players from started games")
            
            # Get early slate results vs projections
            early_results = await self._analyze_early_slate_performance()
            logger.info(f"📊 Early slate analysis: {early_results.get('summary', 'No data')}")
            
            # Filter to late slate players only
            late_slate_players = self._filter_late_slate_players(
                self.current_data.get('players', []), 
                upcoming_games
            )
            
            if len(late_slate_players) < 50:
                logger.warning(f"Only {len(late_slate_players)} late slate players available")
                return
            
            # Generate leverage-focused late slate lineups
            late_swap_configs = [
                {'contest_type': 'gpp', 'num_lineups': 15, 'name': 'late_slate_leverage'},
                {'contest_type': 'contrarian', 'num_lineups': 10, 'name': 'late_slate_contrarian'}
            ]
            
            late_swap_lineups = {}
            
            for config in late_swap_configs:
                lineups = optimize_dfs_lineups(
                    player_data=late_slate_players,
                    weather_data=self.current_data.get('weather', {}),
                    num_lineups=config['num_lineups'],
                    contest_type=config['contest_type']
                )
                
                if lineups:
                    late_swap_lineups[config['name']] = lineups
                    logger.info(f"⚡ Generated {len(lineups)} {config['name']} late swap lineups")
                    
                    # Export late swap lineups
                    from optimizer import EnhancedDFSOptimizer
                    optimizer = EnhancedDFSOptimizer()
                    csv_file = DATA_DIR / "lineups" / f"LATE_SWAP_{config['name']}_{datetime.now().strftime('%H%M')}.csv"
                    optimizer.export_lineups_to_csv(lineups, str(csv_file))
                    logger.info(f"📄 Late swap lineups exported to {csv_file}")
            
            # Update lineup state
            if late_swap_lineups:
                self.latest_lineups.update(late_swap_lineups)
            
            self.last_update['sunday_late_swap'] = datetime.now()
            
            logger.info("⚡ Late swap optimization completed!")
            
        except Exception as e:
            logger.error(f"Error in Sunday late swap: {e}")

    # ============================================================================
    # SUPPORTING METHODS FOR WEEKLY CADENCE
    # ============================================================================
    
    def _calculate_exposure_plan(self, lineups) -> Dict[str, Any]:
        """Calculate player exposure across lineups"""
        if not lineups:
            return {}
        
        player_counts = {}
        total_lineups = len(lineups)
        
        for lineup in lineups:
            for player in lineup.players:
                if player.name not in player_counts:
                    player_counts[player.name] = 0
                player_counts[player.name] += 1
        
        exposure_percentages = {
            name: (count / total_lineups) * 100 
            for name, count in player_counts.items()
        }
        
        return {
            'total_lineups': total_lineups,
            'player_exposures': exposure_percentages,
            'high_exposure_players': {
                name: pct for name, pct in exposure_percentages.items() if pct > 50
            },
            'balanced_exposure_players': {
                name: pct for name, pct in exposure_percentages.items() if 20 <= pct <= 50
            }
        }
    
    def _analyze_data_changes(self, new_data) -> Dict[str, Any]:
        """Compare new data with current data to identify significant changes"""
        if not self.current_data or not self.current_data.get('players'):
            return {'significant_changes': True, 'summary': 'No baseline data for comparison'}
        
        current_players = {p.get('name', ''): p for p in self.current_data.get('players', [])}
        new_players = {p.get('name', ''): p for p in new_data.get('players', [])}
        
        significant_changes = False
        changes = []
        
        # Check for projection changes > 2 points
        for name, new_player in new_players.items():
            if name in current_players:
                old_proj = current_players[name].get('projection', 0)
                new_proj = new_player.get('projection', 0)
                proj_diff = abs(new_proj - old_proj)
                
                if proj_diff > 2.0:
                    significant_changes = True
                    changes.append(f"{name}: {old_proj:.1f} → {new_proj:.1f}")
        
        # Check for new injury reports
        old_injuries = len(self.current_data.get('injuries', []))
        new_injuries = len(new_data.get('injuries', []))
        
        if new_injuries > old_injuries:
            significant_changes = True
            changes.append(f"New injury reports: {new_injuries - old_injuries}")
        
        return {
            'significant_changes': significant_changes,
            'summary': '; '.join(changes[:5]),  # Limit to top 5 changes
            'projection_changes': len([c for c in changes if '→' in c]),
            'injury_changes': new_injuries - old_injuries
        }
    
    def _identify_early_slate_games(self, data) -> List[Dict]:
        """Identify games in the early slate (1PM ET)"""
        # This would integrate with the game time data from data_collector
        # For now, return a placeholder
        return data.get('games_info', {}).get('main_slate', [])
    
    async def _get_current_game_states(self) -> List[Dict]:
        """Get current state of all NFL games (scheduled/live/final)"""
        try:
            # This would typically call ESPN API or similar
            # For now, return placeholder data
            return [
                {'id': 'game_1', 'status': 'live', 'teams': ['PHI', 'WAS']},
                {'id': 'game_2', 'status': 'scheduled', 'teams': ['BAL', 'BUF']},
                {'id': 'game_3', 'status': 'scheduled', 'teams': ['DET', 'GB']},
            ]
        except Exception as e:
            logger.error(f"Error getting game states: {e}")
            return []
    
    def _identify_locked_players(self, started_games) -> List[str]:
        """Identify players who should be locked (from started games)"""
        locked_teams = set()
        for game in started_games:
            locked_teams.update(game.get('teams', []))
        
        if not self.current_data or not self.current_data.get('players'):
            return []
        
        locked_players = [
            player.get('name', '') 
            for player in self.current_data['players'] 
            if player.get('team', '').upper() in locked_teams
        ]
        
        return locked_players
    
    async def _analyze_early_slate_performance(self) -> Dict[str, Any]:
        """Analyze early slate performance vs projections for late slate leverage"""
        try:
            # This would integrate with live scoring APIs
            # For now, return placeholder analysis
            return {
                'summary': 'Early slate analysis placeholder',
                'chalk_performance': 'mixed',
                'leverage_opportunities': ['contrarian_plays_hitting', 'low_owned_outperforming'],
                'pivot_recommendations': []
            }
        except Exception as e:
            logger.error(f"Error analyzing early slate performance: {e}")
            return {'summary': 'Analysis failed', 'error': str(e)}
    
    def _filter_late_slate_players(self, all_players, upcoming_games) -> List[Dict]:
        """Filter players to only those in upcoming (late slate) games"""
        if not upcoming_games:
            return []
        
        late_slate_teams = set()
        for game in upcoming_games:
            late_slate_teams.update(game.get('teams', []))
        
        late_slate_players = [
            player for player in all_players 
            if player.get('team', '').upper() in late_slate_teams
        ]
        
        logger.info(f"🕐 Late slate teams: {late_slate_teams}")
        logger.info(f"🎯 Late slate players: {len(late_slate_players)}")
        
        return late_slate_players

    # ============================================================================
    # EXISTING SCHEDULER METHODS (ENHANCED)
    # ============================================================================
    
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
        """Generate optimized lineups with current data"""
        try:
            if not self.current_data or 'players' not in self.current_data:
                logger.warning("No current data available for optimization")
                return
            
            logger.info("Starting lineup optimization")
            
            # Generate multiple lineup types
            lineup_configs = [
                {'contest_type': 'gpp', 'num_lineups': 20, 'name': 'tournament'},
                {'contest_type': 'cash', 'num_lineups': 5, 'name': 'cash_game'},
                {'contest_type': 'gpp', 'num_lineups': 10, 'name': 'contrarian'}
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
            freshness['data_fresh'] = time_diff.total_seconds() < 3600  # Fresh if < 1 hour
        else:
            freshness['data_age'] = "No data collected yet"
            freshness['data_fresh'] = False
        
        if 'optimization' in self.last_update:
            time_diff = datetime.now() - self.last_update['optimization']
            freshness['lineups_age'] = str(time_diff)
            freshness['lineups_fresh'] = time_diff.total_seconds() < 1800  # Fresh if < 30 min
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
        
        # More frequent updates on game days (implement game day detection logic)
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
        """Setup the automated scheduling with NFL weekly cadence"""
        # ============================================================================
        # WEEKLY NFL CADENCE SCHEDULE (NEW)
        # ============================================================================
        
        # Wednesday: Baseline build
        schedule.every().wednesday.at("09:00").do(lambda: asyncio.run(self.wednesday_baseline_build()))
        
        # Thursday-Saturday: Daily refresh
        schedule.every().thursday.at("10:00").do(lambda: asyncio.run(self.thursday_saturday_refresh()))
        schedule.every().friday.at("10:00").do(lambda: asyncio.run(self.thursday_saturday_refresh()))
        schedule.every().saturday.at("10:00").do(lambda: asyncio.run(self.thursday_saturday_refresh()))
        
        # Sunday AM: Final lineup preparation
        schedule.every().sunday.at("11:30").do(lambda: asyncio.run(self.sunday_am_finalize()))
        
        # Sunday Mid-Slate: Late swap optimization (during 1PM games)
        schedule.every().sunday.at("14:15").do(lambda: asyncio.run(self.sunday_mid_early_late_swap()))
        schedule.every().sunday.at("15:45").do(lambda: asyncio.run(self.sunday_mid_early_late_swap()))
        
        # ============================================================================
        # EXISTING CONTINUOUS SCHEDULE (ENHANCED)
        # ============================================================================
        
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
