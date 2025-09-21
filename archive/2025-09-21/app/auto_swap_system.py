import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os

from app.ai_integration import AIAnalyzer
from app.data_monitor import RealTimeDataMonitor, PlayerUpdate
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.cache_manager import CacheManager
from app import data_ingestion

logger = logging.getLogger(__name__)

class AutoSwapSystem:
    """Automated player swapping system based on real-time data and AI analysis"""
    
    def __init__(self):
        self.ai_analyzer = AIAnalyzer()
        self.data_monitor = RealTimeDataMonitor()
        self.optimizer = EnhancedDFSOptimizer()
        self.cache_manager = CacheManager()
        
        # Swap configuration
        self.max_swaps_per_day = int(os.getenv("MAX_SWAPS_PER_DAY", "3"))
        self.swap_threshold = 0.6  # Severity threshold for auto-swap
        self.min_projection_change = 2.0  # Minimum projection change to trigger swap
        
        # Tracking
        self.daily_swaps = 0
        self.swap_history = []
        
        # Game timing
        self.early_game_cutoff = None  # Will be set based on schedule
        self.late_game_cutoff = None
    
    async def start_monitoring(self):
        """Start the automated swapping monitoring system"""
        logger.info("Starting automated player swapping system...")
        
        # Initialize game timing
        await self._initialize_game_timing()
        
        # Start monitoring loop
        while True:
            try:
                current_time = datetime.now()
                
                # Check if we should process swaps
                if await self._should_process_swaps(current_time):
                    await self._process_potential_swaps()
                
                # Monitor for inactive players before games
                if await self._should_check_inactives(current_time):
                    await self._process_inactive_players()
                
                # Sleep for 5 minutes between checks
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in auto-swap monitoring: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    async def _initialize_game_timing(self):
        """Initialize game timing for swap cutoffs"""
        try:
            # Get current week's game schedule
            # This would integrate with your existing schedule system
            current_time = datetime.now()
            
            # Set default timing (Sunday games)
            self.early_game_cutoff = current_time.replace(
                hour=13, minute=0, second=0, microsecond=0
            )  # 1:00 PM ET
            
            self.late_game_cutoff = current_time.replace(
                hour=16, minute=5, second=0, microsecond=0
            )  # 4:05 PM ET
            
            logger.info(f"Game timing initialized - Early: {self.early_game_cutoff}, Late: {self.late_game_cutoff}")
            
        except Exception as e:
            logger.error(f"Error initializing game timing: {e}")
    
    async def _should_process_swaps(self, current_time: datetime) -> bool:
        """Determine if we should process swaps at current time"""
        
        # Don't swap if we've hit daily limit
        if self.daily_swaps >= self.max_swaps_per_day:
            return False
        
        # Don't swap during games (simplified logic)
        if self.early_game_cutoff and self.late_game_cutoff:
            if self.early_game_cutoff <= current_time <= self.late_game_cutoff:
                return False
        
        # Only swap if auto-swap is enabled
        auto_swap_enabled = os.getenv("AUTO_SWAP_ENABLED", "true").lower() == "true"
        return auto_swap_enabled
    
    async def _should_check_inactives(self, current_time: datetime) -> bool:
        """Check if we should look for inactive player updates"""
        
        # Check inactives 90 minutes before early games
        if self.early_game_cutoff:
            inactive_check_time = self.early_game_cutoff - timedelta(minutes=90)
            return abs((current_time - inactive_check_time).total_seconds()) < 300  # Within 5 minutes
        
        return False
    
    async def _process_potential_swaps(self):
        """Process potential player swaps based on recent updates"""
        
        try:
            # Get recent high-severity updates
            recent_updates = await self.data_monitor.get_recent_updates(hours=2)
            high_severity_updates = [
                update for update in recent_updates 
                if update['severity'] >= self.swap_threshold
            ]
            
            if not high_severity_updates:
                return
            
            # Get current lineup
            current_lineup = await self._get_current_lineup()
            if not current_lineup:
                logger.warning("No current lineup found for auto-swap")
                return
            
            # Check each update for swap potential
            for update in high_severity_updates:
                await self._evaluate_swap_for_update(update, current_lineup)
                
        except Exception as e:
            logger.error(f"Error processing potential swaps: {e}")
    
    async def _process_inactive_players(self):
        """Process inactive player updates and force swaps if needed"""
        
        try:
            # Get recent injury/inactive updates
            recent_updates = await self.data_monitor.get_recent_updates(hours=6)
            inactive_updates = [
                update for update in recent_updates
                if update['update_type'] == 'injury' and update['severity'] >= 0.8
            ]
            
            if not inactive_updates:
                return
            
            current_lineup = await self._get_current_lineup()
            if not current_lineup:
                return
            
            # Force swaps for inactive players
            for update in inactive_updates:
                player_name = update['player_name']
                
                # Check if this player is in our lineup
                lineup_player = next(
                    (p for p in current_lineup if p['name'].lower() == player_name.lower()), 
                    None
                )
                
                if lineup_player:
                    logger.warning(f"Inactive player detected in lineup: {player_name}")
                    await self._force_player_swap(lineup_player, "Player inactive")
                    
        except Exception as e:
            logger.error(f"Error processing inactive players: {e}")
    
    async def _evaluate_swap_for_update(self, update: Dict, current_lineup: List[Dict]):
        """Evaluate if an update should trigger a player swap"""
        
        player_name = update['player_name']
        update_type = update['update_type']
        severity = update['severity']
        
        # Find if this player is in our current lineup
        lineup_player = next(
            (p for p in current_lineup if p['name'].lower() == player_name.lower()), 
            None
        )
        
        if not lineup_player:
            return  # Player not in our lineup
        
        # Calculate potential impact
        current_projection = lineup_player['proj_points']
        
        # Estimate new projection based on update
        if update_type == 'injury':
            new_projection = current_projection * (1 - severity)
        elif update_type == 'weather':
            new_projection = current_projection * (1 - severity * 0.3)
        else:
            new_projection = current_projection * (1 - severity * 0.5)
        
        projection_change = current_projection - new_projection
        
        # Only swap if change is significant
        if projection_change >= self.min_projection_change:
            logger.info(f"Evaluating swap for {player_name}: {current_projection:.1f} â†’ {new_projection:.1f}")
            await self._execute_player_swap(lineup_player, update, new_projection)
    
    async def _execute_player_swap(
        self, 
        player_to_replace: Dict, 
        trigger_update: Dict,
        new_projection: float
    ):
        """Execute a player swap"""
        
        try:
            # Get available players for the position
            current_data = data_ingestion.load_weekly_data()
            if current_data is None:
                logger.error("No player data available for swap")
                return
            
            position = player_to_replace['position']
            salary_budget = player_to_replace['salary']
            team_to_avoid = player_to_replace['team']  # Avoid same team for diversification
            
            # Find replacement candidates
            candidates = current_data[
                (current_data['POS'] == position) & 
                (current_data['SALARY'] <= salary_budget * 1.1) &  # Allow 10% salary increase
                (current_data['TEAM'] != team_to_avoid)
            ].copy()
            
            if candidates.empty:
                logger.warning(f"No replacement candidates found for {player_to_replace['name']}")
                return
            
            # Calculate value scores for candidates
            candidates['value_score'] = candidates['PROJ PTS'] / (candidates['SALARY'] / 1000)
            
            # Use AI to evaluate best replacement
            best_replacement = await self._ai_select_replacement(
                player_to_replace, candidates, trigger_update
            )
            
            if best_replacement is not None:
                await self._confirm_and_execute_swap(
                    player_to_replace, best_replacement, trigger_update
                )
            
        except Exception as e:
            logger.error(f"Error executing swap for {player_to_replace['name']}: {e}")
    
    async def _ai_select_replacement(
        self,
        player_to_replace: Dict,
        candidates: Any,  # DataFrame
        trigger_update: Dict
    ) -> Optional[Dict]:
        """Use AI to select the best replacement player"""
        
        try:
            # Format candidates for AI analysis
            top_candidates = candidates.nlargest(5, 'value_score')
            
            candidate_info = []
            for idx, candidate in top_candidates.iterrows():
                candidate_info.append({
                    'name': candidate['PLAYER NAME'],
                    'team': candidate['TEAM'],
                    'salary': candidate['SALARY'],
                    'projection': candidate['PROJ PTS'],
                    'value_score': candidate['value_score']
                })
            
            # Use AI to select best replacement
            prompt = f"""
            Need to replace {player_to_replace['name']} ({player_to_replace['position']}) due to {trigger_update['description']}.
            Current salary: ${player_to_replace['salary']}
            
            Top replacement candidates:
            {json.dumps(candidate_info, indent=2)}
            
            Select the best replacement considering:
            1. Value (projection/salary)
            2. Game environment
            3. Correlation with existing lineup
            
            Return only the player name of the best replacement.
            """
            
            try:
                best_name = await self.ai_analyzer._call_openai(prompt, max_tokens=50)
                best_name = best_name.strip()
                
                # Find the selected player in candidates
                best_candidate = candidates[candidates['PLAYER NAME'] == best_name].iloc[0]
                
                return {
                    'name': best_candidate['PLAYER NAME'],
                    'position': best_candidate['POS'],
                    'team': best_candidate['TEAM'],
                    'salary': best_candidate['SALARY'],
                    'projection': best_candidate['PROJ PTS']
                }
                
            except Exception as e:
                logger.error(f"AI replacement selection failed: {e}")
                # Fallback to highest value player
                if not candidates.empty:
                    best_idx = candidates['value_score'].idxmax()
                    best = candidates.loc[best_idx]
                    return {
                        'name': best['PLAYER NAME'],
                        'position': best['POS'],
                        'team': best['TEAM'],
                        'salary': best['SALARY'],
                        'projection': best['PROJ PTS']
                    }
                
        except Exception as e:
            logger.error(f"Error in AI replacement selection: {e}")
        
        return None
    
    async def _confirm_and_execute_swap(
        self,
        player_out: Dict,
        player_in: Dict,
        trigger_update: Dict
    ):
        """Confirm and execute the swap"""
        
        try:
            # Log the swap
            swap_record = {
                'timestamp': datetime.now().isoformat(),
                'player_out': player_out,
                'player_in': player_in,
                'trigger': trigger_update,
                'swap_number': self.daily_swaps + 1
            }
            
            self.swap_history.append(swap_record)
            self.daily_swaps += 1
            
            # Update lineup file
            await self._update_lineup_file(player_out, player_in)
            
            # Cache the swap
            cache_key = f"swap_history:{datetime.now().strftime('%Y-%m-%d')}"
            await self.cache_manager.set(cache_key, self.swap_history, ttl=86400)
            
            # Log successful swap
            logger.info(
                f"SWAP EXECUTED: {player_out['name']} → {player_in['name']} "
                f"(Trigger: {trigger_update['update_type']}, Daily swap #{self.daily_swaps})"
            )
            
            # Save to file for persistence
            swap_file = "data/output/swap_history.json"
            os.makedirs(os.path.dirname(swap_file), exist_ok=True)
            
            with open(swap_file, 'a') as f:
                f.write(json.dumps(swap_record) + '\n')
                
        except Exception as e:
            logger.error(f"Error executing swap: {e}")
    
    async def _force_player_swap(self, player: Dict, reason: str):
        """Force a swap for an inactive/injured player"""
        
        try:
            # Create a high-severity update
            trigger_update = {
                'player_name': player['name'],
                'update_type': 'inactive',
                'severity': 1.0,
                'description': reason
            }
            
            # Execute swap with max urgency
            await self._execute_player_swap(player, trigger_update, 0.0)
            
        except Exception as e:
            logger.error(f"Error forcing swap for {player['name']}: {e}")
    
    async def _get_current_lineup(self) -> Optional[List[Dict]]:
        """Get the current lineup from file or cache"""
        
        try:
            # Check cache first
            cache_key = "current_lineup"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return cached
            
            # Try to load from file
            lineup_file = "data/targets/fd_target.csv"
            if os.path.exists(lineup_file):
                import pandas as pd
                df = pd.read_csv(lineup_file)
                
                lineup = []
                for _, row in df.iterrows():
                    lineup.append({
                        'name': row.get('Name', row.get('PLAYER NAME', '')),
                        'position': row.get('Position', row.get('POS', '')),
                        'team': row.get('Team', row.get('TEAM', '')),
                        'salary': row.get('Salary', row.get('SALARY', 0)),
                        'proj_points': row.get('Projection', row.get('PROJ PTS', 0))
                    })
                
                # Cache for quick access
                await self.cache_manager.set(cache_key, lineup, ttl=300)
                return lineup
                
        except Exception as e:
            logger.error(f"Error getting current lineup: {e}")
        
        return None
    
    async def _update_lineup_file(self, player_out: Dict, player_in: Dict):
        """Update the lineup file with the swap"""
        
        try:
            lineup_file = "data/targets/fd_target.csv"
            
            if os.path.exists(lineup_file):
                import pandas as pd
                df = pd.read_csv(lineup_file)
                
                # Find and replace the player
                name_col = 'Name' if 'Name' in df.columns else 'PLAYER NAME'
                player_mask = df[name_col] == player_out['name']
                
                if player_mask.any():
                    idx = df[player_mask].index[0]
                    
                    # Update with new player info
                    df.loc[idx, name_col] = player_in['name']
                    if 'Position' in df.columns:
                        df.loc[idx, 'Position'] = player_in['position']
                    if 'POS' in df.columns:
                        df.loc[idx, 'POS'] = player_in['position']
                    if 'Team' in df.columns:
                        df.loc[idx, 'Team'] = player_in['team']
                    if 'TEAM' in df.columns:
                        df.loc[idx, 'TEAM'] = player_in['team']
                    if 'Salary' in df.columns:
                        df.loc[idx, 'Salary'] = player_in['salary']
                    if 'SALARY' in df.columns:
                        df.loc[idx, 'SALARY'] = player_in['salary']
                    if 'Projection' in df.columns:
                        df.loc[idx, 'Projection'] = player_in['projection']
                    if 'PROJ PTS' in df.columns:
                        df.loc[idx, 'PROJ PTS'] = player_in['projection']
                    
                    # Save updated lineup
                    df.to_csv(lineup_file, index=False)
                    
                    # Clear cache
                    await self.cache_manager.delete("current_lineup")
                    
                    logger.info(f"Lineup file updated: {player_out['name']} → {player_in['name']}")
                    
        except Exception as e:
            logger.error(f"Error updating lineup file: {e}")
    
    async def manual_swap_request(
        self,
        player_out_name: str,
        player_in_name: str,
        reason: str
    ) -> Dict:
        """Process a manual swap request"""
        
        try:
            # Get current lineup
            current_lineup = await self._get_current_lineup()
            if not current_lineup:
                return {"success": False, "error": "No current lineup found"}
            
            # Find player to remove
            player_out = next(
                (p for p in current_lineup if p['name'].lower() == player_out_name.lower()),
                None
            )
            
            if not player_out:
                return {"success": False, "error": f"Player {player_out_name} not in lineup"}
            
            # Get player data
            current_data = data_ingestion.load_weekly_data()
            if current_data is None:
                return {"success": False, "error": "No player data available"}
            
            # Find replacement player
            player_in_data = current_data[
                current_data['PLAYER NAME'].str.lower() == player_in_name.lower()
            ]
            
            if player_in_data.empty:
                return {"success": False, "error": f"Player {player_in_name} not found"}
            
            player_in_row = player_in_data.iloc[0]
            player_in = {
                'name': player_in_row['PLAYER NAME'],
                'position': player_in_row['POS'],
                'team': player_in_row['TEAM'],
                'salary': player_in_row['SALARY'],
                'projection': player_in_row['PROJ PTS']
            }
            
            # Validate positions match
            if player_out['position'] != player_in['position']:
                return {
                    "success": False,
                    "error": f"Position mismatch: {player_out['position']} != {player_in['position']}"
                }
            
            # Check salary cap
            current_total = sum(p['salary'] for p in current_lineup)
            new_total = current_total - player_out['salary'] + player_in['salary']
            
            if new_total > 60000:
                return {
                    "success": False,
                    "error": f"Salary cap exceeded: ${new_total:,} > $60,000"
                }
            
            # Execute swap
            trigger = {
                'player_name': player_out_name,
                'update_type': 'manual',
                'severity': 0.5,
                'description': reason
            }
            
            await self._confirm_and_execute_swap(player_out, player_in, trigger)
            
            return {
                "success": True,
                "swap": {
                    "out": player_out,
                    "in": player_in,
                    "reason": reason,
                    "new_total_salary": new_total,
                    "swap_number": self.daily_swaps
                }
            }
            
        except Exception as e:
            logger.error(f"Manual swap error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_swap_summary(self) -> Dict:
        """Get summary of swap activity"""
        
        try:
            # Load today's swaps from cache
            cache_key = f"swap_history:{datetime.now().strftime('%Y-%m-%d')}"
            today_swaps = await self.cache_manager.get(cache_key) or []
            
            # Calculate impact
            total_proj_change = 0
            total_salary_change = 0
            
            for swap in today_swaps:
                proj_change = swap['player_in']['projection'] - swap['player_out']['proj_points']
                salary_change = swap['player_in']['salary'] - swap['player_out']['salary']
                total_proj_change += proj_change
                total_salary_change += salary_change
            
            return {
                "swaps_executed": self.daily_swaps,
                "swaps_remaining": self.max_swaps_per_day - self.daily_swaps,
                "total_projection_change": round(total_proj_change, 2),
                "total_salary_change": total_salary_change,
                "recent_swaps": today_swaps[-3:] if today_swaps else [],
                "auto_swap_enabled": os.getenv("AUTO_SWAP_ENABLED", "true").lower() == "true"
            }
            
        except Exception as e:
            logger.error(f"Error getting swap summary: {e}")
            return {
                "swaps_executed": 0,
                "swaps_remaining": self.max_swaps_per_day,
                "error": str(e)
            }
    
    def reset_daily_swaps(self):
        """Reset daily swap counter"""
        self.daily_swaps = 0
        self.swap_history = []
        logger.info("Daily swap counter reset")
