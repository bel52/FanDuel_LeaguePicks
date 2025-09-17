import asyncio, logging
from datetime import datetime, timedelta
from typing import List, Dict
from app.ai_integration import AIAnalyzer
from app.data_monitor import RealTimeDataMonitor, PlayerUpdate
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class AutoSwapSystem:
    """Automated late-swap system based on live data and AI suggestions."""
    def __init__(self):
        self.ai_analyzer = AIAnalyzer()
        self.data_monitor = RealTimeDataMonitor()
        self.optimizer = EnhancedDFSOptimizer()
        self.cache_manager = CacheManager()
        self.max_swaps_per_day = settings.MAX_SWAPS_PER_DAY
        self.daily_swaps = 0
        self.swap_history = []
        # Simplistic timing for Sunday games:
        now = datetime.now()
        self.early_game_cutoff = now.replace(hour=13, minute=0, second=0)
        self.late_game_cutoff = now.replace(hour=16, minute=5, second=0)

    async def start_monitoring(self):
        """Continuously check for swap opportunities."""
        logger.info("Auto-swap system started.")
        while True:
            try:
                now = datetime.now()
                if self.daily_swaps < self.max_swaps_per_day and (now < self.early_game_cutoff or now > self.late_game_cutoff):
                    await self._process_updates()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Auto-swap error: {e}")
                await asyncio.sleep(60)

    async def _process_updates(self):
        """Check recent updates and perform swaps if needed."""
        updates = await self.data_monitor.get_recent_updates(hours=2)
        for upd in updates:
            if upd.severity >= 0.6:  # threshold
                await self._consider_swap(upd)

    async def _consider_swap(self, update: PlayerUpdate):
        """Evaluate whether to swap out a player for a given update."""
        player = update.player_name
        # Load current lineup from cache or file
        lineup = await self._get_current_lineup()
        if not lineup: 
            return
        # If the updated player is in lineup
        for p in lineup:
            if p['name'].lower() == player.lower():
                logger.warning(f"High-severity update for {player}: swapping out")
                await self._execute_swap(p, update)
                break

    async def _execute_swap(self, lineup_player: Dict, update: PlayerUpdate):
        """Perform the actual swap using the optimizer with locked players."""
        # Get new candidate lineup excluding this player
        current_lineup = await self._get_current_lineup()
        remaining_players = [p for p in current_lineup if p['name'] != lineup_player['name']]
        # Make sure swapped-out player is not used
        excluded_names = [lineup_player['name']]
        # Re-run optimizer (e.g., league mode) banning this player
        df, _ = load_data_from_input_dir()
        exclude_idxs = {i for i, n in enumerate(df['PLAYER NAME']) if n in excluded_names}
        new_lineup, metadata = await self.optimizer.optimize_lineup(
            df, game_type="league",
            salary_cap=settings.SALARY_CAP,
            enforce_stack=True, min_stack_receivers=1,
            lock_indices=set(), ban_indices=exclude_idxs
        )
        if new_lineup:
            # Save new lineup to file/cache, log swap
            self.swap_history.append({
                "out": lineup_player,
                "in": [df.loc[i,'PLAYER NAME'] for i in new_lineup if i not in exclude_idxs],
                "update": update.__dict__,
                "time": datetime.now().isoformat()
            })
            self.daily_swaps += 1
            logger.info(f"Executed swap: out {lineup_player['name']}, new lineup indices {new_lineup}")

    async def _get_current_lineup(self) -> List[Dict]:
        """Load current lineup from cache or output file."""
        current = await self.cache_manager.get("current_lineup")
        return current or []
