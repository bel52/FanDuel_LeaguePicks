# app/enhanced_optimizer/service.py - Corrected async implementation
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import pandas as pd
from pydfs_lineup_optimizer import get_optimizer, Site, Sport

class OptimizerService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def optimize(self, player_data: List[Dict[str, Any]]) -> List[Dict]:
        """Async wrapper for CPU-intensive optimization"""
        # Convert to DataFrame for optimization
        df = pd.DataFrame(player_data)
        
        # Run CPU-bound optimization in thread pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self._optimize_sync, 
            df
        )
        return result
    
    def _optimize_sync(self, df: pd.DataFrame) -> List[Dict]:
        """Synchronous optimization logic"""
        try:
            optimizer = get_optimizer(Site.FANDUEL, Sport.FOOTBALL)
            
            # Load players into optimizer
            for _, player in df.iterrows():
                optimizer.add_player({
                    'id': player['id'],
                    'first_name': player['first_name'], 
                    'last_name': player['last_name'],
                    'position': player['position'],
                    'salary': player['salary'],
                    'fppg': player['projected_points'],
                    'team': player['team']
                })
            
            # Generate optimal lineup
            lineup = list(optimizer.optimize(1))[0]
            
            return [{
                'player_id': player.id,
                'name': f"{player.first_name} {player.last_name}",
                'position': player.position,
                'salary': player.salary,
                'projected_points': player.fppg,
                'team': player.team
            } for player in lineup.lineup]
            
        except Exception as e:
            raise Exception(f"Optimization failed: {str(e)}")
