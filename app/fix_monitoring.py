# fix_monitoring.py - Add to your app directory
import asyncio
import logging

logger = logging.getLogger(__name__)

class MonitoringManager:
    def __init__(self):
        self.tasks = []
    
    async def start_all(self, data_monitor, auto_swap):
        """Properly start and manage monitoring tasks"""
        try:
            self.tasks = [
                asyncio.create_task(data_monitor.start_monitoring()),
                asyncio.create_task(auto_swap.start_monitoring())
            ]
            logger.info("All monitoring tasks started successfully")
            # Keep tasks alive
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
