# app/dependencies.py - Simplified without database
from typing import Annotated
from fastapi import Depends
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.cache_manager import CacheManager
from app.data_monitor import RealTimeDataMonitor
from app.auto_swap_system import AutoSwapSystem

# Global instances for dependency injection
_optimizer_instance = None
_cache_manager_instance = None
_data_monitor_instance = None
_auto_swap_instance = None

def get_optimizer_service() -> EnhancedDFSOptimizer:
    """Get the DFS optimizer service instance"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = EnhancedDFSOptimizer()
    return _optimizer_instance

def get_cache_manager() -> CacheManager:
    """Get the cache manager instance"""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance

def get_data_monitor() -> RealTimeDataMonitor:
    """Get the data monitor instance"""
    global _data_monitor_instance
    if _data_monitor_instance is None:
        _data_monitor_instance = RealTimeDataMonitor()
    return _data_monitor_instance

def get_auto_swap_system() -> AutoSwapSystem:
    """Get the auto swap system instance"""
    global _auto_swap_instance
    if _auto_swap_instance is None:
        _auto_swap_instance = AutoSwapSystem()
    return _auto_swap_instance

# Type aliases for dependency injection
OptimizerDep = Annotated[EnhancedDFSOptimizer, Depends(get_optimizer_service)]
CacheDep = Annotated[CacheManager, Depends(get_cache_manager)]
DataMonitorDep = Annotated[RealTimeDataMonitor, Depends(get_data_monitor)]
AutoSwapDep = Annotated[AutoSwapSystem, Depends(get_auto_swap_system)]
