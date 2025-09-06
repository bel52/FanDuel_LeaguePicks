"""
Enhanced FanDuel NFL DFS Optimizer
Version: 3.0.0
"""

__version__ = "3.0.0"

# Make package imports cleaner
from .config import settings
from .enhanced_optimizer import EnhancedDFSOptimizer
from .ai_integration import AIAnalyzer
from .data_monitor import RealTimeDataMonitor
from .auto_swap_system import AutoSwapSystem
from .cache_manager import CacheManager

__all__ = [
    'settings',
    'EnhancedDFSOptimizer',
    'AIAnalyzer', 
    'RealTimeDataMonitor',
    'AutoSwapSystem',
    'CacheManager'
]
