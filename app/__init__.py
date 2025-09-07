# app/__init__.py
"""
FanDuel NFL DFS Optimizer
AI-powered lineup optimization with real-time data integration
"""

__version__ = "3.0.0"

# Import main classes for easy access
from .enhanced_optimizer import EnhancedDFSOptimizer
from .cache_manager import CacheManager

__all__ = ["EnhancedDFSOptimizer", "CacheManager"]
