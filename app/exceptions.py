# app/exceptions.py

class OptimizationError(Exception):
    """Raised when optimization fails"""
    pass

class DataIngestionError(Exception):
    """Raised when data ingestion fails"""
    pass

class CacheError(Exception):
    """Raised when cache operations fail"""
    pass
