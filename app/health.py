# app/health.py - Comprehensive health monitoring
from fastapi import HTTPException
from datetime import datetime
import asyncio
from typing import Dict, Any
from app.database import engine
from app.redis_client import redis_manager

async def comprehensive_health_check() -> Dict[str, Any]:
    """Comprehensive system health check"""
    
    health_status = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "healthy",
        "services": {},
        "performance": {}
    }
    
    # Check database connectivity
    try:
        start_time = asyncio.get_event_loop().time()
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        db_response_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        health_status["services"]["database"] = {
            "status": "healthy",
            "response_time_ms": round(db_response_time, 2)
        }
    except Exception as e:
        health_status["services"]["database"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis connectivity
    try:
        start_time = asyncio.get_event_loop().time()
        if redis_manager.redis:
            await redis_manager.redis.ping()
        redis_response_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        health_status["services"]["redis"] = {
            "status": "healthy",
            "response_time_ms": round(redis_response_time, 2)
        }
    except Exception as e:
        health_status["services"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status
