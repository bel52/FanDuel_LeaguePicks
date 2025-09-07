# app/__init__.py
from .enhanced_optimizer.service import OptimizerService
from .data_ingestion.service import IngestionService

__all__ = ["OptimizerService", "IngestionService"]

# app/main.py - Fixed imports
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager
from app.enhanced_optimizer.service import OptimizerService  
from app.data_ingestion.service import IngestionService
from app.dependencies import get_optimizer_service, get_ingestion_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connections
    await startup_connections()
    yield
    # Shutdown: Close connections
    await shutdown_connections()

app = FastAPI(lifespan=lifespan)

# Fixed endpoint with proper dependency injection
@app.post("/optimize-lineup")
async def optimize_lineup(
    optimizer: OptimizerService = Depends(get_optimizer_service),
    ingester: IngestionService = Depends(get_ingestion_service)
):
    try:
        data = await ingester.get_player_data()
        lineup = await optimizer.optimize(data)
        return {"lineup": lineup, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
