#!/usr/bin/env python3
from __future__ import annotations
import uvicorn
from app.config import APP_PORT, LOG_LEVEL

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=APP_PORT, reload=False, log_level=LOG_LEVEL.lower())
