import logging
from fastapi import FastAPI
from app.config import INPUT_DIR
from app.data_ingestion import load_data_from_input_dir

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
def startup_event():
    """
    On startup, log the status of data loading.
    """
    logger.info("Application starting up...")
    _, warnings = load_data_from_input_dir()
    for warning in warnings:
        if "ERROR" in warning:
            logger.error(warning)
        else:
            logger.info(warning)
    logger.info("Application startup complete.")


@app.get("/health")
def health_check():
    """
    A simple endpoint to confirm the API is running.
    """
    return {"status": "ok"}
