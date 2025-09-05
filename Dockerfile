FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/New_York \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tzdata \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/input data/output logs

# Create a non-root user
RUN groupadd -r dfsuser && useradd -r -g dfsuser dfsuser && \
    chown -R dfsuser:dfsuser /app

USER dfsuser

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
