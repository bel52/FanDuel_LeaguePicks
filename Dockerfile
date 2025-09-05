# Multi-stage Docker build for production DFS system:contentReference[oaicite:0]{index=0}  
FROM python:3.11-slim as builder  
RUN apt-get update && apt-get install -y build-essential gcc  
WORKDIR /app  
COPY requirements.txt .  
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt  

FROM python:3.11-slim as production  
RUN apt-get update && apt-get install -y curl && groupadd -r dfsuser && useradd -r -g dfsuser dfsuser  
WORKDIR /app  
COPY --from=builder /wheels /wheels  
RUN pip install --no-cache /wheels/*  
COPY --chown=dfsuser:dfsuser . .  
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1  
ENV PYTHONPATH=/app  
USER dfsuser  
# Healthcheck to ensure service is running:contentReference[oaicite:1]{index=1}  
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \  
    CMD curl -f http://localhost:8000/health || exit 1  
EXPOSE 8000  
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]  
