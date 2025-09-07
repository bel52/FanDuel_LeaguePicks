FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Ensure all subdirectories exist inside container
RUN mkdir -p \
    data/input data/output data/fantasypros data/weekly \
    data/targets data/executed data/season data/bankroll \
    logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ=America/New_York

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8010/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
