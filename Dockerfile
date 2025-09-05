FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ src/
COPY scripts/ scripts/

# Data/logs (bind-mounted at runtime but created for convenience)
RUN mkdir -p data/fantasypros data/weekly data/targets data/executed data/season data/bankroll logs

ENV TZ=America/New_York
ENV PYTHONPATH=/app

# Scripts are run via `docker compose run`; keep container idle otherwise
CMD ["tail","-f","/dev/null"]
