FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends coinor-cbc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart
COPY src/ src/
COPY run.sh bin/ ./bin_src/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
# data/ (distributions, results.db, uploads, lineups) is a bind mount — never baked in
VOLUME /app/data
EXPOSE 8093
CMD ["python3","-m","dfs.web"]
