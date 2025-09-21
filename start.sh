#!/bin/bash

echo "Starting FanDuel DFS Optimizer..."

# Ensure directories exist
mkdir -p data/input data/output logs

# Check for data files
if [ ! -f "data/input/qb.csv" ]; then
    echo "No data files found. Creating sample data..."
    ./create_sample_data.sh
fi

# Stop existing containers
docker compose down

# Build and start
docker compose up -d --build

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 5

# Check health
curl -s http://localhost:8010/health | python3 -m json.tool

echo ""
echo "FanDuel DFS Optimizer is running!"
echo ""
echo "Available endpoints:"
echo "  - Health Check: http://localhost:8010/health"
echo "  - Optimize (JSON): http://localhost:8010/optimize"
echo "  - Optimize (Text): http://localhost:8010/optimize_text"
echo "  - Data Status: http://localhost:8010/data/status"
echo ""
echo "To view logs: docker logs dfs-web"
echo "To stop: docker compose down"
