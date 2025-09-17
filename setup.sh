#!/bin/bash
echo "Setting up FanDuel DFS Optimizer..."
mkdir -p data/input data/output logs
if [ ! -f "data/input/qb.csv" ]; then
    echo "No data files found. Creating sample data..."
    ./create_sample_data.sh
fi
docker compose down
docker compose up -d --build
sleep 5
curl -s http://localhost:8010/health | python3 -m json.tool
echo "Setup complete. Optimizer is running."
