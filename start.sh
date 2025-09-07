#!/bin/bash

echo "Starting FanDuel DFS Optimizer..."

# Ensure directories exist
mkdir -p data/input data/output data/weekly logs

# Check for data files
if [ ! -f "data/input/qb.csv" ]; then
    echo "No data files found. Creating sample data..."
    chmod +x create_sample_data.sh
    ./create_sample_data.sh
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys"
fi

# Stop existing containers
docker compose down

# Build and start
echo "Building and starting containers..."
docker compose up -d --build

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 8

# Check health
echo "Checking application health..."
curl -s http://localhost:8010/health | python3 -m json.tool 2>/dev/null || echo "Health check pending..."

echo ""
echo "✅ FanDuel DFS Optimizer is running!"
echo ""
echo "Available endpoints:"
echo "  - Root: http://localhost:8010/"
echo "  - Health Check: http://localhost:8010/health"
echo "  - Schedule: http://localhost:8010/schedule"
echo "  - Optimize (JSON): http://localhost:8010/optimize"
echo "  - Optimize (Text): http://localhost:8010/optimize_text"
echo "  - Data Status: http://localhost:8010/data/status"
echo ""
echo "Example usage:"
echo "  curl 'http://localhost:8010/optimize_text?width=110'"
echo "  curl 'http://localhost:8010/optimize?game_type=h2h&lock=Josh%20Allen'"
echo ""
echo "To view logs: docker logs fanduel-web -f"
echo "To stop: docker compose down"
