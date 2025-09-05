#!/bin/bash

echo "Cleaning and starting FanDuel DFS Optimizer..."

# Stop and remove existing containers
docker compose down
docker system prune -f

# Ensure directories exist
mkdir -p data/input data/output logs

# Check for data files
if [ ! -f "data/input/qb.csv" ]; then
    echo "Creating sample data..."
    ./create_sample_data.sh
fi

# Build and start fresh
echo "Building Docker image..."
docker compose build --no-cache

echo "Starting services..."
docker compose up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 10

# Check if web container is running
if docker ps | grep -q dfs-web; then
    echo "Container is running!"
    
    # Try health check
    echo "Checking health..."
    curl -s http://localhost:8010/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Health check pending..."
    
    echo ""
    echo "✅ FanDuel DFS Optimizer is running!"
    echo ""
    echo "Available endpoints:"
    echo "  - Root: http://localhost:8010/"
    echo "  - Health: http://localhost:8010/health"
    echo "  - Optimize (Text): http://localhost:8010/optimize_text"
    echo "  - Data Status: http://localhost:8010/data/status"
    echo ""
    echo "Commands:"
    echo "  - View logs: docker logs dfs-web"
    echo "  - Follow logs: docker logs -f dfs-web"
    echo "  - Stop: docker compose down"
else
    echo "❌ Container failed to start. Check logs:"
    docker compose logs web
fi
