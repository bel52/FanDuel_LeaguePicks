#!/bin/bash

# Enhanced FanDuel DFS Optimizer Setup Script
echo "======================================"
echo "Enhanced FanDuel DFS Optimizer Setup"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create project structure
echo "Creating project structure..."
mkdir -p app data/{input,output,targets,fantasypros} logs

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env and add your API keys"
else
    echo ".env file already exists"
fi

# Create sample data if needed
echo "Do you want to create sample data for testing? (y/n)"
read -r create_sample

if [ "$create_sample" = "y" ]; then
    echo "Creating sample data..."
    docker run --rm -v "$PWD:/app" -w /app python:3.11-slim python -c "
import sys
sys.path.insert(0, '/app')
from app.data_ingestion import create_sample_data
create_sample_data()
"
fi

# Build Docker images
echo "Building Docker images..."
docker compose build

# Start services
echo "Starting services..."
docker compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check health
echo "Checking system health..."
curl -s http://localhost:8010/health | python3 -m json.tool

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "The Enhanced DFS Optimizer is now running at http://localhost:8010"
echo ""
echo "Quick Start Commands:"
echo "  - Generate lineup: curl http://localhost:8010/optimize"
echo "  - View health: curl http://localhost:8010/health"
echo "  - Stop services: docker compose down"
echo "  - View logs: docker compose logs -f"
echo ""
echo "Don't forget to:"
echo "  1. Add your OpenAI API key to .env file"
echo "  2. Upload your FantasyPros CSV files to data/input/"
echo "  3. Configure your optimization preferences in .env"
echo ""
