#!/bin/bash

echo "🚀 Starting Working FanDuel DFS Optimizer..."

# Create directories
mkdir -p data/input data/output logs

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# OpenAI API Key (optional but recommended for AI analysis)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional alternative to OpenAI)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
EOF
    echo "✅ Created .env file - edit it with your API keys"
fi

# Check if we have CSV data
csv_count=$(find data/input -name "*.csv" 2>/dev/null | wc -l)
if [ "$csv_count" -eq 0 ]; then
    echo "📊 No CSV files found. Creating sample data..."
    if [ -f "create_sample_data.sh" ]; then
        chmod +x create_sample_data.sh
        ./create_sample_data.sh
    else
        echo "⚠️  create_sample_data.sh not found - sample data will be generated automatically"
    fi
fi

# Test the system first
echo "🧪 Testing system components..."
python3 test_system.py files

if [ $? -ne 0 ]; then
    echo "❌ File structure test failed. Please ensure all required files are present."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose down 2>/dev/null

# Build and start
echo "🔨 Building Docker image..."
docker compose build

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed. Check the output above for errors."
    exit 1
fi

echo "🚀 Starting containers..."
docker compose up -d

# Wait for service to be ready
echo "⏳ Waiting for service to start..."
sleep 10

# Test the service
echo "🔍 Testing service health..."
response=$(curl -s http://localhost:8010/health 2>/dev/null)

if [ $? -eq 0 ]; then
    echo "✅ Service is running!"
    echo "📊 Health check response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    
    echo ""
    echo "🎯 Testing lineup optimization..."
    curl -s "http://localhost:8010/optimize_text?game_type=league" 2>/dev/null | head -20
    
    echo ""
    echo "🎉 SUCCESS! Your DFS optimizer is working!"
    echo ""
    echo "📡 Available endpoints:"
    echo "  • Health Check: http://localhost:8010/health"
    echo "  • Optimize (JSON): http://localhost:8010/optimize"
    echo "  • Optimize (Text): http://localhost:8010/optimize_text"
    echo "  • Data Status: http://localhost:8010/data/status"
    echo ""
    echo "💡 Example usage:"
    echo "  curl 'http://localhost:8010/optimize_text?game_type=h2h'"
    echo "  curl 'http://localhost:8010/optimize?game_type=league&enforce_stack=true'"
    echo ""
    echo "📁 To use real data:"
    echo "  1. Download CSV files from FantasyPros"
    echo "  2. Save them as: data/input/qb.csv, data/input/rb.csv, etc."
    echo "  3. Restart: docker compose restart"
    echo ""
    echo "🔧 To stop: docker compose down"
    echo "📋 To view logs: docker logs dfs-web -f"
    
else
    echo "❌ Service failed to start. Checking logs..."
    docker logs dfs-web
    echo ""
    echo "🔧 Try running the test manually:"
    echo "  python3 test_system.py"
fi
