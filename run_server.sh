# run_server.sh - Server startup script
#!/bin/bash

echo "🚀 Starting DFS Optimization Server..."

# Activate virtual environment
source venv/bin/activate

# Set environment
export PYTHONPATH="$PWD:$PYTHONPATH"

# Load environment variables
set -a
source .env
set +a

# Check if API keys are configured
if [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "⚠️  Warning: OpenAI API key not configured"
fi

# Create log directory
mkdir -p app/logs

# Start server with appropriate configuration
if [ "$DEBUG" = "True" ]; then
    echo "🔧 Starting in development mode..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
else
    echo "🏭 Starting in production mode..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level warning
fi
