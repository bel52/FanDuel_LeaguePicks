# run_tests.sh - Test runner script
#!/bin/bash

echo "🧪 Running DFS System Tests..."

# Activate virtual environment
source venv/bin/activate

# Set environment
export PYTHONPATH="$PWD:$PYTHONPATH"

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Run setup.sh first."
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Run different test types based on argument
case "${1:-all}" in
    "quick")
        echo "🏃 Running quick tests..."
        python3 cli_tester.py config
        python3 cli_tester.py data
        ;;
    "full")
        echo "🔬 Running full test suite..."
        python3 cli_tester.py all
        ;;
    "performance")
        echo "⚡ Running performance tests..."
        python3 cli_tester.py performance
        ;;
    "config")
        echo "⚙️  Testing configuration..."
        python3 cli_tester.py config
        ;;
    "data")
        echo "📊 Testing data ingestion..."
        python3 cli_tester.py data
        ;;
    "weather")
        echo "🌤️  Testing weather integration..."
        python3 cli_tester.py weather
        ;;
    "ai")
        echo "🤖 Testing AI analysis..."
        python3 cli_tester.py ai
        ;;
    "optimization")
        echo "🎯 Testing optimization engine..."
        python3 cli_tester.py optimization
        ;;
    "pipeline")
        echo "🔄 Testing full pipeline..."
        python3 cli_tester.py pipeline
        ;;
    *)
        echo "🎯 Running all tests..."
        python3 cli_tester.py all
        ;;
esac

echo "✅ Tests completed!"
