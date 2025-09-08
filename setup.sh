# setup.sh - Initial setup script
#!/bin/bash

echo "🚀 Setting up FanDuel NFL DFS Optimization System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ Python 3.8+ required. Please upgrade Python."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Install playwright browsers
echo "🎭 Installing Playwright browsers..."
playwright install

# Create directories
echo "📁 Creating directories..."
mkdir -p app/logs
mkdir -p app/data
mkdir -p app/exports
mkdir -p app/cache

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///./app/data/dfs.db

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
MAX_WORKERS=4

# DFS Configuration
DEFAULT_SALARY_CAP=50000
OPTIMIZATION_TIMEOUT=30
MAX_LINEUPS_PER_REQUEST=150

# Rate Limiting
API_RATE_LIMIT=100
WEATHER_RATE_LIMIT=1000
EOF
    echo "📝 Please edit .env with your actual API keys"
fi

# Make scripts executable
chmod +x run_tests.sh
chmod +x run_server.sh
chmod +x monitor.sh

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your API keys"
echo "2. Run tests: ./run_tests.sh"
echo "3. Start server: ./run_server.sh"
echo "4. Monitor system: ./monitor.sh"
