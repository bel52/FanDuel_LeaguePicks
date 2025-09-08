# Makefile for DFS Optimization System
.PHONY: help setup install test test-quick test-full run run-dev run-prod clean monitor dashboard logs backup deploy

# Default target
help:
	@echo "🏈 FanDuel NFL DFS Optimization System"
	@echo ""
	@echo "Available commands:"
	@echo "  setup         - Initial system setup"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run quick tests"
	@echo "  test-quick    - Run configuration and data tests"
	@echo "  test-full     - Run complete test suite"
	@echo "  test-perf     - Run performance tests"
	@echo "  run           - Start development server"
	@echo "  run-dev       - Start development server with reload"
	@echo "  run-prod      - Start production server"
	@echo "  monitor       - Show system status"
	@echo "  monitor-live  - Start continuous monitoring"
	@echo "  dashboard     - Show analytics dashboard"
	@echo "  logs          - Show server logs"
	@echo "  clean         - Clean temporary files"
	@echo "  backup        - Backup data and configs"
	@echo "  deploy        - Deploy to production"
	@echo "  install-service - Install systemd service"

# Setup and installation
setup:
	@echo "🚀 Setting up DFS system..."
	@chmod +x setup.sh run_tests.sh run_server.sh monitor.sh deploy.sh
	@./setup.sh

install:
	@echo "📦 Installing dependencies..."
	@source venv/bin/activate && pip install -r requirements.txt

# Testing
test: test-quick

test-quick:
	@echo "🏃 Running quick tests..."
	@./run_tests.sh quick

test-full:
	@echo "🔬 Running full test suite..."
	@./run_tests.sh full

test-perf:
	@echo "⚡ Running performance tests..."
	@./run_tests.sh performance

test-config:
	@echo "⚙️ Testing configuration..."
	@./run_tests.sh config

test-data:
	@echo "📊 Testing data ingestion..."
	@./run_tests.sh data

test-weather:
	@echo "🌤️ Testing weather integration..."
	@./run_tests.sh weather

test-ai:
	@echo "🤖 Testing AI analysis..."
	@./run_tests.sh ai

test-opt:
	@echo "🎯 Testing optimization..."
	@./run_tests.sh optimization

test-pipeline:
	@echo "🔄 Testing full pipeline..."
	@./run_tests.sh pipeline

# Running the server
run: run-dev

run-dev:
	@echo "🔧 Starting development server..."
	@./run_server.sh

run-prod:
	@echo "🏭 Starting production server..."
	@DEBUG=False ./run_server.sh

# Monitoring and logs
monitor:
	@echo "📊 Checking system status..."
	@./monitor.sh status

monitor-live:
	@echo "🔄 Starting continuous monitoring..."
	@./monitor.sh continuous

dashboard:
	@echo "📊 Opening dashboard..."
	@./monitor.sh dashboard

logs:
	@echo "📋 Showing logs..."
	@./monitor.sh logs

# Maintenance
clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache
	@rm -rf app/cache/*
	@echo "✅ Cleanup complete"

backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@tar -czf backups/dfs-backup-$(shell date +%Y%m%d_%H%M%S).tar.gz \
		app/ .env requirements.txt \
		--exclude=app/logs/* \
		--exclude=app/cache/* \
		--exclude=app/__pycache__/*
	@echo "✅ Backup created in backups/"

# Deployment
deploy:
	@echo "🚀 Deploying to production..."
	@./deploy.sh

install-service:
	@echo "⚙️ Installing system service..."
	@chmod +x install_services.sh
	@./install_services.sh

# Development helpers
format:
	@echo "🎨 Formatting code..."
	@source venv/bin/activate && python -m black app/ --line-length 88
	@source venv/bin/activate && python -m isort app/

lint:
	@echo "🔍 Linting code..."
	@source venv/bin/activate && python -m flake8 app/ --max-line-length=88 --ignore=E203,W503

type-check:
	@echo "🔒 Type checking..."
	@source venv/bin/activate && python -m mypy app/ --ignore-missing-imports

# Quick lineup generation (for testing)
lineups:
	@echo "🎯 Generating test lineups..."
	@source venv/bin/activate && python -c "
import asyncio
import sys
sys.path.insert(0, 'app')

async def quick_lineups():
    try:
        from app.main import app
        from app.optimization_engine import OptimizationEngine
        
        optimizer = OptimizationEngine()
        
        # Sample player data
        players = [
            {'id': 1, 'name': 'Josh Allen', 'position': 'QB', 'salary': 8500, 'projected_points': 22.5},
            {'id': 2, 'name': 'Derrick Henry', 'position': 'RB', 'salary': 6800, 'projected_points': 18.2},
            {'id': 3, 'name': 'Christian McCaffrey', 'position': 'RB', 'salary': 9000, 'projected_points': 20.1},
            {'id': 4, 'name': 'Davante Adams', 'position': 'WR', 'salary': 8000, 'projected_points': 17.5},
            {'id': 5, 'name': 'Stefon Diggs', 'position': 'WR', 'salary': 7200, 'projected_points': 16.8},
            {'id': 6, 'name': 'Tyreek Hill', 'position': 'WR', 'salary': 7800, 'projected_points': 17.2},
            {'id': 7, 'name': 'Travis Kelce', 'position': 'TE', 'salary': 6500, 'projected_points': 15.3},
            {'id': 8, 'name': 'Justin Tucker', 'position': 'K', 'salary': 4800, 'projected_points': 8.5},
            {'id': 9, 'name': 'Buffalo', 'position': 'DST', 'salary': 3200, 'projected_points': 9.2},
        ]
        
        constraints = {
            'salary_cap': 50000,
            'positions': {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1}
        }
        
        lineup = await optimizer.optimize_lineup(players, constraints)
        
        if lineup:
            print('✅ Sample lineup generated:')
            total_salary = 0
            total_projection = 0
            for player in lineup:
                print(f'  {player[\"position\"]:3} | {player[\"name\"]:20} | \$$${player[\"salary\"]:5,} | {player[\"projected_points\"]:5.1f} pts')
                total_salary += player['salary']
                total_projection += player['projected_points']
            print(f'\\nTotal: \$$${total_salary:,} | {total_projection:.1f} pts')
        else:
            print('❌ Failed to generate lineup')
            
    except Exception as e:
        print(f'❌ Error: {e}')

asyncio.run(quick_lineups())
"

# Show system information
info:
	@echo "ℹ️  System Information"
	@echo "===================="
	@echo "Python: $(shell python3 --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "Virtual env: $(shell if [ -d venv ]; then echo 'Present'; else echo 'Missing - run make setup'; fi)"
	@echo "Config file: $(shell if [ -f .env ]; then echo 'Present'; else echo 'Missing - run make setup'; fi)"
	@echo "Log directory: $(shell if [ -d app/logs ]; then echo 'Present'; else echo 'Missing'; fi)"
	@echo "Data directory: $(shell if [ -d app/data ]; then echo 'Present'; else echo 'Missing'; fi)"
	@echo ""
	@echo "Server status:"
	@curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Server not running"

# Development workflow shortcuts
dev-setup: setup test-quick
	@echo "✅ Development environment ready!"

dev-test: test-quick lineups
	@echo "✅ Development testing complete!"

prod-deploy: test-full backup deploy
	@echo "✅ Production deployment complete!"

# Help for specific commands
help-test:
	@echo "🧪 Testing Commands:"
	@echo "  test-quick    - Config and data tests (~30 seconds)"
	@echo "  test-full     - All tests including AI and optimization (~5 minutes)"
	@echo "  test-perf     - Performance benchmarks"
	@echo "  test-config   - Configuration validation only"
	@echo "  test-data     - Data ingestion tests only"
	@echo "  test-weather  - Weather API tests only"
	@echo "  test-ai       - AI analysis tests only"
	@echo "  test-opt      - Optimization engine tests only"
	@echo "  test-pipeline - End-to-end pipeline test"

help-run:
	@echo "🚀 Running Commands:"
	@echo "  run           - Development server with auto-reload"
	@echo "  run-dev       - Same as 'run'"
	@echo "  run-prod      - Production server (4 workers, no reload)"

help-monitor:
	@echo "📊 Monitoring Commands:"
	@echo "  monitor       - One-time status check"
	@echo "  monitor-live  - Continuous monitoring (Ctrl+C to stop)"
	@echo "  dashboard     - Analytics dashboard"
	@echo "  logs          - Real-time log viewing"

# Quick Start Guide
# ================

# README.md content for quick reference
readme:
	@echo "# FanDuel NFL DFS Optimization System"
	@echo ""
	@echo "## Quick Start"
	@echo ""
	@echo "1. **Initial Setup**"
	@echo "   \`\`\`bash"
	@echo "   make setup"
	@echo "   # Edit .env with your API keys"
	@echo "   \`\`\`"
	@echo ""
	@echo "2. **Test System**"
	@echo "   \`\`\`bash"
	@echo "   make test-quick    # Quick validation"
	@echo "   make test-full     # Complete test suite"
	@echo "   \`\`\`"
	@echo ""
	@echo "3. **Run Server**"
	@echo "   \`\`\`bash"
	@echo "   make run           # Development mode"
	@echo "   make run-prod      # Production mode"
	@echo "   \`\`\`"
	@echo ""
	@echo "4. **Monitor System**"
	@echo "   \`\`\`bash"
	@echo "   make monitor       # Check status"
	@echo "   make dashboard     # View analytics"
	@echo "   make logs          # View logs"
	@echo "   \`\`\`"
	@echo ""
	@echo "## API Endpoints"
	@echo ""
	@echo "- Health: http://localhost:8000/health"
	@echo "- Optimize: http://localhost:8000/optimize"
	@echo "- Players: http://localhost:8000/players"
	@echo "- Weather: http://localhost:8000/weather"
	@echo "- Analytics: http://localhost:8000/analytics"
	@echo ""
	@echo "## Troubleshooting"
	@echo ""
	@echo "- **API Key Issues**: Edit .env file"
	@echo "- **Import Errors**: Run \`make install\`"
	@echo "- **Permission Errors**: Run \`chmod +x *.sh\`"
	@echo "- **Port Conflicts**: Change port in run_server.sh"
	@echo ""
	@echo "For detailed help: \`make help\`"
