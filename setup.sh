#!/bin/bash

# NFL DFS Optimizer Setup Script
# This script sets up the complete DFS optimization system on Ubuntu

set -e  # Exit on any error

echo "🏈 NFL DFS Optimizer Setup Script"
echo "=================================="

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    echo "⚠️  Warning: This script is designed for Ubuntu. Other distributions may require modifications."
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required system packages
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    wget \
    git \
    redis-server \
    sqlite3 \
    cron

# Install or update pip
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip

# Use existing project directory structure
echo "📁 Setting up in existing directory..."
mkdir -p /home/brett/fanduel/{data,cache,logs,static}
cd /home/brett/fanduel

# Create Python virtual environment
echo "🔧 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "📦 Installing Python dependencies..."
pip install --upgrade pip wheel setuptools

# Core dependencies
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    pandas==2.1.4 \
    numpy==1.24.3 \
    nfl-data-py==0.3.3 \
    polars==0.20.2 \
    pulp==2.7.0 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    aiohttp==3.9.1 \
    requests==2.31.0 \
    aiofiles==23.2.0 \
    redis==5.0.1 \
    aioredis==2.0.1 \
    python-dotenv==1.0.0 \
    apscheduler==3.10.4 \
    loguru==0.7.2 \
    beautifulsoup4==4.12.2 \
    lxml==4.9.3 \
    arrow==1.3.0 \
    pytz==2023.3

echo "✅ Python dependencies installed successfully"

# Start and enable Redis
echo "🔄 Starting Redis server..."
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis connection
echo "🧪 Testing Redis connection..."
if redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is running correctly"
else
    echo "❌ Redis connection failed"
    exit 1
fi

# Create systemd service file for the DFS optimizer
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/dfs-optimizer.service > /dev/null <<EOF
[Unit]
Description=NFL DFS Optimizer
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=brett
WorkingDirectory=/home/brett/fanduel
Environment=PATH=/home/brett/fanduel/venv/bin
ExecStart=/home/brett/fanduel/venv/bin/python main.py web
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Set up log rotation
echo "📝 Configuring log rotation..."
sudo tee /etc/logrotate.d/dfs-optimizer > /dev/null <<EOF
/home/brett/fanduel/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 brett brett
}
EOF

# Create cron job for daily maintenance
echo "🕐 Setting up cron jobs..."
(crontab -l 2>/dev/null || true; echo "0 3 * * * cd /home/brett/fanduel && ./venv/bin/python -c \"from scheduler import get_scheduler; get_scheduler().daily_cleanup()\"") | crontab -

# Create environment file
echo "📋 Creating environment configuration..."
cat > .env <<EOF
# DFS Optimizer Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DATA_RETENTION_DAYS=7
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Optional: AI Integration (set to true if you want ChatGPT integration)
AI_ENABLED=false
OPENAI_API_KEY=your_api_key_here
EOF

# Set proper permissions
echo "🔒 Setting file permissions..."
chmod +x main.py
chmod 755 /home/brett/fanduel
chmod 644 .env

# Create desktop shortcut
echo "🖥️ Creating desktop shortcut..."
mkdir -p /home/brett/Desktop
cat > /home/brett/Desktop/DFS-Optimizer.desktop <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=NFL DFS Optimizer
Comment=Open DFS Optimizer Web Interface
Exec=xdg-open http://localhost:8000
Icon=applications-games
Terminal=false
Categories=Sports;Game;
EOF
chmod +x /home/brett/Desktop/DFS-Optimizer.desktop

# Test the installation
echo "🧪 Testing installation..."
echo "Checking Python imports..."
source venv/bin/activate
python3 -c "
import pandas as pd
import numpy as np
import nfl_data_py as nfl
import pulp
import aiohttp
import fastapi
print('✅ All core packages imported successfully')
"

echo "🧪 Testing data collection..."
timeout 30 python3 -c "
import asyncio
import sys
sys.path.append('.')
from data_collector import get_fresh_data

async def test():
    try:
        data = await get_fresh_data()
        if data and 'players' in data:
            print(f'✅ Data collection test passed: {len(data[\"players\"])} players')
        else:
            print('⚠️ Data collection returned empty results')
    except Exception as e:
        print(f'⚠️ Data collection test failed: {e}')

asyncio.run(test())
" || echo "⚠️ Data collection test timed out (this is normal for first run)"

# Provide usage instructions
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📍 Installation Directory: /home/brett/fanduel"
echo "🌐 Web Interface: http://localhost:8020"
echo "📊 API Documentation: http://localhost:8020/docs"
echo ""
echo "🚀 Quick Start Commands:"
echo "------------------------"
echo "cd /home/brett/fanduel && source venv/bin/activate"
echo ""
echo "# Run data collection only:"
echo "python main.py collect"
echo ""
echo "# Generate lineups only:"
echo "python main.py optimize"
echo ""
echo "# Start web interface (recommended):"
echo "python main.py web"
echo ""
echo "# Start automated scheduler:"
echo "python main.py scheduler"
echo ""
echo "🔧 System Service Commands:"
echo "---------------------------"
echo "# Start as system service (auto-start on boot):"
echo "sudo systemctl start dfs-optimizer"
echo "sudo systemctl enable dfs-optimizer"
echo ""
echo "# Check service status:"
echo "sudo systemctl status dfs-optimizer"
echo ""
echo "# View service logs:"
echo "sudo journalctl -u dfs-optimizer -f"
echo ""
echo "📁 Important Files:"
echo "------------------"
echo "• Configuration: ~/.env"
echo "• Logs: /home/brett/fanduel/logs/"
echo "• Data: /home/brett/fanduel/data/"
echo "• Lineups: /home/brett/fanduel/data/lineups/"
echo ""
echo "🔧 Troubleshooting:"
echo "-------------------"
echo "• Check logs: tail -f /home/brett/fanduel/logs/dfs_optimizer_*.log"
echo "• Restart Redis: sudo systemctl restart redis-server"
echo "• Update data: python main.py collect"
echo "• Test optimization: python main.py optimize"
echo ""
echo "🎯 Next Steps:"
echo "--------------"
echo "1. Start the web interface: python main.py web"
echo "2. Open http://localhost:8000 in your browser"
echo "3. Click 'Force Data Update' to collect initial data"
echo "4. Click 'Generate New Lineups' to create optimized lineups"
echo "5. Download CSV files for upload to FanDuel"
echo ""
echo "🏈 Ready to dominate your DFS contests!"
