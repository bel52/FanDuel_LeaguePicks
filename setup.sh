#!/bin/bash

echo "Setting up DFS Optimizer..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install playwright browsers
playwright install chromium

# Create directories
mkdir -p data
mkdir -p data/cache
mkdir -p logs

# Check for Redis
if ! command -v redis-server &> /dev/null; then
    echo "Redis not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y redis-server
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
fi

# Create systemd service (optional)
echo "Creating systemd service..."
sudo tee /etc/systemd/system/dfs-optimizer.service > /dev/null <<EOF
[Unit]
Description=DFS Optimizer Service
After=network.target redis.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "Setup complete!"
echo ""
echo "To run the optimizer:"
echo "  1. Add your OpenAI API key to .env file"
echo "  2. Download FanDuel salary CSV to data/ folder"
echo "  3. Run: source venv/bin/activate && python main.py"
echo ""
echo "To run as service:"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl start dfs-optimizer"
echo "  sudo systemctl enable dfs-optimizer"
