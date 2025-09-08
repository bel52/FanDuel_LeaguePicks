# install_services.sh - Install system services
#!/bin/bash

echo "⚙️  Installing DFS system services..."

# Create systemd service file
sudo tee /etc/systemd/system/dfs-server.service > /dev/null << EOF
[Unit]
Description=DFS Optimization Server
After=network.target

[Service]
Type=exec
User=$USER
WorkingDirectory=$PWD
ExecStart=$PWD/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10
Environment=PATH=$PWD/venv/bin
EnvironmentFile=$PWD/.env

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable dfs-server
sudo systemctl start dfs-server

echo "✅ DFS server service installed and started"
echo "   Status: sudo systemctl status dfs-server"
echo "   Logs: sudo journalctl -u dfs-server -f"

# Create logrotate configuration
sudo tee /etc/logrotate.d/dfs-server > /dev/null << EOF
$PWD/app/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 $USER $USER
}
EOF
