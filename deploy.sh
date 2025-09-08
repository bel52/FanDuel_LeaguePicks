# deploy.sh - Production deployment script
#!/bin/bash

echo "🚀 Deploying DFS System to Production..."

# Pull latest changes
git pull origin main

# Activate virtual environment
source venv/bin/activate

# Update dependencies
pip install -r requirements.txt

# Run tests before deployment
echo "🧪 Running pre-deployment tests..."
python3 cli_tester.py config
if [ $? -ne 0 ]; then
    echo "❌ Pre-deployment tests failed. Aborting deployment."
    exit 1
fi

# Backup database
if [ -f app/data/dfs.db ]; then
    cp app/data/dfs.db app/data/dfs.db.backup.$(date +%Y%m%d_%H%M%S)
    echo "💾 Database backed up"
fi

# Set production environment
export DEBUG=False
export LOG_LEVEL=WARNING

# Restart server with supervisor or systemd
if command -v supervisorctl &> /dev/null; then
    supervisorctl restart dfs-server
    echo "🔄 Server restarted with supervisor"
elif command -v systemctl &> /dev/null; then
    systemctl restart dfs-server
    echo "🔄 Server restarted with systemd"
else
    echo "⚠️  Manual server restart required"
fi

echo "✅ Deployment completed!"
