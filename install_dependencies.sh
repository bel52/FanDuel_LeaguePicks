#!/bin/bash

echo "🔧 Installing Missing Dependencies for DFS Optimizer"
echo "===================================================="

# Ensure we're in the right directory
cd /home/brett/fanduel

# Activate virtual environment
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

echo "📦 Installing critical missing packages..."

# Install loguru (the main missing dependency)
pip install loguru>=0.7.0

# Install/upgrade other critical packages
pip install \
    fastapi>=0.104.1 \
    uvicorn[standard]>=0.24.0 \
    pandas>=1.5.0 \
    numpy>=1.24.0 \
    nfl-data-py>=0.3.0 \
    pulp>=2.7.0 \
    scikit-learn>=1.3.0 \
    aiohttp>=3.9.0 \
    requests>=2.31.0 \
    redis>=5.0.0 \
    apscheduler>=3.10.0 \
    python-dotenv>=1.0.0 \
    beautifulsoup4>=4.12.0 \
    scipy>=1.11.0

echo "✅ Core dependencies installed"

# Test critical imports
echo "🧪 Testing critical imports..."
python3 -c "
import loguru
import fastapi
import pandas
import numpy
import nfl_data_py
import pulp
import aiohttp
import requests
print('✅ All critical packages imported successfully')
"

echo "🎯 Testing DFS-specific functionality..."
python3 -c "
import sys
sys.path.append('.')
try:
    from config import API_PORT, DATA_DIR
    from loguru import logger
    print('✅ Config and logging working')
except ImportError as e:
    print(f'❌ Import error: {e}')
    
try:
    import asyncio
    print('✅ Async functionality available')
except:
    print('❌ Async functionality failed')
"

echo ""
echo "🚀 Dependencies installation complete!"
echo ""
echo "Next steps:"
echo "1. Run: python3 main.py web"
echo "2. Open: http://localhost:8020"
echo "3. Test the system with 'Force Data Update'"
echo ""
echo "If you still get errors, run:"
echo "  pip install -r requirements.txt --force-reinstall"
