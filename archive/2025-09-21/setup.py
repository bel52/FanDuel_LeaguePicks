#!/usr/bin/env python3
"""
Setup script for FanDuel DFS Optimizer
Handles environment setup and dependency installation
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Setup the development environment"""
    print("🚀 Setting up FanDuel DFS Optimizer...")
    
    # Create necessary directories
    dirs = ['data', 'cache', 'logs']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/ directory")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create .env file if it doesn't exist
    if not Path('.env').exists():
        print("\n📝 Creating .env file...")
        with open('.env.example', 'r') as example:
            with open('.env', 'w') as env_file:
                env_file.write(example.read())
        print("✅ Created .env file (please add your API keys)")
    
    # Initialize database
    print("\n🗄️ Initializing database...")
    from database import db
    print("✅ Database initialized")
    
    # Check for optional dependencies
    print("\n🔍 Checking optional dependencies...")
    try:
        import openai
        print("✅ OpenAI library installed")
    except ImportError:
        print("ℹ️ OpenAI library not found (AI features limited)")
    
    try:
        import redis
        print("✅ Redis library installed")
    except ImportError:
        print("ℹ️ Redis not found (caching limited)")
    
    print("\n✨ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys (optional)")
    print("2. Run: python main.py --help")
    print("3. For single optimization: python main.py --type gpp --lineups 20")
    print("4. For continuous mode: python main.py --mode continuous")

if __name__ == "__main__":
    setup_environment()
