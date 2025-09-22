#!/usr/bin/env python3
import sys
sys.path.append('.')

def check_system():
    print("🔍 DFS System Status Check")
    print("=" * 40)
    
    # Check imports
    try:
        from loguru import logger
        from config import get_current_nfl_week, NFL_STADIUMS
        from data_collector import EnhancedDataCollector
        from optimizer import EnhancedDFSOptimizer
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check current week detection
    try:
        week = get_current_nfl_week()
        print(f"✅ Current NFL Week: {week}")
    except Exception as e:
        print(f"❌ Week detection error: {e}")
        return False
    
    # Check configuration
    try:
        stadium_count = len(NFL_STADIUMS)
        print(f"✅ Configuration loaded: {stadium_count} stadiums")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    print("\n🎯 System is ready!")
    print("Run: python3 main.py web")
    return True

if __name__ == "__main__":
    success = check_system()
    sys.exit(0 if success else 1)
