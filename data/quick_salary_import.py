#!/usr/bin/env python3
"""
Quick FanDuel Salary Import - 5 minutes weekly
Simple copy/paste from FanDuel to get exact salaries
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

def quick_import_salaries():
    """Quick salary import with validation"""
    
    # Check for manual salary file
    salary_file = 'data/fanduel_salaries_manual.csv'
    
    if not Path(salary_file).exists():
        print("\n" + "="*60)
        print("📋 QUICK FANDUEL SALARY SETUP")
        print("="*60)
        print("1. Go to: https://www.fanduel.com/games/nfl")
        print("2. Click on any 'Main Slate' contest (Sunday games)")
        print("3. You'll see a player list with salaries")
        print("4. Copy about 20-30 players to get started")
        print("5. Format each line as: Name,Position,Team,Salary")
        print("\nExample format:")
        print("Josh Allen,QB,BUF,8800")
        print("Saquon Barkley,RB,PHI,8600")
        print("Tyreek Hill,WR,MIA,8400")
        print("="*60)
        
        # Create template
        create_quick_template()
        print(f"\n📝 Edit this file: {salary_file}")
        print("💡 TIP: You only need ~50-100 players for main slate")
        return False
    
    # Load and validate
    try:
        df = pd.read_csv(salary_file)
        
        # Basic validation
        required_cols = ['Name', 'Position', 'Team', 'Salary']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print("📝 Required format: Name,Position,Team,Salary,FPPG")
            return False
        
        # Clean up data
        df = df.dropna(subset=['Name', 'Salary'])
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        df = df[df['Salary'] > 0]
        
        if len(df) < 20:
            print(f"⚠️ Only {len(df)} valid players found. Need at least 20 for lineups.")
            print("📝 Add more players to the CSV file")
            return False
        
        # Show summary
        print(f"✅ Loaded {len(df)} players from FanDuel")
        print(f"💰 Salary range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
        print(f"🏈 Positions: {df['Position'].value_counts().to_dict()}")
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        backup_file = f'data/fd_salaries_backup_{timestamp}.csv'
        df.to_csv(backup_file, index=False)
        print(f"💾 Backup saved: {backup_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading salary file: {e}")
        print("📝 Check the CSV format in:", salary_file)
        return False

def create_quick_template():
    """Create a quick template with current week teams"""
    template_data = [
        # Week 3 teams playing (update weekly)
        {'Name': 'Josh Allen', 'Position': 'QB', 'Team': 'BUF', 'Salary': 8800, 'FPPG': 22.1},
        {'Name': 'Lamar Jackson', 'Position': 'QB', 'Team': 'BAL', 'Salary': 8600, 'FPPG': 21.8},
        {'Name': 'Saquon Barkley', 'Position': 'RB', 'Team': 'PHI', 'Salary': 8600, 'FPPG': 18.5},
        {'Name': 'ADD_MORE_PLAYERS_FROM_FANDUEL', 'Position': 'RB', 'Team': 'XXX', 'Salary': 0, 'FPPG': 0},
    ]
    
    df = pd.DataFrame(template_data)
    df.to_csv('data/fanduel_salaries_manual.csv', index=False)
    print("📋 Created template file: data/fanduel_salaries_manual.csv")

if __name__ == "__main__":
    print("🏈 FanDuel Quick Salary Import")
    print("-" * 30)
    
    success = quick_import_salaries()
    
    if success:
        print("\n🎯 Ready to generate lineups!")
        print("Run: python main.py web")
    else:
        print("\n📝 Complete the salary file first, then run this again.")
