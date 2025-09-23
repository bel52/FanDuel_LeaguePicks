#!/usr/bin/env python3
"""
Quick FanDuel Salary Import - Works with Native FanDuel Format
Just download from FanDuel and it works automatically
"""
import pandas as pd
import asyncio
from pathlib import Path
from datetime import datetime
import sys

async def main():
    print("🏈 FanDuel Native Format Import")
    print("-" * 40)
    
    # Import the updated scraper
    from fanduel_salary_scraper import get_fanduel_salaries
    
    # Try to load FanDuel data
    df = await get_fanduel_salaries()
    
    if df.empty:
        print("\n" + "="*60)
        print("📋 FANDUEL WEEKLY WORKFLOW")
        print("="*60)
        print("1. Go to: https://www.fanduel.com/games/nfl")
        print("2. Click on any Main Slate contest")
        print("3. Click 'Download players list' (blue link)")
        print("4. Save the downloaded CSV as:")
        print("   ~/fanduel/data/fanduel_salaries_manual.csv")
        print("5. Run this script again")
        print("\n💡 The system now reads FanDuel's native format!")
        print("💡 No conversion needed - just download and go!")
        print("="*60)
        return False
    
    print(f"\n✅ Successfully loaded {len(df)} players")
    print(f"💰 Salary range: ${df['Salary'].min():,} - ${df['Salary'].max():,}")
    print(f"🏈 Position breakdown:")
    
    pos_counts = df['Position'].value_counts()
    for pos, count in pos_counts.items():
        print(f"   {pos}: {count} players")
    
    # Check if we have enough for lineups
    min_requirements = {'QB': 3, 'RB': 8, 'WR': 12, 'TE': 4, 'DST': 4}
    missing = []
    
    for pos, min_count in min_requirements.items():
        actual_count = pos_counts.get(pos, 0)
        if actual_count < min_count:
            missing.append(f"{pos}: need {min_count}, have {actual_count}")
    
    if missing:
        print(f"\n⚠️ May need more players:")
        for msg in missing:
            print(f"   {msg}")
        print("💡 Download from a larger contest if needed")
    
    print(f"\n🎯 Ready to generate lineups!")
    print("Run: python main.py web")
    
    return True

if __name__ == "__main__":
    asyncio.run(main())
