#!/usr/bin/env python3
"""
Debug script to check what players we have by position
"""
import asyncio
import json
from pathlib import Path
import pandas as pd
from collections import Counter

async def debug_players():
    # Find the most recent data file
    data_dir = Path("data")
    data_files = list(data_dir.glob("nfl_data_*.json"))
    
    if not data_files:
        print("No data files found!")
        return
    
    # Get the most recent file
    latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
    print(f"Reading from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    players = data.get('players', [])
    print(f"\nTotal players: {len(players)}")
    
    if not players:
        print("No players found in data!")
        return
    
    # Check positions
    positions = [p.get('position', 'UNKNOWN') for p in players]
    position_counts = Counter(positions)
    
    print("\nPlayers by position:")
    for pos, count in sorted(position_counts.items()):
        print(f"  {pos}: {count}")
    
    # Check FanDuel requirements
    fanduel_requirements = {
        'QB': 1,
        'RB': 2, 
        'WR': 3,
        'TE': 1,
        'FLEX': 1,  # Can be RB/WR/TE
        'DST': 1    # Defense
    }
    
    print(f"\nFanDuel Requirements vs Available:")
    flex_eligible = position_counts.get('RB', 0) + position_counts.get('WR', 0) + position_counts.get('TE', 0)
    
    for pos, needed in fanduel_requirements.items():
        if pos == 'FLEX':
            available = flex_eligible - 2 - 3 - 1  # Already used RBs, WRs, TEs
            print(f"  {pos}: need {needed}, available {max(0, available)} (RB/WR/TE eligible)")
        elif pos == 'DST':
            # Count DEF as DST
            available = position_counts.get('DST', 0) + position_counts.get('DEF', 0) + position_counts.get('D/ST', 0)
            print(f"  {pos}: need {needed}, available {available}")
        else:
            available = position_counts.get(pos, 0)
            feasible = "✅" if available >= needed else "❌"
            print(f"  {pos}: need {needed}, available {available} {feasible}")
    
    # Show sample players
    print(f"\nSample players:")
    df = pd.DataFrame(players)
    if not df.empty:
        for pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF', 'DST']:
            pos_players = df[df['position'] == pos].head(3)
            if not pos_players.empty:
                print(f"\n{pos} players:")
                for _, player in pos_players.iterrows():
                    print(f"  {player.get('player_name', 'Unknown')} ({player.get('team', 'UNK')}) - ${player.get('salary', 0):,} - {player.get('projection', 0):.1f} pts")

if __name__ == "__main__":
    asyncio.run(debug_players())
