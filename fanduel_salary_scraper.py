import pandas as pd
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

async def get_fanduel_salaries(file_path: str = "data/fanduel_salaries_manual.csv") -> List[Dict[str, Any]]:
    """Load FanDuel salary data from manually downloaded CSV file"""
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"FanDuel salary file not found: {file_path}")
        return []
    
    logger.info(f"📂 Reading FanDuel file: {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert to our standard format
        converted = convert_fanduel_format(df)
        
        logger.info(f"✅ Successfully converted {len(converted)} players from FanDuel format")
        return converted
        
    except Exception as e:
        logger.error(f"Error reading FanDuel salary data: {e}")
        return []

def convert_fanduel_format(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert FanDuel CSV format to our standard player format"""
    
    logger.info(f"🔄 Converting FanDuel format with columns: {df.columns.tolist()}")
    
    # Handle different FanDuel export formats
    if 'First Name' in df.columns and 'Last Name' in df.columns:
        logger.info("📊 Detected FanDuel detailed export format")
        
        # Standard detailed export format
        converted = pd.DataFrame({
            'id': df['Id'].astype(str),
            'name': (df['First Name'].fillna('') + ' ' + df['Last Name'].fillna('')).str.strip(),
            'position': df['Position'],
            'team': df['Team'],
            'opponent': df['Opponent'],
            'salary': df['Salary'],
            'projected_points': df['FPPG'].fillna(0),
            'injury_status': df.get('Injury Indicator', '').fillna(''),
            'injury_details': df.get('Injury Details', '').fillna(''),
            'game': df['Game']
        })
        
        # Clean position names (RB/FLEX -> RB, WR/FLEX -> WR, etc.)
        converted['position'] = converted['position'].str.split('/').str[0]
        
        # REMOVED: Position mapping that was causing D->DST conversion
        # Keep original positions as they are in FanDuel data
        
        converted = converted.fillna(0)
        
        # Convert to list of dictionaries
        players = []
        for _, row in converted.iterrows():
            player = {
                'id': str(row['id']),
                'name': row['name'],
                'position': row['position'],  # Keep original: D stays D
                'team': row['team'],
                'opponent': row['opponent'],
                'salary': int(row['salary']),
                'projected_points': float(row['projected_points']),
                'injury_status': row['injury_status'],
                'injury_details': row['injury_details'],
                'game': row['game']
            }
            players.append(player)
            
        logger.info(f"✅ Converted to {len(players)} players")
        logger.info(f"💰 Salary range: ${min(p['salary'] for p in players):,} - ${max(p['salary'] for p in players):,}")
        
        # Position breakdown for verification
        positions = {}
        for p in players:
            pos = p['position']
            positions[pos] = positions.get(pos, 0) + 1
        logger.info(f"🏈 Positions: {positions}")
        
        return players
        
    else:
        logger.error("❌ Unknown FanDuel export format")
        return []

# Synchronous wrapper for compatibility
def get_fanduel_salaries_sync(file_path: str = "data/fanduel_salaries_manual.csv") -> List[Dict[str, Any]]:
    """Synchronous wrapper for get_fanduel_salaries"""
    return asyncio.run(get_fanduel_salaries(file_path))
