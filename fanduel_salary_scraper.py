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
        logger.info(f"Expected file structure: Id,Position,First Name,Nickname,Last Name,FPPG,Played,Salary,Game,Team,Opponent,Injury Indicator,Injury Details,Tier,,,Roster Position")
        return []
    
    logger.info(f"📂 Reading FanDuel file: {file_path}")
    
    try:
        # Read the CSV file with proper handling for the actual FanDuel format
        df = pd.read_csv(file_path)
        
        # Log the structure for debugging
        logger.info(f"🔍 CSV columns: {df.columns.tolist()}")
        logger.info(f"🔍 Total rows: {len(df)}")
        
        # Convert to our standard format
        converted = convert_fanduel_format(df)
        
        logger.info(f"✅ Successfully converted {len(converted)} players from FanDuel format")
        return converted
        
    except Exception as e:
        logger.error(f"Error reading FanDuel salary data: {e}")
        return []

def convert_fanduel_format(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert FanDuel CSV format to our standard player format with CONSERVATIVE injury handling"""
    
    logger.info(f"🔄 Converting FanDuel format with {len(df)} rows")
    
    # The uploaded file has these exact columns:
    # Id,Position,First Name,Nickname,Last Name,FPPG,Played,Salary,Game,Team,Opponent,Injury Indicator,Injury Details,Tier,,,Roster Position
    
    expected_columns = ['Id', 'Position', 'First Name', 'Last Name', 'FPPG', 'Salary', 'Game', 'Team', 'Opponent']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"❌ Missing required columns: {missing_columns}")
        logger.error(f"❌ Available columns: {df.columns.tolist()}")
        return []
    
    # Build player name from First Name + Last Name (ignore Nickname)
    df['full_name'] = (df['First Name'].fillna('') + ' ' + df['Last Name'].fillna('')).str.strip()
    
    # Clean and convert data
    converted_data = []
    
    for idx, row in df.iterrows():
        try:
            # Handle NaN values properly
            fppg_val = row['FPPG']
            if pd.isna(fppg_val):
                fppg_val = 0.0
            else:
                fppg_val = float(fppg_val)
            
            # Clean injury status - CONSERVATIVE handling
            injury_indicator = str(row.get('Injury Indicator', '')).strip()
            injury_details = str(row.get('Injury Details', '')).strip()
            
            # Combine injury info - only include meaningful data
            injury_status = ''
            if injury_indicator and injury_indicator.upper() not in ['NAN', 'NONE', '', 'NULL']:
                injury_status = injury_indicator
                if injury_details and injury_details.upper() not in ['NAN', 'NONE', '', 'NULL']:
                    injury_status += f" - {injury_details}"
            
            player = {
                'id': str(row['Id']),
                'name': row['full_name'],
                'position': str(row['Position']).strip(),  # Use exact Position from file (RB, WR, etc.)
                'team': str(row['Team']).strip().upper(),
                'opponent': str(row['Opponent']).strip().upper(),
                'salary': int(float(row['Salary'])),
                'projected_points': fppg_val,  # FPPG = Fantasy Points Per Game
                'injury_status': injury_status,
                'game': str(row['Game']).strip(),
                'roster_position': str(row.get('Roster Position', '')).strip(),  # RB/FLEX, WR/FLEX, etc.
                'fppg_source': 'real' if fppg_val > 0 else 'estimated'
            }
            
            # Validate required fields - CONSERVATIVE validation
            if not player['name'] or len(player['name'].strip()) < 2:
                logger.warning(f"Skipping player with invalid name at row {idx}")
                continue
                
            if player['salary'] <= 0:
                logger.warning(f"Skipping player with invalid salary at row {idx}: {player['name']}")
                continue
                
            converted_data.append(player)
            
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            continue
    
    logger.info(f"✅ Converted {len(converted_data)} valid players")
    
    # Validation summary
    if converted_data:
        salary_range = (min(p['salary'] for p in converted_data), max(p['salary'] for p in converted_data))
        logger.info(f"💰 Salary range: ${salary_range[0]:,} - ${salary_range[1]:,}")
        
        # Position breakdown
        positions = {}
        for player in converted_data:
            pos = player['position']
            positions[pos] = positions.get(pos, 0) + 1
        logger.info(f"🏈 Position breakdown: {positions}")
        
        # Injury breakdown - CONSERVATIVE reporting
        injured_count = sum(1 for p in converted_data if p['injury_status'])
        ir_count = sum(1 for p in converted_data if 'IR' in p.get('injury_status', '').upper())
        logger.info(f"🏥 Injury reports: {injured_count} players with injury status ({ir_count} on IR)")
        
        # Team breakdown
        teams = {}
        for player in converted_data:
            team = player['team']
            teams[team] = teams.get(team, 0) + 1
        logger.info(f"🏟️ Team breakdown: {dict(sorted(teams.items()))}")
        
        # FPPG source breakdown
        real_fppg_count = sum(1 for p in converted_data if p['fppg_source'] == 'real')
        estimated_count = len(converted_data) - real_fppg_count
        logger.info(f"📊 Projection sources: {real_fppg_count} real FPPG, {estimated_count} estimated")
    
    return converted_data

# Synchronous wrapper for compatibility
def get_fanduel_salaries_sync(file_path: str = "data/fanduel_salaries_manual.csv") -> List[Dict[str, Any]]:
    """Synchronous wrapper for get_fanduel_salaries"""
    return asyncio.run(get_fanduel_salaries(file_path))
