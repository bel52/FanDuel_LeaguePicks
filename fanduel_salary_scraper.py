"""
FanDuel Salary Scraper - Reads Native FanDuel CSV Format
Automatically processes the downloaded FanDuel player list
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger
import re

async def get_fanduel_salaries() -> pd.DataFrame:
    """
    Get FanDuel salaries - reads native FanDuel CSV format
    Looks for downloaded FanDuel files and converts them automatically
    """
    
    # Try multiple possible file names from FanDuel downloads
    possible_files = [
        'data/fanduel_salaries_manual.csv',
        'data/FanDuel-NFL-*.csv',  # FanDuel's typical naming
        'data/players_list.csv',
        'data/contest_players.csv',
        'data/fd_download.csv'
    ]
    
    for file_pattern in possible_files:
        if '*' in file_pattern:
            # Handle wildcard patterns
            files = list(Path('.').glob(file_pattern))
            if files:
                # Use the most recent file
                salary_file = max(files, key=lambda f: f.stat().st_mtime)
            else:
                continue
        else:
            salary_file = Path(file_pattern)
            if not salary_file.exists():
                continue
        
        try:
            logger.info(f"📂 Reading FanDuel file: {salary_file}")
            df = pd.read_csv(salary_file)
            
            # Convert from FanDuel format to our format
            converted_df = convert_fanduel_format(df)
            
            if not converted_df.empty and len(converted_df) > 20:
                logger.info(f"✅ Successfully converted {len(converted_df)} players from FanDuel format")
                return converted_df
                
        except Exception as e:
            logger.warning(f"Error reading {salary_file}: {e}")
            continue
    
    logger.error("❌ No valid FanDuel salary files found!")
    logger.error("📝 Download player list from FanDuel and save as data/fanduel_salaries_manual.csv")
    return pd.DataFrame()

def convert_fanduel_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert FanDuel's native CSV format to our expected format
    Handles multiple possible FanDuel CSV structures
    """
    
    logger.info(f"🔄 Converting FanDuel format with columns: {list(df.columns)}")
    
    # Method 1: FanDuel's detailed export format
    if all(col in df.columns for col in ['First Name', 'Last Name', 'Position', 'Team', 'Salary']):
        logger.info("📊 Detected FanDuel detailed export format")
        
        converted = pd.DataFrame({
            'Name': df['First Name'].astype(str) + ' ' + df['Last Name'].astype(str),
            'Position': df['Position'].astype(str),
            'Team': df['Team'].astype(str),
            'Salary': pd.to_numeric(df['Salary'], errors='coerce'),
            'FPPG': pd.to_numeric(df.get('FPPG', 0), errors='coerce')
        })
        
        # Clean position names (RB/FLEX -> RB, WR/FLEX -> WR, etc.)
        converted['Position'] = converted['Position'].str.split('/').str[0]
        
        # Map FanDuel position names to standard names
        position_mapping = {
            'DST': 'DST',
            'Defense': 'DST', 
            'D/ST': 'DST',
            'K': 'K',
            'Kicker': 'K'
        }
        
        converted['Position'] = converted['Position'].replace(position_mapping)
        
    # Method 2: Simple format (Name, Position, Team, Salary)
    elif all(col in df.columns for col in ['Name', 'Position', 'Team', 'Salary']):
        logger.info("📊 Detected simple format")
        converted = df[['Name', 'Position', 'Team', 'Salary']].copy()
        converted['FPPG'] = df.get('FPPG', 0)
        
    # Method 3: Try to auto-detect columns
    else:
        logger.info("📊 Auto-detecting column structure")
        converted = auto_detect_columns(df)
    
    if converted.empty:
        logger.error("❌ Could not convert FanDuel format")
        return pd.DataFrame()
    
    # Clean and validate data
    converted = clean_salary_data(converted)
    
    # Log summary
    logger.info(f"✅ Converted to {len(converted)} players")
    logger.info(f"💰 Salary range: ${converted['Salary'].min():,} - ${converted['Salary'].max():,}")
    logger.info(f"🏈 Positions: {converted['Position'].value_counts().to_dict()}")
    
    return converted

def auto_detect_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detect column structure for various FanDuel formats
    """
    
    # Look for salary column (contains $ or has "Salary" in name)
    salary_col = None
    for col in df.columns:
        if 'salary' in col.lower() or df[col].astype(str).str.contains(r'\$', na=False).any():
            salary_col = col
            break
    
    # Look for name columns
    name_cols = []
    for col in df.columns:
        if any(word in col.lower() for word in ['name', 'player', 'first', 'last']):
            name_cols.append(col)
    
    # Look for position column
    position_col = None
    for col in df.columns:
        if 'position' in col.lower() or 'pos' in col.lower():
            position_col = col
            break
    
    # Look for team column
    team_col = None
    for col in df.columns:
        if 'team' in col.lower():
            team_col = col
            break
    
    if not salary_col or not position_col:
        logger.error("❌ Cannot auto-detect required columns")
        return pd.DataFrame()
    
    # Build the converted dataframe
    converted = pd.DataFrame()
    
    # Handle name
    if len(name_cols) >= 2:
        # Multiple name columns - combine them
        converted['Name'] = df[name_cols[0]].astype(str) + ' ' + df[name_cols[1]].astype(str)
    elif len(name_cols) == 1:
        converted['Name'] = df[name_cols[0]].astype(str)
    else:
        # Use first column as name
        converted['Name'] = df.iloc[:, 0].astype(str)
    
    converted['Position'] = df[position_col]
    converted['Team'] = df[team_col] if team_col else 'UNK'
    converted['Salary'] = pd.to_numeric(df[salary_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    converted['FPPG'] = 0
    
    return converted

def clean_salary_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the salary data
    """
    
    # Remove rows with missing critical data
    df = df.dropna(subset=['Name', 'Salary'])
    
    # Convert salary to numeric
    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
    
    # Filter to reasonable salary ranges
    df = df[(df['Salary'] >= 3000) & (df['Salary'] <= 15000)]
    
    # Clean names
    df['Name'] = df['Name'].str.strip()
    df = df[df['Name'] != '']
    
    # Clean positions
    df['Position'] = df['Position'].str.strip().str.upper()
    
    # Standardize position names
    position_standards = {
        'QUARTERBACK': 'QB',
        'RUNNING_BACK': 'RB', 
        'RUNNINGBACK': 'RB',
        'WIDE_RECEIVER': 'WR',
        'WIDERECEIVER': 'WR',
        'TIGHT_END': 'TE',
        'TIGHTEND': 'TE',
        'KICKER': 'K',
        'DEFENSE': 'DST',
        'DEFENCE': 'DST',
        'D/ST': 'DST',
        'DEF': 'DST'
    }
    
    df['Position'] = df['Position'].replace(position_standards)
    
    # Clean team names
    df['Team'] = df['Team'].str.strip().str.upper()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Name', 'Team'], keep='first')
    
    # Add source column
    df['Source'] = 'FanDuel_Native'
    
    return df

# Backward compatibility
async def get_current_week_salaries():
    return await get_fanduel_salaries()

# Test function
async def test_fanduel_conversion():
    """Test the FanDuel format conversion"""
    print("🧪 Testing FanDuel CSV conversion...")
    
    salaries = await get_fanduel_salaries()
    
    if not salaries.empty:
        print(f"✅ Success! Converted {len(salaries)} players")
        print(f"💰 Salary range: ${salaries['Salary'].min():,} - ${salaries['Salary'].max():,}")
        print(f"🏈 Positions: {salaries['Position'].value_counts().to_dict()}")
        
        print("\n📋 Sample converted data:")
        print(salaries.head(10)[['Name', 'Position', 'Team', 'Salary']].to_string(index=False))
    else:
        print("❌ Conversion failed")
        print("📝 Make sure you have a FanDuel CSV file in the data/ directory")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_fanduel_conversion())
