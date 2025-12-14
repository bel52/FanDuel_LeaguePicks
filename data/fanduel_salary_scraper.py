"""
FanDuel Salary Scraper - Manual Import Priority
Always tries manual file first, then falls back to scraping
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

async def get_fanduel_salaries() -> pd.DataFrame:
    """
    Get FanDuel salaries - Manual file takes priority
    """
    
    # Try manual file first (most reliable)
    manual_files = [
        'data/fanduel_salaries_manual.csv',
        'data/fd_salaries.csv',
        'data/salaries.csv'
    ]
    
    for manual_file in manual_files:
        if Path(manual_file).exists():
            try:
                df = pd.read_csv(manual_file)
                
                # Validate manual data
                if len(df) > 20 and 'Salary' in df.columns:
                    # Clean the data
                    df = df.dropna(subset=['Name', 'Salary'])
                    df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                    df = df[df['Salary'] > 0]
                    
                    if len(df) >= 20:
                        logger.info(f"✅ Using manual salary file: {manual_file} ({len(df)} players)")
                        
                        # Add source column
                        df['Source'] = 'Manual_FanDuel'
                        
                        # Ensure FPPG column exists
                        if 'FPPG' not in df.columns:
                            df['FPPG'] = 0
                        
                        return df
                        
            except Exception as e:
                logger.warning(f"Error reading manual file {manual_file}: {e}")
                continue
    
    # If no manual file, show instructions
    logger.error("❌ NO MANUAL SALARY FILE FOUND!")
    logger.error("📝 Run: python quick_salary_import.py")
    logger.error("📝 Then add FanDuel salaries to: data/fanduel_salaries_manual.csv")
    
    return pd.DataFrame()

# Backward compatibility
async def get_current_week_salaries():
    return await get_fanduel_salaries()
