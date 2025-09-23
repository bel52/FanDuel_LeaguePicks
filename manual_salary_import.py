"""
Manual FanDuel Salary Import - Most Reliable Method
You copy/paste or download salary data directly from FanDuel
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

def import_manual_salaries(csv_file: str = None) -> pd.DataFrame:
    """Import manually obtained FanDuel salaries"""
    
    if csv_file and Path(csv_file).exists():
        # Load from provided CSV
        df = pd.read_csv(csv_file)
        logger.info(f"✅ Loaded {len(df)} players from {csv_file}")
        return df
    
    # Check for common manual files
    manual_files = [
        'data/fanduel_salaries_manual.csv',
        'data/fd_salaries.csv',
        'data/salaries.csv',
        f'data/fd_week_{datetime.now().strftime("%Y%m%d")}.csv'
    ]
    
    for file_path in manual_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            logger.info(f"✅ Found manual salary file: {file_path} ({len(df)} players)")
            return df
    
    logger.error("❌ No manual salary files found!")
    print("\n" + "="*60)
    print("📋 HOW TO GET FANDUEL SALARIES MANUALLY:")
    print("="*60)
    print("1. Go to https://www.fanduel.com/games/nfl")
    print("2. Click on any Main Slate contest")
    print("3. Copy the player list with salaries")
    print("4. Create a CSV file: data/fanduel_salaries_manual.csv")
    print("5. Format: Name,Position,Team,Salary,FPPG")
    print("6. Example:")
    print("   Josh Allen,QB,BUF,8800,22.1")
    print("   Saquon Barkley,RB,PHI,8600,18.5")
    print("="*60)
    
    return pd.DataFrame()

# Create a template CSV for manual entry
def create_template():
    """Create a template CSV for manual salary entry"""
    template_data = [
        {'Name': 'Josh Allen', 'Position': 'QB', 'Team': 'BUF', 'Salary': 8800, 'FPPG': 22.1},
        {'Name': 'Lamar Jackson', 'Position': 'QB', 'Team': 'BAL', 'Salary': 8600, 'FPPG': 21.8},
        {'Name': 'Saquon Barkley', 'Position': 'RB', 'Team': 'PHI', 'Salary': 8600, 'FPPG': 18.5},
        # Add more template entries...
    ]
    
    df = pd.DataFrame(template_data)
    template_file = 'data/fanduel_salary_template.csv'
    df.to_csv(template_file, index=False)
    logger.info(f"📋 Created salary template: {template_file}")
    print(f"📋 Edit {template_file} with real FanDuel salaries")
    
    return template_file

if __name__ == "__main__":
    create_template()
