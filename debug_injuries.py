# Create this as: debug_injuries.py
import pandas as pd
from pathlib import Path

def check_csv_injuries():
    csv_path = Path("data/fanduel_salaries_manual.csv")
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"📋 CSV loaded: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Check for injury-related columns
    injury_columns = [col for col in df.columns if 'injury' in col.lower() or 'indicator' in col.lower()]
    print(f"\n🚑 Injury-related columns: {injury_columns}")
    
    # Look for Nabers and Hampton specifically
    problem_players = ['Nabers', 'Hampton']
    
    for player in problem_players:
        matches = df[df['Last Name'].str.contains(player, case=False, na=False)]
        if not matches.empty:
            for _, row in matches.iterrows():
                print(f"\n🔍 {player} found:")
                print(f"   Name: {row.get('First Name', '')} {row.get('Last Name', '')}")
                print(f"   Salary: ${row.get('Salary', 0):,}")
                print(f"   Injury Indicator: '{row.get('Injury Indicator', '')}'")
                print(f"   Injury Details: '{row.get('Injury Details', '')}'")
                print(f"   FPPG: {row.get('FPPG', 0)}")
        else:
            print(f"\n❌ {player} not found in CSV")

if __name__ == "__main__":
    check_csv_injuries()
