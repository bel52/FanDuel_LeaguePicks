# Create this as: debug_injury_filter.py
import pandas as pd
from pathlib import Path

def test_injury_filtering():
    csv_path = Path("data/fanduel_salaries_manual.csv")
    df = pd.read_csv(csv_path)
    
    # Test the current logic on our problem players
    problem_players = ['Nabers', 'Hampton']
    
    for player_name in problem_players:
        matches = df[df['Last Name'].str.contains(player_name, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            
            # Convert to the format the code expects
            player_data = {
                'name': f"{row.get('First Name', '')} {row.get('Last Name', '')}".strip(),
                'injury_indicator': row.get('Injury Indicator', ''),
                'injury_details': row.get('Injury Details', ''),
                'salary': row.get('Salary', 0),
                'projected_points': row.get('FPPG', 0)
            }
            
            print(f"\n🔍 Testing {player_data['name']}:")
            print(f"   injury_indicator: '{player_data['injury_indicator']}'")
            print(f"   injury_details: '{player_data['injury_details']}'")
            
            # Test current logic
            injury_status = str(player_data.get('injury_indicator', '')).strip().upper()
            print(f"   Processed status: '{injury_status}'")
            print(f"   Should be filtered: {'IR' in injury_status}")

if __name__ == "__main__":
    test_injury_filtering()
