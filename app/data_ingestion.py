import os
import re
import pandas as pd
import logging
from typing import Tuple, List, Optional, Dict

logger = logging.getLogger(__name__)

# Column mapping for common variations
COLUMN_MAPPING = {
    'PLAYER': 'PLAYER NAME',
    'NAME': 'PLAYER NAME',
    'Player': 'PLAYER NAME',
    'POSITION': 'POS',
    'Position': 'POS',
    'SAL': 'SALARY',
    'PRICE': 'SALARY',
    'Salary': 'SALARY',
    'PROJ': 'PROJ PTS',
    'PROJECTION': 'PROJ PTS',
    'FPTS': 'PROJ PTS',
    'Proj Pts': 'PROJ PTS',
    'PROJ ROSTER %': 'OWN_PCT',
    'Own %': 'OWN_PCT',
    'Ownership': 'OWN_PCT',
    'OPPONENT': 'OPP',
    'Opponent': 'OPP',
    'MATCHUP': 'OPP',
    'Opp': 'OPP',
    'Team': 'TEAM',
    'Tm': 'TEAM'
}

def _clean_salary(val):
    """Clean salary values by removing $ and commas"""
    if pd.isna(val):
        return 0
    s = str(val).replace('$', '').replace(',', '').strip()
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0

def _clean_projection(val):
    """Clean projection values"""
    if pd.isna(val):
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def _clean_ownership(val):
    """Clean ownership percentage values"""
    if pd.isna(val):
        return None
    s = str(val).strip().replace('%', '')
    
    # Handle ranges like "15-20" by taking the midpoint
    if '-' in s:
        try:
            parts = s.split('-')
            if len(parts) == 2:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low + high) / 2
        except:
            pass
    
    # Handle single values
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def _extract_team_and_position(name_str: str) -> Tuple[str, str, str]:
    """
    Extract player name, team, and position from formats like:
    'Josh Allen (BUF - QB)' -> ('Josh Allen', 'BUF', 'QB')
    """
    # Look for pattern like (TEAM - POS) or (TEAM-POS)
    match = re.search(r'\(([^)]+)\)', name_str)
    if match:
        inside = match.group(1)
        # Split on dash or hyphen
        parts = re.split(r'\s*[-–]\s*', inside)
        if len(parts) >= 2:
            team = parts[0].strip().upper()
            pos = parts[-1].strip().upper()
            clean_name = name_str.split('(')[0].strip()
            return clean_name, team, pos
    
    # Fallback - return original name
    return name_str.strip(), "", ""

def load_data_from_input_dir() -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load and parse player data from CSV files in the input directory.
    Returns (DataFrame, warnings_list)
    """
    input_dir = "/app/data/input"
    warnings = []
    all_players = []
    
    if not os.path.exists(input_dir):
        warnings.append(f"ERROR: Input directory not found: {input_dir}")
        return None, warnings
    
    # Expected files
    files_to_process = [
        ("qb.csv", "QB"),
        ("rb.csv", "RB"), 
        ("wr.csv", "WR"),
        ("te.csv", "TE"),
        ("dst.csv", "DST")
    ]
    
    for filename, expected_pos in files_to_process:
        file_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(file_path):
            warnings.append(f"WARNING: Missing file: {filename}")
            continue
        
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            if df.empty:
                warnings.append(f"WARNING: Empty file: {filename}")
                continue
            
            # Normalize column names
            df.columns = df.columns.str.strip()
            df = df.rename(columns=COLUMN_MAPPING)
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Get player name
                    name_raw = row.get('PLAYER NAME', '')
                    if not name_raw or pd.isna(name_raw):
                        continue
                    
                    # Extract name, team, position
                    clean_name, team, pos = _extract_team_and_position(str(name_raw))
                    
                    # Use extracted position or expected position
                    final_pos = pos if pos else expected_pos
                    
                    # Get other fields
                    salary = _clean_salary(row.get('SALARY', 0))
                    proj_pts = _clean_projection(row.get('PROJ PTS', 0))
                    opponent = str(row.get('OPP', '')).replace('@', '').replace('vs', '').strip().upper()
                    own_pct = _clean_ownership(row.get('OWN_PCT'))
                    
                    # Validate required fields
                    if not clean_name or salary <= 0 or proj_pts <= 0:
                        continue
                    
                    # Add player
                    all_players.append({
                        'PLAYER NAME': clean_name,
                        'POS': final_pos,
                        'TEAM': team.upper() if team else '',
                        'OPP': opponent,
                        'SALARY': salary,
                        'PROJ PTS': proj_pts,
                        'OWN_PCT': own_pct
                    })
                
                except Exception as e:
                    warnings.append(f"ERROR processing row in {filename}: {e}")
                    continue
            
            warnings.append(f"INFO: Successfully processed {filename}")
            
        except Exception as e:
            warnings.append(f"ERROR: Failed to process {filename}: {e}")
            continue
    
    if not all_players:
        warnings.append("ERROR: No valid player data found")
        return None, warnings
    
    # Create DataFrame
    final_df = pd.DataFrame(all_players)
    
    # Remove duplicates (same player name and team)
    initial_count = len(final_df)
    final_df = final_df.drop_duplicates(subset=['PLAYER NAME', 'TEAM'], keep='first')
    
    if len(final_df) < initial_count:
        warnings.append(f"INFO: Removed {initial_count - len(final_df)} duplicate players")
    
    # Basic validation
    final_df = final_df[
        (final_df['SALARY'] >= 3000) & 
        (final_df['SALARY'] <= 15000) &
        (final_df['PROJ PTS'] > 0)
    ]
    
    warnings.append(f"SUCCESS: Loaded {len(final_df)} players total")
    
    # Show position breakdown
    if not final_df.empty:
        pos_counts = final_df['POS'].value_counts()
        warnings.append(f"INFO: Position breakdown: {pos_counts.to_dict()}")
    
    return final_df, warnings

def create_sample_data():
    """Create sample data files for testing"""
    input_dir = "/app/data/input"
    os.makedirs(input_dir, exist_ok=True)
    
    # Sample QB data
    qb_data = """PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Josh Allen (BUF - QB),BUF,@MIA,22.5,8500,15-20%
Patrick Mahomes (KC - QB),KC,LV,21.8,8300,12-18%
Jalen Hurts (PHI - QB),PHI,NYG,21.2,8200,10-15%
Lamar Jackson (BAL - QB),BAL,CLE,20.5,8000,8-12%
Dak Prescott (DAL - QB),DAL,@WAS,19.8,7700,6-10%"""
    
    # Sample RB data  
    rb_data = """PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Christian McCaffrey (SF - RB),SF,SEA,18.5,9000,25-30%
Austin Ekeler (LAC - RB),LAC,@DEN,16.2,7500,18-22%
Tony Pollard (DAL - RB),DAL,@WAS,15.8,7200,15-20%
Bijan Robinson (ATL - RB),ATL,TB,15.5,7000,12-18%
Jonathan Taylor (IND - RB),IND,HOU,14.8,6800,10-15%
Najee Harris (PIT - RB),PIT,@CIN,13.5,6500,8-12%"""
    
    # Sample WR data
    wr_data = """PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Tyreek Hill (MIA - WR),MIA,BUF,17.2,9200,20-25%
CeeDee Lamb (DAL - WR),DAL,@WAS,16.8,8800,18-22%
Justin Jefferson (MIN - WR),MIN,@GB,16.5,8700,15-20%
A.J. Brown (PHI - WR),PHI,NYG,15.8,8400,12-18%
Stefon Diggs (BUF - WR),BUF,@MIA,15.2,8200,10-15%
Davante Adams (LV - WR),LV,@KC,14.5,7900,8-12%
Chris Olave (NO - WR),NO,@CAR,13.8,7500,6-10%"""
    
    # Sample TE data
    te_data = """PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
Travis Kelce (KC - TE),KC,LV,14.5,7800,15-20%
T.J. Hockenson (MIN - TE),MIN,@GB,11.2,6200,10-15%
Mark Andrews (BAL - TE),BAL,CLE,10.8,6000,8-12%
Dallas Goedert (PHI - TE),PHI,NYG,10.5,5800,6-10%
George Kittle (SF - TE),SF,SEA,10.2,5700,5-8%"""
    
    # Sample DST data
    dst_data = """PLAYER NAME,TEAM,OPP,PROJ PTS,SALARY,PROJ ROSTER %
San Francisco 49ers (SF - DST),SF,SEA,9.5,5000,12-18%
Buffalo Bills (BUF - DST),BUF,@MIA,9.2,4800,10-15%
Dallas Cowboys (DAL - DST),DAL,@WAS,8.8,4600,8-12%
Baltimore Ravens (BAL - DST),BAL,CLE,8.5,4500,6-10%
Philadelphia Eagles (PHI - DST),PHI,NYG,8.2,4400,5-8%"""
    
    # Write files
    files_data = [
        ("qb.csv", qb_data),
        ("rb.csv", rb_data),
        ("wr.csv", wr_data),
        ("te.csv", te_data),
        ("dst.csv", dst_data)
    ]
    
    for filename, data in files_data:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'w') as f:
            f.write(data)
    
    logger.info("Sample data created successfully")
