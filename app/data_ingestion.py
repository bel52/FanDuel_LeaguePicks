# app/data_ingestion.py - Simplified working version
import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

def load_weekly_data() -> Optional[pd.DataFrame]:
    """Load and combine weekly player data from input directory"""
    try:
        data_dir = "data/input"
        all_data = []
        
        if not os.path.exists(data_dir):
            logger.warning(f"Directory {data_dir} does not exist")
            return None
        
        # Load each position file
        for pos in ["qb", "rb", "wr", "te", "dst"]:
            file_path = os.path.join(data_dir, f"{pos}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        # Add position hint if not in data
                        if 'POS' not in df.columns:
                            df['POS'] = pos.upper() if pos != 'dst' else 'DST'
                        all_data.append(df)
                        logger.info(f"Loaded {len(df)} players from {pos}.csv")
                except Exception as e:
                    logger.warning(f"Failed to load {pos}.csv: {e}")
        
        if not all_data:
            logger.warning("No valid data files found")
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Normalize column names and clean data
        combined_df = normalize_columns(combined_df)
        combined_df = clean_player_data(combined_df)
        
        logger.info(f"Successfully loaded {len(combined_df)} total players")
        return combined_df
        
    except Exception as e:
        logger.error(f"Error loading weekly data: {e}")
        return None

def load_data_from_input_dir() -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Load player data from input directory with warnings"""
    warnings = []
    
    try:
        df = load_weekly_data()
        
        if df is None or df.empty:
            warnings.append("No player data found in data/input/")
            # Try to create sample data as fallback
            df = create_sample_data()
            if df is not None:
                warnings.append("Using sample data - place real CSV files in data/input/")
        
        # Validate data
        if df is not None:
            validation_warnings = validate_player_data(df)
            warnings.extend(validation_warnings)
        
        return df, warnings
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        warnings.append(f"Error loading data: {str(e)}")
        return None, warnings

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected format"""
    column_mappings = {
        'Player': 'PLAYER NAME',
        'Name': 'PLAYER NAME',
        'Player Name': 'PLAYER NAME',
        'Position': 'POS',
        'Pos': 'POS',
        'Team': 'TEAM',
        'Tm': 'TEAM',
        'Opponent': 'OPP',
        'Opp': 'OPP',
        'Salary': 'SALARY',
        'FD Salary': 'SALARY',
        'FanDuel Salary': 'SALARY',
        'Projected Points': 'PROJ PTS',
        'Proj Pts': 'PROJ PTS',
        'Proj. Pts': 'PROJ PTS',
        'FPTS': 'PROJ PTS',
        'Projected Pts': 'PROJ PTS',
        'Ownership': 'OWN_PCT',
        'Own %': 'OWN_PCT',
        'Proj Roster %': 'OWN_PCT',
        'Projected Ownership %': 'OWN_PCT'
    }
    
    # Apply mappings
    df = df.rename(columns=column_mappings)
    
    # Ensure critical columns exist
    if 'PLAYER NAME' not in df.columns and 'PLAYER' in df.columns:
        df['PLAYER NAME'] = df['PLAYER']
    
    return df

def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize player data"""
    
    # Remove rows with missing critical data
    required_cols = ['PLAYER NAME', 'SALARY', 'PROJ PTS']
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]
    
    # Clean salary column
    if 'SALARY' in df.columns:
        df['SALARY'] = df['SALARY'].astype(str).str.replace('$', '').str.replace(',', '')
        df['SALARY'] = pd.to_numeric(df['SALARY'], errors='coerce').fillna(0).astype(int)
        # Filter out invalid salaries
        df = df[(df['SALARY'] >= 3000) & (df['SALARY'] <= 15000)]
    
    # Clean projections
    if 'PROJ PTS' in df.columns:
        df['PROJ PTS'] = pd.to_numeric(df['PROJ PTS'], errors='coerce').fillna(0)
        # Filter out zero projections
        df = df[df['PROJ PTS'] > 0]
    
    # Clean ownership if present
    if 'OWN_PCT' in df.columns:
        # Handle ranges like "15-20%"
        df['OWN_PCT'] = df['OWN_PCT'].astype(str).str.replace('%', '')
        
        def parse_ownership(val):
            if pd.isna(val) or val == 'nan':
                return 0
            val = str(val)
            if '-' in val:
                parts = val.split('-')
                try:
                    return (float(parts[0]) + float(parts[1])) / 2
                except:
                    return 0
            try:
                return float(val)
            except:
                return 0
        
        df['OWN_PCT'] = df['OWN_PCT'].apply(parse_ownership)
    
    # Extract team and position from player name if needed
    if 'TEAM' not in df.columns or 'POS' not in df.columns:
        import re
        
        def extract_from_name(name):
            # Pattern like "Josh Allen (BUF - QB)"
            match = re.search(r'\(([A-Z]{2,3})\s*-\s*([A-Z/]+)\)', str(name))
            if match:
                return match.group(1), match.group(2)
            return None, None
        
        if 'TEAM' not in df.columns:
            df['TEAM'] = df['PLAYER NAME'].apply(lambda x: extract_from_name(x)[0])
        
        if 'POS' not in df.columns:
            df['POS'] = df['PLAYER NAME'].apply(lambda x: extract_from_name(x)[1])
    
    # Clean position values
    if 'POS' in df.columns:
        position_map = {
            'DEF': 'DST',
            'D/ST': 'DST',
            'D': 'DST'
        }
        df['POS'] = df['POS'].replace(position_map)
        
        # Filter to valid positions only
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
        df = df[df['POS'].isin(valid_positions)]
    
    # Clean player names (remove position/team info if in name)
    if 'PLAYER NAME' in df.columns:
        df['PLAYER NAME'] = df['PLAYER NAME'].str.split('(').str[0].str.strip()
    
    # Add calculated columns
    if 'SALARY' in df.columns and 'PROJ PTS' in df.columns:
        df['VALUE'] = df['PROJ PTS'] / (df['SALARY'] / 1000)
        df['VALUE'] = df['VALUE'].round(2)
    
    return df

def validate_player_data(df: pd.DataFrame) -> List[str]:
    """Validate player data and return warnings"""
    warnings = []
    
    # Check for required columns
    required_cols = ['PLAYER NAME', 'POS', 'SALARY', 'PROJ PTS']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.append(f"Missing required columns: {missing_cols}")
    
    # Check position distribution
    if 'POS' in df.columns:
        pos_counts = df['POS'].value_counts()
        
        min_requirements = {
            'QB': 3,
            'RB': 6,
            'WR': 8,
            'TE': 3,
            'DST': 3
        }
        
        for pos, min_count in min_requirements.items():
            if pos not in pos_counts or pos_counts[pos] < min_count:
                warnings.append(f"Low {pos} player count: {pos_counts.get(pos, 0)} (recommend at least {min_count})")
    
    # Check salary distribution
    if 'SALARY' in df.columns:
        avg_salary = df['SALARY'].mean()
        if avg_salary < 4000 or avg_salary > 9000:
            warnings.append(f"Unusual average salary: ${avg_salary:.0f}")
    
    # Check for duplicates
    if 'PLAYER NAME' in df.columns:
        duplicates = df['PLAYER NAME'].duplicated().sum()
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate player names")
    
    return warnings

def create_sample_data() -> pd.DataFrame:
    """Create sample player data for testing"""
    logger.info("Creating sample player data...")
    
    sample_data = {
        'PLAYER NAME': [
            # QBs
            'Josh Allen', 'Patrick Mahomes', 'Jalen Hurts', 'Lamar Jackson', 'Dak Prescott',
            'Joe Burrow', 'Justin Herbert', 'Tua Tagovailoa',
            # RBs
            'Christian McCaffrey', 'Austin Ekeler', 'Derrick Henry', 'Saquon Barkley',
            'Jonathan Taylor', 'Josh Jacobs', 'Tony Pollard', 'Najee Harris',
            'Kenneth Walker', 'Breece Hall', 'James Cook', 'Aaron Jones',
            # WRs
            'Tyreek Hill', 'Stefon Diggs', 'Justin Jefferson', 'Ja\'Marr Chase',
            'CeeDee Lamb', 'A.J. Brown', 'Davante Adams', 'Cooper Kupp',
            'Amon-Ra St. Brown', 'Chris Olave', 'DK Metcalf', 'Mike Evans',
            'DeVonta Smith', 'Jaylen Waddle', 'Calvin Ridley', 'Tee Higgins',
            'Amari Cooper', 'Terry McLaurin', 'Michael Pittman', 'Chris Godwin',
            # TEs
            'Travis Kelce', 'Mark Andrews', 'T.J. Hockenson', 'George Kittle',
            'Dallas Goedert', 'Darren Waller', 'Kyle Pitts', 'Evan Engram',
            # DSTs
            'Buffalo', 'San Francisco', 'Dallas', 'New England', 'Baltimore',
            'Philadelphia', 'Denver', 'Cincinnati'
        ],
        'POS': [
            # QBs
            'QB', 'QB', 'QB', 'QB', 'QB', 'QB', 'QB', 'QB',
            # RBs
            'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB',
            # WRs
            'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR',
            'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR', 'WR',
            # TEs
            'TE', 'TE', 'TE', 'TE', 'TE', 'TE', 'TE', 'TE',
            # DSTs
            'DST', 'DST', 'DST', 'DST', 'DST', 'DST', 'DST', 'DST'
        ],
        'TEAM': [
            # QBs
            'BUF', 'KC', 'PHI', 'BAL', 'DAL', 'CIN', 'LAC', 'MIA',
            # RBs
            'SF', 'LAC', 'TEN', 'NYG', 'IND', 'LV', 'DAL', 'PIT', 'SEA', 'NYJ', 'BUF', 'GB',
            # WRs
            'MIA', 'BUF', 'MIN', 'CIN', 'DAL', 'PHI', 'LV', 'LAR', 'DET', 'NO', 'SEA', 'TB',
            'PHI', 'MIA', 'JAX', 'CIN', 'CLE', 'WAS', 'IND', 'TB',
            # TEs
            'KC', 'BAL', 'MIN', 'SF', 'PHI', 'NYG', 'ATL', 'JAX',
            # DSTs
            'BUF', 'SF', 'DAL', 'NE', 'BAL', 'PHI', 'DEN', 'CIN'
        ],
        'SALARY': [
            # QBs
            8500, 8300, 8200, 8000, 7700, 7900, 7600, 7300,
            # RBs
            9000, 8400, 6800, 8200, 7900, 7500, 6600, 6200, 6900, 6700, 6400, 6100,
            # WRs
            8800, 8600, 8900, 8700, 8400, 8200, 8000, 7800, 7600, 7200, 7000, 7400,
            6800, 6600, 6400, 6200, 5800, 5600, 5400, 6000,
            # TEs
            7000, 6500, 5900, 6200, 5600, 5400, 5200, 5000,
            # DSTs
            4800, 4600, 4400, 4200, 4000, 3800, 3600, 3400
        ],
        'PROJ PTS': [
            # QBs
            22.5, 21.8, 21.2, 20.5, 19.8, 20.2, 19.5, 18.8,
            # RBs
            18.5, 17.2, 16.8, 16.2, 15.8, 15.2, 14.5, 13.8, 14.2, 13.5, 12.8, 12.2,
            # WRs
            17.5, 16.8, 17.2, 16.5, 15.8, 15.2, 14.8, 14.2, 13.8, 13.2, 12.8, 13.5,
            12.2, 11.8, 11.2, 10.8, 10.2, 9.8, 9.5, 10.5,
            # TEs
            12.5, 11.8, 10.5, 10.2, 9.5, 8.8, 8.2, 7.8,
            # DSTs
            9.2, 8.8, 8.5, 8.2, 7.8, 7.5, 7.2, 6.8
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Add ownership percentages
    np.random.seed(42)
    df['OWN_PCT'] = np.random.uniform(2, 30, len(df)).round(1)
    
    # Add calculated columns
    df['VALUE'] = (df['PROJ PTS'] / (df['SALARY'] / 1000)).round(2)
    
    return df
