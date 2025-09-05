import os
import pandas as pd
import re
import logging
from typing import Optional
from app.config import settings

logger = logging.getLogger(__name__)

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def _parse_own(x):
    """Parse ownership percentage from various formats"""
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace("-", " ").replace("–", " ")
    nums = [float(n) for n in re.findall(r"[0-9]+\.?[0-9]*", s)]
    if not nums:
        return None
    return sum(nums) / len(nums)

def _parse_salary(x):
    """Parse salary removing $ and commas"""
    if pd.isna(x):
        return 0
    s = str(x).replace('$', '').replace(',', '').strip()
    try:
        return int(float(s))
    except:
        return 0

def load_weekly_data() -> Optional[pd.DataFrame]:
    """Load and combine all position CSV files"""
    
    files = {
        "QB": os.path.join(settings.input_dir, "qb.csv"),
        "RB": os.path.join(settings.input_dir, "rb.csv"),
        "WR": os.path.join(settings.input_dir, "wr.csv"),
        "TE": os.path.join(settings.input_dir, "te.csv"),
        "DST": os.path.join(settings.input_dir, "dst.csv"),
    }
    
    frames = []
    found_files = []
    
    for pos, path in files.items():
        if not os.path.exists(path):
            logger.warning(f"Missing {pos} file: {path}")
            continue
            
        try:
            df = pd.read_csv(path)
            found_files.append(pos)
            
            # Map column names (case-insensitive)
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower().strip()
                if 'player' in col_lower or 'name' in col_lower:
                    col_mapping[col] = 'PLAYER NAME'
                elif col_lower == 'team':
                    col_mapping[col] = 'TEAM'
                elif col_lower in ['opp', 'opponent']:
                    col_mapping[col] = 'OPP'
                elif 'proj' in col_lower and 'pts' in col_lower:
                    col_mapping[col] = 'PROJ PTS'
                elif 'salary' in col_lower:
                    col_mapping[col] = 'SALARY'
                elif 'roster' in col_lower or 'own' in col_lower:
                    col_mapping[col] = 'PROJ ROSTER %'
            
            df = df.rename(columns=col_mapping)
            
            # Ensure required columns exist
            if 'PLAYER NAME' not in df.columns:
                df['PLAYER NAME'] = df.iloc[:, 0]  # Use first column as name
            
            # Add position
            df['POS'] = pos
            
            # Clean data
            if 'TEAM' not in df.columns:
                df['TEAM'] = ''
            if 'OPP' not in df.columns:
                df['OPP'] = ''
            if 'PROJ PTS' not in df.columns:
                df['PROJ PTS'] = 0.0
            else:
                df['PROJ PTS'] = df['PROJ PTS'].apply(_safe_float)
            
            if 'SALARY' not in df.columns:
                df['SALARY'] = 0
            else:
                df['SALARY'] = df['SALARY'].apply(_parse_salary)
            
            if 'PROJ ROSTER %' not in df.columns:
                df['PROJ ROSTER %'] = ''
            
            # Parse ownership
            df['OWN_PCT'] = df.get('PROJ ROSTER %', pd.Series()).apply(_parse_own)
            
            frames.append(df)
            
        except Exception as e:
            logger.error(f"Error loading {pos} file: {e}")
            continue
    
    if not frames:
        logger.error("No valid CSV files found")
        return None
    
    logger.info(f"Loaded files for positions: {found_files}")
    
    # Combine all frames
    all_df = pd.concat(frames, ignore_index=True)
    
    # Filter valid rows
    all_df = all_df[(all_df['SALARY'] > 0) & (all_df['PROJ PTS'] >= 0)]
    all_df = all_df[(all_df['SALARY'] >= 3000) & (all_df['SALARY'] <= 15000)]
    
    all_df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Total players loaded: {len(all_df)}")
    
    return all_df
