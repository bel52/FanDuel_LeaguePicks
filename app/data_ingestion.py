import os
import re
import pandas as pd
import logging
import requests

from app.config import settings

logger = logging.getLogger(__name__)

# Sleeper API endpoint for NFL players (used to verify active roster status)
SLEEPER_PLAYERS_API = "https://api.sleeper.app/v1/players/nfl"

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def _parse_own(x):
    """Parse ownership percentage from various formats."""
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace("-", " ").replace("–", " ")
    nums = [float(n) for n in re.findall(r"[0-9]+\.?[0-9]*", s)]
    if not nums:
        return None
    return sum(nums) / len(nums)

def _parse_salary(x):
    """Parse salary by removing $ and commas."""
    if pd.isna(x):
        return 0
    s = str(x).replace('$', '').replace(',', '').strip()
    try:
        return int(float(s))
    except:
        return 0

def _filter_active_players_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out players not on active rosters using Sleeper API."""
    try:
        resp = requests.get(SLEEPER_PLAYERS_API, timeout=10)
        resp.raise_for_status()
        players_data = resp.json()
    except Exception as e:
        logger.warning(f" Sleeper API unavailable for roster filter: {e}")
        return df
    active_names = set()
    for player in players_data.values():
        status = player.get('status')
        team = player.get('team')
        if status and status.lower() == "active" and team:
            name = player.get('full_name') or f"{player.get('first_name','')} {player.get('last_name','')}".strip()
            if name:
                norm = re.sub(r'[^A-Za-z0-9 ]+', '', name).strip().lower()
                active_names.add(norm)
    if not active_names:
        return df
    def is_active(name: str) -> bool:
        if not isinstance(name, str):
            return False
        norm = re.sub(r'[^A-Za-z0-9 ]+', '', name).strip().lower()
        return norm in active_names
    filtered_df = df[df['PLAYER NAME'].apply(is_active)].copy()
    if len(filtered_df) < len(df):
        logger.info(f"Filtered inactive players: {len(df)-len(filtered_df)} removed, {len(filtered_df)} remain.")
    return filtered_df

def load_weekly_data() -> pd.DataFrame:
    """Load and combine all position CSV files into a single DataFrame."""
    files = {
        "QB": os.path.join(settings.input_dir, "qb.csv"),
        "RB": os.path.join(settings.input_dir, "rb.csv"),
        "WR": os.path.join(settings.input_dir, "wr.csv"),
        "TE": os.path.join(settings.input_dir, "te.csv"),
        "DST": os.path.join(settings.input_dir, "dst.csv"),
    }
    frames = []
    for pos, path in files.items():
        if not os.path.exists(path):
            logger.warning(f"Missing {pos} file: {path}")
            continue
        try:
            df = pd.read_csv(path)
            # Standardize column names
            col_map = {}
            for col in df.columns:
                clower = col.lower().strip()
                if 'player' in clower or 'name' in clower:
                    col_map[col] = 'PLAYER NAME'
                elif clower == 'team':
                    col_map[col] = 'TEAM'
                elif clower in ['opp', 'opponent']:
                    col_map[col] = 'OPP'
                elif 'proj' in clower and 'pts' in clower:
                    col_map[col] = 'PROJ PTS'
                elif 'salary' in clower:
                    col_map[col] = 'SALARY'
                elif 'roster' in clower or 'own' in clower:
                    col_map[col] = 'PROJ ROSTER %'
            df = df.rename(columns=col_map)
            # Ensure required columns exist
            if 'PLAYER NAME' not in df.columns:
                df['PLAYER NAME'] = df.iloc[:, 0].astype(str)
            df['POS'] = pos  # add position
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
            # Calculate numeric ownership percentage
            df['OWN_PCT'] = df['PROJ ROSTER %'].apply(_parse_own)
            frames.append(df)
        except Exception as e:
            logger.error(f"Error loading {pos} projections: {e}")
    if not frames:
        # No data loaded
        return pd.DataFrame()
    all_df = pd.concat(frames, ignore_index=True)
    # Basic filters: remove players with invalid salary or projection
    all_df = all_df[(all_df['SALARY'] >= 3000) & (all_df['SALARY'] <= 15000)]
    all_df = all_df[all_df['PROJ PTS'] >= 0]
    all_df.reset_index(drop=True, inplace=True)
    # Apply active roster filter
    all_df = _filter_active_players_df(all_df)
    logger.info(f"Total active players loaded: {len(all_df)}")
    return all_df
