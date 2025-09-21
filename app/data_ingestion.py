import os
import re
import pandas as pd
import logging
from typing import Tuple, List, Optional, Dict

from app.config import INPUT_DIR

logger = logging.getLogger(__name__)

# --- Column Aliases to handle variations in CSV headers ---
ALIASES: Dict[str, str] = {
    "PLAYER": "PLAYER NAME", "NAME": "PLAYER NAME",
    "POSITION": "POS",
    "SAL": "SALARY", "PRICE": "SALARY",
    "PROJ": "PROJ PTS", "PROJECTION": "PROJ PTS", "FPTS": "PROJ PTS",
    "PROJ ROSTER %": "OWN_PCT",
    "OPPONENT": "OPP", "MATCHUP": "OPP",
}

def _clean_value(val, is_percent=False):
    """Utility to clean salary and percentage strings into numbers."""
    if pd.isna(val): return pd.NA
    s = str(val).strip().replace("$", "").replace(",", "")
    if is_percent: s = s.replace("%", "")
    if s == "": return pd.NA
    try:
        return float(s)
    except (ValueError, TypeError):
        return pd.NA

def parse_player_details(name_str: str) -> Tuple[str, str, str]:
    """
    Parses 'Player Name (TEAM - POS)' format.
    Returns (Clean Name, Team, Position).
    """
    match = re.search(r'\(([^)]+)\)', name_str)
    if match:
        parts = match.group(1).split(' - ')
        if len(parts) == 2:
            team, pos = parts[0].strip(), parts[1].strip()
            clean_name = name_str.split('(')[0].strip()
            return clean_name, team, pos
    # Fallback for DSTs or other formats
    return name_str.strip(), "UNK", "UNK"

def load_data_from_input_dir() -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Loads, parses, and de-duplicates player data from the data/input directory.
    This is the master function to create a clean player pool.
    """
    warnings = []
    all_players = {} # Use a dictionary to handle duplicates {player_name: player_data}

    if not os.path.exists(INPUT_DIR):
        warnings.append(f"ERROR: Input directory not found: {INPUT_DIR}")
        return None, warnings

    # Process files in a specific order to prioritize position assignments
    filenames = ["qb.csv", "rb.csv", "wr.csv", "te.csv", "dst.csv"]

    for filename in filenames:
        path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(path):
            warnings.append(f"WARNING: Missing expected file: {filename}")
            continue

        try:
            df = pd.read_csv(path)
            df = df.rename(columns={c: c.strip().upper() for c in df.columns})

            # Apply Aliases
            for alias, canonical in ALIASES.items():
                if alias in df.columns and canonical not in df.columns:
                    df.rename(columns={alias: canonical}, inplace=True)

            for _, row in df.iterrows():
                name_str = row.get("PLAYER NAME", "")
                if not name_str:
                    continue

                clean_name, team, pos = parse_player_details(name_str)
                
                # --- This is the key de-duplication logic ---
                if clean_name in all_players:
                    continue # Skip if we've already processed this player

                opponent = str(row.get("OPP", "")).replace('@', '').replace('vs', '').strip()

                all_players[clean_name] = {
                    'PLAYER NAME': clean_name,
                    'POS': pos,
                    'TEAM': team,
                    'OPP': opponent,
                    'SALARY': _clean_value(row.get("SALARY")),
                    'PROJ PTS': _clean_value(row.get("PROJ PTS")),
                    'OWN_PCT': _clean_value(row.get("OWN_PCT"), is_percent=True),
                }
            warnings.append(f"INFO: Processed {filename}.")

        except Exception as e:
            warnings.append(f"ERROR: Failed to process {filename}: {e}")
    
    if not all_players:
        warnings.append("ERROR: No player data could be loaded after processing all files.")
        return None, warnings

    # Convert the dictionary of players into a DataFrame
    final_df = pd.DataFrame(list(all_players.values()))
    final_df.dropna(subset=['PLAYER NAME', 'SALARY', 'PROJ PTS', 'POS', 'TEAM'], inplace=True)
    final_df['SALARY'] = final_df['SALARY'].astype(int)

    warnings.append(f"SUCCESS: Created a clean player pool with {len(final_df)} unique players.")
    return final_df, warnings
