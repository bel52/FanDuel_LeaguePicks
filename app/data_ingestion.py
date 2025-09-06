import os
import logging
import pandas as pd
from typing import Optional, Dict, List
import glob

logger = logging.getLogger(__name__)

def load_weekly_data(input_dir: str = "data/input") -> Optional[pd.DataFrame]:
    """
    Load and combine player data from CSV files
    """
    try:
        # Position files to load
        positions = ['qb', 'rb', 'wr', 'te', 'dst']
        all_data = []
        
        for pos in positions:
            file_path = os.path.join(input_dir, f"{pos}.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"Missing {pos.upper()} data file: {file_path}")
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # Standardize column names
                column_mapping = {
                    'Player': 'PLAYER NAME',
                    'Name': 'PLAYER NAME',
                    'player': 'PLAYER NAME',
                    'Team': 'TEAM',
                    'team': 'TEAM',
                    'Opponent': 'OPP',
                    'Opp': 'OPP',
                    'opponent': 'OPP',
                    'Salary': 'SALARY',
                    'salary': 'SALARY',
                    'Projection': 'PROJ PTS',
                    'Proj': 'PROJ PTS',
                    'Points': 'PROJ PTS',
                    'proj_points': 'PROJ PTS',
                    'Position': 'POS',
                    'Pos': 'POS',
                    'position': 'POS',
                    'Ownership': 'OWN_PCT',
                    'Own%': 'OWN_PCT',
                    'ownership': 'OWN_PCT'
                }
                
                df.rename(columns=column_mapping, inplace=True)
                
                # Ensure position column exists
                if 'POS' not in df.columns:
                    df['POS'] = pos.upper()
                
                # Clean up data
                df = clean_player_data(df)
                
                all_data.append(df)
                logger.info(f"Loaded {len(df)} {pos.upper()} players")
                
            except Exception as e:
                logger.error(f"Error loading {pos} data: {e}")
        
        if not all_data:
            logger.error("No player data loaded")
            return None
        
        # Combine all positions
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add calculated fields
        combined_df = add_calculated_fields(combined_df)
        
        logger.info(f"Loaded total of {len(combined_df)} players")
        return combined_df
        
    except Exception as e:
        logger.error(f"Error loading weekly data: {e}")
        return None

def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize player data"""
    
    # Remove any rows with missing critical data
    required_cols = ['PLAYER NAME', 'SALARY', 'PROJ PTS']
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]
    
    # Convert salary to numeric
    if 'SALARY' in df.columns:
        df['SALARY'] = pd.to_numeric(
            df['SALARY'].astype(str).str.replace('$', '').str.replace(',', ''),
            errors='coerce'
        )
    
    # Convert projections to numeric
    if 'PROJ PTS' in df.columns:
        df['PROJ PTS'] = pd.to_numeric(df['PROJ PTS'], errors='coerce')
    
    # Convert ownership to numeric
    if 'OWN_PCT' in df.columns:
        df['OWN_PCT'] = pd.to_numeric(
            df['OWN_PCT'].astype(str).str.replace('%', ''),
            errors='coerce'
        )
    
    # Fill missing ownership with 0
    if 'OWN_PCT' in df.columns:
        df['OWN_PCT'] = df['OWN_PCT'].fillna(0)
    else:
        df['OWN_PCT'] = 0
    
    # Clean team names
    if 'TEAM' in df.columns:
        df['TEAM'] = df['TEAM'].str.strip().str.upper()
    
    if 'OPP' in df.columns:
        df['OPP'] = df['OPP'].str.strip().str.upper()
        # Remove @ symbol if present
        df['OPP'] = df['OPP'].str.replace('@', '')
    
    # Remove invalid rows
    df = df[df['SALARY'] > 0]
    df = df[df['PROJ PTS'] > 0]
    
    return df

def add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add calculated fields for optimization"""
    
    # Value calculation (points per $1000)
    df['VALUE'] = df['PROJ PTS'] / (df['SALARY'] / 1000)
    
    # Position rank
    df['POS_RANK'] = df.groupby('POS')['PROJ PTS'].rank(ascending=False)
    
    # Salary rank within position
    df['SALARY_RANK'] = df.groupby('POS')['SALARY'].rank(ascending=False)
    
    # Value rank within position
    df['VALUE_RANK'] = df.groupby('POS')['VALUE'].rank(ascending=False)
    
    # Game info parsing (if available)
    if 'GAME INFO' in df.columns:
        df = parse_game_info(df)
    
    return df

def parse_game_info(df: pd.DataFrame) -> pd.DataFrame:
    """Parse game info for additional context"""
    
    try:
        # Extract game time if available
        if 'GAME INFO' in df.columns:
            # Parse format like "TB@ATL 01:00PM ET"
            df['GAME_TIME'] = df['GAME INFO'].str.extract(r'(\d{2}:\d{2}[AP]M)')
            df['HOME_AWAY'] = df.apply(
                lambda row: 'HOME' if row['TEAM'] in str(row.get('GAME INFO', '')) else 'AWAY',
                axis=1
            )
    except Exception as e:
        logger.warning(f"Could not parse game info: {e}")
    
    return df

def load_ownership_data(file_path: str = "data/input/ownership.csv") -> Optional[pd.DataFrame]:
    """Load ownership projections if available"""
    
    if not os.path.exists(file_path):
        return None
    
    try:
        ownership_df = pd.read_csv(file_path)
        
        # Standardize columns
        ownership_df.rename(columns={
            'Player': 'PLAYER NAME',
            'Name': 'PLAYER NAME',
            'Ownership': 'OWN_PCT',
            'Own%': 'OWN_PCT',
            'Projected Ownership': 'OWN_PCT'
        }, inplace=True)
        
        # Clean ownership percentages
        if 'OWN_PCT' in ownership_df.columns:
            ownership_df['OWN_PCT'] = pd.to_numeric(
                ownership_df['OWN_PCT'].astype(str).str.replace('%', ''),
                errors='coerce'
            )
        
        return ownership_df
        
    except Exception as e:
        logger.error(f"Error loading ownership data: {e}")
        return None

def merge_ownership_data(players_df: pd.DataFrame, ownership_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ownership data with player data"""
    
    if ownership_df is None or ownership_df.empty:
        return players_df
    
    try:
        # Merge on player name
        merged_df = players_df.merge(
            ownership_df[['PLAYER NAME', 'OWN_PCT']],
            on='PLAYER NAME',
            how='left',
            suffixes=('', '_NEW')
        )
        
        # Use new ownership if available
        if 'OWN_PCT_NEW' in merged_df.columns:
            merged_df['OWN_PCT'] = merged_df['OWN_PCT_NEW'].fillna(merged_df['OWN_PCT'])
            merged_df.drop('OWN_PCT_NEW', axis=1, inplace=True)
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging ownership data: {e}")
        return players_df

def load_weather_data(file_path: str = "data/input/weather.csv") -> Optional[pd.DataFrame]:
    """Load weather data if available"""
    
    if not os.path.exists(file_path):
        return None
    
    try:
        weather_df = pd.read_csv(file_path)
        return weather_df
    except Exception as e:
        logger.error(f"Error loading weather data: {e}")
        return None

def save_lineup_for_upload(
    lineup_df: pd.DataFrame,
    output_path: str = "data/targets/fd_target.csv"
) -> bool:
    """Save lineup in FanDuel upload format"""
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Format for FanDuel upload
        upload_df = lineup_df[['PLAYER NAME', 'POS', 'TEAM', 'OPP', 'SALARY', 'PROJ PTS']].copy()
        upload_df.columns = ['Name', 'Position', 'Team', 'Opponent', 'Salary', 'Projection']
        
        upload_df.to_csv(output_path, index=False)
        logger.info(f"Lineup saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving lineup: {e}")
        return False

def create_sample_data(output_dir: str = "data/input") -> bool:
    """Create sample data files for testing"""
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample QBs
        qb_data = {
            'PLAYER NAME': ['Josh Allen', 'Patrick Mahomes', 'Lamar Jackson', 'Jalen Hurts'],
            'TEAM': ['BUF', 'KC', 'BAL', 'PHI'],
            'OPP': ['MIA', 'LV', 'CLE', 'DAL'],
            'SALARY': [8500, 8300, 8100, 7900],
            'PROJ PTS': [22.5, 21.8, 21.2, 20.5],
            'OWN_PCT': [15.2, 18.5, 12.3, 14.7]
        }
        pd.DataFrame(qb_data).to_csv(f"{output_dir}/qb.csv", index=False)
        
        # Sample RBs
        rb_data = {
            'PLAYER NAME': [
                'Christian McCaffrey', 'Austin Ekeler', 'Josh Jacobs', 'Tony Pollard',
                'Derrick Henry', 'Saquon Barkley', 'Jonathan Taylor', 'Bijan Robinson'
            ],
            'TEAM': ['SF', 'LAC', 'LV', 'DAL', 'TEN', 'NYG', 'IND', 'ATL'],
            'OPP': ['ARI', 'DEN', 'KC', 'PHI', 'JAX', 'WAS', 'HOU', 'CAR'],
            'SALARY': [9000, 7800, 7200, 6800, 6500, 7500, 7000, 6600],
            'PROJ PTS': [18.5, 15.2, 14.1, 13.5, 12.8, 14.8, 13.2, 12.5],
            'OWN_PCT': [25.3, 18.2, 15.1, 22.5, 10.2, 19.8, 14.3, 11.7]
        }
        pd.DataFrame(rb_data).to_csv(f"{output_dir}/rb.csv", index=False)
        
        # Sample WRs
        wr_data = {
            'PLAYER NAME': [
                'Tyreek Hill', 'CeeDee Lamb', 'Justin Jefferson', 'Stefon Diggs',
                'A.J. Brown', 'Davante Adams', 'Cooper Kupp', 'Jaylen Waddle',
                'DeVonta Smith', 'Chris Olave', 'Mike Evans', 'DK Metcalf'
            ],
            'TEAM': ['MIA', 'DAL', 'MIN', 'BUF', 'PHI', 'LV', 'LAR', 'MIA', 
                    'PHI', 'NO', 'TB', 'SEA'],
            'OPP': ['BUF', 'PHI', 'GB', 'MIA', 'DAL', 'KC', 'SEA', 'BUF',
                   'DAL', 'ATL', 'CAR', 'LAR'],
            'SALARY': [8800, 8200, 8500, 7900, 7600, 7300, 7100, 6900,
                      6500, 6200, 6800, 6600],
            'PROJ PTS': [17.2, 16.1, 16.8, 15.3, 14.8, 14.2, 13.9, 13.5,
                        12.8, 12.2, 13.2, 12.9],
            'OWN_PCT': [22.1, 28.5, 19.3, 17.2, 15.8, 13.4, 11.2, 14.6,
                       10.8, 8.9, 12.3, 9.7]
        }
        pd.DataFrame(wr_data).to_csv(f"{output_dir}/wr.csv", index=False)
        
        # Sample TEs
        te_data = {
            'PLAYER NAME': ['Travis Kelce', 'Mark Andrews', 'T.J. Hockenson', 'George Kittle',
                           'Dallas Goedert', 'Darren Waller'],
            'TEAM': ['KC', 'BAL', 'MIN', 'SF', 'PHI', 'NYG'],
            'OPP': ['LV', 'CLE', 'GB', 'ARI', 'DAL', 'WAS'],
            'SALARY': [7500, 6800, 6200, 6500, 5800, 5500],
            'PROJ PTS': [14.5, 12.8, 11.2, 11.8, 10.5, 9.8],
            'OWN_PCT': [18.5, 15.2, 12.8, 14.1, 10.3, 8.7]
        }
        pd.DataFrame(te_data).to_csv(f"{output_dir}/te.csv", index=False)
        
        # Sample DSTs
        dst_data = {
            'PLAYER NAME': ['Buffalo Bills', 'San Francisco 49ers', 'Dallas Cowboys', 
                           'New England Patriots', 'Baltimore Ravens', 'Philadelphia Eagles'],
            'TEAM': ['BUF', 'SF', 'DAL', 'NE', 'BAL', 'PHI'],
            'OPP': ['MIA', 'ARI', 'PHI', 'NYJ', 'CLE', 'DAL'],
            'SALARY': [3800, 3600, 3400, 3200, 3300, 3100],
            'PROJ PTS': [8.5, 8.2, 7.8, 7.5, 7.6, 7.3],
            'OWN_PCT': [12.5, 15.3, 10.2, 8.7, 9.4, 7.8]
        }
        pd.DataFrame(dst_data).to_csv(f"{output_dir}/dst.csv", index=False)
        
        logger.info(f"Sample data created in {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False
