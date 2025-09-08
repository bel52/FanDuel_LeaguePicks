import os
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re

from app.config import settings, POSITION_MAPPINGS, TEAM_MAPPINGS

logger = logging.getLogger(__name__)

class DataIngestion:
    """Enhanced data ingestion with robust CSV processing and validation"""
    
    def __init__(self):
        self.required_columns = {
            'PLAYER NAME': ['PLAYER NAME', 'Player', 'Name', 'player_name', 'PlayerName'],
            'POS': ['POS', 'Position', 'Pos', 'position'],
            'TEAM': ['TEAM', 'Team', 'Tm', 'team'],
            'OPP': ['OPP', 'Opponent', 'Opp', 'vs', 'opponent'],
            'SALARY': ['SALARY', 'Salary', 'FD Salary', 'FanDuel Salary', 'Cost'],
            'PROJ PTS': ['PROJ PTS', 'Projected Points', 'Proj Pts', 'FPTS', 'Points', 'projection']
        }
        
        self.optional_columns = {
            'OWN_PCT': ['OWN_PCT', 'Ownership', 'Own %', 'Projected Ownership'],
            'CEILING': ['CEILING', 'Ceiling', 'Upside'],
            'FLOOR': ['FLOOR', 'Floor', 'Downside']
        }
        
        self.last_load_time = None
        self.cached_data = None
    
    async def load_weekly_data(self, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Load and process weekly player data from CSV files"""
        try:
            # Check cache first
            if not force_refresh and self.cached_data is not None:
                cache_age = (datetime.now() - self.last_load_time).total_seconds()
                if cache_age < settings.cache_ttl:
                    logger.info("Using cached player data")
                    return self.cached_data.copy()
            
            # Load data from CSV files
            all_data = []
            files_processed = 0
            
            for filename in settings.get_required_files():
                file_path = os.path.join(settings.input_dir, filename)
                
                if not os.path.exists(file_path):
                    logger.warning(f"Required file not found: {filename}")
                    continue
                
                try:
                    df = await self._load_csv_file(file_path)
                    if df is not None and not df.empty:
                        all_data.append(df)
                        files_processed += 1
                        logger.info(f"Loaded {len(df)} players from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    continue
            
            if not all_data:
                logger.error("No valid data files found")
                return None
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Process and validate
            processed_data = await self._process_data(combined_data)
            
            if processed_data is None or processed_data.empty:
                logger.error("Data processing resulted in empty dataset")
                return None
            
            # Cache the results
            self.cached_data = processed_data.copy()
            self.last_load_time = datetime.now()
            
            logger.info(f"Successfully loaded {len(processed_data)} players from {files_processed} files")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in load_weekly_data: {e}")
            return None
    
    async def _load_csv_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and normalize a single CSV file"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Could not decode {file_path} with any encoding")
                return None
            
            if df.empty:
                logger.warning(f"Empty file: {file_path}")
                return None
            
            # Normalize column names
            df = self._normalize_columns(df)
            
            # Infer position from filename if not present
            if 'POS' not in df.columns or df['POS'].isna().all():
                position = self._infer_position_from_filename(file_path)
                if position:
                    df['POS'] = position
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return None
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard format"""
        column_mapping = {}
        
        # Map all known variations to standard names
        for standard_name, variations in self.required_columns.items():
            for col in df.columns:
                if col.strip().upper() in [v.upper() for v in variations]:
                    column_mapping[col] = standard_name
                    break
        
        # Map optional columns
        for standard_name, variations in self.optional_columns.items():
            for col in df.columns:
                if col.strip().upper() in [v.upper() for v in variations]:
                    column_mapping[col] = standard_name
                    break
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return df
    
    def _infer_position_from_filename(self, file_path: str) -> Optional[str]:
        """Infer position from filename"""
        filename = os.path.basename(file_path).lower()
        
        position_map = {
            'qb': 'QB',
            'rb': 'RB', 
            'wr': 'WR',
            'te': 'TE',
            'dst': 'DST',
            'def': 'DST',
            'k': 'K'
        }
        
        for key, pos in position_map.items():
            if key in filename:
                return pos
        
        return None
    
    async def _process_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process and validate the combined dataset"""
        try:
            # Create a copy to avoid modifying original
            processed_df = df.copy()
            
            # Clean player names
            if 'PLAYER NAME' in processed_df.columns:
                processed_df['PLAYER NAME'] = processed_df['PLAYER NAME'].astype(str).str.strip()
                processed_df['PLAYER NAME'] = processed_df['PLAYER NAME'].apply(self._clean_player_name)
            
            # Clean and standardize positions
            if 'POS' in processed_df.columns:
                processed_df['POS'] = processed_df['POS'].astype(str).str.strip().str.upper()
                processed_df['POS'] = processed_df['POS'].map(POSITION_MAPPINGS).fillna(processed_df['POS'])
            
            # Clean and standardize teams
            if 'TEAM' in processed_df.columns:
                processed_df['TEAM'] = processed_df['TEAM'].astype(str).str.strip().str.upper()
                processed_df['TEAM'] = processed_df['TEAM'].map(TEAM_MAPPINGS).fillna(processed_df['TEAM'])
            
            # Clean opponents
            if 'OPP' in processed_df.columns:
                processed_df['OPP'] = processed_df['OPP'].astype(str).str.strip()
                processed_df['OPP'] = processed_df['OPP'].apply(self._clean_opponent)
                processed_df['OPP'] = processed_df['OPP'].map(TEAM_MAPPINGS).fillna(processed_df['OPP'])
            
            # Clean salary data
            if 'SALARY' in processed_df.columns:
                processed_df['SALARY'] = processed_df['SALARY'].apply(self._clean_salary)
            
            # Clean projection data
            if 'PROJ PTS' in processed_df.columns:
                processed_df['PROJ PTS'] = pd.to_numeric(processed_df['PROJ PTS'], errors='coerce')
            
            # Clean ownership data
            if 'OWN_PCT' in processed_df.columns:
                processed_df['OWN_PCT'] = processed_df['OWN_PCT'].apply(self._clean_ownership)
            
            # Add derived columns
            processed_df = self._add_derived_columns(processed_df)
            
            # Filter valid players
            processed_df = self._filter_valid_players(processed_df)
            
            # Remove duplicates
            processed_df = self._remove_duplicates(processed_df)
            
            # Sort by position and projection
            processed_df = processed_df.sort_values(['POS', 'PROJ PTS'], ascending=[True, False])
            processed_df = processed_df.reset_index(drop=True)
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def _clean_player_name(self, name: str) -> str:
        """Clean and standardize player names"""
        if pd.isna(name) or name == 'nan':
            return ''
        
        # Remove special characters but keep basic punctuation
        name = re.sub(r'[^\w\s\'\.\-]', '', str(name))
        
        # Remove position and team info in parentheses
        name = re.sub(r'\([^)]*\)', '', name)
        
        # Clean whitespace
        name = ' '.join(name.split())
        
        return name.strip()
    
    def _clean_opponent(self, opp: str) -> str:
        """Clean opponent data"""
        if pd.isna(opp) or opp == 'nan':
            return ''
        
        opp = str(opp).strip().upper()
        
        # Remove @ symbol
        if opp.startswith('@'):
            opp = opp[1:]
        
        return opp
    
    def _clean_salary(self, salary) -> int:
        """Clean and convert salary to integer"""
        if pd.isna(salary):
            return 4000  # Default minimum salary
        
        # Remove currency symbols and commas
        salary_str = str(salary).replace('$', '').replace(',', '').strip()
        
        try:
            return int(float(salary_str))
        except (ValueError, TypeError):
            return 4000
    
    def _clean_ownership(self, ownership) -> float:
        """Clean ownership percentage data"""
        if pd.isna(ownership):
            return 0.0
        
        ownership_str = str(ownership).replace('%', '').strip()
        
        # Handle ranges like "15-20"
        if '-' in ownership_str:
            try:
                parts = ownership_str.split('-')
                low = float(parts[0])
                high = float(parts[1])
                return (low + high) / 2
            except (ValueError, IndexError):
                return 0.0
        
        try:
            return float(ownership_str)
        except (ValueError, TypeError):
            return 0.0
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated columns"""
        # Value per $1000 spent
        df['VALUE'] = df.apply(
            lambda row: row['PROJ PTS'] / (row['SALARY'] / 1000) if row['SALARY'] > 0 else 0,
            axis=1
        )
        
        # Ceiling and floor estimates if not provided
        if 'CEILING' not in df.columns:
            df['CEILING'] = df['PROJ PTS'] * 1.4
        
        if 'FLOOR' not in df.columns:
            df['FLOOR'] = df['PROJ PTS'] * 0.6
        
        # Standard deviation for Monte Carlo simulation
        df['STD_DEV'] = df['PROJ PTS'] * 0.25  # 25% of projection as std dev
        
        return df
    
    def _filter_valid_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid player records"""
        # Remove rows with missing essential data
        df = df.dropna(subset=['PLAYER NAME', 'POS'])
        
        # Remove rows with empty player names
        df = df[df['PLAYER NAME'].str.len() > 0]
        
        # Filter by valid positions
        valid_positions = ['QB', 'RB', 'WR', 'TE', 'DST']
        df = df[df['POS'].isin(valid_positions)]
        
        # Filter by reasonable salary range
        df = df[(df['SALARY'] >= 3000) & (df['SALARY'] <= 15000)]
        
        # Filter by reasonable projection range
        df = df[(df['PROJ PTS'] >= 0) & (df['PROJ PTS'] <= 50)]
        
        # Filter by minimum value threshold
        df = df[df['VALUE'] >= settings.min_value_threshold]
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate players, keeping the one with highest projection"""
        if df.empty:
            return df
        
        # Sort by projection descending before removing duplicates
        df = df.sort_values('PROJ PTS', ascending=False)
        
        # Remove duplicates based on player name and team
        df = df.drop_duplicates(subset=['PLAYER NAME', 'TEAM'], keep='first')
        
        return df
    
    async def check_data_availability(self) -> str:
        """Check if required data files are available"""
        try:
            required_files = settings.get_required_files()
            available_files = []
            missing_files = []
            
            for filename in required_files:
                file_path = os.path.join(settings.input_dir, filename)
                if os.path.exists(file_path):
                    available_files.append(filename)
                else:
                    missing_files.append(filename)
            
            if len(available_files) == len(required_files):
                return "healthy"
            elif len(available_files) > 0:
                return f"partial ({len(available_files)}/{len(required_files)} files)"
            else:
                return "no_data"
                
        except Exception as e:
            logger.error(f"Error checking data availability: {e}")
            return "error"
    
    async def get_detailed_status(self) -> Dict:
        """Get detailed status of data ingestion"""
        try:
            status = {
                "data_directory": settings.input_dir,
                "files_status": {},
                "last_load_time": self.last_load_time.isoformat() if self.last_load_time else None,
                "cached_players": len(self.cached_data) if self.cached_data is not None else 0,
                "errors": []
            }
            
            for filename in settings.get_required_files():
                file_path = os.path.join(settings.input_dir, filename)
                
                if os.path.exists(file_path):
                    try:
                        # Get file stats
                        stat = os.stat(file_path)
                        status["files_status"][filename] = {
                            "exists": True,
                            "size_bytes": stat.st_size,
                            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                    except Exception as e:
                        status["files_status"][filename] = {
                            "exists": True,
                            "error": str(e)
                        }
                else:
                    status["files_status"][filename] = {"exists": False}
            
            # Try loading data to check for issues
            if self.cached_data is not None:
                pos_counts = self.cached_data['POS'].value_counts().to_dict()
                status["position_counts"] = pos_counts
                
                value_stats = {
                    "min_value": float(self.cached_data['VALUE'].min()),
                    "max_value": float(self.cached_data['VALUE'].max()),
                    "avg_value": float(self.cached_data['VALUE'].mean())
                }
                status["value_statistics"] = value_stats
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting detailed status: {e}")
            return {"error": str(e)}
    
    async def refresh_data(self):
        """Force refresh of cached data"""
        try:
            logger.info("Refreshing player data...")
            self.cached_data = None
            self.last_load_time = None
            
            data = await self.load_weekly_data(force_refresh=True)
            
            if data is not None:
                logger.info(f"Data refresh successful: {len(data)} players loaded")
            else:
                logger.error("Data refresh failed")
                
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
    
    def create_sample_data(self):
        """Create sample data for testing purposes"""
        sample_data = {
            'qb.csv': [
                ['PLAYER NAME', 'TEAM', 'OPP', 'PROJ PTS', 'SALARY', 'OWN_PCT'],
                ['Josh Allen', 'BUF', '@MIA', 22.5, 8500, 15.0],
                ['Patrick Mahomes', 'KC', 'LV', 21.8, 8300, 12.0],
                ['Lamar Jackson', 'BAL', 'CLE', 20.5, 8000, 8.0]
            ],
            'rb.csv': [
                ['PLAYER NAME', 'TEAM', 'OPP', 'PROJ PTS', 'SALARY', 'OWN_PCT'],
                ['Christian McCaffrey', 'SF', 'SEA', 18.5, 9000, 25.0],
                ['Saquon Barkley', 'PHI', 'DAL', 17.2, 8400, 20.0],
                ['Josh Jacobs', 'GB', 'DET', 14.8, 7800, 12.0]
            ],
            'wr.csv': [
                ['PLAYER NAME', 'TEAM', 'OPP', 'PROJ PTS', 'SALARY', 'OWN_PCT'],
                ["Ja'Marr Chase", 'CIN', '@CLE', 17.8, 9200, 22.0],
                ['CeeDee Lamb', 'DAL', '@WAS', 15.0, 8000, 18.0],
                ['Tyreek Hill', 'MIA', 'BUF', 12.6, 7600, 8.0]
            ],
            'te.csv': [
                ['PLAYER NAME', 'TEAM', 'OPP', 'PROJ PTS', 'SALARY', 'OWN_PCT'],
                ['Travis Kelce', 'KC', 'LV', 10.3, 6100, 15.0],
                ['George Kittle', 'SF', 'SEA', 11.4, 6500, 10.0],
                ['Mark Andrews', 'BAL', 'CLE', 8.5, 5500, 6.0]
            ],
            'dst.csv': [
                ['PLAYER NAME', 'TEAM', 'OPP', 'PROJ PTS', 'SALARY', 'OWN_PCT'],
                ['San Francisco 49ers', 'SF', 'SEA', 7.6, 4400, 10.0],
                ['Pittsburgh Steelers', 'PIT', '@NYJ', 8.1, 4600, 6.0],
                ['Buffalo Bills', 'BUF', '@MIA', 5.7, 3800, 3.0]
            ]
        }
        
        try:
            os.makedirs(settings.input_dir, exist_ok=True)
            
            for filename, data in sample_data.items():
                file_path = os.path.join(settings.input_dir, filename)
                
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(data)
                
                logger.info(f"Created sample file: {filename}")
            
            logger.info("Sample data created successfully")
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
