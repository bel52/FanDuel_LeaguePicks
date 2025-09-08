# app/data_ingestion.py
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import random
from faker import Faker

logger = logging.getLogger(__name__)
fake = Faker()

class DataIngestionError(Exception):
    pass

class DataIngestionService:
    def __init__(self):
        self.current_data: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        self.nfl_teams = [
            'KC', 'BAL', 'BUF', 'CIN', 'CLE', 'DEN', 'HOU', 'IND', 'JAX', 'LVR',
            'LAC', 'MIA', 'NE', 'NYJ', 'PIT', 'TEN', 'DAL', 'NYG', 'PHI', 'WAS',
            'ARI', 'ATL', 'CAR', 'CHI', 'DET', 'GB', 'MIN', 'NO', 'SEA', 'SF',
            'LAR', 'TB'
        ]
        
    async def initialize(self):
        """Initialize the data service with sample data"""
        try:
            logger.info("Initializing data ingestion service...")
            await self.create_sample_data()
            logger.info("Data ingestion service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data service: {e}")
            raise DataIngestionError(f"Initialization failed: {e}")

    async def create_sample_data(self, num_players: int = 400) -> pd.DataFrame:
        """Create realistic sample NFL DFS player data"""
        try:
            logger.info(f"Generating {num_players} sample players...")
            
            players = []
            player_id = 1
            
            # Position distribution (realistic for NFL DFS)
            position_distribution = {
                'QB': int(num_players * 0.08),   # ~32 QBs
                'RB': int(num_players * 0.25),   # ~100 RBs  
                'WR': int(num_players * 0.40),   # ~160 WRs
                'TE': int(num_players * 0.12),   # ~48 TEs
                'K': int(num_players * 0.08),    # ~32 Ks
                'DST': int(num_players * 0.07),  # ~28 DSTs
            }
            
            for position, count in position_distribution.items():
                for _ in range(count):
                    player = await self._generate_single_player(player_id, position)
                    players.append(player)
                    player_id += 1
            
            # Create DataFrame and validate
            players_df = pd.DataFrame(players)
            self._validate_sample_data(players_df)
            
            # Store the data
            self.current_data = players_df
            self.last_update = datetime.now()
            
            logger.info(f"Generated {len(players_df)} players across {len(players_df['Position'].unique())} positions")
            return players_df
            
        except Exception as e:
            logger.error(f"Failed to create sample data: {e}")
            raise DataIngestionError(f"Sample data creation failed: {e}")

    async def _generate_single_player(self, player_id: int, position: str) -> Dict[str, Any]:
        """Generate a single realistic NFL player"""
        team = random.choice(self.nfl_teams)
        opponent = random.choice([t for t in self.nfl_teams if t != team])
        
        # Generate position-specific projections
        projections = self._generate_projections(position)
        
        # Calculate salary based on projections (realistic FanDuel pricing)
        salary = self._calculate_realistic_salary(projections['ProjectedPoints'], position)
        
        player = {
            'PlayerID': player_id,
            'Name': self._generate_realistic_name(position),
            'Position': position,
            'Team': team,
            'Opponent': opponent,
            'Salary': salary,
            'ProjectedPoints': projections['ProjectedPoints'],
            'HomeOrAway': random.choice(['HOME', 'AWAY']),
            'GameID': f"2024_W1_{team}_{opponent}",
            'Week': 1,
            'Season': 2024,
            'InjuryStatus': random.choices(
                ['', 'Q', 'D'], 
                weights=[0.85, 0.10, 0.05], 
                k=1
            )[0],
            **projections
        }
        
        return player

    def _generate_projections(self, position: str) -> Dict[str, float]:
        """Generate realistic statistical projections by position"""
        projections = {}
        
        if position == 'QB':
            passing_yards = np.random.normal(250, 80)
            passing_tds = np.random.gamma(2, 1)
            interceptions = np.random.poisson(1.2)
            rushing_yards = np.random.gamma(2, 8)
            rushing_tds = np.random.poisson(0.3)
            
            # FanDuel scoring: 0.04 per pass yard, 4 per pass TD, -2 per INT
            # 0.1 per rush yard, 6 per rush TD
            fantasy_points = (
                passing_yards * 0.04 + 
                passing_tds * 4 - 
                interceptions * 2 +
                rushing_yards * 0.1 + 
                rushing_tds * 6
            )
            
            projections.update({
                'PassingYards': max(0, round(passing_yards, 1)),
                'PassingTDs': max(0, int(passing_tds)),
                'Interceptions': max(0, int(interceptions)),
                'RushingYards': max(0, round(rushing_yards, 1)),
                'RushingTDs': max(0, int(rushing_tds)),
                'ProjectedPoints': max(4, round(fantasy_points, 1))
            })
            
        elif position == 'RB':
            rushing_yards = np.random.gamma(3, 25)
            rushing_tds = np.random.poisson(0.8)
            receptions = np.random.poisson(3)
            receiving_yards = receptions * np.random.gamma(2, 8)
            receiving_tds = np.random.poisson(0.15)
            
            fantasy_points = (
                rushing_yards * 0.1 + 
                rushing_tds * 6 +
                receiving_yards * 0.1 + 
                receptions * 0.5 +  # Half PPR
                receiving_tds * 6
            )
            
            projections.update({
                'RushingYards': max(0, round(rushing_yards, 1)),
                'RushingTDs': max(0, int(rushing_tds)),
                'Receptions': max(0, int(receptions)),
                'ReceivingYards': max(0, round(receiving_yards, 1)),
                'ReceivingTDs': max(0, int(receiving_tds)),
                'ProjectedPoints': max(2, round(fantasy_points, 1))
            })
            
        elif position in ['WR', 'TE']:
            if position == 'WR':
                receptions = np.random.gamma(2, 2.5)
                yards_per_reception = np.random.normal(12, 3)
            else:  # TE
                receptions = np.random.gamma(2, 2)
                yards_per_reception = np.random.normal(10, 2.5)
                
            receiving_yards = receptions * yards_per_reception
            receiving_tds = np.random.poisson(0.4 if position == 'WR' else 0.35)
            
            fantasy_points = (
                receiving_yards * 0.1 + 
                receptions * 0.5 +  # Half PPR
                receiving_tds * 6
            )
            
            projections.update({
                'Receptions': max(0, int(receptions)),
                'ReceivingYards': max(0, round(receiving_yards, 1)),
                'ReceivingTDs': max(0, int(receiving_tds)),
                'ProjectedPoints': max(2, round(fantasy_points, 1))
            })
            
        elif position == 'K':
            # Kickers: Field goals and extra points
            field_goals = np.random.poisson(1.8)
            extra_points = np.random.poisson(2.2)
            fantasy_points = field_goals * 3 + extra_points * 1
            
            projections.update({
                'FieldGoals': max(0, int(field_goals)),
                'ExtraPoints': max(0, int(extra_points)),
                'ProjectedPoints': max(2, round(fantasy_points, 1))
            })
            
        else:  # DST
            # Defense: Points allowed, sacks, turnovers
            sacks = np.random.poisson(2.5)
            interceptions = np.random.poisson(1)
            fumble_recoveries = np.random.poisson(0.8)
            defensive_tds = np.random.poisson(0.2)
            points_allowed = np.random.normal(21, 7)
            
            # FanDuel DST scoring
            fantasy_points = sacks * 1 + interceptions * 2 + fumble_recoveries * 2 + defensive_tds * 6
            
            # Points allowed adjustment
            if points_allowed <= 6:
                fantasy_points += 5
            elif points_allowed <= 13:
                fantasy_points += 3
            elif points_allowed <= 20:
                fantasy_points += 1
            elif points_allowed >= 35:
                fantasy_points -= 3
                
            projections.update({
                'Sacks': max(0, int(sacks)),
                'Interceptions': max(0, int(interceptions)),
                'FumbleRecoveries': max(0, int(fumble_recoveries)),
                'DefensiveTDs': max(0, int(defensive_tds)),
                'PointsAllowed': max(0, round(points_allowed, 1)),
                'ProjectedPoints': max(1, round(fantasy_points, 1))
            })
        
        return projections

    def _calculate_realistic_salary(self, projected_points: float, position: str) -> int:
        """Calculate realistic FanDuel salaries based on projections"""
        # Base salary ranges by position (FanDuel typical ranges)
        base_salaries = {
            'QB': {'min': 6500, 'max': 9000, 'avg': 7500},
            'RB': {'min': 4500, 'max': 9500, 'avg': 6800},
            'WR': {'min': 4500, 'max': 9000, 'avg': 6200},
            'TE': {'min': 4000, 'max': 7500, 'avg': 5500},
            'K': {'min': 4600, 'max': 5200, 'avg': 4900},
            'DST': {'min': 4200, 'max': 5500, 'avg': 4800}
        }
        
        salary_info = base_salaries.get(position, {'min': 4000, 'max': 8000, 'avg': 6000})
        
        # Calculate salary based on projected points with some randomness
        avg_points = {'QB': 18, 'RB': 12, 'WR': 10, 'TE': 8, 'K': 7, 'DST': 6}
        expected_points = avg_points.get(position, 10)
        
        multiplier = projected_points / expected_points
        base_salary = salary_info['avg'] * multiplier
        
        # Add randomness and round to nearest 100
        variance = random.uniform(0.85, 1.15)
        final_salary = int(base_salary * variance / 100) * 100
        
        # Enforce min/max bounds
        final_salary = max(salary_info['min'], min(final_salary, salary_info['max']))
        
        return final_salary

    def _generate_realistic_name(self, position: str) -> str:
        """Generate realistic player names"""
        if position == 'DST':
            # Defense names
            city_names = [
                'Kansas City', 'Baltimore', 'Buffalo', 'Cincinnati', 'Cleveland',
                'Denver', 'Houston', 'Indianapolis', 'Jacksonville', 'Las Vegas',
                'Los Angeles', 'Miami', 'New England', 'New York', 'Pittsburgh',
                'Tennessee', 'Dallas', 'Philadelphia', 'Washington', 'Arizona',
                'Atlanta', 'Carolina', 'Chicago', 'Detroit', 'Green Bay',
                'Minnesota', 'New Orleans', 'Seattle', 'San Francisco', 'Tampa Bay'
            ]
            return f"{random.choice(city_names)} Defense"
        else:
            return fake.name()

    def _validate_sample_data(self, players_df: pd.DataFrame):
        """Validate the generated sample data"""
        required_fields = ['PlayerID', 'Name', 'Position', 'Team', 'Salary', 'ProjectedPoints']
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in players_df.columns]
        if missing_fields:
            raise DataIngestionError(f"Missing required fields: {missing_fields}")
        
        # Validate data types and ranges
        if players_df['Salary'].min() < 3000 or players_df['Salary'].max() > 12000:
            logger.warning("Some salaries are outside expected range")
            
        if players_df['ProjectedPoints'].min() < 0:
            raise DataIngestionError("Negative projected points found")
            
        # Check position distribution
        position_counts = players_df['Position'].value_counts()
        required_positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
        
        for pos in required_positions:
            if pos not in position_counts:
                raise DataIngestionError(f"No {pos} players generated")
                
        logger.info("Sample data validation passed")

    async def get_current_players(self) -> pd.DataFrame:
        """Get current player data"""
        if self.current_data is None:
            await self.create_sample_data()
        return self.current_data.copy()
        
    async def refresh_data(self):
        """Refresh player data"""
        logger.info("Refreshing player data...")
        await self.create_sample_data()
        
    async def get_last_update_time(self) -> Optional[datetime]:
        """Get last update timestamp"""
        return self.last_update
