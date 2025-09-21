import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import config

Base = declarative_base()

class PlayerDB(Base):
    __tablename__ = 'players'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    position = Column(String, nullable=False)
    team = Column(String, nullable=False)
    salary = Column(Integer)
    projected_points = Column(Float)
    actual_points = Column(Float)
    ownership = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    slate_id = Column(String)
    meta_data = Column(Text)  # JSON stored as text for SQLite compatibility

class LineupDB(Base):
    __tablename__ = 'lineups'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    slate_id = Column(String)
    players = Column(Text)  # JSON stored as text
    total_salary = Column(Integer)
    projected_points = Column(Float)
    actual_points = Column(Float)
    finish_position = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    lineup_type = Column(String)
    
class ProjectionDB(Base):
    __tablename__ = 'projections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(String)
    source = Column(String)  # ESPN, NumberFire, etc
    projection = Column(Float)
    floor = Column(Float)
    ceiling = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Database:
    def __init__(self, db_url=None):
        self.db_url = db_url or config.DATABASE_URL
        
        # Handle SQLite connection differently
        if self.db_url.startswith('sqlite'):
            # SQLite doesn't need special connection arguments
            self.engine = create_engine(self.db_url, connect_args={'check_same_thread': False})
        else:
            # For other databases (PostgreSQL, MySQL, etc.)
            self.engine = create_engine(self.db_url)
            
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_player(self, player_data: dict):
        """Save player data to database"""
        # Convert any dict fields to JSON strings for SQLite
        if 'meta_data' in player_data and isinstance(player_data['meta_data'], dict):
            player_data['meta_data'] = json.dumps(player_data['meta_data'])
            
        player = self.session.query(PlayerDB).filter_by(id=player_data.get('id')).first()
        
        if player:
            # Update existing player
            for key, value in player_data.items():
                setattr(player, key, value)
        else:
            # Create new player
            player = PlayerDB(**player_data)
            self.session.add(player)
            
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving player: {e}")
    
    def get_players_by_slate(self, slate_id: str) -> List[dict]:
        """Get all players for a specific slate"""
        players = self.session.query(PlayerDB).filter_by(slate_id=slate_id).all()
        return [self._player_to_dict(p) for p in players]
    
    def save_lineup(self, lineup_data: dict):
        """Save lineup to database"""
        # Convert players list to JSON string for SQLite
        if 'players' in lineup_data and isinstance(lineup_data['players'], list):
            lineup_data['players'] = json.dumps(lineup_data['players'])
            
        lineup = LineupDB(**lineup_data)
        self.session.add(lineup)
        
        try:
            self.session.commit()
            return lineup.id
        except Exception as e:
            self.session.rollback()
            print(f"Error saving lineup: {e}")
            return None
    
    def get_lineups_by_type(self, lineup_type: str, limit: int = 10) -> List[dict]:
        """Get recent lineups by type"""
        lineups = self.session.query(LineupDB)\
            .filter_by(lineup_type=lineup_type)\
            .order_by(LineupDB.created_at.desc())\
            .limit(limit)\
            .all()
        
        result = []
        for lineup in lineups:
            lineup_dict = {
                'id': lineup.id,
                'slate_id': lineup.slate_id,
                'players': json.loads(lineup.players) if lineup.players else [],
                'total_salary': lineup.total_salary,
                'projected_points': lineup.projected_points,
                'actual_points': lineup.actual_points,
                'created_at': lineup.created_at,
                'lineup_type': lineup.lineup_type
            }
            result.append(lineup_dict)
        
        return result
    
    def save_projection(self, projection_data: dict):
        """Save player projection from a specific source"""
        projection = ProjectionDB(**projection_data)
        self.session.add(projection)
        
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            print(f"Error saving projection: {e}")
    
    def get_latest_projections(self, player_id: str) -> Dict[str, float]:
        """Get latest projection for a player from each source"""
        # Get the most recent projection from each source
        from sqlalchemy import func
        
        subquery = self.session.query(
            ProjectionDB.source,
            func.max(ProjectionDB.timestamp).label('max_timestamp')
        ).filter_by(player_id=player_id).group_by(ProjectionDB.source).subquery()
        
        projections = self.session.query(ProjectionDB).join(
            subquery,
            (ProjectionDB.source == subquery.c.source) &
            (ProjectionDB.timestamp == subquery.c.max_timestamp)
        ).filter_by(player_id=player_id).all()
        
        result = {}
        for proj in projections:
            result[proj.source] = {
                'projection': proj.projection,
                'floor': proj.floor,
                'ceiling': proj.ceiling,
                'timestamp': proj.timestamp
            }
        
        return result
    
    def get_historical_performance(self, player_id: str, games: int = 5) -> List[float]:
        """Get player's last N games performance"""
        results = self.session.query(PlayerDB.actual_points)\
            .filter_by(id=player_id)\
            .filter(PlayerDB.actual_points.isnot(None))\
            .order_by(PlayerDB.timestamp.desc())\
            .limit(games)\
            .all()
        return [r[0] for r in results]
    
    def get_player_stats(self, player_id: str) -> Optional[dict]:
        """Get comprehensive player statistics"""
        player = self.session.query(PlayerDB).filter_by(id=player_id).first()
        
        if not player:
            return None
        
        stats = self._player_to_dict(player)
        
        # Add historical performance
        stats['last_5_games'] = self.get_historical_performance(player_id, 5)
        
        # Add projections from various sources
        stats['projections'] = self.get_latest_projections(player_id)
        
        return stats
    
    def _player_to_dict(self, player: PlayerDB) -> dict:
        """Convert player DB object to dictionary"""
        return {
            'id': player.id,
            'name': player.name,
            'position': player.position,
            'team': player.team,
            'salary': player.salary,
            'projected_points': player.projected_points,
            'actual_points': player.actual_points,
            'ownership': player.ownership,
            'timestamp': player.timestamp,
            'slate_id': player.slate_id,
            'meta_data': json.loads(player.meta_data) if player.meta_data else {}
        }
    
    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Delete old players
        self.session.query(PlayerDB).filter(PlayerDB.timestamp < cutoff_date).delete()
        
        # Delete old lineups
        self.session.query(LineupDB).filter(LineupDB.created_at < cutoff_date).delete()
        
        # Delete old projections
        self.session.query(ProjectionDB).filter(ProjectionDB.timestamp < cutoff_date).delete()
        
        try:
            self.session.commit()
            print(f"Cleaned up data older than {days} days")
        except Exception as e:
            self.session.rollback()
            print(f"Error cleaning up data: {e}")
    
    def close(self):
        """Close database session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Create singleton database instance
try:
    db = Database()
except Exception as e:
    print(f"Database initialization warning: {e}")
    # Create a fallback in-memory database
    db = Database("sqlite:///:memory:")
