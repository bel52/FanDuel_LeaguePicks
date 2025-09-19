from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    DST = "DST"
    FLEX = "FLEX"

class Player(BaseModel):
    id: str
    name: str
    position: Position
    team: str
    opponent: str
    salary: int
    projected_points: float
    floor: Optional[float] = None
    ceiling: Optional[float] = None
    ownership_projection: Optional[float] = None
    injury_status: Optional[str] = None
    game_time: Optional[datetime] = None
    is_home: bool = True
    weather_impact: Optional[float] = 1.0
    
    @validator('salary')
    def validate_salary(cls, v):
        if not 3000 <= v <= 15000:
            raise ValueError(f'Salary {v} out of valid range')
        return v
    
    @property
    def value(self) -> float:
        """Points per thousand dollars"""
        return (self.projected_points / self.salary) * 1000 if self.salary > 0 else 0
    
    @property
    def adjusted_projection(self) -> float:
        """Weather-adjusted projection"""
        return self.projected_points * self.weather_impact

class Lineup(BaseModel):
    players: List[Player]
    total_salary: int
    total_projected: float
    stack_score: Optional[float] = 0
    variance: Optional[float] = None
    ownership_sum: Optional[float] = None
    
    @validator('players')
    def validate_lineup_size(cls, v):
        if len(v) != 9:  # FanDuel uses 9 players
            raise ValueError(f'Lineup must have exactly 9 players, got {len(v)}')
        return v
    
    @validator('total_salary')
    def validate_salary_cap(cls, v):
        if v > 60000:
            raise ValueError(f'Lineup salary {v} exceeds cap of 60000')
        return v
    
    def get_position_counts(self) -> Dict[str, int]:
        counts = {}
        for player in self.players:
            pos = player.position.value
            counts[pos] = counts.get(pos, 0) + 1
        return counts

class Game(BaseModel):
    game_id: str
    home_team: str
    away_team: str
    over_under: float
    home_spread: float
    game_time: datetime
    weather: Optional[Dict] = None
    
    @property
    def total(self) -> float:
        return self.over_under
    
    @property
    def is_high_total(self) -> bool:
        return self.over_under >= 47

class OptimizationSettings(BaseModel):
    lineup_type: str = "cash"  # cash, gpp, balanced
    num_lineups: int = 1
    max_exposure: float = 0.5
    min_salary: int = 58000
    stack_rules: Optional[Dict] = None
    unique_players: int = 3  # Minimum unique players between lineups
    correlation_rules: bool = True
    weather_adjustments: bool = True
    
class SlateInfo(BaseModel):
    slate_id: str
    slate_type: str  # main, night, afternoon, etc
    start_time: datetime
    games: List[Game]
    total_players: int
