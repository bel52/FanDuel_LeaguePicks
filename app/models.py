from __future__ import annotations
from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field, validator

GameType = Literal["gpp", "cash"]
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

class Player(BaseModel):
    id: str
    name: str
    team: str
    position: Literal["QB", "RB", "WR", "TE", "DEF"]
    salary: int = Field(ge=2000, le=15000)
    projection: float = Field(ge=0)
    opponent: Optional[str] = None
    game_time: Optional[str] = None
    home: Optional[bool] = None
    notes: Optional[str] = None

    @validator("position")
    def _pos_ok(cls, v):
        if v not in {"QB","RB","WR","TE","DEF"}:
            raise ValueError("Invalid position")
        return v

class OptimizeRequest(BaseModel):
    game_type: GameType = "gpp"
    num_lineups: int = 1
    seed: Optional[int] = None

class Lineup(BaseModel):
    players: List[Player]
    total_salary: int
    projected_points: float
    game_type: GameType
    notes: Optional[str] = None
    ai_commentary: Optional[str] = None

class PlayersResponse(BaseModel):
    week: int
    count: int
    players: List[Player]

class OptimizeResponse(BaseModel):
    lineup: Lineup
    metadata: Dict = {}
