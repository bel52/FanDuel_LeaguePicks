"""Contest specification — first-class config. Nothing downstream hardcodes contest assumptions."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class SlateType(str, Enum):
    FULL = "full"          # Sunday Main classic
    SINGLE_GAME = "single_game"  # showdown MVP+4


class Profile(str, Enum):
    FRIENDS_LEAGUE = "friends_league"
    SHOWDOWN_FRIENDS = "showdown_friends"
    H2H = "h2h"
    PUBLIC_GPP = "public_gpp"


class PayoutTier(BaseModel):
    rank_from: int
    rank_to: int
    amount: float


class ContestSpec(BaseModel):
    """One contest = one spec. Loaded before any build."""
    name: str
    profile: Profile
    slate_type: SlateType
    field_size: int = Field(gt=1)
    entries_per_user: int = 1
    entry_fee: float = 0.0
    payouts: list[PayoutTier] = []          # empty + winner_take_all=True is valid
    winner_take_all: bool = True
    late_swap: bool = True
    salary_cap: int = 60000
    # VERIFY against a real 2026 showdown salary CSV before Week 1: whether FanDuel
    # charges 1.5x salary for the MVP slot (points multiplier is confirmed 1.5x;
    # salary treatment is contested — a second review claims 1.5x salary now applies).
    mvp_salary_mult: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> "ContestSpec":
        if not self.winner_take_all and not self.payouts:
            raise ValueError("non-WTA contest requires explicit payout tiers")
        return self

    def payout_for_rank(self, rank: int) -> float:
        if self.winner_take_all:
            return self.entry_fee * self.field_size if rank == 1 else 0.0
        for t in self.payouts:
            if t.rank_from <= rank <= t.rank_to:
                return t.amount
        return 0.0


# Roster templates (verified against FanDuel rules; re-verify live in 2.1)
ROSTER_FULL = {"QB": (1, 1), "RB": (2, 3), "WR": (3, 4), "TE": (1, 2), "D": (1, 1)}
ROSTER_FULL_SIZE = 9
ROSTER_SHOWDOWN_SIZE = 5  # 1 MVP (1.5x) + 4 utility, any position
MVP_MULTIPLIER = 1.5
