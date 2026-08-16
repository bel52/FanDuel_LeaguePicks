"""Objective functions — one per contest profile.

The core design rule (and the thing v5 got wrong): an objective is a statement about
what the contest PAYS, expressed in dollars. It is never a hand-tuned score built from
stacked multipliers. Every term below traces to a real prize.

Season-league formats pay on two independent axes and we optimize both at once:

  weekly prize   -> depends on the FIELD    -> P(rank = 1) x weekly_prize
  season standing-> depends only on YOU     -> E[points] x marginal $/point

The exchange rate between them is derived, not assumed. Simulating a 12-player,
21-week season (weekly sd ~28, between-player skill sd ~6, prizes 135/81/54) gives
~$3.42 of season equity per +1 expected point per week, versus $0.128 per +1% of
weekly win probability. So one expected point is worth about 1.3 points of win rate.
That ratio is recomputed by season_marginal_value() whenever the prize structure
changes, so switching league settings re-derives the weights rather than needing a
new magic number.

Format differences that matter:
  TOTAL_SCORES : every week counts -> expected points dominates; floors matter,
                 a zero from a late scratch is permanent damage.
  MOST_WINS    : only firsts count -> P(rank=1) is everything; 2nd == 12th, so
                 variance is free and should be bought aggressively.
  BEST_N       : only your top N weeks count -> spike-seeking like MOST_WINS, but
                 with diminishing weight once N good weeks are banked.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Leaderboard(str, Enum):
    TOTAL_SCORES = "total_scores"
    BEST_5 = "best_5"
    BEST_10 = "best_10"
    MOST_WINS = "most_wins"
    NONE = "none"                  # one-off contests with no season standing


@dataclass
class SeasonContext:
    """Where the season stands. Late-season strategy differs from early."""
    leaderboard: Leaderboard = Leaderboard.NONE
    weeks_total: int = 21
    weeks_played: int = 0
    my_points: float = 0.0
    leader_points: float = 0.0
    my_wins: int = 0
    leader_wins: int = 0
    grand_prizes: tuple[float, ...] = (135.0, 81.0, 54.0)
    field_size: int = 12
    weekly_prize: float = 12.84

    @property
    def weeks_left(self) -> int:
        return max(0, self.weeks_total - self.weeks_played)

    def deficit(self) -> float:
        """How far behind the leader, in the currency of this leaderboard."""
        if self.leaderboard == Leaderboard.MOST_WINS:
            return float(self.leader_wins - self.my_wins)
        return self.leader_points - self.my_points


def season_marginal_value(ctx: SeasonContext, weekly_sd: float = 28.0,
                          skill_sd: float = 6.0, n_sims: int = 20000,
                          seed: int = 11) -> float:
    """$ of season prize per +1 expected point per week, for the remaining schedule.

    Derived by simulation rather than assumed. Returns 0 for formats where accumulated
    points do not decide the standing.
    """
    if ctx.leaderboard in (Leaderboard.NONE, Leaderboard.MOST_WINS):
        return 0.0
    W = max(1, ctx.weeks_left)
    P = ctx.field_size
    prizes = np.array(list(ctx.grand_prizes) + [0.0] * max(0, P - len(ctx.grand_prizes)))
    rng = np.random.default_rng(seed)

    def ev(edge: float) -> float:
        skills = rng.normal(0.0, skill_sd, (n_sims, P))
        skills[:, 0] = edge
        totals = skills * W + rng.normal(0, weekly_sd * np.sqrt(W), (n_sims, P))
        totals[:, 0] += ctx.my_points
        totals[:, 1:] += ctx.leader_points          # conservative: field near the leader
        rank = (totals > totals[:, [0]]).sum(axis=1)
        return float(prizes[np.clip(rank, 0, P - 1)].mean())

    return round((ev(1.0) - ev(-1.0)) / 2.0, 4)


@dataclass
class ObjectiveWeights:
    """Dollar-denominated. w_points x E[pts] + w_win x P(win)."""
    w_points: float          # $ per expected point per week
    w_win: float             # $ per unit of win probability
    rationale: str

    def score(self, exp_points: float, p_win: float) -> float:
        return self.w_points * exp_points + self.w_win * p_win


def weights_for(profile: str, ctx: SeasonContext, entry_fee: float = 0.0,
                prize_pool: float | None = None) -> ObjectiveWeights:
    """Resolve a profile + season state into dollar weights."""
    if profile == "friends_league":
        w_pts = season_marginal_value(ctx)
        w_win = ctx.weekly_prize
        if ctx.leaderboard == Leaderboard.MOST_WINS:
            return ObjectiveWeights(
                0.0, w_win + _most_wins_equity(ctx),
                "Most Wins: 2nd pays the same as last. Only firsts have value, so "
                "expected points are worth nothing except as a route to a win — buy variance.")
        if ctx.leaderboard in (Leaderboard.BEST_5, Leaderboard.BEST_10):
            n = 5 if ctx.leaderboard == Leaderboard.BEST_5 else 10
            banked = min(ctx.weeks_played, n) / n
            return ObjectiveWeights(
                (w_pts / max(1, ctx.weeks_left)) * (1 - 0.6 * banked), w_win,
                f"Best {n}: only your top {n} weeks count, so points matter less once "
                f"good weeks are banked ({banked:.0%} banked).")
        # UNITS: season_marginal_value is $ per +1 point per week SUSTAINED for the
        # remaining schedule. A lineup's E[pts] is ONE week, so the per-week value of a
        # point is that figure divided by the weeks it would be sustained over.
        # Multiplying the sustained rate by a single week's score overstates by ~W.
        per_week = w_pts / max(1, ctx.weeks_left)
        return ObjectiveWeights(
            per_week, w_win,
            f"Total Scores: every week counts. ${per_week:.3f} of season equity per point "
            f"THIS week (${w_pts:.2f} per point/week sustained over {ctx.weeks_left} weeks) "
            f"vs ${w_win:.2f} weekly prize — 1 point ~ "
            f"{(per_week / (w_win * 0.01)) if w_win else 0:.2f}% of win rate.")

    if profile == "h2h":
        # One opponent. Beating them is the only thing that pays; margin is worthless.
        pool = prize_pool if prize_pool is not None else entry_fee * 1.8
        return ObjectiveWeights(
            0.0, pool,
            "H2H: maximize P(beat one opponent). That is close to maximizing the median, "
            "NOT the ceiling — ceiling-chasing is actively wrong in a 1v1.")

    if profile == "showdown_friends":
        pool = prize_pool if prize_pool is not None else entry_fee * 8
        return ObjectiveWeights(
            0.0, pool,
            "Single-game friend contest: small field, winner-take-all in practice. "
            "Pure P(rank=1); MVP choice dominates and correlation within one game is strong.")

    if profile == "public_gpp":
        pool = prize_pool if prize_pool is not None else entry_fee * 100
        return ObjectiveWeights(
            0.0, pool,
            "Public GPP: large anonymous field. P(rank=1) with ownership leverage; "
            "expected points are a poor proxy at this field size.")

    raise ValueError(f"unknown profile {profile!r}")


def _most_wins_equity(ctx: SeasonContext) -> float:
    """Extra $ per weekly win, from the season grand prize under Most Wins."""
    if ctx.leaderboard != Leaderboard.MOST_WINS or ctx.weeks_left <= 0:
        return 0.0
    return float(sum(ctx.grand_prizes)) / max(1.0, ctx.weeks_total / ctx.field_size) / ctx.weeks_total


@dataclass
class ScoredLineup:
    exp_points: float
    p_win: float
    p_top3: float
    median: float
    p10: float
    p90: float
    dollars: float
    weights: ObjectiveWeights

    def summary(self) -> str:
        return (f"${self.dollars:6.2f} | E[pts]={self.exp_points:6.1f} P(win)={self.p_win:5.1%} "
                f"P(top3)={self.p_top3:5.1%} | med={self.median:6.1f} "
                f"[{self.p10:5.1f}-{self.p90:6.1f}]")


def score_lineup(mine: np.ndarray, field_totals: np.ndarray,
                 weights: ObjectiveWeights) -> ScoredLineup:
    """Score one lineup's simulated totals against the simulated field.

    All metrics come from the JOINT simulated distribution — never from summing
    individual player percentiles, which overstates a lineup ceiling badly.
    """
    beaten = (mine[None, :] > field_totals).sum(axis=0)
    rank = field_totals.shape[0] + 1 - beaten
    p_win = float((rank == 1).mean())
    p_top3 = float((rank <= 3).mean())
    exp_points = float(mine.mean())
    return ScoredLineup(
        exp_points=round(exp_points, 2), p_win=p_win, p_top3=p_top3,
        median=round(float(np.median(mine)), 2),
        p10=round(float(np.percentile(mine, 10)), 2),
        p90=round(float(np.percentile(mine, 90)), 2),
        dollars=round(weights.score(exp_points, p_win), 4),
        weights=weights,
    )
