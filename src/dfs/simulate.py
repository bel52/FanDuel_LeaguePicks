"""Correlated slate simulator.

Fixes v5 diagnosis #6 (summed marginals) and the independence assumption. Structure:

  team_game_score[g,t] ~ latent game/team script factor (shared by all players on a team)
  player_outcome = projection * ratio_draw(position, tier) * (1 + rho_pos * team_factor)
                                                            * (1 + gamma * game_factor)

Ratio draws come from the EMPIRICAL distributions (distributions.py), not a normal
around the projection. Correlation is injected via shared latent factors, so QBs and
their pass-catchers boom together, opposing skill players share a shootout factor,
and a team's DEF moves inversely to its opponent's offense.

Every lineup metric (median, p90, win probability) is computed from the JOINT
distribution of simulated lineup totals — never from summed player percentiles.
Seeded and deterministic.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


def _std_normal_cdf(z: np.ndarray) -> np.ndarray:
    """Phi(z) via erf — no scipy dependency."""
    from math import sqrt
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 vectorized erf."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return sign * y

from .contest_spec import MVP_MULTIPLIER
from .distributions import TIERS
from .slate import PlayerSlate, SlatePlayer

# --- Gaussian copula loadings ---
# Each player's outcome RANK is driven by: a shared TEAM factor, a shared GAME factor,
# and idiosyncratic noise. corr(i,j) = a_i*a_j (same team) + b_i*b_j (same game).
# Targets from published DFS correlation work: QB<->same-team WR ~0.35-0.45,
# QB<->opposing WR (bring-back) ~0.10-0.20, RB weakly/negatively tied to pass game,
# DEF negatively tied to the opposing offense.
TEAM_LOAD = {"QB": 0.62, "WR": 0.55, "TE": 0.45, "RB": 0.25, "D": 0.35}
GAME_LOAD = {"QB": 0.30, "WR": 0.32, "TE": 0.26, "RB": 0.18, "D": -0.30}
OPP_LOAD = {"D": -0.45}   # loading on the OPPONENT team factor (DEF hurt by opp offense)


@dataclass
class SimResult:
    player_ids: list[str]
    totals: np.ndarray            # shape (n_sims,) — joint lineup totals

    def median(self) -> float:
        return float(np.median(self.totals))

    def pct(self, p: float) -> float:
        return float(np.percentile(self.totals, p))


class SlateSimulator:
    def __init__(self, slate: PlayerSlate, distributions: dict,
                 n_sims: int = 20000, seed: int = 1729):
        self.slate = slate
        self.dist = distributions
        self.n_sims = n_sims
        self.rng = np.random.default_rng(seed)
        self.index = {p.fd_id: i for i, p in enumerate(slate.players)}
        self.matrix = self._simulate_all()

    # ---- ratio pools ----
    def _pool_curve(self, p: SlatePlayer):
        """Empirical ratio percentile curve (quantiles, values, zero_rate) for this player."""
        from .distributions import _nearest_cell
        cell = _nearest_cell(p.projection, p.position, self.dist)
        if cell is None:
            # K and D have no calibrated pools; a generic spread until they are built.
            pcts = {"5": 0.15, "10": 0.35, "25": 0.65, "50": 0.95,
                    "75": 1.35, "90": 1.85, "95": 2.15}
            zero, mean_ratio = 0.02, 1.0
        else:
            pcts, zero = cell["pcts"], cell["zero_rate"]
            mean_ratio = cell.get("mean_ratio")
        qs = np.array([5, 10, 25, 50, 75, 90, 95]) / 100.0
        vals = np.array([pcts[k] for k in ("5", "10", "25", "50", "75", "90", "95")])
        return qs, vals, zero, mean_ratio

    def _draw_from_curve(self, u: np.ndarray, qs, vals, zero: float) -> np.ndarray:
        """Map correlated uniforms through the empirical curve, with tail extension."""
        draws = np.interp(u, qs, vals)
        lo_mask, hi_mask = u < 0.05, u > 0.95
        # linear ramp to 0 below p5; expanding tail above p95
        draws[lo_mask] = vals[0] * (u[lo_mask] / 0.05)
        draws[hi_mask] = vals[-1] * (1.0 + 1.4 * (u[hi_mask] - 0.95) / 0.05)
        # zero games hit the lowest-ranked outcomes, not random ones (injury/ejection)
        if zero > 0:
            draws[u < zero] = 0.0
        return draws

    def _simulate_all(self) -> np.ndarray:
        """Return (n_players, n_sims) matrix of simulated FanDuel points, rank-correlated."""
        players = self.slate.players
        teams = sorted({p.team for p in players})
        games = sorted({"|".join(sorted((p.team, p.opponent))) for p in players})
        team_z = {t: self.rng.standard_normal(self.n_sims) for t in teams}
        game_z = {g: self.rng.standard_normal(self.n_sims) for g in games}

        out = np.empty((len(players), self.n_sims), dtype=np.float32)
        for i, p in enumerate(players):
            g = "|".join(sorted((p.team, p.opponent)))
            a = TEAM_LOAD.get(p.position, 0.35)
            b = GAME_LOAD.get(p.position, 0.20)
            c = OPP_LOAD.get(p.position, 0.0)
            resid = max(0.0, 1.0 - (a * a + b * b + c * c))
            z = (a * team_z[p.team] + b * game_z[g]
                 + c * team_z.get(p.opponent, 0.0)
                 + np.sqrt(resid) * self.rng.standard_normal(self.n_sims))
            u = _std_normal_cdf(z)
            qs, vals, zero, mean_ratio = self._pool_curve(p)
            ratios = self._draw_from_curve(u, qs, vals, zero)
            # MEAN PINNING. Reconstructing a distribution from 7 quantiles plus an
            # interpolated upper tail does not preserve the mean of the residuals it
            # came from — measured drift was +14% (RB +16%, WR +14%). Rescale the
            # sampled ratios so their mean equals the cell's empirical mean ratio.
            # Without this, every ceiling, P(win), and dollar figure is inflated.
            if mean_ratio:
                drawn = float(ratios.mean())
                if drawn > 1e-6:
                    ratios = ratios * (mean_ratio / drawn)
            out[i] = np.clip(p.projection * ratios, 0, None)
        return out

    # ---- lineup scoring ----
    def score(self, player_ids: tuple[str, ...] | list[str],
              mvp_id: str | None = None) -> SimResult:
        rows = [self.index[pid] for pid in player_ids]
        totals = self.matrix[rows].sum(axis=0)
        if mvp_id is not None:
            totals = totals + (MVP_MULTIPLIER - 1) * self.matrix[self.index[mvp_id]]
        return SimResult(player_ids=list(player_ids), totals=totals)

    def score_many(self, lineups: list[tuple[tuple[str, ...], str | None]]) -> np.ndarray:
        """Vectorized: returns (n_lineups, n_sims)."""
        out = np.empty((len(lineups), self.n_sims), dtype=np.float32)
        for k, (ids, mvp) in enumerate(lineups):
            out[k] = self.score(ids, mvp).totals
        return out
