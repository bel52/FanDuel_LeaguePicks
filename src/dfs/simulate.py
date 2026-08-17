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
# ---------------------------------------------------------------------------
# Correlation structure.
#
# These were originally hand-picked loadings on a team/game/opponent factor model.
# Measuring the real thing (3 seasons of nflverse outcomes, residualized within
# player-season — see `corrfit.py`) showed the loadings were materially wrong AND that
# the factor model cannot represent reality:
#
#     pair                 hand-picked    measured
#     QB - same-team WR       +0.37        +0.246
#     QB - same-team TE       +0.30        +0.201
#     QB - opposing QB        +0.08        +0.134
#     WR - same-team WR      (positive)    +0.017   <-- essentially ZERO
#
# A team/game factor model forces every same-team pair to share the same loading
# product, so it cannot make QB-WR strongly positive while WR-WR is ~0. But that is
# what actually happens: the team-total effect and target competition cancel between
# two receivers. Overstating stack correlation inflates stacked-lineup ceilings and
# therefore P(win), biasing every stacking decision.
#
# So we now build an explicit per-game correlation matrix from the measured table and
# factor it with Cholesky. Fallback values below are the measured ones; DEF pairs are
# retained from the factor model because nflverse player stats carry no DST rows
# (a known gap — see PROJECT_STATE "known gaps").
CORR_FALLBACK = {
    ("QB", "WR", True): 0.246, ("QB", "TE", True): 0.201, ("QB", "RB", True): 0.049,
    ("RB", "RB", True): 0.046, ("WR", "WR", True): 0.017, ("TE", "TE", True): 0.031,
    ("TE", "WR", True): 0.012, ("RB", "WR", True): -0.007, ("RB", "TE", True): 0.009,
    ("QB", "QB", False): 0.134, ("QB", "TE", False): 0.054, ("QB", "WR", False): 0.042,
    ("RB", "RB", False): -0.038, ("TE", "TE", False): 0.021, ("WR", "WR", False): 0.020,
    ("TE", "WR", False): 0.015, ("RB", "WR", False): 0.014, ("QB", "RB", False): 0.010,
    ("RB", "TE", False): 0.001,
    # DEF: not measurable from nflverse player stats; carried over from the factor model.
    ("D", "QB", False): -0.26, ("D", "WR", False): -0.26, ("D", "TE", False): -0.20,
    ("D", "RB", False): -0.18, ("D", "QB", True): 0.10, ("D", "WR", True): 0.05,
    ("D", "TE", True): 0.05, ("D", "RB", True): 0.08, ("D", "D", False): -0.10,
}


def _load_corr_table(dist: dict | None = None) -> dict:
    """Measured correlations if data/correlations.json exists, else the fallback."""
    from pathlib import Path
    import json as _json
    f = Path(__file__).resolve().parents[2] / "data" / "correlations.json"
    if not f.exists():
        return dict(CORR_FALLBACK)
    try:
        raw = _json.loads(f.read_text())["pairs"]
    except (OSError, KeyError, ValueError):
        return dict(CORR_FALLBACK)
    out = dict(CORR_FALLBACK)
    for label, v in raw.items():
        parts = label.split(" ", 1)
        if len(parts) != 2:
            continue
        pair, rel = parts
        ps = pair.split("-")
        if len(ps) != 2:
            continue
        out[(ps[0], ps[1], rel.startswith("same"))] = float(v["r"])
    return out


def pair_corr(table: dict, pos_a: str, pos_b: str, same_team: bool) -> float:
    a, b = sorted((pos_a, pos_b))
    return table.get((a, b, same_team), 0.0)


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
        # Correlated latents: block-diagonal by GAME (cross-game correlation measured
        # at -0.022, i.e. zero), with an explicit within-game correlation matrix.
        table = _load_corr_table()
        by_game: dict[str, list[int]] = {}
        for i, p in enumerate(players):
            by_game.setdefault("|".join(sorted((p.team, p.opponent))), []).append(i)

        latent = np.empty((len(players), self.n_sims), dtype=np.float64)
        for g, idxs in by_game.items():
            k = len(idxs)
            C = np.eye(k)
            for x in range(k):
                for y in range(x + 1, k):
                    pa, pb = players[idxs[x]], players[idxs[y]]
                    r = pair_corr(table, pa.position, pb.position, pa.team == pb.team)
                    C[x, y] = C[y, x] = r
            # Nearest positive-definite: clip eigenvalues, then renormalize to unit
            # diagonal so each marginal stays standard normal.
            w, V = np.linalg.eigh(C)
            if w.min() < 1e-8:
                C = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
                d = np.sqrt(np.diag(C))
                C = C / np.outer(d, d)
            L = np.linalg.cholesky(C)
            latent[idxs] = L @ self.rng.standard_normal((k, self.n_sims))

        out = np.empty((len(players), self.n_sims), dtype=np.float32)
        for i, p in enumerate(players):
            u = _std_normal_cdf(latent[i])
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
