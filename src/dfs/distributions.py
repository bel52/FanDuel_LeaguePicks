"""Empirical outcome distributions from nflverse historicals (2022–2025).

Replaces v5's fabricated ±50% floor/ceiling. Method:
1. Compute FanDuel-scored points for every player-week (half-PPR + bonuses).
2. For each player-week, compute expectation proxy = trailing 4-week median (min 3 games).
3. ratio = actual / expectation. Pool ratios by position x expectation tier.
4. A projection's floor/ceiling = projection x pooled ratio percentiles.

This yields calibrated, position- and tier-specific distributions grounded in what
actually happened, including bust/boom asymmetry. Seeds are deterministic downstream.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

FD_SEASONS = [2022, 2023, 2024, 2025]

# FanDuel NFL scoring (verify live in 2.1; current known rules)
def fanduel_points(r: pd.Series) -> float:
    pts = 0.0
    pts += r.get("passing_yards", 0) * 0.04
    pts += r.get("passing_tds", 0) * 4.0
    pts += r.get("passing_interceptions", r.get("interceptions", 0)) * -1.0
    pts += 3.0 if r.get("passing_yards", 0) >= 300 else 0.0
    pts += r.get("rushing_yards", 0) * 0.1
    pts += r.get("rushing_tds", 0) * 6.0
    pts += 3.0 if r.get("rushing_yards", 0) >= 100 else 0.0
    pts += r.get("receptions", 0) * 0.5
    pts += r.get("receiving_yards", 0) * 0.1
    pts += r.get("receiving_tds", 0) * 6.0
    pts += 3.0 if r.get("receiving_yards", 0) >= 100 else 0.0
    pts += r.get("passing_2pt_conversions", 0) * 2.0
    pts += r.get("rushing_2pt_conversions", 0) * 2.0
    pts += r.get("receiving_2pt_conversions", 0) * 2.0
    pts += (r.get("sack_fumbles_lost", 0) + r.get("rushing_fumbles_lost", 0)
            + r.get("receiving_fumbles_lost", 0)) * -2.0
    return round(float(pts), 2)


# Expectation tiers (FanDuel-points scale) per position
TIERS = {
    "QB": [(0, 14), (14, 18), (18, 22), (22, 99)],
    "RB": [(0, 8), (8, 12), (12, 16), (16, 99)],
    "WR": [(0, 7), (7, 11), (11, 15), (15, 99)],
    "TE": [(0, 6), (6, 9), (9, 12), (12, 99)],
}
PCTS = [5, 10, 25, 50, 75, 90, 95]


def build_distributions(seasons: list[int] = FD_SEASONS, out_path: str | Path | None = None) -> dict:
    import nflreadpy as nfl
    df = nfl.load_player_stats(seasons).to_pandas()
    df = df[(df["season_type"] == "REG") & (df["position"].isin(["QB", "RB", "WR", "TE"]))].copy()
    df["fd_pts"] = df.apply(fanduel_points, axis=1)
    df = df.sort_values(["player_id", "season", "week"])

    # trailing 4-game median within season as expectation proxy
    df["expect"] = (
        df.groupby(["player_id", "season"])["fd_pts"]
        .transform(lambda s: s.shift(1).rolling(4, min_periods=3).median())
    )
    obs = df.dropna(subset=["expect"])
    obs = obs[obs["expect"] >= 3.0]  # avoid ratio blowups on near-zero expectations
    obs = obs.assign(ratio=(obs["fd_pts"] / obs["expect"]).clip(0, 6))

    dist: dict = {"meta": {"seasons": seasons, "n_obs": int(len(obs)),
                           "method": "trailing4-median ratio pools", "percentiles": PCTS}}
    for pos, tiers in TIERS.items():
        dist[pos] = {}
        pool_pos = obs[obs["position"] == pos]
        for lo, hi in tiers:
            pool = pool_pos[(pool_pos["expect"] >= lo) & (pool_pos["expect"] < hi)]["ratio"]
            key = f"{lo}-{hi}"
            if len(pool) < 100:
                dist[pos][key] = None
                continue
            dist[pos][key] = {
                "n": int(len(pool)),
                "pcts": {str(p): round(float(np.percentile(pool, p)), 4) for p in PCTS},
                "mean": round(float(pool.mean()), 4),
                "zero_rate": round(float((pool <= 0.05).mean()), 4),
            }
    if out_path:
        Path(out_path).write_text(json.dumps(dist, indent=1))
    return dist


def load_distributions(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def floor_ceiling(projection: float, position: str, dist: dict) -> tuple[float, float]:
    """p10 floor / p90 ceiling for a projection, from empirical ratio pools."""
    tiers = TIERS.get(position)
    if tiers is None:  # D handled separately (defense scoring model TBD Phase 1.7)
        return round(projection * 0.4, 2), round(projection * 1.8, 2)
    for lo, hi in tiers:
        if lo <= projection < hi:
            cell = dist.get(position, {}).get(f"{lo}-{hi}")
            if cell:
                return (round(projection * cell["pcts"]["10"], 2),
                        round(projection * cell["pcts"]["90"], 2))
    cell = dist.get(position, {}).get(f"{tiers[-1][0]}-{tiers[-1][1]}")
    if cell:
        return (round(projection * cell["pcts"]["10"], 2),
                round(projection * cell["pcts"]["90"], 2))
    raise ValueError(f"no distribution cell for {position} proj={projection}")
