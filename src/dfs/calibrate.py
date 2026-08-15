"""Calibration harness — validates the projection layer against real outcomes and
rebuilds the empirical distributions from TRUE residuals.

Why this matters: distributions.py v1 used a trailing-4-week-median proxy for
"expectation" because no projection source was wired up yet. Now that FantasyPros
projections are live, we can compute the real quantity the simulator needs:

    ratio = actual_FanDuel_points / projected_FanDuel_points

pooled by position x projection tier. That is exactly the distribution the Monte Carlo
samples from, so calibrating it on truth removes the last piece of guesswork in the
simulation layer.

Also reports projection error (MAE/RMSE/bias) by position so we know how good the
inputs actually are before trusting any lineup the optimizer emits.

Usage:
    python3 -m dfs.calibrate --season 2025 --weeks 1-17 --out data/calibration_2025.json
    python3 -m dfs.calibrate --rebuild-distributions data/calibration_2025.json \
                             --out data/distributions.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .fantasypros import FantasyProsClient, FantasyProsError
from .matching import norm_name
from .scoring import score
from .distributions import TIERS, PCTS, fanduel_points

POSITIONS = ["QB", "RB", "WR", "TE"]


def actual_points(season: int) -> dict[tuple[str, int], float]:
    """(normalized_name, week) -> actual FanDuel points, from nflverse."""
    import nflreadpy as nfl
    df = nfl.load_player_stats([season]).to_pandas()
    df = df[(df["season_type"] == "REG") & (df["position"].isin(POSITIONS))].copy()
    df["fd"] = df.apply(fanduel_points, axis=1)
    out: dict[tuple[str, int], float] = {}
    for _, r in df.iterrows():
        out[(norm_name(r["player_display_name"]), int(r["week"]))] = float(r["fd"])
    return out


def collect(season: int, weeks: list[int], sleep: float = 0.4) -> dict:
    """Pull FP projections for each week, join to actuals, return paired observations."""
    client = FantasyProsClient()
    actuals = actual_points(season)
    print(f"actuals loaded: {len(actuals)} player-weeks")

    obs: list[dict] = []
    for wk in weeks:
        try:
            projs = client.weekly_projections(season, wk, positions=POSITIONS)
        except FantasyProsError as e:
            print(f"  week {wk}: FAILED ({e})")
            continue
        hit = 0
        for p in projs:
            if p.points <= 0:
                continue
            a = actuals.get((norm_name(p.name), wk))
            if a is None:
                continue
            obs.append({"week": wk, "name": p.name, "pos": p.position,
                        "proj": round(p.points, 2), "actual": round(a, 2)})
            hit += 1
        print(f"  week {wk:2d}: {len(projs):4d} projected, {hit:4d} joined to actuals")
        time.sleep(sleep)
    return {"season": season, "weeks": weeks, "n": len(obs), "obs": obs}


def error_report(data: dict) -> str:
    lines = [f"PROJECTION ACCURACY — {data['season']}, n={data['n']}", ""]
    lines.append(f"  {'pos':4s} {'n':>5s} {'MAE':>7s} {'RMSE':>7s} {'bias':>7s} "
                 f"{'corr':>6s}  {'proj_mean':>9s} {'act_mean':>9s}")
    for pos in POSITIONS + ["ALL"]:
        rows = [o for o in data["obs"] if pos == "ALL" or o["pos"] == pos]
        if len(rows) < 20:
            continue
        p = np.array([r["proj"] for r in rows])
        a = np.array([r["actual"] for r in rows])
        lines.append(
            f"  {pos:4s} {len(rows):5d} {np.abs(a - p).mean():7.2f} "
            f"{np.sqrt(((a - p) ** 2).mean()):7.2f} {(a - p).mean():+7.2f} "
            f"{np.corrcoef(p, a)[0, 1]:6.3f}  {p.mean():9.2f} {a.mean():9.2f}")
    lines.append("")
    lines.append("  bias > 0 = FantasyPros under-projects; corr is the ceiling on how much")
    lines.append("  any optimizer built on these projections can possibly know.")
    return "\n".join(lines)


def rebuild_distributions(data: dict, min_proj: float = 3.0, min_n: int = 80) -> dict:
    """Empirical actual/projected ratio pools by position x projection tier."""
    dist: dict = {"meta": {"source": "calibrated", "season": data["season"],
                           "n_obs": 0, "percentiles": PCTS,
                           "method": "actual/FP-projected ratio pools"}}
    total = 0
    for pos, tiers in TIERS.items():
        dist[pos] = {}
        pool_pos = [o for o in data["obs"] if o["pos"] == pos and o["proj"] >= min_proj]
        for lo, hi in tiers:
            rows = [o for o in pool_pos if lo <= o["proj"] < hi]
            key = f"{lo}-{hi}"
            if len(rows) < min_n:
                dist[pos][key] = None
                continue
            ratios = np.clip(np.array([o["actual"] / o["proj"] for o in rows]), 0, 6)
            total += len(ratios)
            dist[pos][key] = {
                "n": int(len(ratios)),
                "pcts": {str(p): round(float(np.percentile(ratios, p)), 4) for p in PCTS},
                "mean": round(float(ratios.mean()), 4),
                "zero_rate": round(float((ratios <= 0.05).mean()), 4),
            }
    dist["meta"]["n_obs"] = total
    return dist


def compare(old_path: Path, new: dict) -> str:
    """Show how calibration moved the distributions vs the proxy version."""
    if not old_path.exists():
        return ""
    old = json.loads(old_path.read_text())
    lines = ["", "DISTRIBUTION SHIFT (proxy -> calibrated)", "",
             f"  {'pos':4s} {'tier':9s} {'p10 old':>8s} {'p10 new':>8s} "
             f"{'p90 old':>8s} {'p90 new':>8s} {'n':>6s}"]
    for pos, tiers in TIERS.items():
        for lo, hi in tiers:
            k = f"{lo}-{hi}"
            o, n = old.get(pos, {}).get(k), new.get(pos, {}).get(k)
            if not o or not n:
                continue
            lines.append(f"  {pos:4s} {k:9s} {o['pcts']['10']:8.2f} {n['pcts']['10']:8.2f} "
                         f"{o['pcts']['90']:8.2f} {n['pcts']['90']:8.2f} {n['n']:6d}")
    return "\n".join(lines)


def _weeks(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(w) for w in spec.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dfs.calibrate")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--weeks", default="1-17")
    ap.add_argument("--out", default="data/calibration.json")
    ap.add_argument("--rebuild-distributions", metavar="CALIB_JSON", default=None)
    ap.add_argument("--dist-out", default="data/distributions_calibrated.json")
    ap.add_argument("--compare-to", default="data/distributions.json")
    a = ap.parse_args(argv)

    if a.rebuild_distributions:
        data = json.loads(Path(a.rebuild_distributions).read_text())
    else:
        print(f"Collecting {a.season} weeks {a.weeks} ...")
        data = collect(a.season, _weeks(a.weeks))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(data))
        print(f"\nSaved paired observations: {a.out}")

    print()
    print(error_report(data))

    dist = rebuild_distributions(data)
    print(compare(Path(a.compare_to), dist))
    Path(a.dist_out).write_text(json.dumps(dist, indent=1))
    print(f"\nCalibrated distributions: {a.dist_out}  (n={dist['meta']['n_obs']})")
    print("Activate with:  cp %s data/distributions.json" % a.dist_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
