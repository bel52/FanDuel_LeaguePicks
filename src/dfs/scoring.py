"""FanDuel NFL scoring computed from FantasyPros projected stat lines.

Why not use FantasyPros' own `points` / `points_half`? Those are generic fantasy scoring.
FanDuel has its own rules (half-PPR + 300/100/100 yardage bonuses, its own DST points-allowed
ladder). FP hands us the full projected stat line, so we score it ourselves. Zero ambiguity,
and it stays correct if FP changes its default scoring display.

FP weekly stat fields (verified live 2026-08-15 against 2025 wk1):
  QB/RB/WR/TE: pass_att pass_cmp pass_yds pass_tds pass_ints rush_att rush_yds rush_tds
               rec_rec rec_yds rec_tds fumbles ret_tds 2pt_tds
  DST:         def_sack def_int def_td def_pa def_tyda def_safety def_ff def_fr def_retd

Note: FP's pass_yds_300 / rush_yds_100 / rec_yds_100 fields are 0 in weekly projections
(they are season-total counters), so bonus expectation is modelled from projected yardage.
"""
from __future__ import annotations
import math

# --- FanDuel scoring constants (re-verify at fanduel.com/rules before Week 1) ---
PASS_YD = 0.04
PASS_TD = 4.0
PASS_INT = -1.0
RUSH_YD = 0.1
RUSH_TD = 6.0
REC = 0.5           # half PPR
REC_YD = 0.1
REC_TD = 6.0
FUMBLE_LOST = -2.0
TWO_PT = 2.0
RET_TD = 6.0
BONUS = 3.0
BONUS_PASS_YD = 300
BONUS_RUSH_YD = 100
BONUS_REC_YD = 100

# DST
DST_SACK = 1.0
DST_INT = 2.0
DST_FUM_REC = 2.0
DST_TD = 6.0
DST_SAFETY = 2.0
DST_RET_TD = 6.0
# points-allowed ladder
DST_PA_LADDER = [(0, 0, 10.0), (1, 6, 7.0), (7, 13, 4.0), (14, 20, 1.0),
                 (21, 27, 0.0), (28, 34, -1.0), (35, 999, -4.0)]

# Spread of actual outcome around a projected mean, used for bonus/ladder probabilities.
# Calibrated to observed weekly dispersion; refined against residuals in 1.7.
YD_SPREAD = {"pass": 78.0, "rush": 34.0, "rec": 36.0}
PA_SPREAD = 9.5


def p_over(mean: float, threshold: float, spread: float) -> float:
    """P(X >= threshold) for a projected mean, logistic approximation to a normal."""
    if mean <= 0:
        return 0.0
    z = (mean - threshold) / spread
    return 1.0 / (1.0 + math.exp(-1.702 * z))


def _f(stats: dict, key: str) -> float:
    v = stats.get(key, 0)
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def score_skill(stats: dict) -> tuple[float, dict]:
    """FanDuel points for QB/RB/WR/TE from an FP projected stat line.
    Returns (points, breakdown) — breakdown makes every number auditable in the UI/CLI."""
    pass_yds = _f(stats, "pass_yds")
    rush_yds = _f(stats, "rush_yds")
    rec_yds = _f(stats, "rec_yds")

    b = {
        "pass_yds": round(pass_yds * PASS_YD, 3),
        "pass_tds": round(_f(stats, "pass_tds") * PASS_TD, 3),
        "pass_ints": round(_f(stats, "pass_ints") * PASS_INT, 3),
        "rush_yds": round(rush_yds * RUSH_YD, 3),
        "rush_tds": round(_f(stats, "rush_tds") * RUSH_TD, 3),
        "receptions": round(_f(stats, "rec_rec") * REC, 3),
        "rec_yds": round(rec_yds * REC_YD, 3),
        "rec_tds": round(_f(stats, "rec_tds") * REC_TD, 3),
        "fumbles": round(_f(stats, "fumbles") * FUMBLE_LOST, 3),
        "ret_tds": round(_f(stats, "ret_tds") * RET_TD, 3),
        "two_pt": round(_f(stats, "2pt_tds") * TWO_PT, 3),
    }
    # Expected bonus value = P(threshold) * 3
    b["bonus_pass300"] = round(BONUS * p_over(pass_yds, BONUS_PASS_YD, YD_SPREAD["pass"]), 3)
    b["bonus_rush100"] = round(BONUS * p_over(rush_yds, BONUS_RUSH_YD, YD_SPREAD["rush"]), 3)
    b["bonus_rec100"] = round(BONUS * p_over(rec_yds, BONUS_REC_YD, YD_SPREAD["rec"]), 3)
    return round(sum(b.values()), 2), b


def _expected_pa_points(mean_pa: float) -> float:
    """Expected value of the points-allowed ladder, integrated over a normal around mean_pa.
    Using the point estimate directly would be wrong — the ladder is a step function, so
    E[f(PA)] != f(E[PA])."""
    if mean_pa <= 0:
        return DST_PA_LADDER[0][2]
    total = 0.0
    for lo, hi, pts in DST_PA_LADDER:
        p_lo = p_over(mean_pa, lo - 0.5, PA_SPREAD) if lo > 0 else 1.0
        p_hi = p_over(mean_pa, hi + 0.5, PA_SPREAD)
        total += pts * max(0.0, p_lo - p_hi)
    return total


def score_dst(stats: dict) -> tuple[float, dict]:
    """FanDuel points for a DST from an FP projected stat line."""
    b = {
        "sacks": round(_f(stats, "def_sack") * DST_SACK, 3),
        "ints": round(_f(stats, "def_int") * DST_INT, 3),
        "fum_rec": round(_f(stats, "def_fr") * DST_FUM_REC, 3),
        "def_td": round(_f(stats, "def_td") * DST_TD, 3),
        "safety": round(_f(stats, "def_safety") * DST_SAFETY, 3),
        "ret_td": round(_f(stats, "def_retd") * DST_RET_TD, 3),
        "points_allowed": round(_expected_pa_points(_f(stats, "def_pa")), 3),
    }
    return round(sum(b.values()), 2), b


def score(stats: dict, position: str) -> tuple[float, dict]:
    return score_dst(stats) if position in ("D", "DST", "DEF") else score_skill(stats)
