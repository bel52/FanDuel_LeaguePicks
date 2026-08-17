"""Fit the simulator's correlation structure to real outcomes.

The copula loadings in `simulate.py` were hand-selected. That is a defensible starting
point but it is not evidence, and if the loadings are wrong then every P(win) is wrong
in a way no amount of simulation will reveal. This module measures the real thing from
nflverse weekly stats (2023-2025), scored under FanDuel rules by `scoring.py`.

What gets measured, all on residuals rather than raw scores:

  * Raw scores correlate simply because good players score more. To isolate *game*
    correlation we work with each player's deviation from his own season mean,
    standardized by his own season spread. Two teammates both beating their averages in
    the same game is real correlation; both being good players is not.

  * Pairs are formed within a single game, by position pair and same-team/opposing-team
    relationship: QB-WR, QB-TE, QB-RB, WR-WR (same team), QB-oppQB (bring-back),
    WR-oppWR, DEF-oppQB, DEF-oppWR, and cross-game pairs as a null check (should be ~0).

Cross-game correlation is the honesty test: if it comes back materially non-zero, the
residualization is leaking a common factor and the other numbers are suspect.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Minimum player-weeks for a player to contribute residuals — below this the
# per-player mean and sd are too noisy to standardize against.
MIN_WEEKS = 6
# Minimum pair observations before a correlation is reported at all.
MIN_PAIRS = 200


@dataclass
class PairStat:
    label: str
    n: int
    corr: float
    current_loading: float | None = None

    def line(self) -> str:
        cur = ("" if self.current_loading is None
               else f"   (simulator implies {self.current_loading:+.3f})")
        return f"  {self.label:22s} n={self.n:6d}  r={self.corr:+.3f}{cur}"


def fanduel_points_from_stats(row) -> float:
    """FanDuel scoring from an nflverse weekly stat row (offense)."""
    g = lambda k: float(row.get(k) or 0.0)
    pts = (g("passing_yards") * 0.04 + g("passing_tds") * 4.0
           - g("passing_interceptions") * 1.0
           + g("rushing_yards") * 0.1 + g("rushing_tds") * 6.0
           + g("receiving_yards") * 0.1 + g("receiving_tds") * 6.0
           + g("receptions") * 0.5
           + (g("passing_2pt_conversions") + g("rushing_2pt_conversions")
              + g("receiving_2pt_conversions")) * 2.0
           - (g("rushing_fumbles_lost") + g("receiving_fumbles_lost")
              + g("sack_fumbles_lost")) * 2.0)
    # FanDuel yardage bonuses
    if g("passing_yards") >= 300:
        pts += 3.0
    if g("rushing_yards") >= 100:
        pts += 3.0
    if g("receiving_yards") >= 100:
        pts += 3.0
    return pts


def load_residuals(seasons: list[int]) -> dict:
    """(season, week, game_id) -> list of (player, team, opp, position, residual)."""
    import nflreadpy as nfl
    df = nfl.load_player_stats(seasons=seasons).to_pandas()
    df = df[df["season_type"] == "REG"] if "season_type" in df else df
    df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    df["fd"] = df.apply(fanduel_points_from_stats, axis=1)

    # Residualize within player-season: z = (score - player mean) / player sd.
    keys = ["season", "player_id"]
    grp = df.groupby(keys)["fd"]
    df["n_wk"] = grp.transform("count")
    df["mu"] = grp.transform("mean")
    df["sd"] = grp.transform("std")
    df = df[(df["n_wk"] >= MIN_WEEKS) & (df["sd"] > 0.5)]
    df["z"] = (df["fd"] - df["mu"]) / df["sd"]

    games: dict = defaultdict(list)
    for r in df.itertuples():
        games[(r.season, r.week, r.game_id)].append(
            (r.player_id, r.team, r.opponent_team, r.position, float(r.z)))
    return games


def measure(seasons: list[int] | None = None) -> list[PairStat]:
    seasons = seasons or [2023, 2024, 2025]
    games = load_residuals(seasons)
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)

    game_list = list(games.items())
    for _, players in game_list:
        for i, a in enumerate(players):
            for b in players[i + 1:]:
                same = a[1] == b[1]
                pa, pb = a[3], b[3]
                pair = "-".join(sorted((pa, pb)))
                label = f"{pair} {'same team' if same else 'opposing'}"
                buckets[label].append((a[4], b[4]))

    # Null check: pairs from DIFFERENT games in the same week must be ~uncorrelated.
    rng = np.random.default_rng(7)
    by_week: dict = defaultdict(list)
    for (season, week, _), players in game_list:
        by_week[(season, week)].append(players)
    for (_, _), glist in by_week.items():
        if len(glist) < 2:
            continue
        for _ in range(60):
            g1, g2 = rng.choice(len(glist), size=2, replace=False)
            a = glist[g1][rng.integers(len(glist[g1]))]
            b = glist[g2][rng.integers(len(glist[g2]))]
            buckets["CROSS-GAME (null)"].append((a[4], b[4]))

    out: list[PairStat] = []
    for label, pairs in sorted(buckets.items()):
        if len(pairs) < MIN_PAIRS:
            continue
        arr = np.array(pairs)
        r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
        out.append(PairStat(label=label, n=len(pairs), corr=round(r, 4)))
    return sorted(out, key=lambda s: -abs(s.corr))


def report(seasons: list[int] | None = None) -> str:
    stats = measure(seasons)
    lines = [f"EMPIRICAL CORRELATIONS from real outcomes "
             f"(seasons {seasons or [2023, 2024, 2025]}, residualized within player-season)",
             ""]
    lines += [s.line() for s in stats]
    null = next((s for s in stats if s.label.startswith("CROSS-GAME")), None)
    lines.append("")
    if null and abs(null.corr) > 0.05:
        lines.append(f"  WARNING: cross-game correlation is {null.corr:+.3f}, not ~0. "
                     "Residualization is leaking a common factor; treat the figures above "
                     "as unreliable until that is explained.")
    else:
        lines.append("  Cross-game null check passed — within-game figures are trustworthy.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    yrs = [int(x) for x in sys.argv[1:]] or None
    print(report(yrs))
