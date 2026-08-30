import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dfs.contest_spec import ContestSpec, Profile, SlateType, PayoutTier
from dfs.ingest_fanduel import ingest_csv
from dfs.slate import SlateStore, SlateError
from dfs.distributions import floor_ceiling, fanduel_points
from dfs.matching import norm_name, short_key, norm_team
from dfs.vegas import TeamLine

FIX = Path(__file__).parent / "fixtures"
DIST = json.loads((Path(__file__).parent.parent / "data" / "distributions.json").read_text())


# ---- contest spec ----
def test_wta_payout():
    c = ContestSpec(name="league", profile=Profile.FRIENDS_LEAGUE, slate_type=SlateType.FULL,
                    field_size=12, entry_fee=15.0)
    assert c.payout_for_rank(1) == 180.0
    assert c.payout_for_rank(2) == 0.0

def test_tiered_requires_payouts():
    with pytest.raises(ValueError):
        ContestSpec(name="x", profile=Profile.PUBLIC_GPP, slate_type=SlateType.FULL,
                    field_size=100, winner_take_all=False)

def test_tiered_payout():
    c = ContestSpec(name="gpp", profile=Profile.PUBLIC_GPP, slate_type=SlateType.FULL,
                    field_size=100, winner_take_all=False,
                    payouts=[PayoutTier(rank_from=1, rank_to=1, amount=50),
                             PayoutTier(rank_from=2, rank_to=5, amount=10)])
    assert c.payout_for_rank(1) == 50 and c.payout_for_rank(4) == 10 and c.payout_for_rank(6) == 0


# ---- ingest ----
def test_ingest_full_slate():
    slate, rep = ingest_csv(FIX / "fd_full_slate.csv", "test-w1", 2026, 1)
    assert rep.ok
    assert rep.detected_slate_type == SlateType.FULL
    assert rep.ingested == 219            # 220 rows - 1 OUT
    assert len(rep.dropped_injury) == 1
    assert all(p.fd_id.startswith("1234-") for p in slate.players)

def test_ingest_schema_drift_fails_loud(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Id,Position,Salary\n1,QB,7000\n")
    with pytest.raises(SlateError, match="schema drift"):
        ingest_csv(bad, "x", 2026, 1)

def test_single_game_detection(tmp_path):
    src = (FIX / "fd_full_slate.csv").read_text().splitlines()
    header, body = src[0], [l for l in src[1:] if ",PHI," in l or ",DAL," in l]
    f = tmp_path / "sg.csv"; f.write_text("\n".join([header] + body))
    slate, rep = ingest_csv(f, "sg", 2026, 1, strict=False)
    assert rep.detected_slate_type == SlateType.SINGLE_GAME


# ---- persistence ----
def test_store_roundtrip(tmp_path):
    slate, _ = ingest_csv(FIX / "fd_full_slate.csv", "rt-w1", 2026, 1)
    store = SlateStore(tmp_path / "t.db")
    store.save(slate)
    loaded = store.load("rt-w1")
    assert len(loaded.players) == len(slate.players)
    assert loaded.players[0].fd_id == slate.players[0].fd_id

def test_require_projections_fails_loud():
    slate, _ = ingest_csv(FIX / "fd_full_slate.csv", "np", 2026, 1)
    with pytest.raises(SlateError, match="no FPPG/salary fallback"):
        slate.require_projections()


# ---- distributions ----
def test_floor_ceiling_realistic():
    fl, ce = floor_ceiling(15.0, "WR", DIST)
    assert 0 < fl < 6          # empirical WR floors are brutal
    assert 20 < ce < 35        # ceilings well above 1.5x fiction
    fl_qb, ce_qb = floor_ceiling(20.0, "QB", DIST)
    assert fl_qb > fl          # QBs have higher relative floors than WRs

def test_fanduel_scoring():
    import pandas as pd
    r = pd.Series({"passing_yards": 310, "passing_tds": 3, "passing_interceptions": 1,
                   "rushing_yards": 25, "rushing_tds": 0, "receptions": 0,
                   "receiving_yards": 0, "receiving_tds": 0})
    # 12.4 pass + 12 TD - 1 INT + 3 bonus + 2.5 rush = 28.9
    assert fanduel_points(r) == 28.9


# ---- blend helpers ----

def test_name_normalization():
    assert norm_name("A.J. Brown Jr.") == norm_name("AJ Brown") == "aj brown"
    assert norm_name("Amon-Ra St. Brown") == "amon ra st brown"
    assert short_key("Marvin Mims Jr.") == short_key("Marvin Mims") == "m|mims"
    assert norm_team("JAX") == norm_team("JAC") == "JAC"

def test_vegas_implied_total():
    t = TeamLine(team="PHI", opponent="DAL", game_total=47.0, spread=-3.0,
                 implied_total=47/2 + 1.5, kickoff_iso="")
    assert t.implied_total == 25.0


# ---- optimizer + simulator (Phase 2) ----
import random
from dfs.optimize import generate_pool, StackRule
from dfs.simulate import SlateSimulator, _std_normal_cdf
from dfs.field import build_prior_field, rank_candidates
from dfs.contest_spec import ROSTER_FULL, ROSTER_FULL_SIZE
import numpy as np


def _projected_slate():
    from dfs.distributions import floor_ceiling
    slate, _ = ingest_csv(FIX / "fd_full_slate.csv", "t", 2026, 1)
    random.seed(7)
    for p in slate.players:
        p.projection = round(max(1.0, p.fppg * random.uniform(0.8, 1.2)), 2)
        p.proj_source = "test"
        p.floor_p10, p.ceiling_p90 = floor_ceiling(p.projection, p.position, DIST)
    return slate


SPEC = ContestSpec(name="LL", profile=Profile.FRIENDS_LEAGUE, slate_type=SlateType.FULL,
                   field_size=12, entry_fee=15.0)


def test_pool_respects_roster_rules():
    slate = _projected_slate()
    pool = generate_pool(slate, SPEC, n=10, min_unique=3)
    pos = {p.fd_id: p.position for p in slate.players}
    for c in pool:
        assert len(c.player_ids) == ROSTER_FULL_SIZE
        assert c.salary <= SPEC.salary_cap
        counts = {}
        for pid in c.player_ids:
            counts[pos[pid]] = counts.get(pos[pid], 0) + 1
        for p, (lo, hi) in ROSTER_FULL.items():
            assert lo <= counts.get(p, 0) <= hi, f"{p}={counts.get(p,0)}"


def test_pool_diversity_enforced():
    """Phase 1 of the pool spreads across the projection frontier: the first lineups
    must differ from each other by min_unique players."""
    slate = _projected_slate()
    pool = generate_pool(slate, SPEC, n=12, min_unique=3)
    assert len(pool) >= 4
    frontier = pool[:6]
    for i, a in enumerate(frontier):
        for b in frontier[i + 1:]:
            shared = len(set(a.player_ids) & set(b.player_ids))
            assert shared <= len(a.player_ids) - 3, "frontier lineups must be diverse"


def test_pool_includes_near_neighbour_pivots():
    """Phase 2 adds 1-2 player pivots. The diversity constraint excludes exactly these,
    yet at $0.164/pt vs $12.84/win-prob they are the trades most likely to pay."""
    slate = _projected_slate()
    pool = generate_pool(slate, SPEC, n=60, min_unique=3)
    assert len(pool) > 12
    n = len(pool[0].player_ids)
    # at least one pair differing by only 1-2 players must exist
    close = 0
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            d = n - len(set(a.player_ids) & set(b.player_ids))
            if 1 <= d <= 2:
                close += 1
    assert close > 0, "no near-neighbour pivots generated"
    assert len({c.key for c in pool}) == len(pool), "pool must not contain duplicates"


def test_qb_stack_constraint():
    slate = _projected_slate()
    pool = generate_pool(slate, SPEC, n=5, stack=StackRule(require_qb_stack=1), min_unique=3)
    byid = {p.fd_id: p for p in slate.players}
    for c in pool:
        qb = next(byid[i] for i in c.player_ids if byid[i].position == "QB")
        catchers = [i for i in c.player_ids
                    if byid[i].team == qb.team and byid[i].position in ("WR", "TE")]
        assert len(catchers) >= 1


def test_simulator_deterministic():
    slate = _projected_slate()
    a = SlateSimulator(slate, DIST, n_sims=2000, seed=42).matrix
    b = SlateSimulator(slate, DIST, n_sims=2000, seed=42).matrix
    assert np.array_equal(a, b)


def test_correlation_structure():
    """The simulator must reproduce correlations MEASURED from real outcomes
    (nflverse 2023-2025, residualized within player-season — see corrfit.py):

        QB - same-team WR   +0.246
        WR - same-team WR   +0.017   <- the old factor model forced this positive
        QB - opposing QB    +0.134
        cross-game           0.000

    Targets are checked with tolerance because the copula transform through skewed
    marginals attenuates rank correlation somewhat. The critical assertion is the
    ORDERING and that WR-WR is near zero: a factor model cannot do both, and
    overstating stack correlation inflates stacked ceilings and P(win).
    """
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=15000, seed=5)
    M, idx = sim.matrix, sim.index

    def r(a, b):
        return float(np.corrcoef(M[idx[a.fd_id]], M[idx[b.fd_id]])[0, 1])

    qb = next(p for p in slate.players if p.position == "QB")
    same_wr = [p for p in slate.players if p.position == "WR" and p.team == qb.team]
    opp_qb = next((p for p in slate.players
                   if p.position == "QB" and p.team == qb.opponent), None)
    cross = next(p for p in slate.players
                 if p.team not in (qb.team, qb.opponent) and p.position == "WR")

    assert abs(r(qb, cross)) < 0.05, "cross-game correlation must be ~0"
    if same_wr:
        qb_wr = r(qb, same_wr[0])
        assert 0.12 < qb_wr < 0.35, f"QB-WR {qb_wr:+.3f} should approach +0.246"
    if len(same_wr) > 1:
        wr_wr = r(same_wr[0], same_wr[1])
        assert abs(wr_wr) < 0.10, \
            f"WR-WR {wr_wr:+.3f} must be near zero (measured +0.017)"
        assert qb_wr > wr_wr + 0.08, "QB-WR must exceed WR-WR — the whole point of " \
                                     "replacing the factor model"
    if opp_qb:
        assert 0.04 < r(qb, opp_qb) < 0.25, "bring-back should approach +0.134"


def test_correlation_table_loads_measured_values():
    from dfs.simulate import _load_corr_table, pair_corr
    tbl = _load_corr_table()
    assert pair_corr(tbl, "QB", "WR", True) > pair_corr(tbl, "WR", "WR", True)
    assert abs(pair_corr(tbl, "WR", "WR", True)) < 0.08
    assert pair_corr(tbl, "QB", "QB", False) > 0.05      # bring-back is real
    assert pair_corr(tbl, "D", "QB", False) < 0           # DEF hurt by opposing QB


def test_lineup_p90_below_summed_marginals():
    """The v5 bug: summing player p90s massively overstates lineup ceiling."""
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=10000, seed=1729)
    pool = generate_pool(slate, SPEC, n=3, min_unique=3)
    byid = {p.fd_id: p for p in slate.players}
    c = pool[0]
    joint = sim.score(c.player_ids).pct(90)
    summed = sum(byid[i].ceiling_p90 for i in c.player_ids)
    # The v5 bug was summing player p90s to claim a lineup ceiling (~45 pts too high).
    # The joint p90 must be strictly lower. Threshold is 0.95, not 0.90: once simulated
    # means are pinned to the empirical residual mean, tails are less extreme and the
    # honest gap narrows. Tightening below this would test the inflation, not the math.
    assert joint < summed * 0.95


def test_ranking_produces_win_probs():
    from dfs.objectives import Leaderboard, SeasonContext, weights_for
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=5000, seed=1729)
    pool = generate_pool(slate, SPEC, n=12, min_unique=3)
    field = build_prior_field(slate, SPEC, n_opponents=11, seed=99)
    assert len(field.lineups) == 11
    w = weights_for("friends_league",
                    SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES, weeks_total=21))
    ranked = rank_candidates(pool, sim, field, SPEC, w)
    assert len(ranked) == len(pool)
    assert ranked[0].dollars >= ranked[-1].dollars       # sorted by objective
    assert 0.0 < ranked[0].p_win < 1.0
    # objective value must reconcile with the declared weights
    r = ranked[0]
    assert r.dollars == pytest.approx(w.w_points * r.exp_points + w.w_win * r.p_win, rel=1e-4)


def test_objective_changes_lineup_selection():
    """Total Scores (values points) and Most Wins (values only firsts) should not
    agree on the best lineup — if they always did, the objective would be inert."""
    from dfs.objectives import Leaderboard, SeasonContext, weights_for
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=8000, seed=1729)
    pool = generate_pool(slate, SPEC, n=25, min_unique=3)
    field = build_prior_field(slate, SPEC, n_opponents=11, seed=99)
    tot = weights_for("friends_league", SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES))
    win = weights_for("friends_league", SeasonContext(leaderboard=Leaderboard.MOST_WINS))
    a = rank_candidates(pool, sim, field, SPEC, tot)
    b = rank_candidates(pool, sim, field, SPEC, win)
    assert tot.w_points > 0 and win.w_points == 0
    order_a = [r.candidate.player_ids for r in a]
    order_b = [r.candidate.player_ids for r in b]
    assert order_a != order_b, "objectives must produce different rankings"


def test_normal_cdf_accuracy():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0228, 0.1587, 0.5, 0.8413, 0.9772])
    assert np.allclose(_std_normal_cdf(z), expected, atol=1e-3)


# ---- FanDuel scoring from FP stat lines ----
from dfs.scoring import score, p_over, _expected_pa_points

HURTS = {"pass_yds": 218.44, "pass_tds": 1.57, "pass_ints": 0.45, "rush_yds": 40.93,
         "rush_tds": 0.83, "fumbles": 0.28, "ret_tds": 0, "2pt_tds": 0}
CHASE = {"rec_rec": 7.06, "rec_yds": 96.33, "rec_tds": 0.76, "rush_yds": 0.65,
         "fumbles": 0.03, "ret_tds": 0, "2pt_tds": 0}
DEN = {"def_sack": 3.15, "def_int": 0.8, "def_td": 0.19, "def_pa": 17.23,
       "def_safety": 0.1, "def_ff": 1.02, "def_fr": 0.64, "def_retd": 0}


def test_skill_scoring_components():
    pts, b = score(HURTS, "QB")
    assert b["pass_yds"] == pytest.approx(218.44 * 0.04, abs=0.01)
    assert b["pass_tds"] == pytest.approx(1.57 * 4, abs=0.01)
    assert b["rush_tds"] == pytest.approx(0.83 * 6, abs=0.01)
    assert 23 < pts < 25

def test_half_ppr_receptions():
    _, b = score(CHASE, "WR")
    assert b["receptions"] == pytest.approx(7.06 * 0.5, abs=0.01)

def test_fanduel_exceeds_fp_by_bonus():
    """FanDuel adds 300/100/100 bonuses FP's half-PPR number lacks."""
    assert score(CHASE, "WR")[0] > 17.72
    assert score(HURTS, "QB")[0] > 23.10

def test_dst_pa_ladder_is_expectation_not_point_estimate():
    """E[f(PA)] != f(E[PA]) for a step function — 17.2 PA must not simply score 1.0."""
    v = _expected_pa_points(17.23)
    assert 1.0 < v < 4.0
    assert _expected_pa_points(3.0) > _expected_pa_points(24.0)

def test_dst_total():
    pts, b = score(DEN, "D")
    assert b["sacks"] == pytest.approx(3.15)
    assert b["ints"] == pytest.approx(1.6)
    assert 8 < pts < 11

def test_p_over_monotonic():
    assert p_over(250, 300, 78) < p_over(320, 300, 78)
    assert p_over(300, 300, 78) == pytest.approx(0.5, abs=0.001)


# ---- matching (name-first) ----
from dfs.matching import ProjectionIndex, match_slate
from dfs.fantasypros import FPProjection


def _proj(name, team, pos, pts=10.0):
    return FPProjection(player_id=name, name=name, team=team, position=pos,
                        points=pts, stats={}, breakdown={})


def test_match_survives_team_disagreement():
    """The live bug: FD and FP disagreed on team, dropping A.J. Brown."""
    slate, _ = ingest_csv(FIX / "fd_full_slate.csv", "m", 2026, 1)
    sp = slate.players[0]
    projs = [_proj(sp.name, "ZZZ", sp.position, 15.0)]     # wrong team on FP side
    mapping, rep = match_slate([sp], projs)
    assert sp.fd_id in mapping
    assert rep.rate == 1.0
    assert rep.team_disagreements and rep.team_disagreements[0][0] == sp.name


def test_match_disambiguates_same_name_by_team():
    idx = ProjectionIndex([_proj("Mike Williams", "LAC", "WR", 12.0),
                           _proj("Mike Williams", "NYJ", "WR", 6.0)])
    hit, method, amb = idx.lookup("Mike Williams", "NYJ", "WR")
    assert hit.points == 6.0 and not amb


def test_match_suffix_insensitive():
    idx = ProjectionIndex([_proj("Travis Etienne Jr.", "JAC", "RB", 14.0)])
    hit, method, amb = idx.lookup("Travis Etienne", "JAX", "RB")
    assert hit is not None and hit.points == 14.0


def test_unmatched_reported_not_invented():
    idx = ProjectionIndex([_proj("Real Guy", "PHI", "WR")])
    hit, method, amb = idx.lookup("Nonexistent Person", "PHI", "WR")
    assert hit is None and method == "none"


def test_field_model_produces_exact_opponent_count():
    """Under-producing the field silently inflates win probability."""
    slate = _projected_slate()
    for n in (5, 11, 15):
        f = build_prior_field(slate, SPEC, n_opponents=n, seed=3)
        assert len(f.lineups) == n

def test_field_lineups_are_cap_legal():
    slate = _projected_slate()
    sal = {p.fd_id: p.salary for p in slate.players}
    pos = {p.fd_id: p.position for p in slate.players}
    f = build_prior_field(slate, SPEC, n_opponents=11, seed=3)
    for l in f.lineups:
        assert len(l) == 9 and len(set(l)) == 9
        assert sum(sal[i] for i in l) <= SPEC.salary_cap
        counts = {}
        for i in l:
            counts[pos[i]] = counts.get(pos[i], 0) + 1
        assert counts.get("QB") == 1 and counts.get("D") == 1


# ---- contest results parsing (opponent model input) ----
from dfs.contest_parse import parse_contest, _norm

CONTEST_FIX = FIX / "contest_2025_w18.zip"


def test_parse_real_contest_page():
    cap = parse_contest(CONTEST_FIX, 2025, 18)
    assert len(cap.leaderboard) == 10
    assert cap.leaderboard[0].entrant == "xleathy"
    assert cap.leaderboard[0].score == pytest.approx(136.66)
    assert cap.leaderboard[0].won == pytest.approx(15.0)
    # last row must not pick up pagination digits as its score
    assert cap.leaderboard[-1].entrant == "robiz"
    assert cap.leaderboard[-1].score == pytest.approx(72.30)


def test_parse_captures_full_nine_player_lineups():
    """FLEX and DEF are distinct FanDuel labels; missing them silently truncated lineups."""
    cap = parse_contest(CONTEST_FIX, 2025, 18)
    assert len(cap.entries_with_lineups) == 2
    for e in cap.entries_with_lineups:
        assert len(e.players) == 9, f"{e.entrant} has {len(e.players)}"
        assert sum(1 for p in e.players if p.position == "FLEX") == 1
        assert sum(1 for p in e.players if p.position == "D") == 1


def test_measured_ownership_is_twelfths():
    """DRAFTED% in a 12-person league must be a multiple of 1/12 = 8.333%."""
    cap = parse_contest(CONTEST_FIX, 2025, 18)
    own = cap.ownership()
    assert own["trevor lawrence"] == pytest.approx(41.7)
    assert own["travis etienne"] == pytest.approx(50.0)
    for v in own.values():
        n = v / (100 / 12)
        assert abs(n - round(n)) < 0.02, f"{v}% is not n/12"


def test_lineup_entries_get_leaderboard_rank():
    cap = parse_contest(CONTEST_FIX, 2025, 18)
    x = next(e for e in cap.entries_with_lineups if e.entrant == "xleathy")
    assert x.rank == 1 and x.score == pytest.approx(136.66)


def test_player_points_parsed():
    cap = parse_contest(CONTEST_FIX, 2025, 18)
    x = next(e for e in cap.entries_with_lineups if e.entrant == "xleathy")
    chase = next(p for p in x.players if _norm(p.name) == "jamarr chase")
    assert chase.actual_points == pytest.approx(19.6)
    assert chase.drafted_pct == pytest.approx(8.3)


# ---- objectives (contest profiles) ----
from dfs.objectives import (Leaderboard, SeasonContext, season_marginal_value,
                            weights_for, score_lineup, ObjectiveWeights)


def test_total_scores_values_points_and_wins():
    ctx = SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES, weeks_total=21, weeks_played=0)
    w = weights_for("friends_league", ctx)
    assert w.w_points > 0, "Total Scores must value expected points"
    assert w.w_win > 0
    # w_points is $ per point THIS WEEK: ~$3.44 sustained / weeks remaining.
    # This bound is a regression guard against the units bug that inflated the
    # reported edge ~21x by multiplying a sustained rate by one week's score.
    assert 0.05 < w.w_points < 0.50
    late = weights_for("friends_league",
                       SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES,
                                     weeks_total=21, weeks_played=18))
    assert late.w_points > w.w_points, "each point matters more as weeks run out"


def test_most_wins_values_only_wins():
    ctx = SeasonContext(leaderboard=Leaderboard.MOST_WINS, weeks_total=21)
    w = weights_for("friends_league", ctx)
    assert w.w_points == 0.0, "under Most Wins, 2nd == last; points have no standalone value"
    assert w.w_win > 12.0


def test_h2h_is_median_not_ceiling():
    w = weights_for("h2h", SeasonContext(), entry_fee=5.0)
    assert w.w_points == 0.0 and w.w_win > 0
    assert "median" in w.rationale.lower()


def test_best_n_decays_as_weeks_bank():
    early = weights_for("friends_league",
                        SeasonContext(leaderboard=Leaderboard.BEST_5, weeks_played=0))
    late = weights_for("friends_league",
                       SeasonContext(leaderboard=Leaderboard.BEST_5, weeks_played=5))
    assert late.w_points < early.w_points


def test_marginal_value_zero_when_points_dont_decide():
    assert season_marginal_value(SeasonContext(leaderboard=Leaderboard.MOST_WINS)) == 0.0
    assert season_marginal_value(SeasonContext(leaderboard=Leaderboard.NONE)) == 0.0


def test_score_lineup_uses_joint_distribution():
    rng = np.random.default_rng(0)
    mine = rng.normal(120, 25, 5000)
    fieldt = rng.normal(115, 25, (11, 5000))
    w = ObjectiveWeights(3.4, 12.84, "test")
    s = score_lineup(mine, fieldt, w)
    assert 0 < s.p_win < 1 and s.p_top3 > s.p_win
    assert s.p10 < s.median < s.p90
    assert s.dollars == pytest.approx(3.4 * s.exp_points + 12.84 * s.p_win, rel=1e-4)


# ---- injuries ----
from dfs.injuries import (Status, Action, parse_status, InjuryRecord, merge, sweep,
                          records_from_slate, records_from_fantasypros)


def test_parse_status_variants():
    assert parse_status("O") is Status.OUT
    assert parse_status("IR") is Status.IR
    assert parse_status("Q") is Status.QUESTIONABLE
    assert parse_status("Ruled out for Sunday") is Status.OUT
    assert parse_status("Doubtful (hamstring)") is Status.DOUBTFUL
    assert parse_status("") is Status.UNKNOWN


def test_doubtful_is_removed_not_haircut():
    """v5 silently multiplied projections by injury fudge factors; we remove instead."""
    assert ACTION_FOR_DOUBTFUL() is Action.REMOVE


def ACTION_FOR_DOUBTFUL():
    from dfs.injuries import ACTION_FOR
    return ACTION_FOR[Status.DOUBTFUL]


def test_merge_prefers_worse_status():
    a = {"x": InjuryRecord("X", "PHI", Status.QUESTIONABLE, source="csv")}
    b = {"x": InjuryRecord("X", "PHI", Status.OUT, source="fp")}
    assert merge(a, b)["x"].status is Status.OUT
    assert merge(b, a)["x"].status is Status.OUT   # better status never overwrites worse


def test_sweep_removes_and_flags():
    slate = _projected_slate()
    n0 = len(slate.players)
    p_out, p_q = slate.players[0], slate.players[1]
    inj = {norm_name(p_out.name): InjuryRecord(p_out.name, p_out.team, Status.OUT),
           norm_name(p_q.name): InjuryRecord(p_q.name, p_q.team, Status.QUESTIONABLE)}
    res = sweep(slate, inj, lineup_ids={p_out.fd_id})
    assert len(res.removed) == 1 and len(res.flagged) == 1
    assert res.lineup_affected is True
    assert len(slate.players) == n0 - 1        # OUT dropped, Q retained for human call
    assert "REMOVE" in res.summary()


def test_sweep_clean_when_no_injuries():
    slate = _projected_slate()
    res = sweep(slate, {})
    assert res.ok and "clean" in res.summary()


# ---- export ----
from dfs.export import export_upload_csv, lineup_card, _slot_players, CLASSIC_FILL_ORDER


def _one_lineup():
    slate = _projected_slate()
    pool = generate_pool(slate, SPEC, n=1, min_unique=3)
    byid = {p.fd_id: p for p in slate.players}
    return [byid[i] for i in pool[0].player_ids]


def test_slot_assignment_matches_fanduel_order():
    players = _one_lineup()
    slotted = _slot_players(players, SlateType.FULL, None)
    assert [s for s, _ in slotted] == CLASSIC_FILL_ORDER
    assert len({p.fd_id for _, p in slotted}) == 9      # no player used twice


def test_export_warns_without_verified_template(tmp_path):
    res = export_upload_csv(_one_lineup(), tmp_path / "up.csv")
    assert res.path.exists()
    assert any("NOT been verified" in w for w in res.warnings)
    import csv as _csv
    rows = list(_csv.reader(open(res.path)))
    assert len(rows) == 2 and len(rows[0]) == len(rows[1])


def test_export_mirrors_supplied_template(tmp_path):
    tmpl = tmp_path / "t.csv"
    tmpl.write_text("entry_id,QB,RB,RB,WR,WR,WR,TE,FLEX,DEF\n")
    res = export_upload_csv(_one_lineup(), tmp_path / "u.csv", template=tmpl,
                            entry_id="E123")
    assert not res.warnings
    import csv as _csv
    rows = list(_csv.reader(open(res.path)))
    assert rows[0][0] == "entry_id" and rows[1][0] == "E123"
    assert all(rows[1][1:])                              # every slot filled


def test_lineup_card_readable():
    card = lineup_card(_one_lineup(), title="Week 1")
    assert "Week 1" in card and "TOTAL" in card
    assert len([l for l in card.splitlines() if "$" in l]) == 10   # 9 players + total


# ---- result log ----
from dfs.results import ResultLog
from dfs.contest_parse import parse_contest


def test_log_entry_and_outcome(tmp_path):
    log = ResultLog(tmp_path / "r.db")
    players = _one_lineup()
    log.log_entry(2026, 1, "Leather League", players, objective="total_scores",
                  exp_points=128.0, p_win=0.21)
    log.log_outcome(2026, 1, "Leather League", score=131.4, rank=3, field_size=12)
    st = log._c().execute("SELECT * FROM entries").fetchone()
    assert st["actual_score"] == pytest.approx(131.4)
    assert st["final_rank"] == 3
    assert st["exp_points"] == pytest.approx(128.0)     # relog must not wipe outcome


def test_log_capture_and_standings(tmp_path):
    log = ResultLog(tmp_path / "r.db")
    cap = parse_contest(CONTEST_FIX, 2025, 18)
    log.log_capture(cap)
    st = log.standings(2025)
    assert st[0].entrant == "xleathy"
    assert st[0].total_points == pytest.approx(136.66)
    assert st[0].wins == 1
    own = log.measured_ownership(2025)
    assert own["travis etienne"] == pytest.approx(50.0)


def test_season_context_drives_objective(tmp_path):
    """Logged results must feed back into the objective weights."""
    log = ResultLog(tmp_path / "r.db")
    log.log_capture(parse_contest(CONTEST_FIX, 2025, 18))
    ctx = log.season_context(2025, me="xleathy", weeks_total=21)
    assert ctx.weeks_played == 1
    assert ctx.my_points == pytest.approx(136.66)
    assert ctx.leader_points >= ctx.my_points
    w = weights_for("friends_league", ctx)
    assert w.w_points > 0


def test_projection_accuracy_tracks_inseason(tmp_path):
    log = ResultLog(tmp_path / "r.db")
    players = _one_lineup()
    log.log_entry(2026, 1, "LL", players)
    log.log_player_actuals(2026, 1, {p.fd_id: (p.projection or 0) + 2.0 for p in players})
    acc = log.projection_accuracy(2026)
    assert acc["ALL"]["bias"] == pytest.approx(2.0, abs=0.01)
    assert acc["ALL"]["n"] == 9


# ---- kickoffs + late swap ----
from datetime import datetime, timezone, timedelta
from dfs.kickoffs import KickoffSchedule, GameTime
from dfs.lateswap import propose_swap
from dfs.objectives import score_lineup as _sl


def _sched_for(slate, locked_teams=(), now=None):
    """Synthetic schedule: locked teams kicked off an hour ago, rest in 3 hours."""
    now = now or datetime.now(timezone.utc)
    games = {}
    for p in slate.players:
        if p.team in games:
            continue
        ko = now - timedelta(hours=1) if p.team in locked_teams else now + timedelta(hours=3)
        games[p.team] = GameTime(p.team, p.opponent, ko)
    return KickoffSchedule(games)


def test_lock_state_and_next_lock():
    slate = _projected_slate()
    now = datetime.now(timezone.utc)
    sched = _sched_for(slate, locked_teams={"PHI", "DAL"}, now=now)
    assert sched.by_team["PHI"].locked(now) and not sched.by_team["KC"].locked(now)
    assert {"PHI", "DAL"} <= sched.locked_teams(now)
    assert sched.next_lock(now) is not None
    assert "LOCKED" in sched.summary(now) and "open" in sched.summary(now)


def _swap_setup(locked_teams, n_sims=4000):
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=n_sims, seed=1729)
    pool = generate_pool(slate, SPEC, n=5, min_unique=3)
    fieldm = build_prior_field(slate, SPEC, n_opponents=11, seed=99)
    w = weights_for("friends_league",
                    SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES))
    current = pool[-1].player_ids          # deliberately not the best lineup
    sched = _sched_for(slate, locked_teams=locked_teams)
    return slate, sim, fieldm, w, current, sched


def test_swap_never_touches_locked_players():
    slate, sim, fieldm, w, current, sched = _swap_setup(locked_teams={"PHI", "DAL", "KC", "BUF"})
    byid = {p.fd_id: p for p in slate.players}
    locked_in_lineup = {i for i in current if byid[i].team in {"PHI", "DAL", "KC", "BUF"}}
    prop = propose_swap(slate, current, SPEC, sched, sim, fieldm, w)
    assert locked_in_lineup <= set(prop.proposed_ids), "locked players must survive any swap"
    for out_p, in_p in prop.swaps:
        assert out_p.team not in {"PHI", "DAL", "KC", "BUF"}


def test_swap_excludes_started_games_from_replacements():
    slate, sim, fieldm, w, current, sched = _swap_setup(locked_teams={"PHI", "DAL"})
    byid = {p.fd_id: p for p in slate.players}
    prop = propose_swap(slate, current, SPEC, sched, sim, fieldm, w)
    for _, in_p in prop.swaps:
        assert in_p.team not in {"PHI", "DAL"}, "cannot swap IN a player whose game started"


def test_swap_all_locked_is_noop():
    slate, sim, fieldm, w, current, sched = _swap_setup(
        locked_teams={p.team for p in _projected_slate().players})
    prop = propose_swap(slate, current, SPEC, sched, sim, fieldm, w)
    assert not prop.swaps and prop.proposed_ids == current
    assert "all players locked" in prop.reason


def test_swap_only_proposes_improvement():
    slate, sim, fieldm, w, current, sched = _swap_setup(locked_teams=set())
    prop = propose_swap(slate, current, SPEC, sched, sim, fieldm, w)
    if prop.swaps:
        assert prop.new_dollars > prop.old_dollars
    else:
        assert prop.new_dollars == prop.old_dollars


# ---- adversarial-review fixes (2026-08-16) ----
from dfs.field import build_field_ensemble, _prior_field_showdown


def _showdown_slate():
    """Two-team slate with K, built from the fixture's PHI/DAL players + synth kickers."""
    slate, _ = ingest_csv(FIX / "fd_full_slate.csv", "sd", 2026, 1, strict=False)
    from dfs.slate import SlatePlayer
    import random as _r
    _r.seed(3)
    players = [p for p in slate.players if p.team in ("PHI", "DAL")]
    for team, opp in (("PHI", "DAL"), ("DAL", "PHI")):
        players.append(SlatePlayer(fd_id=f"k-{team}", name=f"Kicker {team}", position="K",
                                   team=team, opponent=opp, salary=4000, game="DAL@PHI"))
    slate.players = players
    from dfs.distributions import floor_ceiling
    for p in slate.players:
        p.projection = round(max(1.0, (p.fppg or 6.0) * _r.uniform(0.8, 1.2)), 2)
        p.floor_p10, p.ceiling_p90 = floor_ceiling(p.projection, p.position, DIST)
    slate.slate_type = SlateType.SINGLE_GAME
    return slate


SD_SPEC = ContestSpec(name="SNF", profile=Profile.SHOWDOWN_FRIENDS,
                      slate_type=SlateType.SINGLE_GAME, field_size=6, entry_fee=5.0)


def test_kickers_ingest_on_showdown():
    slate = _showdown_slate()
    assert sum(1 for p in slate.players if p.position == "K") == 2


def test_showdown_field_generates_exact_count():
    """Reviewer repro: classic field generator died on 2-team slates (needs >=4 QB)."""
    slate = _showdown_slate()
    fm = build_prior_field(slate, SD_SPEC, n_opponents=5, seed=7)
    assert len(fm.lineups) == 5 and fm.mvp_ids and len(fm.mvp_ids) == 5
    sal = {p.fd_id: p.salary for p in slate.players}
    for l, mvp in zip(fm.lineups, fm.mvp_ids):
        assert len(l) == 6 and mvp in l          # FanDuel single game: 1 MVP + 5 FLEX
        assert (sum(sal[i] for i in l)
                + 0.5 * sal[mvp]) <= SD_SPEC.salary_cap   # MVP premium counts
        teams = {next(p.team for p in slate.players if p.fd_id == i) for i in l}
        assert len(teams) == 2


def test_showdown_end_to_end_ranks():
    slate = _showdown_slate()
    pool = generate_pool(slate, SD_SPEC, n=8, min_unique=1)
    assert pool and all(len(c.player_ids) == 6 and c.mvp_id for c in pool)
    sim = SlateSimulator(slate, DIST, n_sims=3000, seed=5)
    fields = build_field_ensemble(slate, SD_SPEC, n_opponents=5, n_fields=5, seed=9)
    w = weights_for("showdown_friends", SeasonContext(), entry_fee=5.0)
    ranked = rank_candidates(pool, sim, fields, SD_SPEC, w)
    assert ranked[0].dollars >= ranked[-1].dollars


def test_mvp_salary_multiplier_enforced():
    slate = _showdown_slate()
    spec15 = ContestSpec(name="SNF", profile=Profile.SHOWDOWN_FRIENDS,
                         slate_type=SlateType.SINGLE_GAME, field_size=6,
                         mvp_salary_mult=1.5)
    pool = generate_pool(slate, spec15, n=3, min_unique=1)
    sal = {p.fd_id: p.salary for p in slate.players}
    for c in pool:
        charged = sum(sal[i] for i in c.player_ids) + 0.5 * sal[c.mvp_id]
        assert charged <= spec15.salary_cap + 1e-6


def test_ties_split_win_credit():
    from dfs.objectives import score_lineup, ObjectiveWeights
    mine = np.full(1000, 100.0)
    fieldt = np.full((3, 1000), 100.0)           # 3 opponents, all exact ties
    s = score_lineup(mine, fieldt, ObjectiveWeights(0, 1, "t"))
    assert s.p_win == pytest.approx(0.25)        # 4-way tie -> quarter credit, not zero


def test_field_ensemble_reduces_seed_sensitivity():
    """Reviewer repro: single-field P(win) swung 14.5%->21.7% across seeds."""
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=4000, seed=1729)
    pool = generate_pool(slate, SPEC, n=6, min_unique=3)
    w = weights_for("friends_league", SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES))
    singles, ensembles = [], []
    for seed in (1, 2, 3):
        f1 = build_prior_field(slate, SPEC, 11, seed=seed)
        fe = build_field_ensemble(slate, SPEC, 11, n_fields=15, seed=seed)
        singles.append(rank_candidates(pool, sim, f1, SPEC, w)[0].p_win)
        ensembles.append(rank_candidates(pool, sim, fe, SPEC, w)[0].p_win)
    assert np.std(ensembles) < np.std(singles), \
        f"ensemble {np.std(ensembles):.4f} should be steadier than single {np.std(singles):.4f}"


def test_lateswap_survives_swept_lineup_player():
    """Reviewer repro: sweep removed a rostered player, swap aborted. Now his slot opens."""
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=3000, seed=1729)
    pool = generate_pool(slate, SPEC, n=3, min_unique=3)
    current = pool[0].player_ids
    gone = current[0]
    slate.players = [p for p in slate.players if p.fd_id != gone]   # ruled out + swept
    sim2 = SlateSimulator(slate, DIST, n_sims=3000, seed=1729)
    fieldm = build_prior_field(slate, SPEC, n_opponents=11, seed=99)
    w = weights_for("friends_league", SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES))
    sched = _sched_for(slate, locked_teams=set())
    prop = propose_swap(slate, current, SPEC, sched, sim2, fieldm, w)
    assert prop.swaps, "a guaranteed zero must force a swap proposal"
    assert gone not in prop.proposed_ids
    assert "ruled OUT after entry" in prop.reason


def test_baseline_and_ranking_use_same_evaluation_path():
    """A baseline scored on one field draw while candidates are scored on a 25-field
    ensemble compares different measurements and can invert the sign of the edge.
    Both must run through rank_candidates."""
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=6000, seed=1729)
    pool = generate_pool(slate, SPEC, n=20, min_unique=3)
    fields = build_field_ensemble(slate, SPEC, 11, n_fields=10, seed=100)
    w = weights_for("friends_league",
                    SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES, weeks_total=21))
    ranked = rank_candidates(pool, sim, fields, SPEC, w)
    mx = max(pool, key=lambda c: c.proj_sum)
    mp = rank_candidates([mx], sim, fields, SPEC, w)[0]
    assert ranked[0].dollars >= mp.dollars - 1e-9, \
        "the selected lineup cannot score below a candidate it was ranked against"


# ---- web API (offline smoke) ----
def test_web_health_and_index():
    from fastapi.testclient import TestClient
    from dfs.web import app
    c = TestClient(app)
    h = c.get("/health")
    assert h.status_code == 200 and h.json()["ok"] is True
    assert "calibrated_distributions" in h.json()
    i = c.get("/")
    assert i.status_code == 200 and "Leather League" in i.text


def test_web_capture_roundtrip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import dfs.web as web
    monkeypatch.setattr(web, "DB", tmp_path / "r.db")
    monkeypatch.setattr(web, "UPLOADS", tmp_path)
    c = TestClient(web.app)
    r = c.post("/api/capture",
               files={"page": ("w18.zip", open(CONTEST_FIX, "rb"), "application/zip")},
               data={"season": 2025, "week": 18})
    assert r.status_code == 200
    job = r.json()["job"]
    import time
    for _ in range(60):
        j = c.get(f"/api/job/{job}").json()
        if j["status"] != "running":
            break
        time.sleep(0.3)
    assert j["status"] == "done", j["output"][-400:]
    s = c.get("/api/standings", params={"season": 2025, "me": "xleathy"}).json()
    assert s["standings"][0]["entrant"] == "xleathy"
    assert s["objective"]["w_points"] > 0


def test_web_download_traversal_blocked():
    from fastapi.testclient import TestClient
    from dfs.web import app
    c = TestClient(app)
    assert c.get("/api/download/..%2F..%2F.env").status_code == 404


# ---- learning-from-league + operational fixes (2026-08-16) ----
from dfs.field import ownership_weights


def test_entry_history_csv_rejected_with_clear_message(tmp_path):
    bad = tmp_path / "hist.csv"
    bad.write_text("Entry Id,Sport,Date,Title,SalaryCap,Score,Opp Score,Position,Entries,"
                   "Opponent,Entry ($),Winnings ($),Link\nS1,nfl,2026/01/01,x,$60k,"
                   "100,,1,12,league,0,15,/entry/X\n")
    with pytest.raises(SlateError, match="ENTRY HISTORY"):
        ingest_csv(bad, "x", 2026, 1)


def _fake_odds_board(games):
    """Odds-API-shaped payload: [(home_full, away_full, total, home_spread), ...]"""
    out = []
    for home, away, total, hsp in games:
        out.append({
            "home_team": home, "away_team": away, "commence_time": "2026-09-13T17:00:00Z",
            "bookmakers": [{"markets": [
                {"key": "totals", "outcomes": [{"point": total}, {"point": total}]},
                {"key": "spreads", "outcomes": [{"name": home, "point": hsp},
                                                {"name": away, "point": -hsp}]},
            ]}]})
    return out


def test_vegas_missing_slate_teams_is_warning_not_fatal(monkeypatch):
    """REAL code path: team_lines() must return the teams it has and populate
    missing_teams for the ones it doesn't — never raise for a partial board.
    (The previous version of this test re-implemented the filter inline and passed
    while production raised; every regression test must call the real function.)"""
    from dfs.vegas import OddsClient
    oc = OddsClient(api_key="test")
    board = _fake_odds_board([("Philadelphia Eagles", "Dallas Cowboys", 47.0, -3.0)])
    monkeypatch.setattr(oc, "_get", lambda *a, **k: board)
    out = oc.team_lines(slate_teams={"PHI", "DAL", "JAC"})   # JAC kicked off Thursday
    assert set(out) == {"PHI", "DAL"}
    assert oc.missing_teams == ["JAC"]
    assert out["PHI"].implied_total == 25.0


def test_vegas_jax_jac_normalization(monkeypatch):
    """A FanDuel 'JAC' slate must match the Odds board's 'Jacksonville Jaguars' —
    both sides route through matching.norm_team, no hardcoded abbreviations."""
    from dfs.vegas import OddsClient
    oc = OddsClient(api_key="test")
    board = _fake_odds_board([("Jacksonville Jaguars", "Tennessee Titans", 41.5, -2.5)])
    monkeypatch.setattr(oc, "_get", lambda *a, **k: board)
    out = oc.team_lines(slate_teams={"JAC", "TEN"})
    assert set(out) == {"JAC", "TEN"} and oc.missing_teams == []


def test_ownership_weights_blend_and_shrink():
    slate = _projected_slate()
    from dfs.matching import norm_name
    heavy = slate.players[0]
    measured = {norm_name(heavy.name): 50.0}          # half the league on this player
    w1 = ownership_weights(slate, measured, n_weeks=1)
    w8 = ownership_weights(slate, measured, n_weeks=8)
    i = slate.players.index(heavy)
    base = ownership_weights(slate, measured, n_weeks=0)
    assert base is None                                # no evidence -> no override
    assert w8[i] > w1[i] > 0, "more weeks -> measured ownership dominates the prior"
    assert abs(w1.sum() - 1) < 1e-9 and abs(w8.sum() - 1) < 1e-9


def test_ownership_field_labels_source():
    slate = _projected_slate()
    from dfs.matching import norm_name
    measured = {norm_name(p.name): 25.0 for p in slate.players[:20]}
    fields = build_field_ensemble(slate, SPEC, 11, n_fields=3, seed=5,
                                  measured_ownership=measured, ownership_weeks=2)
    assert fields[0].source == "ownership(2wk)"
    assert all(len(f.lineups) == 11 for f in fields)


def test_auto_context_flows_from_log(tmp_path):
    """The build must price points using the LIVE season standing, not week-0 defaults."""
    log = ResultLog(tmp_path / "r.db")
    log.log_capture(parse_contest(CONTEST_FIX, 2025, 18))
    ctx = log.season_context(2025, me="brettleath", weeks_total=21)
    assert ctx.weeks_played == 1
    assert ctx.deficit() > 0                          # brettleath trails xleathy
    w0 = weights_for("friends_league",
                     SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES, weeks_total=21))
    w1 = weights_for("friends_league", ctx)
    assert w1.w_points > w0.w_points                   # fewer weeks left -> pricier points


# ---- FantasyPros injuries: verified live schema (2026-08-16) ----
def test_fp_injury_schema_parsed():
    rows = [{"name": "Alec Pierce", "status": "PUP", "team_id": "IND",
             "injury_type": "", "comment": "", "practice_1": None,
             "probability_of_playing": None},
            {"name": "Q Full", "status": "Q", "team_id": "PHI", "injury_type": "Knee",
             "practice_1": "LP", "practice_2": "FP", "practice_3": "FP",
             "probability_of_playing": 80},
            {"name": "Q Nopractice", "status": "Q", "team_id": "DAL",
             "injury_type": "Hamstring", "practice_1": "DNP", "practice_2": "DNP",
             "practice_3": "DNP", "probability_of_playing": 20}]
    recs = records_from_fantasypros(rows)
    assert recs[_norm("Alec Pierce")].status is Status.IR
    # practice participation must separate two identically-tagged Questionables
    assert recs[_norm("Q Full")].status is Status.PROBABLE
    assert recs[_norm("Q Nopractice")].status is Status.DOUBTFUL
    assert recs[_norm("Q Nopractice")].action is Action.REMOVE
    assert "practice DNP/DNP/DNP" in recs[_norm("Q Nopractice")].detail


def test_injury_detail_is_human_readable():
    recs = records_from_fantasypros([{"name": "X Y", "status": "Q", "team_id": "KC",
                                      "injury_type": "Ankle", "practice_1": "LP",
                                      "probability_of_playing": 50,
                                      "comment": "expected to test pregame"}])
    d = recs[_norm("X Y")].detail
    assert "Ankle" in d and "practice LP" in d and "50% to play" in d


# ---- NFL calendar: the system knows what week it is ----
from dfs.nflcal import current_week, WeekInfo
from zoneinfo import ZoneInfo as _ZI

_ET = _ZI("America/New_York")


def _at(y, m, d, h=12):
    return datetime(y, m, d, h, tzinfo=_ET).astimezone(timezone.utc)


def test_preseason_prepares_week_one():
    w = current_week(_at(2026, 8, 16))
    assert w.season == 2026 and w.week == 1 and w.is_preseason
    assert w.days_to_kickoff(_at(2026, 8, 16)) > 20


def test_week_holds_through_monday_night():
    """MNF still belongs to the week that just played."""
    assert current_week(_at(2026, 9, 14, 23)).week == 1


def test_week_rolls_tuesday_morning():
    """New FanDuel slates post Tue/Wed — that is the handoff."""
    assert current_week(_at(2026, 9, 15, 7)).week == 2


def test_january_belongs_to_prior_season():
    w = current_week(_at(2027, 1, 5))
    assert w.season == 2026 and w.week >= 17


def test_midseason_week_is_sane():
    w = current_week(_at(2026, 10, 20))
    assert w.season == 2026 and 5 <= w.week <= 8


def test_season_hint_overrides_detection():
    w = current_week(_at(2026, 8, 16), season_hint=2025)
    assert w.season == 2025


def test_calendar_endpoint_and_override():
    from fastapi.testclient import TestClient
    from dfs.web import app
    c = TestClient(app)
    r = c.get("/api/calendar").json()
    assert r["season"] >= 2025 and 1 <= r["week"] <= 22
    assert r["reason"] and r["label"].startswith(str(r["season"]))


# ---- league vs non-league isolation ----
def test_public_contests_cannot_pollute_league_standings(tmp_path):
    """A captured public H2H must not appear in Leather League standings or drive
    the league objective."""
    log = ResultLog(tmp_path / "r.db")
    cap = parse_contest(CONTEST_FIX, 2025, 18, contest="Leather League")
    log.log_capture(cap)
    pub = parse_contest(CONTEST_FIX, 2025, 17, contest="Public H2H")
    log.log_capture(pub)
    league = log.standings(2025, contest_like="Leather League")
    everything = log.standings(2025)
    assert all(s.weeks == 1 for s in league), "league rows must count league weeks only"
    assert sum(s.weeks for s in everything) == 2 * len(league)
    ctx = log.season_context(2025, me="brettleath", contest_like="Leather League")
    assert ctx.weeks_played == 1


def test_league_ownership_scoped_to_league(tmp_path):
    log = ResultLog(tmp_path / "r.db")
    log.log_capture(parse_contest(CONTEST_FIX, 2025, 18, contest="Leather League"))
    log.log_capture(parse_contest(CONTEST_FIX, 2025, 17, contest="Public GPP"))
    league_own = log.measured_ownership(2025, contest_like="Leather League")
    assert league_own
    assert log.ownership_week_count(2025, contest_like="Leather League") == 1
    assert log.ownership_week_count(2025) == 2


def test_slate_library_reuse(tmp_path, monkeypatch):
    """Upload once, build again without a file: the stored CSV is reused; a missing
    week fails with a clear message."""
    import dfs.web as web
    from fastapi.testclient import TestClient
    monkeypatch.setattr(web, "UPLOADS", tmp_path)
    c = TestClient(web.app)
    r = c.post("/api/build", data={"season": 2031, "week": 3, "pool": 5})
    assert r.status_code == 400 and "No stored salary CSV" in r.text
    (tmp_path / "2031-w03-abc123.csv").write_text(
        open(FIX / "fd_full_slate.csv", encoding="utf-8-sig").read())
    s = c.get("/api/slates", params={"season": 2031, "week": 3}).json()
    assert len(s) == 1 and s[0]["rows"] > 100


# ---- adversarial review round 2 (2026-08-16) ----
def test_simulated_means_match_projections():
    """The reconstructed distribution ran ~14% hot (RB +16%). Simulated player means
    must equal projection x the cell's empirical mean ratio, not drift above it."""
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=20000, seed=11)
    ratios = [sim.score([p.fd_id]).totals.mean() / p.projection
              for p in slate.players[:80] if p.projection > 3]
    m = float(np.mean(ratios))
    assert 0.93 < m < 1.07, f"simulated/projected mean ratio {m:.3f} — mean pinning broken"


def test_mvp_salary_premium_charged_and_reported():
    """FanDuel charges 1.5x salary for the showdown MVP. Default must be 1.5, the
    premium must count against the cap, and reported salary must include it."""
    spec = ContestSpec(name="SD", profile=Profile.SHOWDOWN_FRIENDS,
                       slate_type=SlateType.SINGLE_GAME, field_size=6)
    assert spec.mvp_salary_mult == 1.5
    slate = _showdown_slate()
    sal = {p.fd_id: p.salary for p in slate.players}
    for c in generate_pool(slate, spec, n=3, min_unique=1):
        charged = sum(sal[i] for i in c.player_ids) + 0.5 * sal[c.mvp_id]
        assert charged <= spec.salary_cap + 1
        assert c.salary == pytest.approx(charged, abs=1), \
            "reported salary must include the MVP premium"


def test_showdown_field_respects_mvp_premium():
    slate = _showdown_slate()
    spec = ContestSpec(name="SD", profile=Profile.SHOWDOWN_FRIENDS,
                       slate_type=SlateType.SINGLE_GAME, field_size=6)
    fm = build_prior_field(slate, spec, n_opponents=5, seed=7)
    sal = {p.fd_id: p.salary for p in slate.players}
    for l, mvp in zip(fm.lineups, fm.mvp_ids):
        assert sum(sal[i] for i in l) + 0.5 * sal[mvp] <= spec.salary_cap


def test_candidate_identity_includes_mvp():
    """Same five players, different MVP = a different lineup."""
    from dfs.optimize import Candidate
    a = Candidate(player_ids=("1", "2", "3"), salary=100, proj_sum=10.0, mvp_id="1")
    b = Candidate(player_ids=("1", "2", "3"), salary=100, proj_sum=10.0, mvp_id="2")
    assert a.key != b.key


# ---- adversarial review round 3: operational-correctness fixes (2026-08-17) ----
from dfs.optimize import max_projection_lineup


def test_max_projection_baseline_beats_pool_max():
    """The reported 'edge vs max-projection' is only meaningful against the TRUE
    unconstrained cap-legal optimum. Selecting the baseline from the constrained
    candidate pool measured 1.96 projected points short on this fixture — enough
    to flip the sign of the claimed edge."""
    slate = _projected_slate()
    pool = generate_pool(slate, SPEC, n=120,
                         stack=StackRule(require_qb_stack=1), min_unique=3)
    pool_max = max(c.proj_sum for c in pool)
    true_max = max_projection_lineup(slate, SPEC)
    assert true_max.proj_sum >= pool_max - 1e-6
    # regression pin on the fixture: the gap is real and material
    assert true_max.proj_sum - pool_max > 1.0
    # and the baseline is actually legal
    assert len(true_max.player_ids) == 9
    byid = {p.fd_id: p for p in slate.players}
    assert sum(byid[i].salary for i in true_max.player_ids) <= SPEC.salary_cap


def test_kicker_scorer_real_path():
    """score() must route K through a real kicker scorer — score_skill reads a
    kicker's FG stat line as ~0, which silently zeroes every K projection."""
    from dfs.scoring import score
    pts, breakdown = score({"fg": 2.0, "xpt": 3.0}, "K")
    assert pts > 8.0                       # 2 FG + 3 XP is never ~0
    assert breakdown["fg"] > 0 and breakdown["xp"] == 3.0
    zero, _ = score({"fg": 2.0, "xpt": 3.0}, "RB")   # skill scorer ignores FG stats
    assert zero == 0.0


def test_fantasypros_requests_kickers():
    from dfs.fantasypros import POSITIONS
    assert "K" in POSITIONS


def test_kicker_never_fills_a_classic_slot():
    """Classic FanDuel rosters have no K slot. The position ranges sum to a minimum
    of 8 against a roster size of 9, so without an explicit exclusion a projected
    kicker could take the 9th slot the moment K projections go live."""
    slate = _projected_slate()
    from dfs.slate import SlatePlayer
    from dfs.distributions import floor_ceiling
    k = SlatePlayer(fd_id="k-test", name="Test Kicker", position="K", team="PHI",
                    opponent="DAL", salary=4000, game="DAL@PHI")
    k.projection = 999.0                   # irresistible unless excluded
    k.floor_p10, k.ceiling_p90 = floor_ceiling(9.0, "K", DIST)
    slate.players.append(k)
    true_max = max_projection_lineup(slate, SPEC)
    assert "k-test" not in true_max.player_ids
    pool = generate_pool(slate, SPEC, n=4, min_unique=3)
    for c in pool:
        assert "k-test" not in c.player_ids


def test_log_entry_mvp_roundtrip(tmp_path):
    """Showdown entry must persist MVP identity and CHARGED salary, and swap must
    be able to recover the MVP from the log."""
    import json as _json
    slate = _showdown_slate()
    pool = generate_pool(slate, SD_SPEC, n=2, min_unique=1)
    cand = pool[0]
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in cand.player_ids]
    db = tmp_path / "r.db"
    rl = ResultLog(db)
    rl.log_entry(2026, 1, "SNF", players, mvp_id=cand.mvp_id, mvp_salary_mult=1.5)
    with rl._c() as c:
        row = c.execute("SELECT salary, lineup_json FROM entries").fetchone()
    lineup = _json.loads(row["lineup_json"])
    recovered_mvp = next(p["fd_id"] for p in lineup if p.get("mvp"))
    assert recovered_mvp == cand.mvp_id
    base = sum(p.salary for p in players)
    mvp_sal = byid[cand.mvp_id].salary
    assert row["salary"] == base + int(round(0.5 * mvp_sal))   # charged, not base


def test_showdown_export_has_six_player_columns(tmp_path):
    from dfs.export import export_upload_csv, DEFAULT_SHOWDOWN_COLS
    assert DEFAULT_SHOWDOWN_COLS.count("AnyFLEX") == 5 and "MVP" in DEFAULT_SHOWDOWN_COLS
    slate = _showdown_slate()
    pool = generate_pool(slate, SD_SPEC, n=1, min_unique=1)
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in pool[0].player_ids]
    out = tmp_path / "u.csv"
    ex = export_upload_csv(players, out, slate_type=SlateType.SINGLE_GAME,
                           mvp_id=pool[0].mvp_id)
    import csv as _csv
    rows = list(_csv.reader(out.open()))
    hdr, row = rows[0], rows[1]
    ids = [v for c, v in zip(hdr, row) if c in ("MVP", "AnyFLEX")]
    assert len(ids) == 6 and ids[0] == pool[0].mvp_id and all(ids)


def test_build_end_to_end_produces_artifacts(tmp_path, monkeypatch):
    """THE test that was missing: a full `build` run must leave behind (1) the upload
    CSV, (2) a logged entry that `swap` can find (with MVP recoverable), (3) the
    pushover card, (4) the lineup JSON. 96 unit tests passed while all four were
    silently gone — never again."""
    import json as _json
    from dfs import cli as cli_mod
    from dfs.fantasypros import FPProjection

    slate_probe, _ = ingest_csv(FIX / "fd_full_slate.csv", "e2e", 2026, 1)

    def fake_projections(self, season, week, positions=None):
        import random as _r
        _r.seed(11)
        return [FPProjection(player_id=f"fp{i}", name=p.name, team=p.team,
                             position=p.position,
                             points=round(max(1.0, (p.fppg or 5.0) * _r.uniform(0.9, 1.1)), 2),
                             stats={}, breakdown={})
                for i, p in enumerate(slate_probe.players)]

    monkeypatch.setattr("dfs.fantasypros.FantasyProsClient.weekly_projections",
                        fake_projections)
    monkeypatch.setenv("FANTASYPROS_API_KEY", "test")   # ctor check; method is stubbed

    export = tmp_path / "upload.csv"
    log_db = tmp_path / "results.db"
    push = tmp_path / "push.txt"
    outj = tmp_path / "lineups.json"
    snapdir = tmp_path / "snaps"
    rc = cli_mod.main([
        "build", "--csv", str(FIX / "fd_full_slate.csv"),
        "--season", "2026", "--week", "1", "--slate-id", "e2e",
        "--no-vegas", "--no-injuries",
        "--pool", "10", "--sims", "400", "--fields", "3", "--show", "1",
        "--snapshot-dir", str(snapdir),
        "--export", str(export), "--log-db", str(log_db),
        "--pushover-out", str(push), "--out", str(outj),
    ])
    assert rc == 0
    # (1) upload CSV: header + one lineup row with 9 populated player columns
    import csv as _csv
    rows = list(_csv.reader(export.open()))
    assert len(rows) == 2
    player_cells = [v for c, v in zip(rows[0], rows[1])
                    if c not in ("entry_id", "contest_id", "contest_name")]
    assert len(player_cells) == 9 and all(player_cells)
    # (2) logged entry, findable exactly the way cmd_swap looks for it — and it must
    # be the MAX-PROJECTION arm for the friends league (--entry auto), with the
    # model lineup logged as a shadow row under a suffixed contest name
    from dfs.optimize import max_projection_lineup as _mpl
    from dfs.contest_spec import ContestSpec as _CS, Profile as _P
    from dfs.slate import SlateType as _ST
    rl = ResultLog(log_db)
    with rl._c() as c:
        row = c.execute("""SELECT lineup_json, salary, objective FROM entries
                           WHERE season=2026 AND week=1
                           AND contest='Leather League'""").fetchone()
        shadow = c.execute("""SELECT lineup_json, objective FROM entries
                              WHERE season=2026 AND week=1
                              AND contest LIKE 'Leather League [shadow:%'""").fetchone()
    assert row is not None
    lineup = _json.loads(row["lineup_json"])
    assert len(lineup) == 9 and all(p["fd_id"] for p in lineup)
    assert row["salary"] == sum(p["salary"] for p in lineup)  # classic: no MVP premium
    assert row["objective"].startswith("arm=max-proj")
    assert shadow is not None and shadow["objective"].startswith("arm=model")
    assert len(_json.loads(shadow["lineup_json"])) == 9
    # entered lineup IS the true unconstrained max-projection lineup: rebuild an
    # identically-projected slate (same seed as the stub), solve independently,
    # and require the same player set — a REAL comparison, not a re-implementation
    import random as _r
    _r.seed(11)
    proj_by_name = {p2.name: round(max(1.0, (p2.fppg or 5.0) * _r.uniform(0.9, 1.1)), 2)
                    for p2 in slate_probe.players}
    slate2, _ = ingest_csv(FIX / "fd_full_slate.csv", "e2e", 2026, 1)
    from dfs.distributions import floor_ceiling as _fc
    for p2 in slate2.players:
        p2.projection = proj_by_name.get(p2.name, 1.0)
        p2.proj_source = "test"
        p2.floor_p10, p2.ceiling_p90 = _fc(p2.projection, p2.position, DIST)
    spec2 = _CS(name="Leather League", profile=_P.FRIENDS_LEAGUE,
                slate_type=_ST.FULL, field_size=12, entry_fee=15.0)
    expected = _mpl(slate2, spec2)
    assert set(p["fd_id"] for p in lineup) == set(expected.player_ids)
    # (3) pushover card names the arm; (4) lineup JSON exists
    assert "[max-proj]" in push.read_text()
    assert _json.loads(outj.read_text())["lineups"]
    # (5) at-lock snapshot written and readable, with audit fields
    from dfs.snapshots import read_snapshot
    snaps = list(snapdir.glob("fp-2026-w01-*.json.gz"))
    assert len(snaps) == 1
    sd = read_snapshot(snaps[0])
    assert sd["kind"] == "fp_at_lock" and sd["payload_sha256"] and sd["scorer_source_sha256"]
    assert sd["season"] == 2026 and sd["week"] == 1



def test_entry_arm_model_flag(tmp_path, monkeypatch):
    """--entry model must enter the sim-ranked lineup and shadow max-proj."""
    import json as _json
    from dfs import cli as cli_mod
    from dfs.fantasypros import FPProjection

    slate_probe2, _ = ingest_csv(FIX / "fd_full_slate.csv", "arm", 2026, 1)

    def fake_projections(self, season, week, positions=None):
        import random as _r
        _r.seed(11)
        return [FPProjection(player_id=f"fp{i}", name=p.name, team=p.team,
                             position=p.position,
                             points=round(max(1.0, (p.fppg or 5.0) * _r.uniform(0.9, 1.1)), 2),
                             stats={}, breakdown={})
                for i, p in enumerate(slate_probe2.players)]

    monkeypatch.setattr("dfs.fantasypros.FantasyProsClient.weekly_projections",
                        fake_projections)
    monkeypatch.setenv("FANTASYPROS_API_KEY", "test")
    log_db = tmp_path / "r.db"
    rc = cli_mod.main([
        "build", "--csv", str(FIX / "fd_full_slate.csv"),
        "--season", "2026", "--week", "1", "--slate-id", "arm",
        "--no-vegas", "--no-injuries", "--no-snapshot",
        "--pool", "8", "--sims", "300", "--fields", "2", "--show", "1",
        "--entry", "model",
        "--log-db", str(log_db), "--out", str(tmp_path / "l.json"),
    ])
    assert rc == 0
    rl = ResultLog(log_db)
    with rl._c() as c:
        row = c.execute("SELECT objective FROM entries WHERE contest='Leather League'").fetchone()
        shadow = c.execute("SELECT objective FROM entries WHERE contest LIKE '%[shadow:%'").fetchone()
    assert row["objective"].startswith("arm=model")
    assert shadow["objective"].startswith("arm=max-proj")


# ---- defects found by the first REAL FanDuel single-game CSV (2026-08-23) ----
REAL_SD = FIX / "fd_showdown_real.csv"


def _real_showdown_slate():
    """Genuine FanDuel single-game salary file (SEA@TEN preseason, 2026-08-23).
    Projections proxied from FPPG — the point is the real slate SHAPE, not values."""
    from dfs.distributions import floor_ceiling
    slate, rep = ingest_csv(REAL_SD, "real-sd", 2026, 0, strict=False)
    for p in slate.players:
        p.projection = max(1.0, p.fppg or 4.0)
        p.proj_source = "fppg-proxy"
        p.floor_p10, p.ceiling_p90 = floor_ceiling(p.projection, p.position, DIST)
    return slate, rep


REAL_SD_SPEC = ContestSpec(name="RealSD", profile=Profile.SHOWDOWN_FRIENDS,
                           slate_type=SlateType.SINGLE_GAME, field_size=100,
                           entry_fee=5.0)


def test_real_fanduel_showdown_csv_ingests():
    slate, rep = _real_showdown_slate()
    assert rep.detected_slate_type == SlateType.SINGLE_GAME
    assert len(rep.teams) == 2 and set(rep.teams) == {"SEA", "TEN"}
    assert rep.ingested == 60 and len(rep.dropped_injury) == 4
    assert any(p.position == "K" for p in slate.players)
    assert any(p.position == "D" for p in slate.players)


def test_mvp_multiplier_confirmed_by_fanduel_csv():
    """FanDuel's own 'MVP 1.5x Salary' column is authoritative — read the multiplier
    from their file instead of trusting our constant, so a rules change surfaces as
    a mismatch rather than silently mispricing every lineup against the cap."""
    _, rep = _real_showdown_slate()
    assert rep.mvp_salary_mult_observed == 1.5
    assert rep.mvp_salary_mult_observed == REAL_SD_SPEC.mvp_salary_mult
    assert not any("multiplier is not constant" in v
                   for v in rep.validation_problems)


def test_showdown_pool_is_not_silently_halved():
    """Phase 2 of generate_pool was gated to SlateType.FULL, so every showdown pool
    was capped at n//2 — half the requested candidates never existed. The old test
    asked for 8 and only asserted the pool was non-empty, so it never caught this."""
    slate, _ = _real_showdown_slate()
    for n in (5, 10, 20):
        pool = generate_pool(slate, REAL_SD_SPEC, n=n, min_unique=1)
        assert len(pool) == n, f"requested {n}, got {len(pool)}"
        assert all(len(c.player_ids) == 6 and c.mvp_id for c in pool)
        keys = [c.key for c in pool]
        assert len(set(keys)) == len(keys)          # no duplicate candidates


def test_showdown_allows_five_one_team_split():
    """FanDuel single-game requires only that both teams appear, so 5-1 is legal on a
    6-man roster. The per-team cap was 4 — correct for the old 5-man roster, and it
    silently made every 5-1 stack unreachable after the roster fix."""
    from dfs.optimize import _solve_showdown, StackRule as _SR
    slate, _ = _real_showdown_slate()
    # make one team overwhelmingly attractive so the optimum WANTS a 5-1 split
    for p in slate.players:
        if p.team == "SEA":
            p.projection *= 10
    cand = _solve_showdown(slate.players, REAL_SD_SPEC, _SR(require_qb_stack=0),
                           set(), [], 0)
    assert cand is not None and len(cand.player_ids) == 6
    byid = {p.fd_id: p for p in slate.players}
    from collections import Counter
    split = Counter(byid[i].team for i in cand.player_ids)
    assert max(split.values()) == 5 and min(split.values()) == 1
    assert len(split) == 2                      # both teams still represented


def test_showdown_mvp_rotations_present_and_legal():
    """MVP choice is the biggest single-game decision; rotations over an existing
    player set are distinct legal lineups and must appear in the pool."""
    slate, _ = _real_showdown_slate()
    pool = generate_pool(slate, REAL_SD_SPEC, n=20, min_unique=1)
    byid = {p.fd_id: p for p in slate.players}
    by_set = {}
    for c in pool:
        by_set.setdefault(frozenset(c.player_ids), set()).add(c.mvp_id)
    assert any(len(mvps) > 1 for mvps in by_set.values()), "no MVP rotation generated"
    for c in pool:
        charged = (sum(byid[i].salary for i in c.player_ids)
                   + int(round(0.5 * byid[c.mvp_id].salary)))
        assert charged <= REAL_SD_SPEC.salary_cap
        assert c.salary == charged              # reported salary is the charged one


# ---- real FanDuel Week 1 classic slate (Sun 2026-09-13 main, 12 games) ----
REAL_W1 = FIX / "fd_classic_real_w1.csv"


def _real_w1_slate():
    from dfs.distributions import floor_ceiling
    slate, rep = ingest_csv(REAL_W1, "real-w1", 2026, 1, strict=False)
    for p in slate.players:
        p.projection = max(1.0, p.fppg or 3.0)
        p.proj_source = "fppg-proxy"
        p.floor_p10, p.ceiling_p90 = floor_ceiling(p.projection, p.position, DIST)
    return slate, rep


W1_SPEC = ContestSpec(name="H2H", profile=Profile.H2H, slate_type=SlateType.FULL,
                      field_size=2, entry_fee=5.0)


def test_real_w1_slate_ingests():
    slate, rep = _real_w1_slate()
    assert rep.detected_slate_type == SlateType.FULL
    assert len(rep.teams) == 24 and rep.ingested == 703
    assert rep.mvp_salary_mult_observed is None      # classic CSV has no MVP column


def test_real_classic_slate_has_no_kickers():
    """FanDuel classic rosters have no K slot and the salary file carries no kickers.
    Kickers only appear on single-game slates. This is why max_projection_lineup and
    generate_pool must exclude non-classic positions rather than relying on the CSV."""
    slate, _ = _real_w1_slate()
    assert not any(p.position == "K" for p in slate.players)
    assert {p.position for p in slate.players} == {"QB", "RB", "WR", "TE", "D"}


def test_real_w1_roster_shape_matches_fanduel_flex():
    """FanDuel classic: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 DEF = 9. Our
    ROSTER_FULL encodes the FLEX as widened ranges; verify a real solve lands inside
    them and that the CSV's own Roster Position column agrees."""
    import csv as _csv
    from collections import Counter
    from dfs.contest_spec import ROSTER_FULL
    rps = Counter(r["Roster Position"] for r in _csv.DictReader(REAL_W1.open()))
    assert set(rps) == {"QB", "RB/FLEX", "WR/FLEX", "TE/FLEX", "DEF"}
    slate, _ = _real_w1_slate()
    mp = max_projection_lineup(slate, W1_SPEC)
    byid = {p.fd_id: p for p in slate.players}
    counts = Counter(byid[i].position for i in mp.player_ids)
    assert len(mp.player_ids) == 9
    assert mp.salary <= W1_SPEC.salary_cap
    for pos, (lo, hi) in ROSTER_FULL.items():
        assert lo <= counts.get(pos, 0) <= hi, f"{pos}={counts.get(pos,0)} outside {lo}-{hi}"
    # exactly one flex beyond the base 1/2/3/1/1
    assert sum(counts.values()) == 9


def test_real_w1_pool_fills_to_requested_size():
    slate, _ = _real_w1_slate()
    for n in (10, 25):
        pool = generate_pool(slate, W1_SPEC, n=n,
                             stack=StackRule(require_qb_stack=1), min_unique=3)
        assert len(pool) == n
        assert all(len(c.player_ids) == 9 for c in pool)
        assert all(c.salary <= W1_SPEC.salary_cap for c in pool)


def test_real_w1_contains_jac_the_vegas_regression_team():
    """The live Week 1 slate really does use 'JAC' — the abbreviation that broke the
    Vegas layer twice. Odds boards say Jacksonville/JAX; both sides must normalize."""
    from dfs.matching import norm_team
    slate, rep = _real_w1_slate()
    assert "JAC" in rep.teams
    assert norm_team("JAX") == "JAC" == norm_team("JAC")
    assert {norm_team(t) for t in rep.teams} == set(rep.teams)   # already canonical


# ---- matcher collision found by the LIVE Week 1 dry run (2026-08-23) ----
def _fpp(name, team, pos, pts):
    from dfs.fantasypros import FPProjection
    return FPProjection(player_id=name, name=name, team=team, position=pos,
                        points=pts, stats={}, breakdown={})


def _sp(name, team, pos, salary, fd_id=None):
    from dfs.slate import SlatePlayer
    return SlatePlayer(fd_id=fd_id or name, name=name, position=pos, team=team,
                       opponent="X", salary=salary, game="A@B")


def test_short_key_collision_rejected_cross_team():
    """LIVE FAILURE: 'Jalon Daniels' (TB backup, $6000) received Jayden Daniels'
    (WAS) projection via first-initial+lastname+position, then anchored all three
    winning lineups. Different first name + different team must NOT match."""
    from dfs.matching import match_slate
    fp = [_fpp("Jayden Daniels", "WAS", "QB", 19.0),
          _fpp("Jayden Reed", "GB", "WR", 9.6)]
    slate = [_sp("Jalon Daniels", "TB", "QB", 6000),
             _sp("Ja'seem Reed", "CAR", "WR", 4000),
             _sp("Jayden Daniels", "WAS", "QB", 8500, fd_id="real-jd")]
    mapping, rep = match_slate(slate, fp)
    assert "Jalon Daniels" not in mapping            # collision rejected
    assert "Ja'seem Reed" not in mapping             # collision rejected
    assert mapping["real-jd"].name == "Jayden Daniels"   # the real one still matches
    assert {u[0] for u in rep.unmatched} == {"Jalon Daniels", "Ja'seem Reed"}


def test_short_key_nickname_variants_still_match():
    """The tier's legitimate purpose survives the guard: prefix-compatible first
    names (nicknames) match, and team agreement rescues same-team initial hits."""
    from dfs.matching import match_slate
    fp = [_fpp("Cameron Ward", "TEN", "QB", 15.0),
          _fpp("Kenneth Walker III", "SEA", "RB", 13.5),
          _fpp("Joshua Palmer", "BUF", "WR", 8.0)]
    slate = [_sp("Cam Ward", "TEN", "QB", 7000),
             _sp("Ken Walker", "SEA", "RB", 6800),
             _sp("Josh Palmer", "BUF", "WR", 5200)]
    mapping, rep = match_slate(slate, fp)
    assert len(mapping) == 3 and not rep.unmatched
    assert rep.by_method.get("short+pos", 0) >= 2


def test_exact_name_cross_team_still_matches():
    """Trades/stale rosters: an EXACT name match across teams must keep matching
    (that is the documented reason team is not part of the primary key), and it is
    recorded as a team disagreement."""
    from dfs.matching import match_slate
    fp = [_fpp("A.J. Brown", "PHI", "WR", 14.0)]
    slate = [_sp("AJ Brown", "DAL", "WR", 8700)]     # hypothetical trade
    mapping, rep = match_slate(slate, fp)
    assert len(mapping) == 1
    assert rep.team_disagreements and rep.team_disagreements[0][0] == "AJ Brown"


# ---- arm-aware late swap (first live swap drill, 2026-08-23) ----
def test_maxproj_swap_never_trades_projection_away():
    """LIVE FAILURE: on a max-proj entry, the model-objective swap proposed giving up
    1.0 projected points for +$0.13 of simulator objective. A max-projection entry
    must be re-solved on PROJECTION under lock constraints — the sim has no say."""
    from dfs.lateswap import propose_swap_maxproj
    from dfs.kickoffs import KickoffSchedule
    slate = _projected_slate()
    mp = max_projection_lineup(slate, SPEC)
    sched = KickoffSchedule({})                      # nothing locked
    prop = propose_swap_maxproj(slate, mp.player_ids, SPEC, sched)
    assert not prop.swaps                            # already optimal: no churn
    assert prop.criterion == "projected pts"
    assert prop.old_dollars == prop.new_dollars

    # make a better lineup exist: bump a non-lineup player's projection hard
    byid = {p.fd_id: p for p in slate.players}
    outsider = next(p for p in slate.players
                    if p.fd_id not in mp.player_ids and p.position == "WR")
    outsider.projection = 60.0
    prop2 = propose_swap_maxproj(slate, mp.player_ids, SPEC, sched)
    assert prop2.swaps and prop2.new_dollars > prop2.old_dollars
    assert outsider.fd_id in prop2.proposed_ids
    # and the proposal never loses projection
    new_proj = sum(byid[i].projection for i in prop2.proposed_ids)
    old_proj = sum(byid[i].projection for i in mp.player_ids)
    assert new_proj > old_proj


def test_maxproj_swap_forced_when_player_swept():
    """A lineup player removed by the inactives sweep forces a swap even if the
    replacement lineup projects lower than the (now fictional) original total."""
    from dfs.lateswap import propose_swap_maxproj
    from dfs.kickoffs import KickoffSchedule
    slate = _projected_slate()
    mp = max_projection_lineup(slate, SPEC)
    gone = mp.player_ids[0]
    slate.players = [p for p in slate.players if p.fd_id != gone]   # swept
    prop = propose_swap_maxproj(slate, mp.player_ids, SPEC, KickoffSchedule({}))
    assert prop.swaps
    assert gone not in prop.proposed_ids
    assert any(out_p.fd_id == gone for out_p, _ in prop.swaps)
    assert len(prop.proposed_ids) == 9


def test_web_test_mode_routes_to_sandbox(tmp_path, monkeypatch):
    """The 'Test run' checkbox must route EVERY write (entry log, exports, snapshots)
    into the test sandbox, so a UI test can never touch season records — and an
    unchecked box must route to production paths."""
    from fastapi.testclient import TestClient
    import dfs.web as web
    captured = {}
    monkeypatch.setattr(web, "_start_job",
                        lambda kind, argv: captured.setdefault(kind, argv) or "job-x")
    monkeypatch.setattr(web, "UPLOADS", tmp_path / "up")
    monkeypatch.setattr(web, "TEST", tmp_path / "test")
    monkeypatch.setattr(web, "DB_TEST", tmp_path / "test" / "results.db")
    monkeypatch.setattr(web, "LINEUPS_TEST", tmp_path / "test" / "lineups")
    monkeypatch.setattr(web, "DB", tmp_path / "prod.db")
    monkeypatch.setattr(web, "LINEUPS", tmp_path / "lineups")
    c = TestClient(web.app)

    r = c.post("/api/build",
               files={"csv": ("w1.csv", REAL_W1.open("rb"), "text/csv")},
               data={"season": 2026, "week": 1, "test_mode": "true"})
    assert r.status_code == 200
    argv = captured["build"]
    dbi = argv.index("--log-db") + 1
    assert "test" in argv[dbi] and "prod.db" not in argv[dbi]
    assert "--snapshot-dir" in argv
    exi = argv.index("--export") + 1
    assert str(tmp_path / "test" / "lineups") in argv[exi]

    captured.clear()
    r = c.post("/api/build",
               files={"csv": ("w1.csv", REAL_W1.open("rb"), "text/csv")},
               data={"season": 2026, "week": 1})
    assert r.status_code == 200
    argv = captured["build"]
    assert argv[argv.index("--log-db") + 1].endswith("prod.db")
    assert "--snapshot-dir" not in argv

    captured.clear()
    r = c.post("/api/swap",
               files={"csv": ("w1.csv", REAL_W1.open("rb"), "text/csv")},
               data={"season": 2026, "week": 1, "test_mode": "true"})
    assert r.status_code == 200
    argv = captured["swap"]
    assert "test" in argv[argv.index("--log-db") + 1]


# ---- wrong-CSV-for-contest guard (2026-08-23) ----
def test_build_rejects_showdown_csv_for_league_profile(tmp_path, monkeypatch):
    """A single-game player list must not silently build a 6-man lineup logged under
    the league contest. Stored CSVs are reused by season/week alone, so a leftover
    showdown file from the same week is a live hazard."""
    from dfs import cli as cli_mod
    monkeypatch.setenv("FANTASYPROS_API_KEY", "test")
    rc = cli_mod.main([
        "build", "--csv", str(REAL_SD),          # single-game file
        "--season", "2026", "--week", "1", "--slate-id", "bad",
        "--profile", "friends_league",            # league = full slate
        "--no-vegas", "--no-injuries", "--no-snapshot",
        "--out", str(tmp_path / "x.json"),
    ])
    assert rc == 2                                # BUILD STOPPED, not a wrong lineup


def test_build_rejects_full_csv_for_showdown_profile(tmp_path, monkeypatch):
    from dfs import cli as cli_mod
    monkeypatch.setenv("FANTASYPROS_API_KEY", "test")
    rc = cli_mod.main([
        "build", "--csv", str(REAL_W1),           # full slate
        "--season", "2026", "--week", "1", "--slate-id", "bad2",
        "--profile", "showdown_friends",          # showdown = single game
        "--no-vegas", "--no-injuries", "--no-snapshot",
        "--out", str(tmp_path / "x.json"),
    ])
    assert rc == 2


def test_ambiguous_profiles_accept_either_slate():
    """h2h and public_gpp are played both ways — they must assert nothing."""
    from dfs.contest_spec import expected_slate_type
    assert expected_slate_type(Profile.FRIENDS_LEAGUE) == SlateType.FULL
    assert expected_slate_type(Profile.SHOWDOWN_FRIENDS) == SlateType.SINGLE_GAME
    assert expected_slate_type(Profile.H2H) is None
    assert expected_slate_type(Profile.PUBLIC_GPP) is None


def test_web_stored_csv_picker_is_contest_aware(tmp_path, monkeypatch):
    """Reusing a stored CSV must match the contest type, not just the week — even
    when the wrong-type file is newer."""
    import shutil, time
    from fastapi.testclient import TestClient
    import dfs.web as web
    up = tmp_path / "up"; up.mkdir()
    shutil.copy(REAL_W1, up / "2026-w01-aaaaaa.csv")          # full slate (older)
    time.sleep(0.01)
    shutil.copy(REAL_SD, up / "2026-w01-bbbbbb.csv")          # showdown (NEWER)
    captured = {}
    monkeypatch.setattr(web, "_start_job",
                        lambda kind, argv: captured.setdefault(kind, argv) or "j")
    monkeypatch.setattr(web, "UPLOADS", up)
    monkeypatch.setattr(web, "LINEUPS", tmp_path / "l")
    monkeypatch.setattr(web, "DB", tmp_path / "p.db")
    c = TestClient(web.app)

    r = c.post("/api/build", data={"season": 2026, "week": 1,
                                   "profile": "friends_league"})
    assert r.status_code == 200
    assert r.json()["slate_csv"].endswith("aaaaaa.csv")       # picked FULL, not newest

    captured.clear()
    r = c.post("/api/build", data={"season": 2026, "week": 1,
                                   "profile": "showdown_friends"})
    assert r.status_code == 200
    assert r.json()["slate_csv"].endswith("bbbbbb.csv")       # picked SINGLE_GAME


def test_web_no_matching_stored_csv_errors_clearly(tmp_path, monkeypatch):
    import shutil
    from fastapi.testclient import TestClient
    import dfs.web as web
    up = tmp_path / "up"; up.mkdir()
    shutil.copy(REAL_SD, up / "2026-w01-only.csv")            # showdown only
    monkeypatch.setattr(web, "UPLOADS", up)
    c = TestClient(web.app)
    r = c.post("/api/build", data={"season": 2026, "week": 1,
                                   "profile": "friends_league"})
    assert r.status_code == 400 and "different contest type" in r.text


def test_web_job_latest_enables_reconnect(tmp_path, monkeypatch):
    """When the POST response is lost in transit the run still starts server-side
    (observed: app logged 200, Safari reported 'Load failed'). /api/job-latest lets
    the page find and attach to that run instead of reporting a dead error."""
    from fastapi.testclient import TestClient
    import dfs.web as web
    c = TestClient(web.app)
    assert c.get("/api/job-latest/build").status_code == 404      # none yet

    with web._jobs_lock:
        web._jobs.clear()
        web._jobs["old"] = {"id": "old", "kind": "build", "status": "done",
                            "output": "old", "argv": [], "started": "2026-01-01T00:00:00"}
        web._jobs["new"] = {"id": "new", "kind": "build", "status": "running",
                            "output": "new", "argv": [], "started": "2026-06-01T00:00:00"}
        web._jobs["swp"] = {"id": "swp", "kind": "swap", "status": "running",
                            "output": "s", "argv": [], "started": "2026-05-01T00:00:00"}
    try:
        j = c.get("/api/job-latest/build").json()
        assert j["id"] == "new" and j["status"] == "running"      # newest, not oldest
        assert c.get("/api/job-latest/swap").json()["id"] == "swp"  # kind-scoped
        assert c.get("/api/job-latest/capture").status_code == 404
    finally:
        with web._jobs_lock:
            web._jobs.clear()


def test_web_api_entry_returns_entry_and_shadow(tmp_path, monkeypatch):
    """/api/entry powers the readable lineup cards: it must return the logged entry
    with parsed lineup + arm, the shadow row, and honor the test-DB switch."""
    from fastapi.testclient import TestClient
    import dfs.web as web
    slate = _projected_slate()
    mp = max_projection_lineup(slate, SPEC)
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in mp.player_ids]
    prod = tmp_path / "prod.db"
    ResultLog(prod).log_entry(2026, 1, "Leather League", players,
                              objective="arm=max-proj; test", exp_points=123.0,
                              p_win=0.17)
    ResultLog(prod).log_entry(2026, 1, "Leather League [shadow:model]", players,
                              objective="arm=model; test", exp_points=122.0,
                              p_win=0.16)
    monkeypatch.setattr(web, "DB", prod)
    monkeypatch.setattr(web, "DB_TEST", tmp_path / "missing.db")
    c = TestClient(web.app)

    d = c.get("/api/entry", params={"season": 2026, "week": 1}).json()
    assert d["entry"]["arm"] == "max-proj" and len(d["entry"]["lineup"]) == 9
    assert all(k in d["entry"]["lineup"][0] for k in ("name", "pos", "salary",
                                                      "projection", "inj"))
    assert d["shadow"]["arm"] == "model"
    # missing week -> clean 404, not a crash
    assert c.get("/api/entry", params={"season": 2026, "week": 9}).status_code == 404
    # test switch points at the (absent) sandbox DB -> 404
    assert c.get("/api/entry", params={"season": 2026, "week": 1,
                                       "test": "true"}).status_code == 404


# ================= external review round 4 fixes (2026-08-30) =================
def test_same_team_short_key_collision_rejected():
    """REVIEW P0-1: Bijan Robinson and Brian Robinson Jr. are BOTH ATL RBs on the
    real Week 1 slate (short key b|robinson). When FP lacks the backup's row, the
    team-agreement branch handed him the star's projection at a reported 100% match
    rate. First-name compatibility is now unconditional."""
    from dfs.matching import match_slate
    fp = [_fpp("Bijan Robinson", "ATL", "RB", 18.4)]
    slate = [_sp("Bijan Robinson", "ATL", "RB", 8800, fd_id="bijan"),
             _sp("Brian Robinson Jr.", "ATL", "RB", 5200, fd_id="brian")]
    mapping, rep = match_slate(slate, fp)
    assert mapping["bijan"].name == "Bijan Robinson"
    assert "brian" not in mapping
    assert any(u[0] == "Brian Robinson Jr." for u in rep.unmatched)


def test_max_projection_lineup_fanduel_team_legality():
    """REVIEW P0-2: the entered lineup must satisfy FanDuel legality — at most 4
    players from one team (which over 9 slots forces >= 3 teams). Verified with an
    INDEPENDENT count, on the real slate, with projections adversarially stacked on
    one team; the old solver returned nine same-team players under this setup."""
    from collections import Counter
    slate, _ = _real_w1_slate()
    for p in slate.players:                       # make CIN irresistible
        if p.team == "CIN":
            p.projection *= 10
    mp = max_projection_lineup(slate, W1_SPEC)
    byid = {p.fd_id: p for p in slate.players}
    counts = Counter(byid[i].team for i in mp.player_ids)
    assert max(counts.values()) <= 4, counts
    assert len(counts) >= 3
    assert counts["CIN"] == 4                     # it should still max the cap out


def test_maxproj_swap_respects_team_legality():
    """The lock-constrained max-proj re-solve shares the same legality."""
    from collections import Counter
    from dfs.lateswap import propose_swap_maxproj
    from dfs.kickoffs import KickoffSchedule
    slate = _projected_slate()
    for p in slate.players:
        if p.team == "PHI":
            p.projection *= 10
    mp = max_projection_lineup(slate, SPEC)
    prop = propose_swap_maxproj(slate, mp.player_ids, SPEC, KickoffSchedule({}))
    byid = {p.fd_id: p for p in slate.players}
    counts = Counter(byid[i].team for i in prop.proposed_ids)
    assert max(counts.values()) <= 4


def test_blend_zero_projection_on_matched_star_is_critical():
    """REVIEW P0-3: a matched player whose FP payload scores to zero was silently
    dropped at a 100%% reported match rate. Nonpositive matched projections at or
    above the critical salary must stop the build like an unmatched star."""
    import pytest
    from dfs.blend import apply_projections
    slate, _ = _real_w1_slate()
    star = max(slate.players, key=lambda p: p.salary)
    fp = [_fpp(p.name, p.team, p.position, 0.0 if p.fd_id == star.fd_id
               else max(1.0, p.fppg or 3.0)) for p in slate.players]
    with pytest.raises(SlateError) as e:
        apply_projections(slate, fp, {}, DIST, min_match_rate=0.5,
                          critical_salary=7000)
    assert "roster-relevant" in str(e.value)


def test_swap_accept_updates_entry_and_archives_prior(tmp_path):
    """REVIEW P0-4: an accepted swap must update the logged entry (and archive the
    prior lineup) so later swap windows and Monday capture grade the REAL roster."""
    import json as _json
    from dfs import cli as cli_mod
    slate = _projected_slate()
    mp = max_projection_lineup(slate, SPEC)
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in mp.player_ids]
    db = tmp_path / "r.db"
    ResultLog(db).log_entry(2026, 1, "Leather League", players,
                            objective="arm=max-proj; t")
    # a proposal that swaps one player
    out_p = players[-1]
    repl = next(p for p in slate.players
                if p.fd_id not in mp.player_ids and p.position == out_p.position)
    new_players = players[:-1] + [repl]
    proposal = {
        "season": 2026, "week": 1, "contest": "Leather League",
        "created_utc": "2026-09-13T15:30:00+00:00",
        "criterion": "projected pts", "old": 120.0, "new": 121.0,
        "lineup": [{"fd_id": p.fd_id, "name": p.name, "pos": p.position,
                    "team": p.team, "salary": p.salary, "projection": p.projection,
                    "proj_source": "t", "implied_total": None, "mvp": False,
                    "inj": "", "inj_detail": ""} for p in new_players],
        "salary": sum(p.salary for p in new_players),
    }
    pf = tmp_path / "prop.json"
    pf.write_text(_json.dumps(proposal))
    rc = cli_mod.main(["swap-accept", "--season", "2026", "--week", "1",
                       "--contest", "Leather League", "--log-db", str(db),
                       "--proposal", str(pf)])
    assert rc == 0
    rl = ResultLog(db)
    with rl._c() as c:
        row = c.execute("SELECT lineup_json, objective FROM entries").fetchone()
        rev = c.execute("SELECT prior_lineup_json, note FROM entry_revisions").fetchone()
    ids = {p["fd_id"] for p in _json.loads(row["lineup_json"])}
    assert repl.fd_id in ids and out_p.fd_id not in ids
    assert "swap accepted" in row["objective"]
    prior = {p["fd_id"] for p in _json.loads(rev["prior_lineup_json"])}
    assert out_p.fd_id in prior
    # idempotent: re-accepting the same proposal is a no-op, not a second revision
    assert cli_mod.main(["swap-accept", "--season", "2026", "--week", "1",
                         "--contest", "Leather League", "--log-db", str(db),
                         "--proposal", str(pf)]) == 0
    with rl._c() as c:
        assert c.execute("SELECT COUNT(*) FROM entry_revisions").fetchone()[0] == 1


def test_showdown_cli_build_with_kicker_in_lineup(tmp_path, monkeypatch):
    """REVIEW P0-5: the real showdown slate carries kickers; a K in a displayed
    candidate crashed the print loop (position missing from the sort order). Full
    CLI build on the REAL single-game fixture, with a kicker forced projectable."""
    from dfs import cli as cli_mod
    from dfs.fantasypros import FPProjection
    sd_slate, _ = ingest_csv(REAL_SD, "sdcli", 2026, 1)

    def fake_projections(self, season, week, positions=None):
        import random as _r
        _r.seed(3)
        out = []
        for i, p in enumerate(sd_slate.players):
            pts = round(max(1.0, (p.fppg or 4.0) * _r.uniform(0.9, 1.1)), 2)
            if p.position == "K":
                pts = 40.0                        # force the kicker into the lineup
            out.append(FPProjection(player_id=f"fp{i}", name=p.name, team=p.team,
                                    position=p.position, points=pts,
                                    stats={}, breakdown={}))
        return out

    monkeypatch.setattr("dfs.fantasypros.FantasyProsClient.weekly_projections",
                        fake_projections)
    monkeypatch.setenv("FANTASYPROS_API_KEY", "test")
    rc = cli_mod.main([
        "build", "--csv", str(REAL_SD), "--season", "2026", "--week", "1",
        "--slate-id", "sdcli", "--profile", "showdown_friends",
        "--contest", "SD", "--field", "100", "--entry-fee", "5",
        "--no-vegas", "--no-injuries", "--no-snapshot",
        "--pool", "6", "--sims", "300", "--fields", "2", "--show", "2",
        "--log-db", str(tmp_path / "r.db"), "--out", str(tmp_path / "l.json"),
    ])
    assert rc == 0
    import json as _json
    entry = _json.loads((tmp_path / "l.json").read_text())
    rl = ResultLog(tmp_path / "r.db")
    with rl._c() as c:
        row = c.execute("SELECT lineup_json FROM entries WHERE contest='SD'").fetchone()
    lineup = _json.loads(row["lineup_json"])
    assert len(lineup) == 6
    assert any(p["pos"] == "K" for p in lineup)   # the kicker made the entry


def test_template_preserves_entry_and_contest_ids(tmp_path):
    """REVIEW P0-6: a real entries template carries entry_id/contest_id that tell
    FanDuel which entry to EDIT; blanking them turns an edit into a rejected or
    duplicate upload. Template values fill blanks; explicit args still win."""
    from dfs.export import export_upload_csv
    import csv as _csv
    tmpl = tmp_path / "tmpl.csv"
    tmpl.write_text("entry_id,contest_id,contest_name,QB,RB,RB,WR,WR,WR,TE,FLEX,DEF\n"
                    "E9001,C7777,Leather League W1,x,x,x,x,x,x,x,x,x\n")
    slate, _ = _real_w1_slate()
    mp = max_projection_lineup(slate, W1_SPEC)
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in mp.player_ids]
    out = tmp_path / "up.csv"
    export_upload_csv(players, out, slate_type=SlateType.FULL, template=tmpl)
    hdr, row = list(_csv.reader(out.open()))[:2]
    d = dict(zip(hdr, row))
    assert d["entry_id"] == "E9001" and d["contest_id"] == "C7777"
    assert d["contest_name"] == "Leather League W1"
    assert sum(1 for c, v in zip(hdr, row)
               if c not in ("entry_id", "contest_id", "contest_name") and v) == 9
    # explicit args beat template values
    export_upload_csv(players, out, slate_type=SlateType.FULL, template=tmpl,
                      entry_id="E-EXPLICIT")
    d2 = dict(zip(*list(_csv.reader(out.open()))[:2]))
    assert d2["entry_id"] == "E-EXPLICIT" and d2["contest_id"] == "C7777"


def test_single_dnp_with_incomplete_report_stays_questionable():
    """REVIEW hardening: one Wednesday DNP with Thu/Fri still blank is routine rest,
    not a removal signal; two recorded DNPs still escalate to DOUBTFUL."""
    from dfs.injuries import records_from_fantasypros, Status
    one = records_from_fantasypros([{"name": "A Vet", "team_id": "PHI",
                                     "status": "Questionable", "injury_type": "rest",
                                     "practice_1": "DNP"}])
    two = records_from_fantasypros([{"name": "B Hurt", "team_id": "PHI",
                                     "status": "Questionable", "injury_type": "knee",
                                     "practice_1": "DNP", "practice_2": "DNP"}])
    assert one["a vet"].status is Status.QUESTIONABLE
    assert two["b hurt"].status is Status.DOUBTFUL


def test_vegas_network_failure_becomes_vegas_error(monkeypatch):
    """REVIEW hardening: the build's soft-skip catches VegasError; a raw timeout or
    JSON error crashed the build instead. All transport failures must arrive as
    VegasError through the REAL _get path."""
    import urllib.request, urllib.error, pytest
    from dfs.vegas import OddsClient, VegasError
    oc = OddsClient(api_key="test")
    def boom(*a, **k):
        raise urllib.error.URLError("timed out")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(VegasError):
        oc.team_lines(slate_teams={"PHI"})


# ============ external review round 5 fixes (2026-08-30) ============
def test_swap_form_has_every_field_its_js_reads():
    """REVIEW #1: the swap-accept button read f.contest while the swap form had no
    contest field, so every click threw before reaching the API — the backend feature
    was unreachable. Assert the form supplies every field the handler reads."""
    import re
    html = (Path(__file__).parent.parent / "src" / "dfs" / "static" /
            "index.html").read_text()
    swap_form = html[html.index('id="f-swap"'):html.index('id="o-swap"')]
    names = set(re.findall(r'name="([a-z_]+)"', swap_form))
    handler = html[html.index("#swap-accept-btn"):]
    handler = handler[:handler.index("};")]
    for field in re.findall(r"f\.([a-z_]+)\.value", handler):
        assert field in names, f"handler reads f.{field} but the form has no such field"
    for field in re.findall(r"of \[([^\]]+)\]", handler):
        for k in re.findall(r"'([a-z_]+)'", field):
            assert k in names, f"handler appends '{k}' but the form has no such field"
    assert {"season", "week", "contest", "profile"} <= names


def test_injury_sweep_writes_status_onto_player():
    """REVIEW #3: sweep() flagged players but never wrote the merged record back, so
    a player flagged ONLY by the fresher FantasyPros feed carried no badge on the
    lineup card — the card showed the Wednesday-stale FanDuel CSV field."""
    from dfs.injuries import InjuryRecord, Status, sweep
    slate = _projected_slate()
    target = slate.players[0]
    target.injury_indicator = ""           # nothing from the FanDuel CSV
    target.injury_details = ""
    recs = {norm_name(target.name): InjuryRecord(
        name=target.name, team=target.team, status=Status.QUESTIONABLE,
        detail="hamstring · practice DNP/LP", source="fantasypros")}
    res = sweep(slate, recs)
    assert res.flagged and res.flagged[0][0].fd_id == target.fd_id
    assert target.injury_indicator                    # now badge-able
    assert "hamstring" in target.injury_details
    assert target.injury_source == "fantasypros" and target.injury_ts


def test_relevance_gate_is_position_relative():
    """REVIEW #5: a flat $7,000 cutoff ignores a promoted $5,600 WR starter (the
    entered lineup routinely rosters four players under $6,500) while treating a
    $6,000 third-string QB as critical. Gate on the top quartile within position."""
    import pytest
    from dfs.blend import apply_projections
    slate, _ = _real_w1_slate()
    wr = next(p for p in slate.players
              if p.position == "WR" and 5400 <= p.salary <= 6000)
    qb = next(p for p in slate.players
              if p.position == "QB" and p.salary <= 6000)
    # a mid-priced WR starter missing must STOP the build even though < $7000
    fp = [_fpp(p.name, p.team, p.position, max(1.0, p.fppg or 3.0))
          for p in slate.players if p.fd_id != wr.fd_id]
    with pytest.raises(SlateError) as e:
        apply_projections(slate, fp, {}, DIST, min_match_rate=0.5, critical_salary=7000)
    assert wr.name in str(e.value) and "top quartile" in str(e.value)
    # a cheap backup QB missing must NOT stop it
    slate2, _ = _real_w1_slate()
    fp2 = [_fpp(p.name, p.team, p.position, max(1.0, p.fppg or 3.0))
           for p in slate2.players if p.name != qb.name]
    rep = apply_projections(slate2, fp2, {}, DIST, min_match_rate=0.5,
                            critical_salary=7000)
    assert rep.rate > 0.9


def test_swap_proposal_rejected_when_entry_changed(tmp_path):
    """REVIEW #4: swap-accept validated only season/week/contest, so a proposal made
    before a rebuild (or before an earlier accept) would silently REVERT the entry to
    a stale roster."""
    import json as _json
    from dfs import cli as cli_mod
    from datetime import datetime, timezone
    slate = _projected_slate()
    mp = max_projection_lineup(slate, SPEC)
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in mp.player_ids]
    db = tmp_path / "r.db"
    ResultLog(db).log_entry(2026, 1, "Leather League", players, objective="arm=max-proj")

    def _pl(ps):
        return [{"fd_id": p.fd_id, "name": p.name, "pos": p.position, "team": p.team,
                 "salary": p.salary, "projection": p.projection, "proj_source": "t",
                 "implied_total": None, "mvp": False, "inj": "", "inj_detail": ""}
                for p in ps]

    repl = next(p for p in slate.players
                if p.fd_id not in mp.player_ids and p.position == players[-1].position)
    proposal = {"season": 2026, "week": 1, "contest": "Leather League",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "criterion": "projected pts", "old": 1.0, "new": 2.0,
                "lineup": _pl(players[:-1] + [repl]),
                "salary": sum(p.salary for p in players[:-1] + [repl]),
                "source_lineup_hash": cli_mod._lineup_hash(_pl(players))}
    pf = tmp_path / "p.json"
    pf.write_text(_json.dumps(proposal))

    # entry gets REBUILT to something else before the swap is accepted
    other = next(p for p in slate.players
                 if p.fd_id not in mp.player_ids and p.fd_id != repl.fd_id
                 and p.position == players[-2].position)
    ResultLog(db).log_entry(2026, 1, "Leather League",
                            players[:-2] + [other, players[-1]], objective="arm=max-proj")
    rc = cli_mod.main(["swap-accept", "--season", "2026", "--week", "1",
                       "--contest", "Leather League", "--log-db", str(db),
                       "--proposal", str(pf)])
    assert rc == 2                                    # refused, entry untouched
    with ResultLog(db)._c() as c:
        n = c.execute("SELECT COUNT(*) FROM entry_revisions").fetchone()[0]
    assert n == 0


def test_swap_proposal_rejected_when_stale(tmp_path):
    """A proposal older than the swap window is refused — projections and inactives
    have moved since it was computed."""
    import json as _json
    from dfs import cli as cli_mod
    slate = _projected_slate()
    mp = max_projection_lineup(slate, SPEC)
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in mp.player_ids]
    db = tmp_path / "r.db"
    ResultLog(db).log_entry(2026, 1, "Leather League", players, objective="arm=max-proj")
    lineup = [{"fd_id": p.fd_id, "name": p.name, "pos": p.position, "team": p.team,
               "salary": p.salary, "projection": p.projection, "proj_source": "t",
               "implied_total": None, "mvp": False, "inj": "", "inj_detail": ""}
              for p in players]
    pf = tmp_path / "p.json"
    pf.write_text(_json.dumps({
        "season": 2026, "week": 1, "contest": "Leather League",
        "created_utc": "2026-08-29T02:00:00+00:00",      # a day+ before now
        "criterion": "projected pts", "old": 1.0, "new": 2.0,
        "lineup": lineup, "salary": sum(p.salary for p in players),
        "source_lineup_hash": cli_mod._lineup_hash(lineup)}))
    assert cli_mod.main(["swap-accept", "--season", "2026", "--week", "1",
                         "--contest", "Leather League", "--log-db", str(db),
                         "--proposal", str(pf)]) == 2


def test_proposal_path_includes_contest():
    """Main-slate and showdown proposals for the same week must not collide."""
    from dfs.cli import _proposal_path
    a = _proposal_path(2026, 1, "Leather League")
    b = _proposal_path(2026, 1, "League Showdown")
    assert a != b and "leather-league" in a and "league-showdown" in b


def test_template_multiple_entries_selectable(tmp_path):
    """A real entries template can hold several entries; picking the first silently
    edits the wrong one. --entry-id selects, and the ambiguity is warned about."""
    from dfs.export import export_upload_csv
    import csv as _csv
    tmpl = tmp_path / "t.csv"
    tmpl.write_text("entry_id,contest_id,contest_name,QB,RB,RB,WR,WR,WR,TE,FLEX,DEF\n"
                    "E1,C1,Contest One,x,x,x,x,x,x,x,x,x\n"
                    "E2,C2,Contest Two,x,x,x,x,x,x,x,x,x\n")
    slate, _ = _real_w1_slate()
    mp = max_projection_lineup(slate, W1_SPEC)
    byid = {p.fd_id: p for p in slate.players}
    players = [byid[i] for i in mp.player_ids]
    out = tmp_path / "u.csv"
    ex = export_upload_csv(players, out, slate_type=SlateType.FULL, template=tmpl)
    assert any("2 entries" in w for w in ex.warnings)
    ex2 = export_upload_csv(players, out, slate_type=SlateType.FULL, template=tmpl,
                            entry_id="E2")
    d = dict(zip(*list(_csv.reader(out.open()))[:2]))
    assert d["entry_id"] == "E2" and d["contest_id"] == "C2"


def test_injuries_module_does_not_claim_an_official_inactives_layer():
    """REVIEW #2: the module documented a layer-3 'Sunday inactives sweep — official
    actives lists' that does not exist. Documentation that overstates protection is
    worse than none: it invites assuming a zero-point risk is handled."""
    import dfs.injuries as inj
    doc = inj.__doc__ or ""
    assert "NOT IMPLEMENTED" in doc
    assert "by hand" in doc or "verify" in doc.lower()
    assert hasattr(inj, "sweep")
