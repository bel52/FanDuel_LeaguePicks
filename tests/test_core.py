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
    slate = _projected_slate()
    pool = generate_pool(slate, SPEC, n=8, min_unique=3)
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            overlap = len(set(pool[i].player_ids) & set(pool[j].player_ids))
            assert overlap <= ROSTER_FULL_SIZE - 3


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
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=20000, seed=1729)
    g = lambda pos, team: next(p for p in slate.players if p.position == pos and p.team == team)
    c = lambda a, b: np.corrcoef(sim.matrix[sim.index[a.fd_id]], sim.matrix[sim.index[b.fd_id]])[0, 1]
    qb, wr = g("QB", "PHI"), g("WR", "PHI")
    assert 0.25 < c(qb, wr) < 0.55                      # same-team stack
    assert abs(c(qb, g("WR", "KC"))) < 0.05             # different game ~ independent
    assert c(g("D", "PHI"), g("WR", "DAL")) < -0.10     # DEF vs opposing offense
    assert 0.0 < c(qb, g("WR", "DAL")) < 0.20           # bring-back


def test_lineup_p90_below_summed_marginals():
    """The v5 bug: summing player p90s massively overstates lineup ceiling."""
    slate = _projected_slate()
    sim = SlateSimulator(slate, DIST, n_sims=10000, seed=1729)
    pool = generate_pool(slate, SPEC, n=3, min_unique=3)
    byid = {p.fd_id: p for p in slate.players}
    c = pool[0]
    joint = sim.score(c.player_ids).pct(90)
    summed = sum(byid[i].ceiling_p90 for i in c.player_ids)
    assert joint < summed * 0.9


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
    # w_points is $ per point THIS WEEK: ~$3.44 sustained / 21 weeks remaining.
    # Asserting the per-week magnitude guards against re-introducing the units bug
    # that inflated the reported edge ~21x.
    assert 0.05 < w.w_points < 0.50
    ctx_late = SeasonContext(leaderboard=Leaderboard.TOTAL_SCORES, weeks_total=21,
                             weeks_played=18)
    w_late = weights_for("friends_league", ctx_late)
    assert w_late.w_points > w.w_points, "each point matters more as weeks run out"


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
