"""dfs build — CSV in, ranked lineups out.

    python3 -m dfs.cli build --csv <path> --season 2026 --week 1 [--no-vegas]
                             [--entry-fee 15] [--field 12] [--pool 120] [--sims 20000]

Every number in the output is traceable: projections come from FantasyPros stat lines
scored under FanDuel rules, floors/ceilings from empirical 2022-2025 distributions,
lineup metrics from the joint correlated simulation, EV from the contest payout curve.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

from .contest_spec import ContestSpec, Profile, SlateType
from .ingest_fanduel import ingest_csv
from .fantasypros import FantasyProsClient, FantasyProsError
from .vegas import OddsClient, VegasError
from .blend import apply_projections
from .distributions import load_distributions, build_distributions
from .optimize import generate_pool, StackRule, max_projection_lineup
from .simulate import SlateSimulator
from .field import (build_prior_field, build_field_ensemble, rank_candidates,
                    baseline_lineups)
from .objectives import Leaderboard, SeasonContext, weights_for
from .slate import SlateStore, SlateError
from .injuries import (records_from_fantasypros, records_from_slate, merge,
                       sweep as injury_sweep)
from .export import export_upload_csv, lineup_card, pushover_body
from .results import ResultLog
from .kickoffs import KickoffSchedule
from .nflcal import current_week
from .lateswap import propose_swap, propose_swap_maxproj

DATA = Path(__file__).resolve().parents[2] / "data"


def _dist():
    p = DATA / "distributions.json"
    if not p.exists():
        print("Building empirical distributions (one-time, ~60s)...")
        DATA.mkdir(exist_ok=True)
        build_distributions(out_path=p)
    d = load_distributions(p)
    method = d.get("meta", {}).get("method", "?")
    if "calibrated" not in str(d.get("meta", {}).get("source", "")) and "actual/FP" not in method:
        print("!" * 72)
        print(f"!! ACTIVE DISTRIBUTIONS ARE THE PROXY ({method}), NOT THE 2025 CALIBRATION.")
        print("!! Regenerate: python3 -m dfs.calibrate --rebuild-distributions "
              "data/calibration_2025.json --dist-out data/distributions.json — then COMMIT it.")
        print("!" * 72)
    else:
        print(f"distributions: {method} (n={d['meta'].get('n_obs')})")
    sample = next((v for pos in ("QB", "RB", "WR", "TE")
                   for v in (d.get(pos) or {}).values() if v), None)
    if sample is not None and "mean_ratio" not in sample:
        print("!" * 72)
        print("!! DISTRIBUTIONS PREDATE MEAN PINNING — simulated outcomes run ~14% HOT.")
        print("!! Every ceiling, P(win) and dollar figure below is inflated.")
        print("!! Rebuild: python3 -m dfs.calibrate --rebuild-distributions "
              "data/calibration_2025.json --dist-out data/distributions.json")
        print("!" * 72)
    return d


def cmd_build(a) -> int:
    dist = _dist()

    print("=" * 72)
    slate, irep = ingest_csv(a.csv, a.slate_id, a.season, a.week, strict=False)
    print(irep.summary())
    if irep.validation_problems:
        print("\n!! validation problems above — continuing, review before entering lineups")

    print("\n" + "=" * 72)
    print("Pulling FantasyPros projections...")
    fp_client = FantasyProsClient()
    fp = fp_client.weekly_projections(a.season, a.week)
    print(f"  {len(fp)} players projected (FanDuel-scored from stat lines)")
    if not a.no_snapshot:
        from .snapshots import write_snapshot
        snap = write_snapshot(a.snapshot_dir, a.season, a.week, a.slate_id or "",
                              getattr(fp_client, "last_raw", {}) or {})
        if snap:
            print(f"  at-lock snapshot -> {snap}")

    team_lines = {}
    if not a.no_vegas:
        try:
            oc = OddsClient()
            team_lines = oc.team_lines(slate_teams={p.team for p in slate.players})
            print(f"  Vegas: {len(team_lines)} team totals "
                  f"(quota remaining {oc.last_quota.get('remaining','?')})")
            if oc.missing_teams:
                print(f"  Vegas missing for {oc.missing_teams} (kicked off or off-board) — "
                      "those teams build without a Vegas tilt")
            hot = sorted(team_lines.values(), key=lambda t: -t.implied_total)[:3]
            print("  highest implied totals: " +
                  ", ".join(f"{t.team} {t.implied_total}" for t in hot))
        except VegasError as e:
            print(f"  Vegas SKIPPED: {e}")
    else:
        print("  Vegas skipped (--no-vegas)")

    mrep = apply_projections(slate, fp, team_lines, dist,
                             min_match_rate=a.min_match, critical_salary=a.critical_salary)
    print("\n" + mrep.summary(top_n=8))
    print(f"\nOptimizable pool: {len(slate.players)} players")

    top = sorted(slate.players, key=lambda p: -p.projection)[:8]
    print("\nTop projections:")
    for p in top:
        print(f"  {p.projection:6.2f}  {p.position:3s} {p.team:3s} ${p.salary:5d} "
              f"[{p.floor_p10:5.1f} - {p.ceiling_p90:5.1f}]  {p.name}")

    # ---- injury / inactives sweep ----
    # A player who is inactive scores 0. Under Total Scores that zero is permanent
    # damage to the season standing, so this runs before any lineup is built.
    if not a.no_injuries:
        try:
            fp_inj = records_from_fantasypros(FantasyProsClient().injuries(a.season))
        except FantasyProsError as e:
            fp_inj = {}
            print(f"\n  injury feed unavailable: {e}")
        inj = merge(records_from_slate(slate), fp_inj)
        sw = injury_sweep(slate, inj)
        print("\n" + sw.summary())
        if sw.flagged and a.strict_injuries:
            raise SlateError("questionable players in pool and --strict-injuries set; "
                             "resolve before building")

    spec = ContestSpec(name=a.contest, profile=Profile(a.profile),
                       slate_type=slate.slate_type, field_size=a.field,
                       entry_fee=a.entry_fee, late_swap=not a.no_late_swap)

    print("\n" + "=" * 72)
    print(f"Generating candidate pool (target {a.pool})...")
    pool = generate_pool(slate, spec, n=a.pool,
                         stack=StackRule(require_qb_stack=a.qb_stack,
                                         require_bring_back=a.bring_back),
                         min_unique=a.min_unique)
    print(f"  {len(pool)} distinct lineups; projection range "
          f"{min(c.proj_sum for c in pool):.1f} - {max(c.proj_sum for c in pool):.1f}")

    print(f"Simulating {a.sims} correlated slate outcomes...")
    sim = SlateSimulator(slate, dist, n_sims=a.sims, seed=a.seed)
    measured, own_weeks = None, 0
    if a.log_db and Path(a.log_db).exists() and a.profile in ("friends_league",
                                                               "showdown_friends"):
        rl_own = ResultLog(a.log_db)
        measured = rl_own.measured_ownership(a.season, contest_like=a.contest) or None
        if measured:
            own_weeks = rl_own.ownership_week_count(a.season, contest_like=a.contest)
    field = build_field_ensemble(slate, spec, n_opponents=spec.field_size - 1,
                                 n_fields=a.fields, seed=a.seed + 1,
                                 measured_ownership=measured,
                                 ownership_weeks=own_weeks)
    print(f"  field ensemble: {a.fields} draws x {spec.field_size - 1} opponents "
          f"({field[0].source})")
    if measured:
        print(f"  league ownership: {len(measured)} players from {own_weeks} captured "
              f"week(s), shrinkage {own_weeks/(own_weeks+4):.0%} toward measured")

    ctx = None
    if a.auto_context and a.log_db and Path(a.log_db).exists():
        # Scoped to THIS contest's history: the league objective must be driven by
        # league results only, never by public/side contests sharing the database.
        ctx = ResultLog(a.log_db).season_context(a.season, me=a.me,
                                                 weeks_total=a.weeks_total,
                                                 contest_like=a.contest)
        ctx.weekly_prize = a.weekly_prize
        ctx.grand_prizes = tuple(float(x) for x in a.grand_prizes.split(",") if x)
        ctx.field_size = spec.field_size
        print(f"\nSeason context (from log, me={a.me}): week {ctx.weeks_played + 1}, "
              f"{ctx.my_points:.1f} pts, {ctx.deficit():+.1f} vs leader")
    if ctx is None:
        ctx = SeasonContext(leaderboard=Leaderboard(a.leaderboard),
                            weeks_total=a.weeks_total, weeks_played=a.weeks_played,
                            my_points=a.my_points, leader_points=a.leader_points,
                            my_wins=a.my_wins, leader_wins=a.leader_wins,
                            field_size=spec.field_size, weekly_prize=a.weekly_prize,
                            grand_prizes=tuple(float(x) for x in a.grand_prizes.split(",") if x))
    weights = weights_for(a.profile, ctx, entry_fee=a.entry_fee, prize_pool=a.prize_pool)
    print("\n" + "=" * 72)
    print("OBJECTIVE")
    print(f"  {weights.rationale}")
    print(f"  weights: ${weights.w_points:.2f} per expected point | "
          f"${weights.w_win:.2f} per unit win probability")

    ranked = rank_candidates(pool, sim, field, spec, weights)
    byid = {p.fd_id: p for p in slate.players}

    # ---- honest baselines: a win probability alone is uninterpretable ----
    n_opp = len(field[0].lineups)
    naive = 1.0 / (n_opp + 1)

    def _obj(ids, mvp=None):
        """Objective value via the SAME ensemble path used for ranking.

        Scoring a baseline against one field draw while ranking candidates against a
        25-field ensemble compares two different measurements and can invert the sign
        of the reported edge.
        """
        from .optimize import Candidate
        cand = Candidate(player_ids=tuple(ids), salary=0, proj_sum=0.0, mvp_id=mvp)
        return rank_candidates([cand], sim, field, spec, weights)[0]

    bl = baseline_lineups(slate, spec, n=200, seed=a.seed + 2)
    rand = [_obj(l) for l in bl["random"]]
    # TRUE max-projection: its own legality-only solve. Taking max() over the
    # candidate pool inherits the pool's stack/diversity constraints and measured
    # 1.96 projected points short of the real optimum on the test fixture — an
    # invalid yardstick that flips the sign of small edges.
    max_proj = max_projection_lineup(slate, spec)
    pool_best_proj = max(c.proj_sum for c in pool)
    if max_proj.proj_sum > pool_best_proj + 1e-6:
        print(f"  (pool's best projection {pool_best_proj:.2f} vs unconstrained "
              f"{max_proj.proj_sum:.2f} — constraints cost "
              f"{max_proj.proj_sum - pool_best_proj:.2f} pts)")
    mp = _obj(max_proj.player_ids, max_proj.mvp_id)
    best = ranked[0]

    ar_d = sum(r.dollars for r in rand) / len(rand)
    ar_p = sum(r.p_win for r in rand) / len(rand)
    ar_e = sum(r.exp_points for r in rand) / len(rand)
    print("\n" + "=" * 72)
    print(f"BASELINES (vs the same {n_opp}-opponent simulated field)")
    print(f"  {'':32s} {'$obj':>8s} {'E[pts]':>7s} {'P(win)':>7s}")
    print(f"  coin flip among {n_opp + 1:<16d} {'-':>8s} {'-':>7s} {naive:7.1%}")
    print(f"  random valid lineup (n={len(rand):<3d})       {ar_d:8.2f} {ar_e:7.1f} {ar_p:7.1%}")
    print(f"  max-projection (unconstrained)   {mp.dollars:8.2f} {mp.exp_points:7.1f} {mp.p_win:7.1%}")
    print(f"  our #1 lineup                    {best.dollars:8.2f} {best.exp_points:7.1f} {best.p_win:7.1%}")
    edge = best.dollars - mp.dollars
    print(f"  delta vs max-projection          {edge:+8.2f}")
    print("  NOTE: selection and this comparison share simulation paths. See the")
    print("        INDEPENDENT EVALUATION below for the number to actually trust.")

    print("\n" + "=" * 72)
    print(f"TOP {a.show} LINEUPS — ranked by objective vs {spec.field_size - 1}-opponent field")
    print("=" * 72)
    for i, r in enumerate(ranked[:a.show], 1):
        c = r.candidate
        print(f"\n#{i}  {r.summary()}")
        for pid in sorted(c.player_ids, key=lambda x: ("QB RB WR TE D".split().index(byid[x].position),
                                                       -byid[x].salary)):
            p = byid[pid]
            mvp = " (MVP)" if pid == c.mvp_id else ""
            print(f"     {p.position:3s} {p.name:24s} {p.team:3s} ${p.salary:5d} "
                  f"{p.projection:5.1f}{mvp}")

    best = ranked[0]
    best_players = [byid[i] for i in best.candidate.player_ids]

    # ---- independent evaluation ----
    # Selection and evaluation on the same simulation paths is self-grading: the argmax
    # of noisy estimates is biased high. Fresh paths remove that Monte Carlo winner's
    # curse. Two requirements the first version got wrong:
    #   (1) IDENTICAL CONDITIONING. If selection used measured ownership, evaluation
    #       must too; silently dropping it changes the model, not just the seed, and
    #       makes the two numbers incomparable.
    #   (2) EVALUATE THE BASELINES on the same fresh paths. Comparing a fresh estimate
    #       of our lineup against a stale estimate of max-projection is not a delta.
    # This still only corrects simulation noise. It does NOT validate the model or
    # demonstrate predictive edge.
    eval_sim = SlateSimulator(slate, dist, n_sims=a.sims, seed=a.seed + 777)
    eval_fields = build_field_ensemble(slate, spec, n_opponents=spec.field_size - 1,
                                       n_fields=a.fields, seed=a.seed + 7777,
                                       measured_ownership=measured,
                                       ownership_weeks=own_weeks)
    ev_rows = rank_candidates([best.candidate, max_proj], eval_sim, eval_fields,
                              spec, weights)
    ev_best = next(r for r in ev_rows if r.candidate.key == best.candidate.key)
    ev_mp = next(r for r in ev_rows if r.candidate.key == max_proj.key)
    print("\nINDEPENDENT EVALUATION (fresh sims + fresh fields, same conditioning)")
    print(f"  {'':22s} {'$obj':>8s} {'E[pts]':>7s} {'P(win)':>8s}")
    print(f"  max-projection        {ev_mp.dollars:8.2f} {ev_mp.exp_points:7.1f} "
          f"{ev_mp.p_win:7.1%} (field spread ±{ev_mp.mean_rank:.1%})")
    print(f"  our #1                {ev_best.dollars:8.2f} {ev_best.exp_points:7.1f} "
          f"{ev_best.p_win:7.1%} (field spread ±{ev_best.mean_rank:.1%})")
    fresh_edge = ev_best.dollars - ev_mp.dollars
    sel_edge = best.dollars - mp.dollars
    print(f"  fresh edge vs max-proj {fresh_edge:+8.2f}   "
          f"(selection claimed {sel_edge:+.2f}; optimism {sel_edge - fresh_edge:+.2f})")
    if fresh_edge <= 0.02:
        print("  VERDICT: no edge demonstrated over max-projection on this slate.")
        print("           The lineup is defensible; the claim of an edge is not.")
    else:
        print(f"  VERDICT: {fresh_edge:+.2f}/week vs max-projection on fresh paths. Still")
        print("           model-graded — not out-of-sample validation.")
    print(f"  ownership conditioning: "
          f"{'measured (' + str(own_weeks) + 'wk)' if measured else 'generic prior'} "
          "— identical in selection and evaluation")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slate_id": a.slate_id, "season": a.season, "week": a.week,
        "contest": {"name": spec.name, "profile": spec.profile.value,
                    "field_size": spec.field_size, "entry_fee": spec.entry_fee},
        "match_rate": mrep.rate, "seed": a.seed, "sims": a.sims,
        "objective": {"leaderboard": a.leaderboard, "w_points": weights.w_points,
                      "w_win": weights.w_win, "rationale": weights.rationale},
        "lineups": [{
            "rank": i, "p_win": r.p_win, "p_top3": r.p_top3, "dollars": r.dollars,
            "exp_points": r.exp_points, "median": r.median, "p10": r.p10, "p90": r.p90,
            "salary": r.candidate.salary, "projection": r.candidate.proj_sum,
            "mvp_id": r.candidate.mvp_id,
            "players": [{"fd_id": pid, "name": byid[pid].name, "pos": byid[pid].position,
                         "team": byid[pid].team, "salary": byid[pid].salary,
                         "projection": byid[pid].projection,
                         "source": byid[pid].proj_source}
                        for pid in r.candidate.player_ids],
        } for i, r in enumerate(ranked[:a.show], 1)],
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nSaved: {out}")

    # ---- entry-arm selection ----
    # Agreed strategy (review round 3): the LEAGUE entry is the true max-projection
    # lineup; the simulator's pick rides along as a SHADOW so the two-arm record
    # accumulates every week. The sim's win-probability is not yet trusted enough to
    # pay projection points for (its claimed edge was measured against an invalid
    # baseline). --entry model flips this for contests where the sim is load-bearing.
    entry_arm = a.entry
    if entry_arm == "auto":
        entry_arm = ("max-proj" if spec.profile == Profile.FRIENDS_LEAGUE else "model")
    if entry_arm == "max-proj":
        entry_eval, shadow_eval, shadow_arm = ev_mp, ev_best, "model"
    else:
        entry_eval, shadow_eval, shadow_arm = ev_best, ev_mp, "max-proj"
    entry_cand = entry_eval.candidate
    entry_players = [byid[i] for i in entry_cand.player_ids]
    shadow_players = [byid[i] for i in shadow_eval.candidate.player_ids]
    print(f"\nENTRY ARM: {entry_arm}  (proj {entry_cand.proj_sum:.1f}, "
          f"E[pts] {entry_eval.exp_points:.1f}, P(win) {entry_eval.p_win:.0%})  "
          f"[shadow: {shadow_arm}]")

    # ---- artifacts: upload CSV, entry log, phone card ----
    # This block IS the operating loop. `swap` reads the logged entry; the upload CSV
    # is what actually gets entered. (A block-replace edit deleted this once — the
    # end-to-end test in tests/test_core.py now asserts these artifacts exist.)
    if a.export:
        ex = export_upload_csv(entry_players, a.export, slate_type=slate.slate_type,
                               mvp_id=entry_cand.mvp_id, template=a.template,
                               contest_name=spec.name)
        print("\n" + ex.summary())
    if a.log_db:
        rl_out = ResultLog(a.log_db)
        rl_out.log_entry(a.season, a.week, spec.name, entry_players,
                         slate_id=a.slate_id or "",
                         objective=f"arm={entry_arm}; {weights.rationale}",
                         exp_points=entry_eval.exp_points, p_win=entry_eval.p_win,
                         mvp_id=entry_cand.mvp_id,
                         mvp_salary_mult=spec.mvp_salary_mult)
        # Shadow arm: logged under a suffixed contest name so it can never collide
        # with the real entry (PK is season/week/contest) and never receives money
        # results — capture only grades the real contest name. projection_accuracy
        # and the two-arm comparison read it by the [shadow:...] suffix.
        rl_out.log_entry(a.season, a.week, f"{spec.name} [shadow:{shadow_arm}]",
                         shadow_players, slate_id=a.slate_id or "",
                         objective=f"arm={shadow_arm}; {weights.rationale}",
                         exp_points=shadow_eval.exp_points, p_win=shadow_eval.p_win,
                         mvp_id=shadow_eval.candidate.mvp_id,
                         mvp_salary_mult=spec.mvp_salary_mult)
        print(f"Entry logged ({entry_arm}) + shadow ({shadow_arm}) -> {a.log_db}")
    if a.pushover_out:
        metrics = (f"{spec.name} w{a.week} [{entry_arm}]: proj {entry_cand.proj_sum:.1f}, "
                   f"E[pts] {entry_eval.exp_points:.1f}, P(win) {entry_eval.p_win:.0%}")
        po = Path(a.pushover_out)
        po.parent.mkdir(parents=True, exist_ok=True)
        po.write_text(pushover_body(entry_players, metrics,
                                    mvp_id=entry_cand.mvp_id))
        print(f"Pushover card -> {po}")

    if a.db:
        SlateStore(a.db).save(slate)
        print(f"Slate persisted: {a.db}")
    return 0


def cmd_swap(a) -> int:
    """Sunday workflow: fresh injuries + fresh Vegas + lock-aware re-optimization of the
    logged lineup. Run at 11:30 ET, again before the 4:00 window, and before SNF."""
    import json as _json
    dist = _dist()
    rl = ResultLog(a.log_db)
    with rl._c() as conn:
        row = conn.execute("""SELECT lineup_json, objective FROM entries
                              WHERE season=? AND week=? AND contest=?""",
                           (a.season, a.week, a.contest)).fetchone()
    if not row:
        print(f"No logged entry for {a.contest} {a.season} w{a.week}. "
              "Run `build --log-db` first.")
        return 2
    logged_lineup = _json.loads(row["lineup_json"])
    current_ids = tuple(p["fd_id"] for p in logged_lineup)
    current_mvp = next((p["fd_id"] for p in logged_lineup if p.get("mvp")), None)
    # The arm the entry was BUILT with decides the swap criterion. A max-proj entry
    # must never be churned Sunday morning by the model objective it was chosen over
    # (first live drill: the model proposed trading 1.0 projected points for +$0.13
    # of simulator objective on a max-proj entry).
    entry_arm = "model"
    if (row["objective"] or "").startswith("arm=max-proj"):
        entry_arm = "max-proj"

    slate, irep = ingest_csv(a.csv, a.slate_id, a.season, a.week, strict=False)
    print(irep.summary())
    fp = FantasyProsClient().weekly_projections(a.season, a.week)

    team_lines = {}
    try:
        oc = OddsClient()
        team_lines = oc.team_lines(slate_teams={p.team for p in slate.players})
        print(f"Vegas refreshed: {len(team_lines)} team totals")
    except VegasError as e:
        print(f"Vegas skipped: {e}")

    # fresh injuries BEFORE matching, so removed players leave the pool
    try:
        fp_inj = records_from_fantasypros(FantasyProsClient().injuries(a.season))
    except FantasyProsError:
        fp_inj = {}
    inj = merge(records_from_slate(slate), fp_inj)
    lineup_set = set(current_ids)
    sw = injury_sweep(slate, inj, lineup_ids=lineup_set)
    print("\n" + sw.summary())

    mrep = apply_projections(slate, fp, team_lines, dist,
                             min_match_rate=a.min_match,
                             critical_salary=a.critical_salary)
    print(f"Projections refreshed: {mrep.matched}/{mrep.total}")

    try:
        sched = KickoffSchedule.from_nflverse(a.season, a.week)
    except Exception as e:
        if team_lines:
            print(f"nflverse schedule unavailable ({e}); falling back to Odds kickoff times")
            sched = KickoffSchedule.from_team_lines(team_lines)
        else:
            raise
    print("\n" + sched.summary())

    spec = ContestSpec(name=a.contest, profile=Profile(a.profile),
                       slate_type=slate.slate_type, field_size=a.field,
                       entry_fee=a.entry_fee)
    ctx = SeasonContext(leaderboard=Leaderboard(a.leaderboard),
                        weeks_total=a.weeks_total, field_size=a.field,
                        weekly_prize=a.weekly_prize,
                        grand_prizes=tuple(float(x) for x in a.grand_prizes.split(",") if x))
    weights = weights_for(a.profile, ctx, entry_fee=a.entry_fee, prize_pool=a.prize_pool)

    sim = SlateSimulator(slate, dist, n_sims=a.sims, seed=a.seed)
    fieldm = build_prior_field(slate, spec, n_opponents=spec.field_size - 1,
                               seed=a.seed + 1)
    reason = ("inactives affected the lineup" if sw.lineup_affected
              else "scheduled late-swap check")
    if entry_arm == "max-proj":
        print("\nSwap criterion: PROJECTED POINTS (entry arm: max-proj)")
        prop = propose_swap_maxproj(slate, current_ids, spec, sched, reason=reason)
    else:
        print("\nSwap criterion: model objective (entry arm: model)")
        prop = propose_swap(slate, current_ids, spec, sched, sim, fieldm, weights,
                            reason=reason, mvp_id=current_mvp)
    byid = {p.fd_id: p for p in slate.players}
    print("\n" + "=" * 72)
    print(prop.summary(byid))
    if prop.improves and a.export:
        best_players = [byid[i] for i in prop.proposed_ids]
        ex = export_upload_csv(best_players, a.export, slate_type=slate.slate_type,
                               mvp_id=prop.proposed_mvp, contest_name=spec.name)
        print("\n" + ex.summary())
    return 0


def cmd_capture(a) -> int:
    import json as _json
    from .contest_parse import parse_contest
    from .matching import norm_name
    capture = parse_contest(a.path, a.season, a.week, a.contest)
    print(capture.summary())
    rl = ResultLog(a.log_db)
    rl.log_capture(capture)
    print(f"\nLogged {len(capture.leaderboard)} entrants and "
          f"{len(capture.ownership())} ownership records -> {a.log_db}")

    # ---- close the loop on MY entry: outcome + per-player actuals ----
    # Without this, entries.actual_score and player_results.actual stay NULL and
    # projection_accuracy() is never fed — the advertised workflow must feed it.
    mine = next((e for e in capture.leaderboard if e.entrant == a.me), None)
    if mine is not None and mine.score is not None:
        rl.log_outcome(a.season, a.week, a.contest, mine.score, mine.rank or 0,
                       len(capture.leaderboard), winnings=mine.won or 0.0)
        print(f"Outcome recorded for {a.me}: {mine.score:.2f} pts, "
              f"rank {mine.rank} of {len(capture.leaderboard)}")
    else:
        print(f"NOTE: entrant '{a.me}' not on this page — outcome not recorded "
              "(pass --me if your FanDuel handle differs).")

    lu = next((e for e in capture.entries_with_lineups if e.entrant == a.me), None)
    if lu and lu.players:
        with rl._c() as conn:
            row = conn.execute("""SELECT lineup_json FROM entries
                                  WHERE season=? AND week=? AND contest=?""",
                               (a.season, a.week, a.contest)).fetchone()
        if row:
            by_name = {norm_name(p.name): p.actual_points for p in lu.players
                       if p.actual_points is not None}
            actuals = {p["fd_id"]: by_name[norm_name(p["name"])]
                       for p in _json.loads(row["lineup_json"])
                       if norm_name(p["name"]) in by_name}
            n = rl.log_player_actuals(a.season, a.week, actuals)
            print(f"Player actuals recorded: {n} (feeds projection_accuracy)")
        else:
            print("NOTE: no logged entry to attach player actuals to.")
    return 0


def cmd_standings(a) -> int:
    rl = ResultLog(a.log_db)
    st = rl.standings(a.season, contest_like=a.contest)
    if not st:
        print("No logged results yet. Run `capture` on a contest results page first.")
        return 0
    print(f"SEASON STANDINGS — {a.season} (Total Points)")
    print(f"  {'#':>2s} {'entrant':18s} {'total':>8s} {'avg':>7s} {'wks':>4s} {'wins':>5s} {'best':>7s}")
    for i, s in enumerate(st, 1):
        me = " <-- you" if s.entrant == a.me else ""
        print(f"  {i:2d} {s.entrant:18s} {s.total_points:8.1f} {s.avg:7.1f} "
              f"{s.weeks:4d} {s.wins:5d} {s.best:7.1f}{me}")
    ctx = rl.season_context(a.season, a.me, a.weeks_total, contest_like=a.contest)
    w = weights_for("friends_league", ctx)
    print(f"\nOBJECTIVE for next build (week {ctx.weeks_played + 1} of {ctx.weeks_total}):")
    print(f"  {w.rationale}")
    print(f"  behind leader by {ctx.deficit():.1f} points with {ctx.weeks_left} weeks left")
    print(f"  weights: ${w.w_points:.2f}/pt | ${w.w_win:.2f}/win-prob")
    acc = rl.projection_accuracy(a.season)
    if acc:
        print("\nIN-SEASON PROJECTION ACCURACY:")
        for pos, v in acc.items():
            print(f"  {pos:4s} n={v['n']:4d} MAE={v['mae']:5.2f} bias={v['bias']:+5.2f} "
                  f"corr={v['corr']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dfs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build ranked lineups from a FanDuel slate CSV")
    b.add_argument("--csv", required=True)
    b.add_argument("--season", type=int, default=None,
                   help="default: current NFL season (auto-detected)")
    b.add_argument("--week", type=int, default=None,
                   help="default: current NFL week (auto-detected)")
    b.add_argument("--slate-id", default=None)
    b.add_argument("--contest", default="Leather League")
    b.add_argument("--profile", default="friends_league",
                   choices=[p.value for p in Profile])
    b.add_argument("--field", type=int, default=12)
    b.add_argument("--entry-fee", type=float, default=15.0)
    b.add_argument("--pool", type=int, default=120)
    b.add_argument("--sims", type=int, default=20000)
    b.add_argument("--auto-context", action="store_true",
                   help="derive season standing (weeks played, deficit) from the result log")
    b.add_argument("--me", default="brettleath")
    b.add_argument("--fields", type=int, default=25,
                   help="field draws in the ensemble; P(win) is averaged across them")
    b.add_argument("--seed", type=int, default=1729)
    b.add_argument("--show", type=int, default=3)
    b.add_argument("--min-unique", type=int, default=3)
    b.add_argument("--qb-stack", type=int, default=1)
    b.add_argument("--bring-back", type=int, default=0)
    b.add_argument("--min-match", type=float, default=0.60)
    b.add_argument("--critical-salary", type=int, default=6500)
    b.add_argument("--no-vegas", action="store_true")
    b.add_argument("--no-late-swap", action="store_true")
    b.add_argument("--leaderboard", default="total_scores",
                   choices=[l.value for l in Leaderboard],
                   help="season standing format — drives the objective weights")
    b.add_argument("--weeks-total", type=int, default=21)
    b.add_argument("--weeks-played", type=int, default=0)
    b.add_argument("--my-points", type=float, default=0.0)
    b.add_argument("--leader-points", type=float, default=0.0)
    b.add_argument("--my-wins", type=int, default=0)
    b.add_argument("--leader-wins", type=int, default=0)
    b.add_argument("--weekly-prize", type=float, default=12.84)
    b.add_argument("--grand-prizes", default="135,81,54")
    b.add_argument("--prize-pool", type=float, default=None,
                   help="one-off contests: total prize pool (h2h/showdown/gpp)")
    b.add_argument("--no-injuries", action="store_true",
                   help="skip the inactives sweep (testing only)")
    b.add_argument("--strict-injuries", action="store_true",
                   help="refuse to build while questionable players remain in the pool")
    b.add_argument("--export", default=None, help="write a FanDuel upload CSV here")
    b.add_argument("--template", default=None,
                   help="a real FanDuel entries template to mirror column headers from")
    b.add_argument("--log-db", default=None, help="SQLite result log to record this entry in")
    b.add_argument("--pushover-out", default=None, help="write a phone-sized lineup here")
    b.add_argument("--snapshot-dir", default="data/snapshots",
                   help="where at-lock raw projection snapshots are written")
    b.add_argument("--no-snapshot", action="store_true",
                   help="skip the at-lock snapshot (tests/offline)")
    b.add_argument("--entry", choices=["auto", "max-proj", "model"], default="auto",
                   help="which arm gets exported/logged as THE entry. auto: max-proj "
                        "for the friends league, model otherwise. The other arm is "
                        "always logged as a shadow row.")
    b.add_argument("--out", default="data/lineups/latest.json")
    b.add_argument("--db", default=None)
    b.set_defaults(func=cmd_build)

    cap = sub.add_parser("capture", help="ingest a pasted FanDuel contest results page")
    cap.add_argument("path")
    cap.add_argument("--season", type=int, required=True)
    cap.add_argument("--week", type=int, required=True)
    cap.add_argument("--contest", default="Leather League")
    cap.add_argument("--me", default="brettleath",
                     help="your FanDuel handle on the results page")
    cap.add_argument("--log-db", default="data/results.db")
    cap.set_defaults(func=cmd_capture)

    st = sub.add_parser("standings", help="season standings + objective weights from the log")
    st.add_argument("--season", type=int, required=True)
    st.add_argument("--me", default="brettleath")
    st.add_argument("--weeks-total", type=int, default=21)
    st.add_argument("--contest", default="Leather League",
                    help="scope standings to this contest (league by default)")
    st.add_argument("--log-db", default="data/results.db")
    st.set_defaults(func=cmd_standings)

    sw = sub.add_parser("swap", help="Sunday finalize: lock-aware late-swap check of the logged lineup")
    sw.add_argument("--csv", required=True)
    sw.add_argument("--season", type=int, default=None,
                   help="default: current NFL season (auto-detected)")
    sw.add_argument("--week", type=int, default=None,
                   help="default: current NFL week (auto-detected)")
    sw.add_argument("--slate-id", default=None)
    sw.add_argument("--contest", default="Leather League")
    sw.add_argument("--profile", default="friends_league", choices=[p.value for p in Profile])
    sw.add_argument("--leaderboard", default="total_scores", choices=[l.value for l in Leaderboard])
    sw.add_argument("--field", type=int, default=12)
    sw.add_argument("--entry-fee", type=float, default=0.0)
    sw.add_argument("--weekly-prize", type=float, default=12.84)
    sw.add_argument("--grand-prizes", default="135,81,54")
    sw.add_argument("--prize-pool", type=float, default=None)
    sw.add_argument("--weeks-total", type=int, default=21)
    sw.add_argument("--sims", type=int, default=15000)
    sw.add_argument("--seed", type=int, default=1729)
    sw.add_argument("--min-match", type=float, default=0.60)
    sw.add_argument("--critical-salary", type=int, default=6500)
    sw.add_argument("--log-db", default="data/results.db")
    sw.add_argument("--export", default=None)
    sw.set_defaults(func=cmd_swap)

    a = ap.parse_args(argv)
    # Season/week default to the live NFL calendar; an explicit flag always wins so
    # Brett can build or re-check any week for testing or review.
    if hasattr(a, "week"):
        if getattr(a, "season", None) is None or getattr(a, "week", None) is None:
            wi = current_week(season_hint=getattr(a, "season", None))
            if getattr(a, "season", None) is None:
                a.season = wi.season
            if getattr(a, "week", None) is None:
                a.week = wi.week
            print(f"Calendar: {wi.summary()}")
    if getattr(a, "slate_id", "SKIP") is None and hasattr(a, "week"):
        a.slate_id = f"{a.season}-w{a.week:02d}"
    try:
        return a.func(a)
    except (SlateError, FantasyProsError, VegasError) as e:
        # stdout as well as stderr: the web dashboard streams stdout, and an error the
        # user cannot see is indistinguishable from a hang.
        print(f"\nBUILD STOPPED: {e}")
        print(f"\nBUILD STOPPED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
