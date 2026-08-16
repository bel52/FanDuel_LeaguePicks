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
from .optimize import generate_pool, StackRule
from .simulate import SlateSimulator
from .field import build_prior_field, rank_candidates, baseline_lineups
from .objectives import Leaderboard, SeasonContext, weights_for
from .slate import SlateStore, SlateError
from .injuries import (records_from_fantasypros, records_from_slate, merge,
                       sweep as injury_sweep)
from .export import export_upload_csv, lineup_card, pushover_body
from .results import ResultLog

DATA = Path(__file__).resolve().parents[2] / "data"


def _dist():
    p = DATA / "distributions.json"
    if not p.exists():
        print("Building empirical distributions (one-time, ~60s)...")
        DATA.mkdir(exist_ok=True)
        build_distributions(out_path=p)
    return load_distributions(p)


def cmd_build(a) -> int:
    dist = _dist()

    print("=" * 72)
    slate, irep = ingest_csv(a.csv, a.slate_id, a.season, a.week, strict=False)
    print(irep.summary())
    if irep.validation_problems:
        print("\n!! validation problems above — continuing, review before entering lineups")

    print("\n" + "=" * 72)
    print("Pulling FantasyPros projections...")
    fp = FantasyProsClient().weekly_projections(a.season, a.week)
    print(f"  {len(fp)} players projected (FanDuel-scored from stat lines)")

    team_lines = {}
    if not a.no_vegas:
        try:
            oc = OddsClient()
            team_lines = oc.team_lines(slate_teams={p.team for p in slate.players})
            print(f"  Vegas: {len(team_lines)} team totals "
                  f"(quota remaining {oc.last_quota.get('remaining','?')})")
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
    field = build_prior_field(slate, spec, n_opponents=spec.field_size - 1, seed=a.seed + 1)
    print(f"  field model: {len(field.lineups)} opponents ({field.source})")

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
    n_opp = len(field.lineups)
    naive = 1.0 / (n_opp + 1)
    field_totals = sim.score_many([(l, None) for l in field.lineups])

    def _obj(ids, mvp=None):
        """Objective value in dollars, so baselines compare on the same axis."""
        from .objectives import score_lineup
        return score_lineup(sim.score(ids, mvp).totals, field_totals, weights)

    bl = baseline_lineups(slate, spec, n=200, seed=a.seed + 2)
    rand = [_obj(l) for l in bl["random"]]
    max_proj = max(pool, key=lambda c: c.proj_sum)
    mp = _obj(max_proj.player_ids, max_proj.mvp_id)
    best = ranked[0]

    print("\n" + "=" * 72)
    print(f"BASELINES (vs the same {n_opp}-opponent simulated field)")
    print(f"  chance if all 12 were coin flips : {naive:6.1%}")
    print(f"  random valid lineup (n=%d)       : {sum(rand_p)/len(rand_p):6.1%}" % len(rand_p))
    print(f"  max-projection lineup            : {maxproj_p:6.1%}")
    print(f"  our #1 lineup                    : {best_p:6.1%}")
    edge = best_p - maxproj_p
    print(f"  edge over max-projection         : {edge:+6.1%} "
          f"({'REAL' if edge > 0.005 else 'NOT DEMONSTRATED — do not trust this build'})")

    print("\n" + "=" * 72)
    print(f"TOP {a.show} LINEUPS — ranked by EV vs {spec.field_size - 1}-opponent field")
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

    if a.export:
        ex = export_upload_csv(best_players, a.export, slate_type=slate.slate_type,
                               mvp_id=best.candidate.mvp_id, template=a.template,
                               contest_name=spec.name)
        print("\n" + ex.summary())

    if a.log_db:
        rl = ResultLog(a.log_db)
        rl.log_entry(a.season, a.week, spec.name, best_players, slate_id=a.slate_id,
                     objective=a.leaderboard, exp_points=best.exp_points,
                     p_win=best.p_win)
        print(f"Logged entry -> {a.log_db}")

    if a.pushover_out:
        Path(a.pushover_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.pushover_out).write_text(pushover_body(
            best_players,
            f"W{a.week} ${best.dollars:.2f} | E[pts] {best.exp_points:.1f} | "
            f"P(win) {best.p_win:.0%}",
            mvp_id=best.candidate.mvp_id))
        print(f"Pushover body -> {a.pushover_out}")

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

    if a.db:
        SlateStore(a.db).save(slate)
        print(f"Slate persisted: {a.db}")
    return 0


def cmd_capture(a) -> int:
    from .contest_parse import parse_contest
    capture = parse_contest(a.path, a.season, a.week, a.contest)
    print(capture.summary())
    rl = ResultLog(a.log_db)
    rl.log_capture(capture)
    mine = next((e for e in capture.leaderboard if e.rank == 1), None)
    print(f"\nLogged {len(capture.leaderboard)} entrants and "
          f"{len(capture.ownership())} ownership records -> {a.log_db}")
    return 0


def cmd_standings(a) -> int:
    rl = ResultLog(a.log_db)
    st = rl.standings(a.season)
    if not st:
        print("No logged results yet. Run `capture` on a contest results page first.")
        return 0
    print(f"SEASON STANDINGS — {a.season} (Total Points)")
    print(f"  {'#':>2s} {'entrant':18s} {'total':>8s} {'avg':>7s} {'wks':>4s} {'wins':>5s} {'best':>7s}")
    for i, s in enumerate(st, 1):
        me = " <-- you" if s.entrant == a.me else ""
        print(f"  {i:2d} {s.entrant:18s} {s.total_points:8.1f} {s.avg:7.1f} "
              f"{s.weeks:4d} {s.wins:5d} {s.best:7.1f}{me}")
    ctx = rl.season_context(a.season, a.me, a.weeks_total)
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
    b.add_argument("--season", type=int, required=True)
    b.add_argument("--week", type=int, required=True)
    b.add_argument("--slate-id", default=None)
    b.add_argument("--contest", default="Leather League")
    b.add_argument("--profile", default="friends_league",
                   choices=[p.value for p in Profile])
    b.add_argument("--field", type=int, default=12)
    b.add_argument("--entry-fee", type=float, default=15.0)
    b.add_argument("--pool", type=int, default=120)
    b.add_argument("--sims", type=int, default=20000)
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
    b.add_argument("--out", default="data/lineups/latest.json")
    b.add_argument("--db", default=None)
    b.set_defaults(func=cmd_build)

    cap = sub.add_parser("capture", help="ingest a pasted FanDuel contest results page")
    cap.add_argument("path")
    cap.add_argument("--season", type=int, required=True)
    cap.add_argument("--week", type=int, required=True)
    cap.add_argument("--contest", default="Leather League")
    cap.add_argument("--log-db", default="data/results.db")
    cap.set_defaults(func=cmd_capture)

    st = sub.add_parser("standings", help="season standings + objective weights from the log")
    st.add_argument("--season", type=int, required=True)
    st.add_argument("--me", default="xleathy")
    st.add_argument("--weeks-total", type=int, default=21)
    st.add_argument("--log-db", default="data/results.db")
    st.set_defaults(func=cmd_standings)

    a = ap.parse_args(argv)
    if getattr(a, "slate_id", "SKIP") is None and hasattr(a, "week"):
        a.slate_id = f"{a.season}-w{a.week:02d}"
    try:
        return a.func(a)
    except (SlateError, FantasyProsError, VegasError) as e:
        print(f"\nBUILD STOPPED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
