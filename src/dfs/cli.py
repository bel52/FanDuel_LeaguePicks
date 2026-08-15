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
from .slate import SlateStore, SlateError

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

    ranked = rank_candidates(pool, sim, field, spec)
    byid = {p.fd_id: p for p in slate.players}

    # ---- honest baselines: a win probability alone is uninterpretable ----
    n_opp = len(field.lineups)
    naive = 1.0 / (n_opp + 1)
    field_totals = sim.score_many([(l, None) for l in field.lineups])

    def _pwin(ids, mvp=None):
        mine = sim.score(ids, mvp).totals
        return float(((mine[None, :] > field_totals).sum(axis=0) == n_opp).mean())

    bl = baseline_lineups(slate, spec, n=200, seed=a.seed + 2)
    rand_p = [_pwin(l) for l in bl["random"]] or [naive]
    max_proj = max(pool, key=lambda c: c.proj_sum)
    maxproj_p = _pwin(max_proj.player_ids, max_proj.mvp_id)
    best_p = ranked[0].p_win

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

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slate_id": a.slate_id, "season": a.season, "week": a.week,
        "contest": {"name": spec.name, "profile": spec.profile.value,
                    "field_size": spec.field_size, "entry_fee": spec.entry_fee},
        "match_rate": mrep.rate, "seed": a.seed, "sims": a.sims,
        "lineups": [{
            "rank": i, "p_win": r.p_win, "ev": r.ev, "median": r.median, "p90": r.p90,
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
    b.add_argument("--out", default="data/lineups/latest.json")
    b.add_argument("--db", default=None)
    b.set_defaults(func=cmd_build)

    a = ap.parse_args(argv)
    if getattr(a, "slate_id", None) is None:
        a.slate_id = f"{a.season}-w{a.week:02d}"
    try:
        return a.func(a)
    except (SlateError, FantasyProsError, VegasError) as e:
        print(f"\nBUILD STOPPED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
