# FanDuel LeaguePicks — v6

DFS optimizer for a 12-person family league (Total Points scoring) plus head-to-head,
single-game showdown, and public contests.

> **Branch `v6-rebuild` is the active line of development.** `main` and the older `v6`
> branch are the v5-era code and are retained only for reference. Do not build on them.

## Status — honest

The data layer, modeling core, and Sunday operating loop are complete and tested
(69 offline tests). **Edge is not yet demonstrated.** The system reports a positive
objective delta over a max-projection baseline, but that number is produced by the
same simulator that selects the lineup. Until it is validated against out-of-sample
outcomes, treat it as an internal consistency check, not evidence of profitability.

Every build prints an INDEPENDENT EVALUATION block — the selected lineup re-scored on
fresh simulations and fresh opponent fields — and reports the selection optimism gap.
Trust that number, not the selection estimate.

## Quick start

    pip install -r requirements.txt
    cp .env.example .env        # add FANTASYPROS_API_KEY and ODDS_API_KEY
    python3 -m pytest tests/ -q # 69 tests, offline, no keys required

    # Wednesday: build from the FanDuel salary CSV
    ./run.sh build --csv <fanduel_salaries.csv> --season 2026 --week 1 \
        --leaderboard total_scores --field 12 --weekly-prize 12.84 \
        --grand-prizes 135,81,54 --weeks-total 21 \
        --export data/upload.csv --log-db data/results.db

    # Sunday 11:30 / 15:45 / 19:45 ET: lock-aware late-swap check
    bin/sunday-swap.sh 2026 1 <fanduel_salaries.csv>

    # Monday: ingest the contest results page (Cmd-A / Cmd-C into a text file)
    ./run.sh capture <pasted_page.txt> --season 2026 --week 1
    ./run.sh standings --season 2026 --me brettleath

## Design rules (each learned from a specific v5 failure)

1. **No fabricated projections.** Values come from FantasyPros stat lines scored under
   FanDuel rules. There is no FPPG or salary-derived fallback; a missing projection
   halts the build. *(v5 used season-average FPPG, or `salary/550` when absent.)*
2. **Vegas enters exactly once**, as implied team totals with a bounded tilt. No
   downstream game-total boosts. *(v5 multiplied by Vegas, then boosted again.)*
3. **Lineup metrics come from the joint simulated distribution**, never from summing
   player percentiles. *(v5 overstated one lineup's ceiling by ~45 points.)*
4. **Objectives are denominated in dollars** and derived from the contest's real prize
   structure. *(v5 stacked hand-tuned multipliers.)*
5. **Injuries produce an action** — keep, flag, or remove — never a silent projection
   haircut. Doubtful is treated as remove.
6. **Win probability is averaged over a field ensemble.** A single opponent draw moved
   the estimate 14.5% → 21.7% across seeds; that instability is now integrated out and
   the residual spread is displayed.
7. **Ties split win credit** rather than counting as losses.
8. **Selection and evaluation are separated.** The argmax of noisy estimates is biased
   high, so the reported number comes from independent simulations.
9. **Fail loud.** Schema drift, low match rates, missing Vegas lines, and uncalibrated
   distributions all produce visible warnings or stop the build.

## Modules

| module | role |
|---|---|
| `contest_spec` | contest config: field, payouts, cap, late-swap, MVP salary rule |
| `slate` | canonical `PlayerSlate` + SQLite store, fail-loud validation |
| `ingest_fanduel` | FanDuel salary CSV → slate, with a validation report |
| `fantasypros` | FantasyPros public API v2 client |
| `scoring` | FanDuel points from FP stat lines (half-PPR, bonuses, DST PA ladder) |
| `matching` | name-first player matching; team is a disambiguator only |
| `vegas` | The Odds API → implied **team** totals |
| `distributions` | empirical outcome ratios, calibrated on 2025 residuals |
| `calibrate` | projection-accuracy harness + distribution rebuild |
| `blend` | projection assembly |
| `injuries` | three-layer injury pipeline + Sunday inactives sweep |
| `kickoffs` | nflverse kickoff times and per-game lock state |
| `optimize` | MIP candidate pools with diversity, stack, and showdown rules |
| `simulate` | correlated Monte Carlo (Gaussian copula over empirical ratios) |
| `objectives` | per-profile dollar-denominated objective weights |
| `field` | opponent field ensemble, baselines, candidate ranking |
| `lateswap` | lock-aware re-optimization of unlocked slots |
| `export` | FanDuel upload CSV + lineup cards |
| `contest_parse` | pasted contest results → lineups + measured league ownership |
| `results` | result log, season standings, in-season accuracy tracking |
| `cli` | `build` / `swap` / `capture` / `standings` |

## Known gaps

- No walk-forward backtest (historical FanDuel salary archives are patchy).
- Opponent field is a generic chalk-weighted prior; measured league ownership is
  captured but not yet driving it.
- Kickers and defenses have no calibrated distributions — both use a generic spread.
- FanDuel upload CSV headers are unverified; the export warns until `--template`
  supplies a real entries file.
- Whether FanDuel charges 1.5× salary for the showdown MVP slot is unconfirmed;
  `ContestSpec.mvp_salary_mult` toggles it in one place.

## Data files

`data/distributions.json` is the active calibration and is **not** shipped in deploy
tarballs, so a deploy cannot silently replace it with the older proxy version.
Rebuild with:

    python3 -m dfs.calibrate --rebuild-distributions data/calibration_2025.json \
        --dist-out data/distributions.json
