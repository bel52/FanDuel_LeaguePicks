# FanDuel LeaguePicks — v6

DFS optimizer for a 12-person family league (Total Points scoring) plus head-to-head,
single-game showdown, and public contests.

> **Branch `v6-rebuild` is the active line of development.** `main` and the older `v6`
> branch are the v5-era code and are retained only for reference. Do not build on them.

## Status — honest

The data layer, modeling core, and Sunday operating loop are complete and tested
(235 offline tests, including an end-to-end build test that asserts the upload CSV,
entry log, and pushover card actually exist). **Edge is not yet demonstrated.** The system reports a positive
objective delta over a max-projection baseline, but that number is produced by the
same simulator that selects the lineup. Until it is validated against out-of-sample
outcomes, treat it as an internal consistency check, not evidence of profitability.

Every build prints an INDEPENDENT EVALUATION block — the selected lineup re-scored on
fresh simulations and fresh opponent fields — and reports the selection optimism gap.
Trust that number, not the selection estimate.

## Quick start

    pip install -r requirements.txt
    cp .env.example .env        # add FANTASYPROS_API_KEY and ODDS_API_KEY
    python3 -m pytest tests/ -q # offline, no keys required

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
   haircut. Doubtful is treated as remove. *Refined for league play (2026-09-07):*
   under Total Points a Questionable player's honest expected contribution is
   `P(plays) x projection`, so `--avail-adjust` (default on for `friends_league`)
   discounts it. The haircut is never silent — `p_active` is stored on the player, the
   undiscounted number stays in `proj_blend`, every adjusted row is printed, and the
   lineup card marks it. Non-league profiles are unchanged.
6. **Win probability is averaged over a field ensemble.** A single opponent draw moved
   the estimate 14.5% → 21.7% across seeds; that instability is now integrated out and
   the residual spread is displayed.
7. **Ties split win credit** rather than counting as losses.
8. **Selection and evaluation are separated.** The argmax of noisy estimates is biased
   high, so the reported number comes from independent simulations.
9. **Fail loud.** Schema drift, low match rates, missing Vegas lines, and uncalibrated
   distributions all produce visible warnings or stop the build.

10. **Markets set the distribution, consensus sets the level.** Player props supply
   relative player-to-player signal; each position's props points are then scaled by
   the median FantasyPros/props ratio for that position on that slate. A prop line
   sits near the *median* outcome while FanDuel points need the *mean*, and that skew
   is not identifiable from one threshold — so it is cancelled by anchoring rather
   than papered over with a fabricated constant. Anchoring also keeps
   `distributions.json`, fit against FP-scaled projections, valid.

## Modules

| module | role |
|---|---|
| `contest_spec` | contest config: field, payouts, cap, late-swap, MVP salary rule |
| `slate` | canonical `PlayerSlate` + SQLite store, fail-loud validation |
| `ingest_fanduel` | FanDuel salary CSV → slate, with a validation report |
| `fantasypros` | FantasyPros public API v2 client |
| `scoring` | FanDuel points from FP stat lines (half-PPR, bonuses, DST PA ladder) |
| `matching` | name-first player matching; team is a disambiguator only |
| `vegas` | The Odds API → implied **team** totals (per-book pairing, FanDuel first) |
| `props` | The Odds API → market-implied per-player stat lines (league play) |
| `distributions` | empirical outcome ratios, calibrated on 2025 FP residuals |
| `calibrate` | projection-accuracy harness + distribution rebuild |
| `blend` | projection assembly |
| `injuries` | three-layer injury pipeline + Sunday inactives sweep |
| `kickoffs` | nflverse kickoff times and per-game lock state |
| `optimize` | MIP candidate pools with diversity, stack, and showdown rules |
| `simulate` | correlated Monte Carlo (Gaussian copula over empirical ratios) |
| `objectives` | per-profile dollar-denominated objective weights |
| `field` | opponent field ensemble, baselines, candidate ranking |
| `lateswap` | lock-aware re-optimization of unlocked slots |
| `export` | lineup cards (+ an upload CSV FanDuel has no way to consume) |
| `contest_parse` | pasted contest results → lineups + measured league ownership |
| `results` | result log, standings, projection-component and availability calibration |
| `cli` | `build` / `swap` / `capture` / `standings` |

## League play: player props

Under Total Points the season objective is (near enough) the sum of projections, so
projection accuracy is the only lever that matters — the correlated simulator and the
opponent field affect the weekly-prize sliver alone. Betting markets are the sharpest
public per-player signal available, so `props` blends them in.

```
# default for the friends_league profile; ~6 credits per game
./run.sh build --csv <fanduel.csv> --season 2026 --week 1
./run.sh build --csv <fanduel.csv> --season 2026 --week 1 --no-props   # FP only
```

Markets consumed: `player_pass_yds`, `player_pass_tds`, `player_rush_yds`,
`player_reception_yds`, `player_receptions`, `player_anytime_td`. Each becomes part of
a market-implied **stat line**, which is then scored by the same `scoring.score()` used
for FantasyPros — one scoring path, identical units, same bonus model, same auditable
breakdown. Categories no props board prices (interceptions, fumbles, return TDs,
two-point conversions) come from the player's FP line; leaving them at zero would
inflate every QB by about a point.

Pass TDs are the one market priced at a single threshold, so lambda is solved from a
Poisson tail. That is defensible rather than a guess because it cross-checks: on the
live 2026 Week 1 board DraftKings priced Joe Burrow at 1.5 and FanDuel at 2.5, and both
solve to lambda ~2.2. `tests/test_props.py` asserts that agreement.

**Cost.** One credit per market per event: 6 per game, ~72 for a 12-game slate, against
a 500/month free tier. Boards are cached on disk (`data/props/`, `--props-max-age`,
default 6h) so a rebuild after the inactives sweep is free, and props are **off by
default on `swap`** — three Sunday windows at full price would exceed the tier.

**Safety.** Every failure degrades to FantasyPros-only, which is the pre-props
behaviour, so this layer cannot cost a build. Cache freshness is judged on a
`_fetched_at` stamp written inside each board file, never on the file's mtime — a
clone or pull rewrites mtimes and would otherwise serve week-old lines as live. Per-position scale factors are printed
with their sample size and gated: a factor outside ±25% warns, outside 0.55–1.80 is
treated as a board or parser failure and props are discarded for that position. A
position with too few priced players is skipped rather than scaled on noise.

**DEF.** A defense is not scaled by its own offense's total — it is scored against the
opposing one. `score_dst` accepts the opponent's implied team total in place of
FantasyPros' projected points-allowed, which is the dominant term in the FanDuel ladder
(10 for a shutout down to −4 for 35+). Before 2026-09-07 the D slot received no market
signal at all: `blend.py` excluded position D from the Vegas tilt, rightly, and nothing
replaced it — so the one position whose projection is essentially a single market-priced
quantity was the only one modelled without the market. Vegas still enters exactly once:
the D never gets a tilt, skill players never see an opponent total.

**Measuring it.** `log_projection_components` writes `proj_fp`, `proj_props`,
`proj_blend` and `p_active` for the whole priced pool every build (~370 rows a week, not
the nine entered), and `capture` attaches actuals for every player on the contest results
page (~60–100 distinct, matched by name; players never projected are skipped rather than
invented). `component_accuracy` then grades the components against each other on the same
players and reports a least-squares props weight. That number is **reported, never
applied** — `PROPS_WEIGHT` stays a human decision, because a weight fitted on four weeks
of one season is not evidence. No walk-forward backtest is possible (the salary archives
are patchy), so this in-season log is the only route to a demonstrated edge, and it has
to start in Week 1 or the data does not exist.

**Inactives.** There is still no official inactives feed, but there is a free one: FanDuel
marks scratched players `O` on its own player list, and `ingest_fanduel` already drops them
before they can reach a lineup. So a CSV re-downloaded after the inactives post *is* an
inactives source, and a Wednesday CSV is not. `swap` therefore prints the CSV's age
against the next unlocked kickoff and, with `--require-fresh-csv HOURS` (set to 3 in
`bin/sunday-swap.sh`), refuses rather than swapping on a lineup it cannot verify.

**Persistence (the learning loop's prerequisite).** Three things are written down every
build because an unrecorded week is permanently unlearnable, and there are 21 of them:

* **`props-<season>-w<NN>-<slate>.json.gz`** — the raw prop boards, frozen at lock
  beside the FantasyPros snapshot, stamped with SHA-256 of both `scoring.py` and
  `props.py` so a later re-derivation can tell a constant change from a market move.
  The disk cache expires in six hours and The Odds API has no free historical props
  endpoint, so this file is the only way a past week's market projection can ever be
  re-derived.
* **`props_lines`** — the market-implied *stat line* per player-week, not just its
  points total. `MULTI_TD_FACTOR` and `ANYTIME_TD_OVERROUND` are fittable only by
  comparing an expected quantity to the actual one (expected TDs against TDs scored);
  from a points total they are unrecoverable, so both would stay coarse priors forever.
* **`player_results.played` / `.status_at_lock`** — who actually suited up, taken from
  the post-lock FanDuel player list (`O` flag) during `swap`. `actual` cannot stand in:
  it is NULL for a scratch, NULL for a player nobody rostered, and 0.0 for a player who
  played and did nothing — three different facts. This is what turns the flat 0.72
  questionable prior into a measured number.

`standings` prints both learning reads — `component_accuracy` (market vs consensus on
the same players, plus a least-squares props weight) and `availability_accuracy`
(predicted `p_active` against observed play rate, bucketed). Both are **read-only by
design**: they report what the data supports, and changing `PROPS_WEIGHT` or the
questionable prior stays a human decision until the sample justifies it.

**Scope.** `friends_league` only. Showdown and head-to-head paths are untouched: they
are scored on P(win), where a discounted mean is the wrong treatment, and a single game
yields too few priced players per position to fit a scale factor.

## Known gaps

- No walk-forward backtest (historical FanDuel salary archives are patchy), which is
  why the in-season component log exists — it is the only available route to a
  measured edge.
- **The actuals sample is biased, not merely small.** Actuals come from the contest
  results page, which contains only players the twelve entrants rostered — roughly
  50–70 distinct players a week out of ~371 projected, and skewed toward high
  projections by construction. Fitting a blend weight on a projection-selected
  subsample gives a biased estimate. Fix (queued for weeks 2–3): pull weekly actuals
  for the whole pool from `nflreadpy` (already a dependency for kickoffs) and score
  them through `scoring.py`, which also makes the `played` flag definitive via snap
  counts and supersedes the `fanduel_csv`-sourced rows.
- **`PROPS_WEIGHT` is still a hardcoded constant.** `component_accuracy` reports a
  fitted weight but nothing consumes it, and there is no threshold at which it would
  ever be adopted — so "reported only" becomes "never used" unless a versioned weights
  file (same shape as `distributions.json`) is wired in. Queued for week 6+, gated on
  sample size.
- **`distributions.json` is fitted on 2025 FantasyPros residuals, but projections are
  now blends.** Scale anchoring keeps it approximately valid — that is precisely why
  the props component is anchored to the FP level rather than used raw — but the pools
  should be re-fitted on blend residuals as weeks accumulate, with versioning and
  rollback (this file has been clobbered three times historically).
- **No holdout discipline for fitted parameters yet.** Anything fitted in-season needs
  expanding-window or leave-one-week-out evaluation before it is trusted, or it will
  reproduce exactly the optimism the INDEPENDENT EVALUATION block exists to catch.
- Opponent field is a generic chalk-weighted prior; measured league ownership is
  captured but not yet driving it.
- Two props constants remain coarse priors, labelled as such in `props.py`:
  `ANYTIME_TD_OVERROUND` (TD boards are one-sided and cannot be devigged pairwise) and
  `MULTI_TD_FACTOR` (E[TDs] given P(>=1 TD)). Both become empirical once logged
  actuals accumulate; neither affects the yardage or pass-TD terms.
- No LLM/news layer yet. The candidate job is structuring beat-writer and
  practice-report text into `{play_prob, role_change, confidence, source}` to feed
  `p_active` — a logged structured input, never a silent projection edit.
- Kickers and defenses have no calibrated distributions — both use a generic spread.
- FanDuel bulk upload is **closed as not-possible**, not pending (verified on
  fanduel.com 2026-09-07): "Export this Lineup" copies a lineup between your own
  entries inside the site, and there is no CSV download and no blank entries template
  anywhere in the flow. So the built-in column layout can never be validated against a
  real file, and the exported row would carry no entry_id/contest_id even if the
  headers were right. Hand entry from the lineup card is the permanent workflow.
  `--template` still works if a real entries file ever becomes available.
- FanDuel single-game format (confirmed, 2025 rules onward): 6 slots — 1 MVP + 5 FLEX;
  the MVP costs AND scores 1.5×. `ContestSpec.mvp_salary_mult` toggles the salary rule
  in one place if FanDuel ever changes it.

## Data files

| Path | What it is | Tracked? |
|------|-----------|----------|
| `data/distributions.json` | active outcome calibration (2025 residuals) | yes |
| `data/aliases.json` | hand-confirmed FanDuel→FantasyPros name overrides | yes |
| `data/props/` | Odds API prop-board cache, `_fetched_at`-stamped | **no** (gitignored) |
| `data/snapshots/` | at-lock FantasyPros and prop-board payloads, gzipped | no |
| `data/results.db` | entries, `player_results` components, `props_lines` | no |

Everything under `data/` except the two JSON files above is runtime state. The cache is
gitignored because a tracked cache is rewritten by every clone and pull; freshness comes
from a stamp inside each board file rather than its mtime for the same reason.

`data/distributions.json` is the active calibration and is **not** shipped in deploy
tarballs, so a deploy cannot silently replace it with the older proxy version.
Rebuild with:

    python3 -m dfs.calibrate --rebuild-distributions data/calibration_2025.json \
        --dist-out data/distributions.json
