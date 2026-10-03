# FanDuel LeaguePicks — v6

DFS optimizer for a 12-person family league (Total Points scoring) plus head-to-head,
single-game showdown, and public contests.

> **Branch `v6-rebuild` is the active line of development.** `main` and the older `v6`
> branch are the v5-era code and are retained only for reference. Do not build on them.

## Status — honest

The data layer, modeling core, and Sunday operating loop are complete and tested
(281 offline tests, including an end-to-end build test that asserts the upload CSV,
entry log, and pushover card actually exist). **Edge is not yet demonstrated.** The system reports a positive
objective delta over a max-projection baseline, but that number is produced by the
same simulator that selects the lineup. Until it is validated against out-of-sample
outcomes, treat it as an internal consistency check, not evidence of profitability.

Every build prints an INDEPENDENT EVALUATION block — the selected lineup re-scored on
fresh simulations and fresh opponent fields — and reports the selection optimism gap.
Trust that number, not the selection estimate.

## Quick start

**Normal operation is the web UI at `https://dfs.leathfam.com`** (behind Cloudflare
Access): Saturday build → enter by hand on FanDuel → "I entered this lineup"; Sunday
11:30 and ~3:00 ET fresh player list → Run Swap Check → do what its first line says
(`DECISION: NO CHANGE` or `DECISION: MAKE THIS SWAP ON FANDUEL`); Monday capture. The
CLI below is what the UI runs.

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

    # Whole-pool actuals from nflverse — runs automatically inside build (prior weeks)
    # and capture (all weeks); this is only for a manual backfill
    ./run.sh actuals --season 2026

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
| `sleeper` | layer 3: official NFL game-status designations via Sleeper (LOG mode) |
| `teammates` | vacated production from absent players → teammates (measured rates) |
| `actuals` | nflverse whole-pool actuals, snap-count `played`, shadow-arm grading |
| `kickoffs` | nflverse kickoff times and per-game lock state |
| `optimize` | MIP candidate pools with diversity, stack, and showdown rules |
| `simulate` | correlated Monte Carlo (Gaussian copula over empirical ratios) |
| `objectives` | per-profile dollar-denominated objective weights |
| `field` | opponent field ensemble, baselines, candidate ranking |
| `lateswap` | lock-aware re-optimization of unlocked slots |
| `export` | lineup cards (+ an upload CSV FanDuel has no way to consume) |
| `contest_parse` | pasted contest results → lineups + measured league ownership |
| `results` | result log, standings, projection-component and availability calibration |
| `cli` | `build` / `swap` / `capture` / `standings` / `actuals` |

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
the nine entered), and **every projected player is graded from nflverse** — see
*Whole-pool actuals* below. `component_accuracy` then grades the components against each other on the same
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

**Sunday swap decision.** The swap output leads with one line a human acts on —
`DECISION: NO CHANGE` or `DECISION: MAKE THIS SWAP ON FANDUEL` — so no rule has to be
remembered on game day. A speculative swap must gain at least `MIN_SWAP_GAIN_PTS` = 6
projected points (per-player projection error is ~6–7 pts MAE; Week 1 2026's +2.5 swap
rewrote three slots and cost 10.7 actual points). A player ruled out always forces a
swap, but only his slot changes unless reshuffling the healthy players clears the same
bar. The web swap passes `--require-fresh-csv 3`, so a forgotten upload is refused with
instructions instead of silently re-using Saturday's player list.

**Whole-pool actuals (`actuals.py`).** Contest-page actuals cover only the ~50–60
players somebody rostered — a projection-selected sample. Every projected player
(~390 a week) is now graded from nflverse: players join through nflverse's own id
(`gsis_id`), never name-to-name; a roster player with no stat row genuinely scored 0; an
unmatched pool player stays NULL and is reported (never a false zero); only final games
are graded. DST sacks and interceptions come from the opponent's offense (defender
credits sum short on split sacks), and blocked kicks count 2. Validated 2026-09-26:
the scorer reproduced the Week-2 contest page to the hundredth for 50 of 52 players
including every defense; the two misses were hand-transcribed capture cells, where
nflverse was right. `actual` prefers nflverse and `actual_fd` keeps the page value;
disagreements are printed, which surfaces capture errors. Shadow arms grade themselves
and `standings` prints entered-vs-shadow by week. Runs automatically on build (prior
weeks) and capture (all weeks); a feed outage prints one line and retries next run.

**Teammate effects (`teammates.py`).** When a player is out or may be, part of his
production flows to teammates, and FantasyPros consensus is slow to move them. The
rates are measured on nflverse 2023–25: teammates absorb 61% of a sitting receiver's
receiving points (271 games, 95% CI 45–77%) and 58% of a sitting back's rushing points
(103 games, CI 40–76%), split by projected volume with same-position teammates taking
~2.5× per unit (2,096 recipient-games). Applied conservatively: the low CI bounds
(45% / 40%), to the FantasyPros component only (markets already reprice teammates),
with "vacated" = the absent player's own FP projection × P(he sits) so it switches off
when consensus has caught up; capped at +5 pts / +35% (so it alone can never clear the
swap threshold); logged per player in `player_results.teammate_adj`; fail-safe (any
error leaves projections unchanged); `--no-teammates` disables it. A QB out is a
team-level effect already carried by the Vegas implied total, so it is excluded.

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
* **`player_results.played` / `.status_at_lock`** — who actually suited up: from nflverse
  snap counts once a game's snaps publish (`played_src='snaps'`), which overrides the
  post-lock FanDuel player list (`O` flag) recorded during `swap` — that flag cannot see
  a late scratch. `actual` cannot stand in:
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
- **FantasyPros DOUBTFUL de-escalation (fixed 2026-10-03).** The pessimistic `merge()`
  let a FantasyPros DOUBTFUL (which removes the player) override everything: Week 3 2026
  removed Mike Evans at 87% to play, Keon Coleman at 70% and Tyjae Spears at 62%; Ladd
  McConkey (removed as DOUBTFUL, Week 2) played. Now a FantasyPros DOUBTFUL with an
  explicit play probability >= `DEESCALATE_MIN_PROB` (50%) is held as Questionable, and
  an explicit official Q/P from Sleeper overrules a FantasyPros DOUBTFUL in GATE mode
  (LOG mode reports it only). Both only ever put a player back in the pool; `p_active` prices the risk and the
  Sunday fresh-CSV gate still catches a real scratch. Official D, OUT and IR are untouched,
  and probability still cannot create a removal. The 50% threshold is a prior — tune it
  from `availability_accuracy`.
- **Odds API credit budget (guard added 2026-10-03).** Week 3 2026 priced props for only
  42 of 359 players after the free tier ran dry. Prop fetches are now all-or-nothing
  against the quota header on the free `/events` call: if pricing the slate would leave
  fewer than `--props-reserve` (20) credits for Vegas lines, the run uses cached boards
  only. Web "Test run" builds pass `--props-cache-only` and never spend credits. Whether
  500/month is enough for a 21-week season plus showdown play is still an open decision.
- **Teammate-effect accuracy is unmeasured.** The rates are historical; whether the
  adjustment improves this season's projections is readable from `teammate_adj` against
  nflverse actuals after ~3 weeks.
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
- The opponent field is conditioned on measured league ownership (n/(n+4) shrinkage
  toward measured DRAFTED%); per-opponent tendency modeling is not built.
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
| `data/results.db` | entries, `player_results` (components, actuals + provenance, `played`, `teammate_adj`), `props_lines` | no |

Everything under `data/` except the two JSON files above is runtime state. The cache is
gitignored because a tracked cache is rewritten by every clone and pull; freshness comes
from a stamp inside each board file rather than its mtime for the same reason.

`data/distributions.json` is the active calibration and is **not** shipped in deploy
tarballs, so a deploy cannot silently replace it with the older proxy version.
Rebuild with:

    python3 -m dfs.calibrate --rebuild-distributions data/calibration_2025.json \
        --dist-out data/distributions.json
