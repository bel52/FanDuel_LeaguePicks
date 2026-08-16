# DFS Optimizer v6 — Project State

**Living document. Replace this file in the project library whenever it changes.**
Last updated: 2026-08-16 (rev 2 — post second adversarial review) · Target: NFL Week 1, Sunday 2026-09-13 (~4 weeks out)

A new session should be able to read this file and pick up without re-deriving anything.

---

## 1. What this is

A DFS optimizer for Brett Leatherman's contests, rebuilt from scratch after v5 was
found to have no defensible edge. Two distinct use cases:

- **League play** — "Leather League": 12 family members, $50 season entry, 21 contests,
  $12.84 weekly prize plus $135/$81/$54 season grand prizes. Season leaderboard is
  **Total Points** (switched from Most Wins — see §5).
- **Non-league play** — public head-to-head, single-game showdown, public Sunday Main,
  tournaments. Brett plays roughly 2x more single-game contests than league games, and
  showdown is historically his only profitable segment (20.5% win rate).

Brett is `brettleath` on FanDuel. (Early sessions incorrectly used `xleathy`, who is an
opponent — corrected 2026-08-16.)

## 2. Current status — honest

**Deployed and working.** Live at `https://dfs.leathfam.com` behind Cloudflare Access.
Full pipeline runs end to end through the web UI: FanDuel CSV → FantasyPros projections
→ FanDuel scoring → Vegas team totals → injury sweep → correlated simulation → ranked
lineups → upload CSV → result logging → season standings → objective weights.

**Edge is NOT demonstrated.** The system reports a positive objective delta over a
max-projection baseline, but that number comes from the same simulator that selects the
lineup. Until validated against out-of-sample outcomes it is an internal consistency
check, not evidence of profit. Every build prints an INDEPENDENT EVALUATION block
(fresh sims + fresh fields) and a selection-optimism gap; trust that number.

**A correction to an earlier conclusion.** I previously argued that because expected
points is ~89% of the league objective, league play reduces to "maximize projections."
The second adversarial review showed this reasoning is wrong: when comparing two
lineups the common value cancels, so what matters is the *marginal* trade —

    Δobjective = $0.164 × Δ(expected points) + $12.84 × Δ P(win)

One projected point trades against ~1.28 percentage points of win probability, and
among near-max lineups that margin genuinely can change the winner. Correlation
modeling and the field model are therefore not decorative in league play. The
conclusion that survives is narrower: showdown and H2H are where the machinery matters
*most*, and projection quality is the single largest lever — but not the only one.

Code: ~4,500 lines, **94 tests**, all passing.

## 3. Where things live

| What | Where |
|---|---|
| Repo | `github.com/bel52/FanDuel_LeaguePicks`, branch **`v6-rebuild`** (main/v6 = stale v5) |
| Dev working copy | `/home/brett/dfs-v6` on ubuntserv (venv at `.venv`) |
| Container build source | `/srv/appdata/dfs/src-checkout` (git checkout, HTTPS remote for root pulls) |
| Compose | `/srv/compose/dfs/docker-compose.yml` |
| Persistent data (bind mount) | `/srv/appdata/dfs/data` — `distributions.json`, `calibration_2025.json`, `results.db`, `settings.json`, `uploads/`, `lineups/` |
| Secrets | `/srv/appdata/dfs/.env` (chmod 600) |
| Container | `dfs`, port `127.0.0.1:8093`, healthcheck on `/health` |
| Public URL | `https://dfs.leathfam.com` (CF tunnel → CF Access app "DFS", email OTP, 1-month session) |

**Deploy path (tarballs are retired):**
```bash
# dev: edit in sandbox → tarball → ~/dfs-v6 → pytest → commit → push
cd /srv/appdata/dfs/src-checkout && sudo git pull -q
cd /srv/compose/dfs && sudo docker compose up -d --build
```
Data is a bind mount, so a code deploy can never clobber the calibration or results log.

## 4. Architecture

`src/dfs/` — each module one job:

| module | role |
|---|---|
| `contest_spec` | contest config: field, payouts, cap, late swap, `mvp_salary_mult` |
| `slate` | canonical `PlayerSlate` + SQLite store, fail-loud validation |
| `ingest_fanduel` | FD salary CSV → slate; rejects entry-history files with a clear message |
| `fantasypros` | FP public API v2 client |
| `scoring` | FanDuel points from FP stat lines (half-PPR, bonuses, DST PA ladder over a distribution) |
| `matching` | name-first matching; team is a disambiguator only (FD/FP disagree ~18%) |
| `vegas` | Odds API → implied **team** totals; missing teams degrade per-team, not fatally |
| `distributions` | empirical actual/projected ratio pools, calibrated on 2025 (n=3,602) |
| `calibrate` | accuracy harness + distribution rebuild |
| `injuries` | 3-layer pipeline; practice participation + play probability escalate/de-escalate |
| `kickoffs` | nflverse kickoff times, per-game lock state |
| `nflcal` | season/week auto-detection, Tuesday 6am ET rollover |
| `optimize` | MIP candidate pools, diversity/stack/showdown constraints |
| `simulate` | correlated Monte Carlo (Gaussian copula over empirical ratios) |
| `objectives` | per-profile dollar-denominated weights |
| `field` | opponent field **ensemble** + ownership calibration + baselines + ranking |
| `lateswap` | lock-aware re-optimization of unlocked slots |
| `export` | FanDuel upload CSV + lineup cards |
| `contest_parse` | pasted contest page → lineups + measured ownership |
| `results` | result log, standings, in-season accuracy; league-scoped queries |
| `web` | FastAPI dashboard (calls the CLI internally — one code path) |
| `cli` | `build` / `swap` / `capture` / `standings` |

**Dashboard tabs:** Week (schedule picker, click a game or Sunday Main to configure) →
Build → Sunday swap → Capture results → Season.

## 5. Design rules (each from a specific v5 failure)

1. **No fabricated projections.** FP stat lines scored under FanDuel rules; no FPPG or
   salary-derived fallback. Missing projection halts the build. (v5 used `salary/550`.)
2. **Vegas enters once**, as implied team totals, bounded ±9%. (v5 double-counted.)
3. **Metrics come from the joint simulated distribution**, never summed player
   percentiles. (v5 overstated a lineup ceiling ~45 pts.)
4. **Objectives in dollars**, derived from real prize structure. (v5 stacked magic numbers.)
5. **Injuries produce an action** (keep/flag/remove), never a silent haircut.
6. **P(win) averaged over a field ensemble** (25 draws). A single draw swung 14.5%→21.7%.
7. **Ties split win credit.**
8. **Selection and evaluation separated** — argmax of noisy estimates is biased high.
9. **League and non-league are isolated** — league standings, season context, and
   ownership are scoped to the league contest name; public contests never pollute them.
10. **Fail loud** — schema drift, low match rate, proxy distributions all surface.

## 6. Key numbers

- Objective weights: **$3.44 season equity per +1 pt/week sustained**; divided by weeks
  remaining for the per-week weight (~$0.164/pt at week 0, rising as weeks run out).
  $12.84 per unit of weekly win probability. So **1 expected point ≈ 1.28% of win rate**.
- Winning a league week takes ~135–140 FD points (league average ~117).
- Brett's history: 8.5% weekly win rate (vs 8.2% random), 0.55–0.63 percentile finish —
  consistently good, rarely spiky. This is why Total Points suits him.
- Calibration: n=3,602 (2025 residuals). K and D have **no** calibrated pools — generic
  spread. Cells now carry `mean_ratio`; the simulator pins sampled means to it.
- FP projections correlate ~0.60 with outcomes (~36% of variance).

## 7. Hard-won lessons (do not re-derive)

- **Simulated means ran 14% hot** (RB +16%, TE +15%, WR +14%) because a distribution
  reconstructed from 7 quantiles plus an interpolated tail does not preserve the mean
  of its residuals. Fixed by storing `mean_ratio` per cell at calibration time and
  rescaling sampled ratios to match. **Distributions built before this must be rebuilt**
  — the CLI prints a loud banner if `mean_ratio` is absent. Regression-tested.
- **FanDuel showdown MVP costs 1.5x salary** (and scores 1.5x). Default was 1.0, which
  produced cap-illegal lineups FanDuel would reject. Now enforced in the optimizer, the
  opponent field, and reported salary.
- **2026 opens on WEDNESDAY Sep 9** (NE@SEA) because the Thursday game is in Melbourne
  (LAR/SF). Weekday bucketing must cover Wed and Fri, not just Thu/Sat/Sun/Mon.
- **FP injuries endpoint:** `nfl/injuries?season=YYYY` — season is a **query param**.
  The path form returns a CloudFront 403 that looks like an auth failure but is a
  nonexistent route.
- **Odds API team codes:** JAX vs FanDuel's JAC — canonicalize through `matching.norm_team`.
- **Tarball deploys clobbered the calibration three times.** Tarballs now exclude
  `data/distributions*.json` and `data/calibration*.json` entirely.
- **Box-only patches got reverted by the next tarball, repeatedly.** All fixes go into
  the sandbox source first, never directly on the box.
- **Units bug:** multiplying a *sustained* per-week rate by a single week's score
  inflated the reported edge ~21×. Regression-guarded in tests.
- **Baseline consistency:** scoring a baseline on one field draw while ranking
  candidates on a 25-field ensemble inverted the sign of the edge. Both go through
  `rank_candidates` now.
- **Root can't use Brett's SSH key** — the container's checkout uses an HTTPS remote.
- Git ownership: `git config --global --add safe.directory /srv/appdata/dfs/src-checkout`.

## 8. Remaining before Week 1 (2026-09-13)

Reordered per the second adversarial review: **fix correctness before automating it.**
Automating a flawed swap or showdown path just makes the flaw run unattended.

1. **Rotate both API keys** (FantasyPros + Odds) — exposed in chat transcripts.
2. **Rebuild distributions on the box** so `mean_ratio` exists (see §7). Until then
   every simulated figure is ~14% high.
3. **Independent evaluation must be genuinely comparable** — it currently drops
   measured ownership and evaluates only the selected lineup. Fix: identical
   conditioning, and evaluate the max-projection baseline on the same fresh paths.
4. **Showdown correctness** — enumerate every feasible (5-player set, MVP) combination
   rather than using a projection-only MIP pool; add a kicker scorer and request `K`
   from FantasyPros (kickers currently ingest but have no production projection path);
   MVP-aware candidate identity (done).
5. **Near-neighbour candidate search** — the pool requires 3-player differences, which
   excludes exactly the 1–2 player pivots that trade points for win probability.
6. **Late swap: slot-faithful, season context preserved, ensemble field, and condition
   on completed players' ACTUAL scores** rather than re-simulating them.
7. **Validate real salary and entry templates** for both classic and showdown.
8. **Full failure-mode dry run** before the Week 1 slate posts.
9. **Only then** automate: n8n reminders and alerts. Do **not** auto-apply swap
   recommendations.
10. **Backup coverage + NOC delta.**

**Post-launch backlog:**

- Immutable pre-lock snapshots (projections, raw API responses, salaries, injuries,
  Vegas, calibration hash, candidates, seeds) starting immediately — this is the
  foundation of any real validation.
- Prospective pre-registration of 2026 lineups and ablations: max-projection, no-Vegas,
  independent-outcomes, generic-field, measured-field. Bootstrap by week, never by
  simulation row.
- Empirical correlation estimation (current loadings are hand-selected, not fitted).
- Vegas tilt: keep only if a walk-forward ablation wins — it may duplicate information
  already inside FantasyPros projections.
- Per-opponent tendencies (~Week 6), gated on beating the generic prior out-of-sample.
- Season equity currently assigns all opponents the leader's total and ignores candidate
  variance; model E[season prize] and weekly prize separately, then sum.

**Cut from the four-week scope:** AI-assigned numerical deltas (a source-linked staleness
flagger only, and only if logs show real stale-projection failures), drive-level showdown
simulation, per-opponent random effects, and the "trailing → ceiling / leading → floor"
heuristic (a variance trap unless the late swap directly simulates remaining players
against actual scores and revealed opponent rosters).

## 9. Weekly operating rhythm (once n8n lands)

- **Tue/Wed** — slates post. Week tab → click Sunday Main (or a single game). Download
  that contest's player list from FanDuel, drop it on Build. Upload the CSV once; the
  slate library reuses it all week.
- **Thu–Sat** — nothing.
- **Sun 11:30 ET** — swap check (auto): fresh injuries, fresh Vegas, lock-aware
  re-optimization. Pushover alert if a swap is proposed.
- **Sun 15:45 / 19:45** — same, for the late and SNF windows.
- **Mon/Tue** — open the contest results page, ⌘A ⌘C, paste into Capture. Standings and
  objective weights update themselves.

**Manual by design:** FanDuel salary CSVs are downloaded by hand (ToS — no credentialed
automation), and lineups are entered/uploaded by Brett.

## 10. Open questions for review

- Is the "maximize projections for league play" conclusion right, or is there value
  being left on the table in the 11% weekly-prize component?
- What is the strongest validation achievable without historical FanDuel salary
  archives (they are patchy)?
- For showdown, what matters most that a full-slate optimizer gets wrong?
