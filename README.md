# FanDuel LeaguePicks v6

Rebuilt DFS optimizer for Brett's Leather League (12 players, Total Points) plus
head-to-head, showdown, and public contests.

**Status:** data layer + modeling core + operational loop complete and tested.
Not yet containerized. See DFS_Optimizer_v6_Project_Plan_v1.2.md.

## Commands

    ./run.sh build --csv <fanduel.csv> --season 2026 --week 1 \
        --leaderboard total_scores --field 12 --weekly-prize 12.84 \
        --grand-prizes 135,81,54 --weeks-total 21 \
        --export data/upload.csv --log-db data/results.db

    ./run.sh capture <pasted_contest_page> --season 2026 --week 1
    ./run.sh standings --season 2026 --me xleathy

## Design rules (learned the hard way from v5)

1. Projections come from FantasyPros stat lines scored under FanDuel rules.
   There is no FPPG or salary-derived fallback anywhere — a missing projection
   stops the build.
2. Vegas enters the system exactly once, as implied TEAM totals. No downstream
   game-total boosts. (v5 double-counted this.)
3. Lineup metrics come from the joint simulated distribution, never from summing
   individual player percentiles. (v5 overstated a lineup ceiling by ~45 pts.)
4. Objectives are denominated in dollars and derived from the contest's real prize
   structure, not hand-tuned multipliers.
5. Injuries produce an ACTION (keep/flag/remove), never a silent projection haircut.
6. Every build reports baselines. An edge that isn't demonstrated is reported as
   NOT DEMONSTRATED.

## Modules

    contest_spec  contest config (field, payouts, late swap)
    slate         canonical PlayerSlate + SQLite store, fail-loud validation
    ingest_fanduel  FD salary CSV -> slate, with validation report
    fantasypros   FP public API v2 client
    scoring       FanDuel points from FP stat lines (half-PPR + bonuses + DST ladder)
    matching      name-first player matching (team is unreliable across sources)
    vegas         Odds API -> implied team totals
    distributions empirical outcome ratios, calibrated on 2025 residuals
    blend         projection assembly
    injuries      3-layer injury pipeline + Sunday inactives sweep
    optimize      MIP candidate-pool generation with diversity + stack constraints
    simulate      correlated Monte Carlo (Gaussian copula over empirical ratios)
    objectives    per-profile dollar-denominated objective weights
    field         opponent field model + baselines
    export        FanDuel upload CSV + lineup cards
    contest_parse pasted contest results -> lineups + measured ownership
    results       result log, season standings, in-season accuracy tracking
    cli           build / capture / standings

## Tests

    python3 -m pytest tests/ -q      # 57 offline tests, no keys needed
