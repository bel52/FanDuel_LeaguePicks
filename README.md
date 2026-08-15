# FanDuel LeaguePicks v6 — rebuild (Phase 1 in progress)

Honest status: data layer + projection engine under construction. Nothing here is
validated for real-money use yet. See DFS_Optimizer_v6_Project_Plan_v1.2.md.

## Modules (Phase 1)
- src/dfs/contest_spec.py — contest config (field, payouts, late-swap) as first-class input
- src/dfs/slate.py        — canonical PlayerSlate + SQLite store; fail-loud validation
- src/dfs/ingest_fanduel.py — FD salary CSV ingester + validation report
- src/dfs/fantasypros.py  — FP public API v2 client (projections/injuries/news/points)
- src/dfs/vegas.py        — Odds API → implied TEAM totals (Vegas applied exactly once)
- src/dfs/distributions.py — empirical outcome distributions, 2022–2025 nflverse (10,688 obs)
- src/dfs/blend.py        — projection blend: FP half-PPR + FD bonus expectation + bounded Vegas tilt

## Run tests (offline, no keys needed)
    pip install -r requirements.txt
    python3 -m pytest tests/ -q

## Smoke-test FantasyPros key
    export FANTASYPROS_API_KEY=...
    python3 -c "import sys; sys.path.insert(0,'src'); from dfs.fantasypros import FantasyProsClient; print(FantasyProsClient().smoke_test())"
