# FanDuel_LeaguePicks — DFS Optimizer (v6)

> **Status: v6 rebuild in progress.** The DFS optimizer core is being rebuilt.
> This README describes what is actually in the tree today and marks everything
> not yet built as **planned**. No backtest numbers or win-rate claims are made
> here because none have been validated.

## Overview

An on-demand NFL Daily Fantasy Sports (DFS) lineup optimizer for FanDuel,
oriented around a private friends league. It pulls free/low-cost data sources,
builds projections, runs an optimizer plus Monte Carlo simulation, and serves
results through a small FastAPI web app and a CLI. There is no scheduler — you
run it when you want lineups.

The v6 pass so far is **repo hygiene only**: dead modules pruned, the env
template cleaned up, and this honest README. Module rewrites come in later
phases.

## Architecture

| Layer | Where it lives today | Notes |
|-------|----------------------|-------|
| **Data layer** | `data_collector.py`, `vegas_data_collector.py`, `news_monitor.py`, `injury_opportunity_detector.py` | Free sources (nfl_data_py, ESPN hidden APIs, weather.gov, Sleeper), FantasyPros, and The Odds API for Vegas lines. Player salaries come from a manually downloaded FanDuel CSV (see automation boundary). |
| **Projection engine** | Embedded in `data_collector.py` | Builds base projections and feeds the optimizer. Not yet a standalone module — **the dedicated projection engine is planned** (the old `enhanced_projections.py` was removed as dead code in v6). |
| **Optimizer** | `optimizer.py` | Lineup construction with contest-type-aware logic, stacking, and ownership prediction. |
| **Simulation** | `monte_carlo_engine.py` | Monte Carlo scoring simulation; loaded behind a guarded import with a clean fallback. |
| **AI layer** | `WinningAIAnalyzer` in `data_collector.py` | Claude-only (Anthropic). Adjusts projections via must-play / must-fade signals. The richer `ai_analyzer_enhanced` module was removed in v6; its import is guarded and degrades to the basic analyzer. **Enhanced AI analysis is planned.** |
| **Delivery** | `app.py` (FastAPI web), `main.py` (CLI), `late_swap.py` / `late_swap_engine.py` | Web dashboard + CSV export; CLI modes for generating lineups. Late-swap engine present but not fully wired — **treat as planned.** |

### Web endpoints (`app.py`)

- `GET /` — dashboard (HTML)
- `GET /players` — current player pool
- `POST /optimize` — generate optimized lineups
- `GET /health` — health check

### CLI (`main.py`)

Modes: `gpp`, `cash`, `contrarian`, `friends_league`, `web`, `test`.

## Contest profiles

The v6 target profiles:

- **`friends_league`** — **active.** Has dedicated optimizer logic (GPP-style
  ceiling/stacking tuned for a small private league). This is the primary
  supported profile today.
- **`showdown_friends`** — **planned.** Single-game/showdown scoring is not yet
  a named profile. Single-game strategy weights exist in config, but the
  profile itself is not built.
- **`public_gpp`** — **planned.** A generic `gpp` mode exists in the CLI/optimizer,
  but a distinct large-field public GPP profile is not yet built.

> The code also currently exposes generic `gpp`, `cash`, `contrarian`,
> `bestball`, and `h2h` contest types (`config.py: CONTEST_TYPES`). These
> predate the v6 profile scheme and will be reconciled with the three profiles
> above in a later phase.

## Status (per phase)

- **Phase 0 — Repo hygiene (this pass): in progress.** Prune dead modules,
  clean `.env.example`, honest README. Deletion + cleanup only.
- **Phase 1 — Data + projection engine: planned.** Extract a standalone
  projection engine; harden data sources.
- **Phase 2 — Optimizer + simulation: partially present, needs review.**
  `optimizer.py` and `monte_carlo_engine.py` exist and import cleanly; no v6
  validation yet.
- **Phase 3 — AI layer (Claude-only): partial.** Basic `WinningAIAnalyzer`
  present; enhanced analysis planned.
- **Phase 4 — Contest profiles: partial.** `friends_league` active;
  `showdown_friends` and `public_gpp` planned.
- **Phase 5 — Delivery (web/CLI/late-swap): partial.** Web app + CLI work;
  late-swap not fully wired.

No backtests have been run for v6; there are no performance numbers to report.

## Setup

Requires Python 3.10+. Copy the env template and fill in keys:

```bash
cp .env.example .env
pip install -r requirements.txt
```

Key environment variables (see `.env.example` for the full list):

- `FANTASYPROS_API_KEY` — FantasyPros projections/rankings.
- `ANTHROPIC_API_KEY` — Claude (the only AI provider in v6).
- `ODDS_API_KEY` — The Odds API for Vegas lines. **Rotate this** — older keys
  leaked; generate a fresh free key at https://the-odds-api.com.
- `AI_WEEKLY_BUDGET_USD` — weekly AI spend cap (default `10`).

The app runs on free data sources with no keys set; AI analysis is skipped
without `ANTHROPIC_API_KEY`.

Run:

```bash
python main.py web              # start the web app (http://localhost:8020)
python main.py friends_league   # generate lineups from the CLI
```

## Honest automation boundary

**The FanDuel salary CSV is a manual download.** There is no FanDuel scraping
and no FanDuel credentials in v6. Export the salary CSV from FanDuel yourself
and save it as:

```
data/fanduel_salaries_manual.csv
```

(H2H slates use `data/fanduel_h2h_salaries.csv`.) Everything downstream —
projections, optimization, simulation, export — runs from that file. If the
CSV is missing, lineup generation stops with an error.
