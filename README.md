# FanDuel Sunday Main — Development README

> **Status:** Active development. Docker-first, local cron-friendly.
>
> **Scope:** FanDuel NFL **Sunday Main** only (1pm + 4:05/4:25pm ET). No TNF/SNF/MNF.
>
> **League:** 12-person friends league (beat 11 opponents weekly).
>
> **Data:** Paid = **FantasyPros (manual CSVs only)**. Free = **The Odds API** (spreads/totals), **NWS api.weather.gov** (weather), **ESPN scoreboard JSON** (live snapshot).
>
> **Philosophy:** **Simple, transparent, scriptable** (no opaque ML). Single-stack rule (QB + 1 WR/TE). Bring-back optional.

---

## 0) What this project does

This pipeline builds, monitors, and late-swaps a single FanDuel **Sunday Main** lineup each week:

1. **Wednesday 9:00 ET** — Deep Build: ingest FantasyPros cheat sheets (CSV), fetch odds/weather → make `board.csv` + first target lineup `data/targets/fd_target.csv`.
2. **Thu–Sat 10:00 ET** — Deltas: small bumps/penalties from line moves/weather → update target lineup.
3. **Sunday 11:30 ET** — Inactives: apply pivots, write final **executed** lineup `data/executed/fd_executed.csv`.
4. **\~2:15 ET** — Mid-Slate Review (at early-game halftime): compute status **AHEAD/EVEN/BEHIND** → `late_swap_plan.json`.
5. **3:55–4:05 ET** — Late-Swap Suggestions: cap/slot/schedule-aware swaps for late games → suggestions log.
6. **Monday 10:00 ET** — Post-mortem & standings/bankroll updates.

All times are **Eastern (Washington, DC)**.

---

## 1) Project structure (containers + bind-mounts)

This repo now supports both the **script pipeline** you’ve been using *and* a lightweight **FastAPI web service** for quick checks/testing.

```
~/fanduel/
├─ Dockerfile
├─ docker-compose.yml
├─ Makefile
├─ requirements.txt
├─ .env.example        # copy to .env and put your ODDS_API_KEY there
├─ README.md           # THIS FILE
├─ app/                # FastAPI service + shared analysis code
│  ├─ __init__.py
│  ├─ main.py          # API endpoints (/health, /schedule, /optimize, /optimize_text, /data/status)
│  ├─ config.py        # settings (input_dir/output_dir/log level/timezone)
│  ├─ data_ingestion.py
│  ├─ optimization.py
│  ├─ analysis.py
│  ├─ formatting.py    # text report builder for /optimize_text
│  ├─ kickoff_times.py # kickoff map + late-swap helpers (save/load last lineup)
│  └─ player_match.py  # fuzzy/utility matching
├─ src/                # Scripted pipeline (unchanged intent)
│  ├─ __init__.py
│  ├─ util.py          # logging, csv helpers, current_week, ValuePer1k
│  ├─ data_fetch.py    # FantasyPros CSV loader, Odds (The Odds API)
│  ├─ lineup_rules.py  # thresholds, weather/vegas adjustments
│  ├─ lineup_builder.py# greedy search (single stack, cap, validation hooks)
│  ├─ postmortem.py    # standings & bankroll writers
├─ scripts/
│  ├─ deep_build.py            # Wed build (board + target)
│  ├─ update_deltas.py         # Thu–Sat deltas
│  ├─ process_inactives.py     # Sun 11:30 ET → executed lineup
│  ├─ review_early_games.py    # ~2:15 ET mid-slate status -> late_swap_plan.json
│  ├─ late_swap.py             # 3:55–4:05 ET suggestions (cap/slot/late-only)
│  ├─ validate_target.py       # schema/salary/stack checks
│  ├─ inspect_fp_raw.py        # quick peek at FP CSV columns/samples
│  └─ inspect_candidate_pool.py# counts after value thresholds
├─ data/
│  ├─ fantasypros/             # DROP 5 CSVs HERE: qb.csv rb.csv wr.csv te.csv dst.csv
│  ├─ weekly/2025_wXX/         # per-week board + late swap artifacts
│  ├─ targets/                 # fd_target.csv (latest target lineup)
│  ├─ executed/                # fd_executed.csv (final submitted)
│  ├─ season/                  # standings.csv (header or rolling log)
│  └─ bankroll/                # bankroll.csv (header or rolling log)
└─ logs/                       # script logs by week
```

> **Note:** The `app/` folder is new for the API. The script flow in `src/` + `scripts/` remains the source of truth for the weekly pipeline.

---

## 2) Data inputs (FantasyPros) — how to export and where to drop

**FantasyPros → Cheat Sheets → FanDuel → NFL → Positions (QB/RB/WR/TE/DST)**

1. For each position page, set **Site = FanDuel**, **Slate filter = Sun Main**, then **copy the grid to CSV** (FantasyPros shows a **“Copy CSV”** action near the table header on Cheat Sheets).
2. Save files exactly as:

   * `data/fantasypros/qb.csv`
   * `data/fantasypros/rb.csv`
   * `data/fantasypros/wr.csv`
   * `data/fantasypros/te.csv`
   * `data/fantasypros/dst.csv`
3. Each file should have columns like:
   `"PLAYER NAME","OPP","KICKOFF","WX","SPREAD","O/U","PRED SCORE","PROJ RANK","$ RANK","RANK DIFF","PROJ PTS","SALARY","CPP","PROJ ROSTER %"`

**Filtering done by the loader**

* Only **Sun 1:00PM**, **Sun 4:05PM**, **Sun 4:25PM** rows are kept (no TNF/SNF/MNF).
* Positions are inferred from `PLAYER NAME` (e.g., `(BAL - QB)` → `QB`).
* `ProjFP` and `Salary` are parsed from `PROJ PTS` and `SALARY`.

**Quick verification**

```bash
# raw column/header check
docker compose run --rm bot python scripts/inspect_fp_raw.py

# loader output quick glance
docker compose run --rm bot python scripts/inspect_loader_output.py
```

---

## 3) Free endpoints wired in

* **Odds (The Odds API)**: spreads + totals (FanDuel market)
  `https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?regions=us&oddsFormat=american&markets=spreads,totals&apiKey=$ODDS_API_KEY`
  → Used to infer **implied team totals** per game (stored into `board.csv`).
* **Weather (NWS)**: `https://api.weather.gov/points/{lat},{lon}` → forecast JSON; wind/precip tiers (hook points exist; optional until we add stadium coords).
* **Live snapshot (ESPN)**: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard` (unofficial) → used for **mid-slate status** heuristic.

Set your key:

```bash
cp .env.example .env
# put your Odds key
echo 'ODDS_API_KEY=YOUR_KEY_HERE' > .env
```

---

## 4) Scoring & math (implemented)

* `ValuePer1k = Proj / (Salary/1000)`
* Weather modifiers applied in `lineup_rules.py` (QB/WR down in wind ≥15; RB/DST up in heavy precip/wind).
* `adjusted_score()` adds light vegas/weather nudges; thresholds tunable in `lineup_rules.py`.
* **Candidate pool gate:** `ValuePer1k ≥ 1.8` (TE/DST ≥ 1.7). (You may temporarily relax to 1.6/1.5 for testing.)

**Lineup construction rules**

* **Single stack**: exactly one pass-catcher (WR/TE) with your QB.
* Bring-back optional if projection within \~1 pt and salary fits.
* Roles: RB1 volume; RB2 value; WR1 = stack mate; WR2 best AdjScore at/under median; WR3 ceiling or value; TE elite if fits else value; FLEX best remaining; avoid DST vs your QB.
* Cap ≤ **\$60,000**; ok to leave \$200–\$800.

**Ownership proxy (tiebreaker only)**

```
OwnershipEstimate = BasePosRate * (Salary/PosAvgSalary)^0.7
                    * (1 + (ImpliedTeamTotal - 21)/100) * NewsMultiplier
BasePosRate: QB 0.10, RB 0.18, WR 0.15, TE 0.08
NewsMultiplier: 0.8 .. 1.1 (manual)
```

---

## 5) Run cadence (ET) + scripts

| Time (ET)     | Script                          | Output                                          | Purpose                                 |
| ------------- | ------------------------------- | ----------------------------------------------- | --------------------------------------- |
| Wed 09:00     | `scripts/deep_build.py`         | `weekly/.../board.csv`, `targets/fd_target.csv` | First target lineup + vegas board       |
| Thu–Sat 10:00 | `scripts/update_deltas.py`      | updated `targets/fd_target.csv`                 | Small adjustments from odds/weather     |
| Sun 11:30     | `scripts/process_inactives.py`  | `executed/fd_executed.csv`                      | Applies inactives; locks lineup file    |
| Sun \~2:15    | `scripts/review_early_games.py` | `late_swap_plan.json`                           | Status AHEAD/EVEN/BEHIND for late swaps |
| Sun 3:55–4:05 | `scripts/late_swap.py`          | suggestions log                                 | Slot/cap/schedule-aware pivots          |
| Mon 10:00     | `scripts/postmortem_run.py`     | season/bankroll rows                            | Post-mortem & bookkeeping               |

**Manual steps**

* **Wed**: Export FP CSVs → drop in `data/fantasypros/` → run Deep Build (or let cron).
* **Sun 11:30**: read inactives summary in log; confirm lineup in FanDuel.
* **\~2:15**: open plan; if **BEHIND** and you have late slots → apply upside pivots; if **AHEAD** → apply safer pivots.
* Keep the **single stack** unless forced.

---

## 6) Docker usage (no venv needed)

Build once:

```bash
docker compose build
```

Common runs (script pipeline):

```bash
docker compose run --rm bot python scripts/deep_build.py
docker compose run --rm bot python scripts/update_deltas.py
docker compose run --rm bot python scripts/process_inactives.py
docker compose run --rm bot python scripts/review_early_games.py
docker compose run --rm bot python scripts/late_swap.py
docker compose run --rm bot python scripts/validate_target.py
```

Inspect outputs quickly:

```bash
tail -n 120 logs/deep_build_week*.log
column -s, -t < data/targets/fd_target.csv | sed -n '1,50p'
column -s, -t < data/executed/fd_executed.csv | sed -n '1,50p'
```

---

## 7) Web API (FastAPI service)

A small API is included for health checks, quick text output, and future UI hooks.

**Compose port mapping**: container listens on `8000`; host exposes **`8010`** (see `docker-compose ps`).

**Start/refresh**

```bash
docker compose up -d --build
# check health
curl -s http://localhost:8010/health | python3 -m json.tool
```

**Endpoints**

* `GET /health` → JSON status, including player data availability
* `GET /data/status` → JSON file presence counts and position breakdowns
* `GET /schedule` → JSON `{ timezone, kickoffs, auto_locked_from_last_lineup }`
* `GET /optimize` → JSON lineup object (accepts query params like `salary_cap`, `enforce_stack`, `min_stack_receivers`, `lock`, `ban`, `auto_late_swap`)
* `GET /optimize_text` → **plain-text** lineup report (query param `width`)

**Examples**

```bash
# JSON health (use jq or json.tool)
curl -s http://localhost:8010/health | python3 -m json.tool

# Schedule (JSON)
curl -s http://localhost:8010/schedule | python3 -m json.tool

# Text lineup (plain text)
curl -s "http://localhost:8010/optimize_text?width=110"

# JSON lineup (with optional locks/bans)
curl -s "http://localhost:8010/optimize?lock=Patrick%20Mahomes&ban=Some%20DST" | python3 -m json.tool
```

> **Why you might see** `Expecting value: line 1 column 1 (char 0)`
>
> That error means you tried to pipe **plain text** or a 404 **HTML/Not found** page into a JSON parser. Fixes:
>
> * Use the correct **host port 8010** (not 8000 on host).
> * Only pipe JSON endpoints (`/health`, `/optimize`, `/data/status`, `/schedule`) into `jq`/`json.tool`.
> * Call text endpoints like `/optimize_text` **without** piping to a JSON tool.

---

## 8) Cron (ET) with Docker

Edit user crontab (`crontab -e`) and add:

```
# Wed Deep Research
0 9 * * WED   cd ~/fanduel && docker compose run --rm bot python scripts/deep_build.py         >> logs/build_board.log 2>&1
# Thu–Sat daily deltas
0 10 * * THU,FRI,SAT  cd ~/fanduel && docker compose run --rm bot python scripts/update_deltas.py >> logs/pull_deltas.log 2>&1
# Sun inactives (90 min pre-lock)
30 11 * * SUN  cd ~/fanduel && docker compose run --rm bot python scripts/process_inactives.py  >> logs/inactives.log 2>&1
# Sun mid-slate review (~2:15 ET)
15 14 * * SUN  cd ~/fanduel && docker compose run --rm bot python scripts/review_early_games.py >> logs/mid_slate.log 2>&1
# Sun late-swap suggestions (before 4:05/4:25)
55 15 * * SUN  cd ~/fanduel && docker compose run --rm bot python scripts/late_swap.py          >> logs/late_swap.log 2>&1
```

> **Note:** Host cron is simpler and more reliable than running cron inside a container for this workflow.

---

## 9) Troubleshooting & QA

* **No lineup produced?**

  * Verify 5 CSVs exist and have rows: `ls -lh data/fantasypros && wc -l data/fantasypros/*.csv`
  * Inspect headers/samples: `docker compose run --rm bot python scripts/inspect_fp_raw.py`
  * See candidate pool after thresholds: `docker compose run --rm bot python scripts/inspect_candidate_pool.py`
  * Temporarily relax thresholds in `src/lineup_rules.py` (e.g., 1.8→1.7) and rebuild.

* **Validate constraints**
  `docker compose run --rm bot python scripts/validate_target.py`

* **Late-only suggestions**
  The builder writes a **Kick** column (e.g., `Sun 4:05PM`). Late-swap script restricts to those.

* **API 404 or JSON parse errors**

  * Confirm the mapped port: `docker compose ps` should show `0.0.0.0:8010->8000/tcp`.
  * If you curl `http://localhost:8000` on host, you’ll hit **Not found**; use `http://localhost:8010` instead.
  * Only pipe JSON endpoints to `jq`/`json.tool`; `/optimize_text` is plain text by design.

---

## 10) Security & data hygiene

* **Never commit secrets**. `.env` is git-ignored.
* **Never commit raw FantasyPros CSVs**. `data/fantasypros/` is git-ignored.
* Logs and generated artifacts can grow; rotate or prune old weeks as needed.

---

## 11) Roadmap (nice-to-haves)

* Stadium coordinates → NWS wind/precip tiers applied per game.
* Ownership proxy surfaced in tie-breakers.
* Bring-back logic (optional within \~1 pt).
* Simple UI summary (`rich`/markdown) and HTML export.

---

## 12) License & disclaimers

For personal/league use. Respect all third-party terms (FantasyPros, The Odds API, NWS, ESPN). No guarantees; this is an engineering helper, not betting advice.
