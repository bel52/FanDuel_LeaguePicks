
# FanDuel NFL DFS Automation (Docker) — Updated

This project builds a **reproducible, low-cost, Dockerized DFS stack** that:
- Normalizes your weekly **FantasyPros** CSV exports (QB/RB/WR/TE/DST).
- **Optimizes** FanDuel lineups via **OR-Tools** under roster & cap rules.
- Runs a **Monte Carlo simulation** to quantify risk/ceiling (mean, std dev, percentiles, Sharpe).
- Produces **readable console & API reports** with **AI analysis** (GPT-4o-mini) behind a cache to control cost.
- Optionally enforces **stacking** rules (e.g., QB must have ≥1 WR/TE from same team).
- Isolates secrets via `.env` and **never** commits them to Git.

> Cost focus: defaults to `gpt-4o-mini` + multi-layer caching (in-memory + Redis). Typical usage stays at or under **$10–$15/week** with repeated calls benefiting from cache hits.

---

## Quick Start

**Prereqs**: Docker & Docker Compose

```bash
# 1) Place weekly CSVs
data/input/qb.csv
data/input/rb.csv
data/input/wr.csv
data/input/te.csv
data/input/dst.csv

# 2) Set secrets in .env (do NOT commit this file)
# REQUIRED
OPENAI_API_KEY=sk-...

# OPTIONAL (only if you later enable scraping)
FANTASYPROS_USER=...
FANTASYPROS_PASS=...

# Redis inside compose network
REDIS_URL=redis://redis:6379
CACHE_TTL=300
GPT_MODEL=gpt-4o-mini

# 3) Build & run
docker compose up -d --build

# 4) Healthcheck
curl -s http://localhost:8010/health
# -> {"status":"ok"}
```

---

## Using the App

### Endpoints
- **`GET /optimize`** → JSON (lineup, totals, simulation, analysis)
- **`GET /optimize_text`** → **formatted plain text** table + analysis (great for CLI)
  - Query params on both endpoints:
    - `salary_cap` (default `60000`)
    - `enforce_stack` (`true|false`, default `false`)
    - `min_stack_receivers` (`1..3`, default `1`)
    - `width` (`70..160`, for `/optimize_text` only)

**Examples**:
```bash
# Pretty text with default cap
curl -s "http://localhost:8010/optimize_text?width=110"

# Force a stack (QB with at least 1 same-team WR/TE)
curl -s "http://localhost:8010/optimize_text?enforce_stack=true&min_stack_receivers=1&width=110"

# Reduced cap example (59.5k) — useful to test sensitivity
curl -s "http://localhost:8010/optimize_text?salary_cap=59500&width=110"
```

### CLI (inside the container)
```bash
docker compose exec web python -m app.cli --enforce-stack --min-stack-receivers 1 --width 110
```

---

## Data Flow

1) You export weekly **FantasyPros** CSVs and drop them into `data/input/` using the exact names:
   `qb.csv, rb.csv, wr.csv, te.csv, dst.csv`.
2) The **normalizer** (`app/normalizer.py`) reconciles column names/variants, parses player info from
   `"Player Name (TEAM - POS)"`, cleans salary & projection fields, and estimates numeric ownership (`OWN_PCT`)
   when the CSV carries strings like `"20-30%-"`.
3) The **optimizer** (`app/optimization.py`) builds a CP-SAT model for FanDuel NFL:
   - Roster: `QB(1), RB(2), WR(3), TE(1), FLEX(1 of RB/WR/TE), DST(1)` → **9 total**
   - Cap: default **$60,000** (configurable)
   - Optional **stacking**: if on, the chosen QB must have ≥ `min_stack_receivers` WR/TE from the same team.
   - Objective: **maximize projected points**.
4) The **simulator** (`app/analysis.py`) runs Monte Carlo on the chosen lineup:
   - Each player’s outcome ~ Normal(mean = projection, stdev ≈ 0.15 × projection; floored at 0).
   - Aggregates lineup totals across 10,000 runs.
5) **AI analysis** (`app/openai_utils.py`) produces a concise, sectioned readout (correlation, leverage, risk/ceiling,
   strengths, weaknesses, swap ideas). Calls are cached in memory and Redis to control cost.
6) Results are rendered as either **JSON** or a **console-friendly table with headings and bullets**.

> If you prefer automated collection from FantasyPros later, we can add Playwright-based scraping that only runs when CSVs are missing. It will use `FANTASYPROS_USER`/`FANTASYPROS_PASS` from `.env` and keep auth state out of Git via `.gitignore`.

---

## Simulation Summary — What the Numbers Mean

- **Mean** — The average total points across all simulations. Higher mean = stronger central expectation.
- **StdDev** — How volatile the lineup is. Higher std dev = wider range of outcomes (riskier).
- **P50 (Median)** — Middle outcome (50% of sims score above this; 50% below).
- **P90** — A ceiling-ish outcome (top ~10% of sims beat this). Good proxy for tournament spike.
- **P95** — Even more ceiling (top ~5%).
- **Sharpe (heuristic)** — `Mean / StdDev`. A quick risk-adjusted score; higher is “more points per risk.”

### StdDev Rule-of-Thumb Scale (NFL, per lineup)

| StdDev | Interpretation                              | Typical Use Case             |
|:------:|---------------------------------------------|------------------------------|
|  0–5   | **Low** volatility (cash-game / safe-ish)   | Head-to-head, double-ups     |
|  5–7   | **Moderate** volatility                     | Small/medium GPPs            |
|  7–9   | **High** volatility                         | Large-field tournaments       |
|  9+    | **Very high** volatility (boom/bust)        | Milli-maker style shots       |

> These are heuristics; they shift with slate size and projection sources. Compare **relative** std devs between your candidate lineups on the same slate.

---

## How the Analysis Works (under the hood)

- **Projections & Ownership**: from your FantasyPros CSVs (normalized). Ownership strings are converted to numeric estimates when possible.
- **Optimization**: OR-Tools CP-SAT solver maximizes projected points subject to FanDuel constraints and any optional stacking constraints you pass.
- **Simulation**: 10,000-run Monte Carlo with player-level variance to quantify risk and tail outcomes.
- **AI Layer**: GPT-4o-mini (cheap + capable) with a structured prompt that returns five sections:
  - **CORRELATION** (stacks/bring-backs), **LEVERAGE** (chalk vs contrarian), **RISK & CEILING**, **STRENGTHS**, **WEAKNESSES**, and **SWAP IDEAS** (max 2 legal swaps).
- **Caching**: Two-level cache (RAM + Redis) deduplicates identical prompts to keep your OpenAI bill low.

---

## Swap Ideas — What Are They? Do They Replace the Optimal?

- The optimizer returns the **single best** lineup for the given constraints (max projected points).
- **Swap Ideas** come from the AI layer. They are **human-style alternatives** that may:
  - improve **correlation** (e.g., pair QB with WR/TE),
  - change **leverage** (reduce chalk, raise uniqueness),
  - or address **risk/ceiling** preferences.
- They are **not guaranteed** to be the strict “second-best” solution by projection.
- If you want true **next-best (K-best)** lineups by the solver, we can add a feature that iteratively finds the top N optimal solutions via “no-good” cuts, and list them (e.g., Top-10 by projection).

**Bottom line**: The model “prefers” the reported optimal lineup under the current objective (max projection). **Swap ideas** surface *strategic* alternatives that may **trade a small amount of projection** for **better correlation/leverage** — useful for GPPs.

---

## Late Swap Workflow

When news breaks (inactives, role changes), you can **late swap** quickly:

1) **Lock already-started players**: add a constraint to keep them fixed (feature can be toggled in optimization; we can expose this as flags such as `--lock "Player Name"` or via an endpoint — easy to add).
2) Re-run the optimizer for remaining slots (same cap rules, optionally update stacking).
3) Re-check **SIMULATION SUMMARY** to ensure risk/ceiling fits your contest type.
4) Use **/optimize_text** for a quick, readable report you can act on immediately.

> If you want fully automated late-swap with a watcher (e.g., check injuries/actives every 5 minutes and re-optimize), say the word — we can add a scheduler container and “lock by kickoff” logic.

---

## Security & Cost Control

- Secrets live only in `.env`. **Never** commit `.env` or auth state files; `.gitignore` already excludes them.
- Caching (RAM + Redis) means **repeat AI calls are almost free**.
- Default model `gpt-4o-mini` balances quality/cost. You can change `GPT_MODEL` in `.env` if needed.

**Check for accidental secrets before pushing**:
```bash
# Look for common API key patterns
grep -RIn --binary-files=without-match -E 'sk-[A-Za-z0-9_\-]+' .

# If you ever pasted your FantasyPros password or email in code, scan for it too:
grep -RIn --binary-files=without-match -E 'fantasypros|password|@' app || true
```

---

## Repo Hygiene / Updating GitHub

Make sure your **remote** is set to the GitHub repo (example: `bel52/FanDuel_LeaguePicks`):
```bash
git remote -v
# If needed:
# git remote add origin https://github.com/bel52/FanDuel_LeaguePicks.git
```

Verify `.gitignore` includes:
```
.env
fantasypros_auth.json
data/input/*
data/output/*
```

Commit & push **only safe files**:
```bash
git add -A
git status         # sanity check (ensure .env is NOT listed)
git commit -m "Enhance DFS stack: readable CLI + /optimize_text, Monte Carlo docs, AI sections, caching"
git push origin HEAD:main  # or 'master' depending on your repo
```

> If your default branch is different, replace `main` accordingly. If you get a rejection, pull first: `git pull --rebase` and retry the push.

---

## Troubleshooting

- **`/optimize` returns “No data available”** → ensure all 5 CSVs exist with the exact names in `data/input/`.
- **OpenAI errors** → check `OPENAI_API_KEY` in `.env` and that the container was restarted after edits.
- **Port conflicts** → if 8010 is busy on your host, change the left side of the mapping in `docker-compose.yml`:
  ```yaml
  ports:
    - "8015:8000"
  ```
- **Redis not reachable** → the app will still run with in-memory cache only; performance/cost may be slightly worse.

---

## Roadmap (optional enhancements)

- True **Top-N (K-best)** lineups via solver enumeration.
- **Late-swap** helper flags and a scheduled watcher service.
- Vegas lines & weather context.
- Exposure/ownership constraints and team-level stacking caps.
- Export to CSV/Markdown/Slack bot message.
