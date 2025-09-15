# 🏈 FanDuel League Picks – NFL DFS Optimizer

This project is a **production-ready Daily Fantasy Sports (DFS) optimizer** for FanDuel NFL contests.  
It generates **the best possible lineups** for **Head-to-Head (H2H)** and **GPP (tournaments)** using real-time data and advanced analysis:

- **AI-Powered Analysis** – GPT-4o or Claude Sonnet explain lineup choices (stacking, leverage, contrarian picks, weather, etc.)
- **Monte Carlo Simulation** – 50,000+ simulations estimate upside, variance, and risk (mean score, percentiles, Sharpe ratio).
- **Multi-Source Data** – ESPN, Sleeper, Vegas odds, Weather.gov, and free APIs (no FantasyPros dependency).
- **Linear Programming Optimizer** – Ensures valid lineups under FanDuel rules (salary cap, roster positions, stacking).
- **User-Friendly CLI** – No raw curl commands. You get an interactive text menu to pick contest type and game slate.

---

## 🚀 Features

- ✅ **No FantasyPros dependency** – all data is from free APIs.
- ✅ **Real active rosters only** – practice squad / retired players filtered out.
- ✅ **Contest type selection** – optimize for GPP (upside) or H2H (floor).
- ✅ **Advanced lineup logic** – QB-WR stacking, max 3 teammates per QB, leverage adjustments.
- ✅ **AI explanations** – why the lineup works, what risks exist, contrarian plays.
- ✅ **Simulation summary** – variance profile and ceiling outcomes.
- ✅ **Dockerized** – runs consistently on Ubuntu/Docker Compose.
- ✅ **Transparent cost tracking** – shows API calls and estimated AI cost (usually <$0.10/week).

---

## 📂 Project Structure

```
app/
  cli.py              # Interactive CLI tool
  data_ingestion.py   # Load projections, filter active players
  enhanced_optimizer.py
  ai_analysis.py
  formatting.py
  ...
data/
  input/              # CSVs for QB/RB/WR/TE/DST projections + salaries
  output/             # Generated lineups
config/               # Settings and .env
logs/                 # Runtime logs
```

---

## ⚙️ Setup Instructions

### 1. System Requirements
- Ubuntu 20.04+ (or compatible Linux/Mac)
- **8 GB RAM, 4 CPU cores** recommended
- Docker + Docker Compose installed

```bash
docker --version
docker-compose --version
```

### 2. Clone and Prepare Repo
```bash
git clone https://github.com/bel52/FanDuel_LeaguePicks.git
cd FanDuel_LeaguePicks
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
nano .env
```

At minimum, add your **OpenAI API key**:
```
OPENAI_API_KEY=sk-your-key
```

Optional keys (recommended):
- **ODDS_API_KEY** – [the-odds-api.com](https://the-odds-api.com) (Vegas odds)
- **ANTHROPIC_API_KEY** – for Claude Sonnet 4 analysis (optional, OpenAI is enough)

### 4. Add Weekly Projection Data
Place FanDuel salary/projection CSVs in `data/input/`:

```
data/input/qb.csv
data/input/rb.csv
data/input/wr.csv
data/input/te.csv
data/input/dst.csv
```

> These can come from FanDuel contest exports or scraped cheat sheets.  
> The system will normalize column names automatically.

---

## ▶️ Running the Optimizer

### Option A: CLI (inside repo)

```bash
# Activate virtual environment or run inside Docker container
python -m app.cli
```

You will see:

```
0. All Games (Full Slate)
1. DET @ CHI - Sun 1:00 PM ET
2. DAL @ PHI - Sun 4:25 PM ET
...

Select a game number for the lineup slate (or 'q' to quit):
```

Then select **contest type**:

```
Choose contest type - (G)PP Tournament or (H)ead-to-Head:
```

The optimizer will:
1. Filter inactive players
2. Optimize best lineup
3. Simulate 50k outcomes
4. Provide an AI explanation

### Example Output
```
==============================================
🏈 FanDuel DFS Lineup (GPP)
----------------------------------------------
QB   Jalen Hurts    PHI   $8600   Proj: 24.3
RB   Tony Pollard   DAL   $7300   Proj: 18.2
RB   Breece Hall    NYJ   $7400   Proj: 16.7
WR   A.J. Brown     PHI   $8400   Proj: 22.1
WR   CeeDee Lamb    DAL   $8800   Proj: 23.4
WR   Drake London   ATL   $6700   Proj: 14.8
TE   Dallas Goedert PHI   $5200   Proj: 11.0
FLEX James Cook     BUF   $6800   Proj: 15.6
DST  ARI DST        ARI   $3000   Proj: 7.0
----------------------------------------------
Salary Used: $59,800 | Remaining: $200
Total Proj Points: 152.1
----------------------------------------------
📊 Simulation Summary:
Mean: 149.8 | Std Dev: 18.2
50th: 150.2 | 90th: 172.4 | 95th: 180.3
Sharpe Ratio: 8.2
----------------------------------------------
🤖 AI Analysis:
- Strong QB-WR correlation stack (Hurts + Brown + Goedert).
- Game total PHI-DAL projects as highest on slate.
- Contrarian FLEX play (Cook) adds leverage.
- DST punt allows high ceiling while staying under cap.
----------------------------------------------
(AI API calls used: 1, approx. cost: $0.0012)
==============================================
```

---

## 🐳 Running with Docker

```bash
docker compose up -d --build
docker compose exec app python -m app.cli
```

Health check:
```bash
curl -s http://localhost:8000/health | jq .
```

---

## 🧪 Testing

### 1. Health & Data
```bash
curl -s http://localhost:8000/players/current | jq .
# Should list 100+ players
```

### 2. Optimization
```bash
python -m app.cli
# Run through menu → should return a full lineup
```

### 3. AI Integration
```bash
# Ensure OPENAI_API_KEY in .env
python -m app.cli
# Output should include "🤖 AI Analysis"
```

---

## 🔧 Troubleshooting

- **“No players found”**  
  → Check CSVs in `data/input/` are present and valid.

- **AI analysis missing**  
  → Ensure `OPENAI_API_KEY` is in `.env` and container restarted.

- **Optimization failed**  
  → Verify player pool isn’t too small (single-game slates may revert to full).

- **Cost concerns**  
  → GPT-4o-mini is used by default (~$0.10/week). Costs shown after each run.

---

## 📈 Roadmap

- [ ] Web UI (FastAPI frontend)
- [ ] Live injury/ownership updates
- [ ] Automated weekly lineup export
- [ ] Bankroll/contest management

---

## 📜 License

MIT License – free to use and adapt. Not affiliated with FanDuel, ESPN, or Sleeper.
