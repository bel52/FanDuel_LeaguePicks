# Project File Structure

This is the complete file structure for the Enhanced FanDuel DFS Optimizer. Place files in these locations:

```
FanDuel_LeaguePicks/
│
├── app/
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # FastAPI application (enhanced_main.py)
│   ├── config.py                   # Settings configuration (enhanced_config.py)
│   ├── enhanced_optimizer.py       # Core optimization engine
│   ├── ai_integration.py          # AI analysis module (enhanced_ai_integration.py)
│   ├── data_monitor.py            # Real-time data monitoring
│   ├── auto_swap_system.py        # Automated player swapping
│   ├── cache_manager.py           # Redis cache management
│   ├── data_ingestion.py          # Data loading and processing
│   ├── formatting.py              # Output formatting utilities
│   ├── player_match.py            # Player name matching
│   └── kickoff_times.py           # Game schedule management
│
├── data/
│   ├── input/                     # Upload your CSV files here
│   │   ├── qb.csv                 # FantasyPros QB projections
│   │   ├── rb.csv                 # FantasyPros RB projections
│   │   ├── wr.csv                 # FantasyPros WR projections
│   │   ├── te.csv                 # FantasyPros TE projections
│   │   └── dst.csv                # FantasyPros DST projections
│   ├── output/                    # Generated outputs
│   │   ├── last_lineup.json       # Last generated lineup
│   │   └── swap_history.json      # Swap activity log
│   ├── targets/                   # FanDuel upload files
│   │   └── fd_target.csv          # Current lineup for upload
│   └── fantasypros/               # Additional FantasyPros data
│
├── logs/                          # Application logs
│
├── docker-compose.yml             # Docker services configuration
├── Dockerfile                     # Docker image definition
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .env                          # Your environment variables (create from .example)
├── setup.sh                      # Quick setup script
└── README.md                     # Your existing README (enhanced_readme.txt)
```

## Setup Instructions

1. **Clone/Update your repository:**
   ```bash
   cd ~/FanDuel_LeaguePicks
   ```

2. **Copy all the provided files to their locations:**
   - Rename `enhanced_main.py` to `main.py` in the app folder
   - Rename `enhanced_config.py` to `config.py` in the app folder
   - Rename `enhanced_ai_integration.py` to `ai_integration.py` in the app folder

3. **Create your .env file:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Make setup script executable:**
   ```bash
   chmod +x setup.sh
   ```

5. **Run the setup:**
   ```bash
   ./setup.sh
   ```

6. **Upload your FantasyPros CSV files to `data/input/`**

7. **Start optimizing:**
   ```bash
   # Generate lineup
   curl http://localhost:8010/optimize

   # Or with specific parameters
   curl "http://localhost:8010/optimize?game_type=h2h&lock=Josh%20Allen"
   ```

## Important Notes

- The `app/main.py` file should be the content from `enhanced_main.py`
- The `app/config.py` file should be the content from `enhanced_config.py`
- The `app/ai_integration.py` file should be the content from `enhanced_ai_integration.py`
- All other files use their names as provided
- Make sure Redis is running (handled by docker-compose)
- The system will automatically create sample data if needed

## Troubleshooting

If you encounter issues:

1. Check Docker is running: `docker ps`
2. View logs: `docker-compose logs -f`
3. Verify file permissions: `ls -la app/`
4. Ensure .env has your API keys
5. Test health endpoint: `curl http://localhost:8010/health`
