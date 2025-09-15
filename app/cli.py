import os
import json
import re
import asyncio
import logging
import requests
import pandas as pd

# Load data ingestion and optimization modules
from app.data_ingestion import load_weekly_data
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.formatting import build_text_report

# Optional: reduce log verbosity for CLI output
logging.basicConfig(level=logging.ERROR)

# Sleeper API endpoint for NFL players
SLEEPER_PLAYERS_API = "https://api.sleeper.app/v1/players/nfl"

def _fetch_game_options():
    """Fetches upcoming NFL games from ESPN to allow slate selection."""
    options = [{"teams": None, "label": "All Games (Full Slate)"}]
    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        # Use Eastern Time for consistency with DFS lock times
        from dateutil import parser as dateparser
        import pytz
        eastern = pytz.timezone("America/New_York")
        for event in data.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            teams = comp.get("competitors", [])
            if len(teams) == 2:
                away = teams[0].get("team", {}).get("abbreviation", "").upper()
                home = teams[1].get("team", {}).get("abbreviation", "").upper()
                if not away or not home:
                    continue
                # Format game time
                dt_str = comp.get("date", "")
                try:
                    dt_obj = dateparser.parse(dt_str).astimezone(eastern)
                    time_str = dt_obj.strftime("%a %I:%M %p ET")
                except Exception:
                    time_str = "TBD"
                options.append({
                    "teams": [away, home],
                    "label": f"{away} @ {home} - {time_str}"
                })
    except Exception as e:
        print(f"WARNING: Could not fetch games from ESPN: {e}. Defaulting to full slate.")
    return options

def _filter_active_players(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out players not on active NFL rosters using Sleeper API data."""
    try:
        resp = requests.get(SLEEPER_PLAYERS_API, timeout=10)
        resp.raise_for_status()
        players_data = resp.json()
    except Exception as e:
        print(f"WARNING: Could not retrieve player status from Sleeper API ({e}). Skipping active roster filter.")
        return df  # return unfiltered if API call fails

    # Build set of active player names (normalized)
    active_names = set()
    for player in players_data.values():
        status = player.get('status')
        team   = player.get('team')
        if status is None:
            continue  # skip if no status info
        # Consider players who are on an active roster (status "Active") and have a team
        if status.lower() == "active" and team:
            # Construct full name and normalize (remove punctuation, lowercase)
            full_name = player.get('full_name')
            if not full_name:
                first = player.get('first_name', '')
                last = player.get('last_name', '')
                full_name = f"{first} {last}".strip()
            name_key = re.sub(r'[^A-Za-z0-9 ]+', '', full_name).strip().lower()
            if name_key:
                active_names.add(name_key)
    if not active_names:
        # If Sleeper returned nothing useful, skip filtering
        return df

    # Filter dataframe to only keep players whose normalized name is in active_names
    def is_active_player(name: str) -> bool:
        if not isinstance(name, str):
            return False
        norm = re.sub(r'[^A-Za-z0-9 ]+', '', name).strip().lower()
        return norm in active_names

    initial_count = len(df)
    df_active = df[df["PLAYER NAME"].apply(is_active_player)].copy()
    filtered_count = len(df_active)
    removed = initial_count - filtered_count
    if removed > 0:
        print(f"Filtered out {removed} inactive/invalid players. Active players remaining: {filtered_count}.")
    return df_active

def main():
    # 1. Load projection data
    df, warnings = None, []
    try:
        df = load_weekly_data()
    except Exception as e:
        warnings.append(f"ERROR: Failed to load player data - {e}")
    # Print any data load warnings (e.g., missing files)
    for w in warnings:
        print(w)
    if df is None or df.empty:
        print("\nFATAL: No player projection data available. Please add data to 'data/input' and retry.")
        return

    # 2. Filter out non-active players (practice squad, retired, etc.)
    df = _filter_active_players(df)
    if df.empty:
        print("\nFATAL: No active roster players available after filtering. Exiting.")
        return

    # 3. Let user select game slate (full slate or a specific game)
    game_options = _fetch_game_options()
    for i, opt in enumerate(game_options):
        print(f"{i}. {opt['label']}")
    while True:
        choice = input("Select a game number for the lineup slate (or 'q' to quit): ").strip().lower()
        if choice == 'q':
            return  # user wants to exit
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(game_options):
                selected_game = game_options[idx]
                break
        print("Invalid choice. Please enter a valid game number or 'q'.")

    # 4. Choose contest type: GPP (Tournament) or H2H
    while True:
        ct = input("Choose contest type - (G)PP Tournament or (H)ead-to-Head: ").strip().lower()
        if ct in ['g', 'h']:
            break
        print("Invalid choice. Enter 'g' for GPP or 'h' for H2H.")
    contest_type = "league" if ct == 'g' else "h2h"

    # 5. Prepare player pool based on selected game
    player_pool = df.copy()
    if selected_game.get("teams"):
        teams = selected_game["teams"]
        player_pool = df[df["TEAM"].isin(teams)].copy()
        if len(player_pool) < 18:
            # If not enough players (e.g., single game), revert to full slate
            print(f"WARNING: Only {len(player_pool)} players available for {selected_game['label']}. Using full slate instead.")
            player_pool = df.copy()
        else:
            print(f"\nSelected slate: {selected_game['label']} ({len(player_pool)} players)")

    # 6. Optimize lineup using advanced optimizer
    print(f"Optimizing lineup for a {'GPP' if contest_type=='league' else 'H2H'} contest...\n")
    optimizer = EnhancedDFSOptimizer()
    try:
        lineup_indices, metadata = asyncio.run(
            optimizer.optimize_lineup(player_pool, game_type=contest_type)
        )
    except Exception as e:
        print(f"ERROR: Optimization failed - {e}")
        return
    if not lineup_indices:
        print("ERROR: No feasible lineup could be generated with the given data and constraints.")
        return

    # 7. Build result object for output formatting
    lineup_df = player_pool.loc[lineup_indices]
    lineup_players = json.loads(lineup_df.to_json(orient='records'))
    total_salary = int(lineup_df["SALARY"].sum())
    total_proj = float(lineup_df["PROJ PTS"].sum())
    result = {
        "game_type": contest_type.upper(),
        "lineup": lineup_players,
        "cap_usage": {
            "total_salary": total_salary,
            "remaining": max(0, 60000 - total_salary)
        },
        "total_projected_points": round(total_proj, 2),
        "simulation": {},   # filled below
        "analysis": ""
    }
    # Include simulation results if available
    sim = metadata.get("simulation_results") or {}
    result["simulation"] = {
        "mean_score": sim.get("mean_score", 0.0),
        "std_dev": sim.get("std_dev", 0.0),
        "percentiles": {
            "50th": float(sim.get("percentiles", {}).get("90th", 0.0) * 0.8),  # approximate median
            "90th": sim.get("percentiles", {}).get("90th", 0.0),
            "95th": sim.get("percentiles", {}).get("95th", 0.0)
        },
        "sharpe_ratio": sim.get("sharpe_ratio", 0.0)
    }
    # Use AI analysis if present, otherwise fallback explanation
    ai_text = metadata.get("ai_analysis") or "No AI analysis available."
    # If contest was GPP, replace "league" terminology with "GPP" for clarity
    if ct == 'g':
        ai_text = ai_text.replace("League", "GPP").replace("LEAGUE", "GPP").replace("league", "GPP")
    result["analysis"] = ai_text

    # 8. Output the lineup, simulation summary, and AI analysis in a readable format
    report = build_text_report(result, width=110)
    print(report)
    # 9. Show AI API usage and cost
    if optimizer.ai_analyzer and optimizer.ai_analyzer.call_count:
        calls = optimizer.ai_analyzer.call_count
        cost = optimizer.ai_analyzer.daily_cost
        print(f"(AI API calls used: {calls}, approx. cost: ${cost:.4f})")

if __name__ == "__main__":
    main()
