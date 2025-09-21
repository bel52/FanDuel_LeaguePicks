import os
import json
import pandas as pd
import requests
from dateutil import parser as dateparser
import pytz

from app.data_ingestion import load_data_from_input_dir
from app.optimization import run_optimization
from app.formatting import build_text_report
from app.config import SALARY_CAP

def _fetch_games():
    """Fetches game schedules from ESPN."""
    options = [{"teams": None, "label": "All Games (Full Slate)"}]
    try:
        r = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard", timeout=5)
        r.raise_for_status()
        data = r.json()
        eastern = pytz.timezone("America/New_York")
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            teams_data = comp.get("competitors", [])
            if len(teams_data) == 2:
                away = teams_data[0].get("team", {}).get("abbreviation", "TBD")
                home = teams_data[1].get("team", {}).get("abbreviation", "TBD")
                dt_str = comp.get("date", "")
                dt_obj = dateparser.parse(dt_str).astimezone(eastern)
                time_str = dt_obj.strftime("%a %I:%M %p ET")
                options.append({"teams": [away, home], "label": f"{away} vs {home} - {time_str}"})
    except Exception as e:
        print(f"WARNING: Could not fetch games from ESPN: {e}. Defaulting to full slate.")
    return options

def _prompt_user(prompt: str, choices: list) -> str:
    """Generic user prompt helper."""
    while True:
        val = input(prompt).strip().lower()
        if val in choices:
            return val
        print(f"Invalid choice. Please enter one of {choices}.")

def main():
    df, warnings = load_data_from_input_dir()
    for w in warnings:
        print(w)
    
    if df is None or df.empty:
        print("\nFATAL: No player data available. Exiting.")
        return

    game_options = _fetch_games()
    for i, opt in enumerate(game_options):
        print(f"{i}. {opt['label']}")
    
    while True:
        choice = input("Enter game number (or 'q' to quit): ").strip()
        if choice == 'q': return
        if choice.isdigit() and 0 <= int(choice) < len(game_options):
            selected_game = game_options[int(choice)]
            break
        print("Invalid number.")

    game_type = _prompt_user("Choose game type - (L)eague or (H)ead-to-Head: ", ['l', 'h'])
    game_mode = "league" if game_type == 'l' else "h2h"
    
    player_pool = df.copy()
    if selected_game['teams']:
        player_pool = df[df['TEAM'].isin(selected_game['teams'])]
        if len(player_pool) < 18: # Need at least 2x players for a lineup
            print(f"WARNING: Insufficient players for {selected_game['label']}. Using full slate.")
            player_pool = df.copy()

    print(f"\nOptimizing {game_mode} lineup for: {selected_game['label']}")
    print(f"Players available: {len(player_pool)}")

    lineups, errors = run_optimization(player_df=player_pool, game_mode=game_mode)

    if errors:
        print(f"ERROR: {errors[0]}")
        return

    lineup_indices = lineups[0]['players']
    lineup_df = player_pool.loc[lineup_indices]
    
    result = {
        "game_type": game_mode.upper(),
        "lineup": json.loads(lineup_df.to_json(orient='records')),
        "cap_usage": {
            "total_salary": int(lineup_df['SALARY'].sum()),
            "remaining": SALARY_CAP - int(lineup_df['SALARY'].sum())
        }
    }

    print("\n" + build_text_report(result))

if __name__ == "__main__":
    main()
