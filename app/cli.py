import requests, json, pandas as pd
from dateutil import parser as dateparser
import pytz
from app.data_ingestion import load_data_from_input_dir
from app.formatting import build_text_report
from app.config import settings

def fetch_games():
    """Get NFL game options from ESPN scoreboard."""
    options = [{"teams": None, "label": "All Games"}]
    try:
        resp = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard", timeout=5)
        data = resp.json()
        eastern = pytz.timezone("America/New_York")
        for event in data.get("events", []):
            teams = event["competitions"][0]["competitors"]
            if len(teams)==2:
                away = teams[0]["team"]["abbreviation"]
                home = teams[1]["team"]["abbreviation"]
                dt = dateparser.parse(event["competitions"][0]["date"]).astimezone(eastern)
                options.append({"teams":[away,home], "label":f"{away} vs {home} - {dt:%a %I:%M %p ET}"})
    except Exception as e:
        print(f"WARNING: Could not fetch games: {e}")
    return options

def main():
    df, warnings = load_data_from_input_dir()
    for w in warnings: print(w)
    if df is None or df.empty:
        print("No data. Exiting.")
        return
    games = fetch_games()
    for i, opt in enumerate(games):
        print(f"{i}: {opt['label']}")
    choice = input("Enter game number (or 'q'): ").strip()
    if choice.lower()=='q': return
    idx = int(choice) if choice.isdigit() else None
    if idx is None or idx>=len(games):
        print("Invalid choice."); return
    selected = games[idx]
    game_type = input("League or H2H? (l/h): ").strip().lower()
    game_type = "league" if game_type=='l' else "h2h"
    player_pool = df.copy()
    if selected['teams']:
        player_pool = df[df['TEAM'].isin(selected['teams'])]
        if len(player_pool)<18:
            print("Not enough players for this game. Using full slate.")
            player_pool = df.copy()
    print(f"Optimizing {game_type} lineup for {selected['label']}")
    print(f"Players available: {len(player_pool)}")
    # Call API (requires running web server on localhost:8010)
    url = f"http://localhost:8010/optimize?game_type={game_type}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print("Optimization failed:", resp.text)
        return
    result = resp.json()
    print("\n" + build_text_report(result))

if __name__ == "__main__":
    main()
