import os
import logging
import requests
from datetime import datetime, timezone

TEAM_WHITELIST = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND",
    "JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
}

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

class DataCollector:
    def __init__(self):
        now_year = datetime.now(timezone.utc).year
        self.season = int(os.getenv("NFL_SEASON", now_year))
        self.week = int(os.getenv("NFL_WEEK", 0))
        logging.info(f"DataCollector initialized for {self.season} week={self.week or 'auto'}")

    def fetch_player_pool(self):
        """Fetch all active NFL players from Sleeper API"""
        try:
            r = requests.get(SLEEPER_PLAYERS_URL, timeout=20)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logging.error(f"Error fetching Sleeper players: {e}")
            return []

        players = []
        for pid, info in data.items():
            if not info or not isinstance(info, dict):
                continue
            team = info.get("team")
            pos = info.get("position")
            status = info.get("status")
            full_name = info.get("full_name")
            if team in TEAM_WHITELIST and pos in {"QB","RB","WR","TE","K","DEF"}:
                if status == "Inactive":
                    continue
                players.append({
                    "id": pid,
                    "name": full_name,
                    "team": team,
                    "position": pos,
                    # Placeholder projections/salary until model enriches
                    "projection": float(info.get("fantasy_positions", [0])[0] != 0) * 5.0,
                    "salary": 3000
                })
        logging.info(f"Collected {len(players)} players from Sleeper")
        return players
