from datetime import datetime
import pytz, requests

def get_next_game_time(team: str) -> datetime:
    """Example: fetch ESPN scoreboard and find next game involving `team`."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        eastern = pytz.timezone("America/New_York")
        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            teams = [c['team']['abbreviation'] for c in comp.get('competitors', [])]
            if team in teams:
                dt = datetime.fromisoformat(comp['date'].replace('Z','+00:00')).astimezone(eastern)
                return dt
    except:
        pass
    return None
