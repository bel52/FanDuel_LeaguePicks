# app/data_collector.py
import os
import requests
import json
import logging
from datetime import datetime, timezone

# DataCollector handles fetching NFL data from various APIs (ESPN, Sleeper, Weather, Odds)
class DataCollector:
    def __init__(self):
        self.season = int(os.getenv('NFL_SEASON', datetime.now().year))
        self.week = int(os.getenv('NFL_WEEK', 0))
        # API keys from environment (if required)
        self.odds_api_key = os.getenv('ODDS_API_KEY')
        # Yahoo API would require OAuth integration (not implemented)
        self.weather_enabled = True
        # Memory cache for heavy data
        self._all_players = None
        logging.info(f"DataCollector initialized for season {self.season}, week {self.week or 'current'}")

    def fetch_scoreboard(self):
        """Fetch NFL schedule and scores from ESPN API (no auth required)"""
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"dates": self.season}
        if self.week:
            params.update({"seasontype": 2, "week": self.week})
        response = requests.get(url, params=params)
        if response.status_code != 200:
            # If failure, could retry with backoff
            logging.error(f"ESPN scoreboard API error: {response.status_code}")
            return None
        return response.json()

    def fetch_odds(self):
        """Fetch Vegas odds (spreads, totals) using The Odds API (if API key provided)"""
        if not self.odds_api_key:
            return None
        try:
            url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
            params = {
                "apiKey": self.odds_api_key,
                "regions": "us",
                "markets": "spreads,totals",
                "oddsFormat": "american"
            }
            resp = requests.get(url, params=params)
            if resp.status_code != 200:
                logging.warning(f"Odds API error: {resp.status_code}, {resp.text}")
                return None
            return resp.json()
        except Exception as e:
            logging.error(f"Failed to fetch odds: {e}")
            return None

    def fetch_trending_players(self, hours=24):
        """Get trending players from Sleeper (no auth required)"""
        url = f"https://api.sleeper.app/v1/players/nfl/trending/add"
        params = {"lookback_hours": hours, "limit": 25}
        try:
            resp = requests.get(url, params=params)
            return resp.json()
        except Exception as e:
            logging.error(f"Error fetching trending players: {e}")
            return []

    def fetch_all_players(self):
        """Fetch complete NFL player database from Sleeper (cached after first call)"""
        if self._all_players is None:
            # Use caching: Load all_players once and store for reuse
            try:
                resp = requests.get("https://api.sleeper.app/v1/players/nfl")
                resp.raise_for_status()
                self._all_players = resp.json()
                logging.info(f"Loaded {len(self._all_players)} players from Sleeper API.")
            except Exception as e:
                logging.error(f"Error fetching Sleeper players: {e}")
                self._all_players = {}
        return self._all_players

    def fetch_weather(self, games):
        """Fetch weather forecasts for stadiums via Weather.gov API"""
        weather_info = {}
        # Coordinates for some stadiums (extend as needed)
        NFL_STADIUMS = {
            'GB': {'lat': 44.5013, 'lon': -88.0622},
            'CHI': {'lat': 41.8623, 'lon': -87.6167},
            'BUF': {'lat': 42.7738, 'lon': -78.7870},
            # ... add remaining stadium coordinates as needed
        }
        headers = {"User-Agent": "DFS Optimizer Bot (contact@example.com)"}
        for game in games:
            home = game['home_team']
            if home in NFL_STADIUMS:
                coords = NFL_STADIUMS[home]
                try:
                    points_url = f"https://api.weather.gov/points/{coords['lat']},{coords['lon']}"
                    resp = requests.get(points_url, headers=headers)
                    if resp.status_code != 200:
                        continue
                    grid_data = resp.json()
                    forecast_url = grid_data['properties'].get('forecast')
                    if forecast_url:
                        forecast_resp = requests.get(forecast_url, headers=headers)
                        if forecast_resp.status_code == 200:
                            forecast = forecast_resp.json()
                            if 'periods' in forecast.get('properties', {}):
                                period = forecast['properties']['periods'][0]
                                weather_info[home] = period.get('shortForecast', '')
                except Exception as e:
                    logging.error(f"Weather API error for {home}: {e}")
        return weather_info

    def fetch_yahoo_data(self):
        """Placeholder for Yahoo API integration (OAuth required)."""
        if os.getenv('YAHOO_CLIENT_ID'):
            logging.info("Yahoo API integration not implemented.")
        return None

    def collect_weekly_data(self):
        """Collect data for the current NFL week (games, odds, weather, players)."""
        scoreboard = self.fetch_scoreboard()
        if not scoreboard:
            logging.error("No scoreboard data fetched.")
            return None
        games = []
        team_name_map = {}
        team_game_info = {}
        current_week = None
        try:
            for event in scoreboard.get('events', []):
                comp = event['competitions'][0]
                teams = comp['competitors']
                home_team = away_team = None
                for t in teams:
                    team_data = t['team']
                    abbr = team_data.get('abbreviation')
                    disp_name = team_data.get('displayName')
                    if t.get('homeAway') == 'home':
                        home_team = abbr
                    else:
                        away_team = abbr
                    team_name_map[disp_name] = abbr
                start = comp.get('date')
                dt = None
                if start:
                    try:
                        dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    except Exception:
                        dt = None
                games.append({'home_team': home_team, 'away_team': away_team, 'start_time': dt})
                if home_team and away_team:
                    team_game_info[home_team] = {'opponent': away_team, 'start_time': dt}
                    team_game_info[away_team] = {'opponent': home_team, 'start_time': dt}
            current_week = scoreboard.get('week', {}).get('number') or scoreboard.get('week', {}).get('weekNumber')
        except Exception as e:
            logging.error(f"Error parsing scoreboard: {e}")
        week_number = self.week or current_week
        logging.info(f"Collecting data for Week {week_number}.")
        # Vegas odds
        odds_data = self.fetch_odds()
        if odds_data:
            for game in odds_data:
                home_name = game.get('home_team')
                away_name = game.get('away_team')
                total = None
                spread_home = None
                for book in game.get('bookmakers', []):
                    for market in book.get('markets', []):
                        if market['key'] == 'totals':
                            outcomes = market.get('outcomes', [])
                            if outcomes:
                                total = outcomes[0].get('point')
                        if market['key'] == 'spreads':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == home_name:
                                    spread_home = outcome.get('point')
                if total is not None and spread_home is not None:
                    try:
                        total = float(total)
                        spread_home = float(spread_home)
                        home_score = (total / 2) - (spread_home / 2)
                        away_score = total - home_score
                        abbr_home = team_name_map.get(home_name)
                        abbr_away = team_name_map.get(away_name)
                        if abbr_home in team_game_info:
                            team_game_info[abbr_home]['implied_total'] = round(home_score, 1)
                        if abbr_away in team_game_info:
                            team_game_info[abbr_away]['implied_total'] = round(away_score, 1)
                    except Exception as e:
                        logging.warning(f"Could not compute implied totals: {e}")
        # Weather data
        weather_info = {}
        if self.weather_enabled:
            weather_info = self.fetch_weather(games)
            for team, forecast in weather_info.items():
                if team in team_game_info:
                    team_game_info[team]['weather'] = forecast
        # Trending players
        trending = self.fetch_trending_players(hours=72)
        trending_ids = [str(p['player_id']) for p in trending] if trending else []
        # Build player pool
        players = []
        all_players = self.fetch_all_players()
        teams_playing = {g['home_team'] for g in games} | {g['away_team'] for g in games}
        for pid, player in all_players.items():
            team = player.get('team')
            pos = player.get('position')
            if not team or not pos or team not in teams_playing:
                continue
            if team not in team_game_info:
                team_game_info[team] = {'opponent': None, 'start_time': None, 'count': {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}}
            if pos not in ['QB', 'RB', 'WR', 'TE']:
                continue
            if player.get('injury_status') in ['IR', 'O']:
                continue
            count = team_game_info[team]['count']
            limit = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}[pos]
            if count[pos] >= limit:
                continue
            team_game_info[team]['count'][pos] += 1
            proj = 0.0
            if pos == 'QB':
                implied = team_game_info.get(team, {}).get('implied_total')
                proj = 8 + 0.5 * implied if implied is not None else 18.0
            elif pos == 'RB':
                proj = 15.0 if count[pos] == 1 else 8.0
            elif pos == 'WR':
                proj = 12.0 if count[pos] == 1 else (10.0 if count[pos] == 2 else 8.0)
            elif pos == 'TE':
                proj = 8.0
            player_entry = {
                'id': pid,
                'name': player.get('full_name') or player.get('name'),
                'team': team,
                'position': pos,
                'projection': proj,
                'salary': 0
            }
            players.append(player_entry)
        # Add DST entries for each team
        for team in teams_playing:
            opp = team_game_info.get(team, {}).get('opponent')
            if opp:
                opp_implied = team_game_info.get(opp, {}).get('implied_total')
                dst_proj = 6.0
                if opp_implied is not None:
                    dst_proj = 6 + (20 - opp_implied) / 2
                if dst_proj < 3:
                    dst_proj = 3.0
                if dst_proj > 10:
                    dst_proj = 10.0
                players.append({
                    'id': f"{team}_DST",
                    'name': f"{team} DST",
                    'team': team,
                    'position': 'DST',
                    'projection': round(dst_proj, 1),
                    'salary': 0
                })
        # Assign salaries based on projection
        for p in players:
            proj = p['projection']
            pos = p['position']
            if pos == 'QB':
                sal = int(proj * 400)
            elif pos == 'RB':
                sal = int(proj * 500)
            elif pos == 'WR':
                sal = int(proj * 500)
            elif pos == 'TE':
                sal = int(proj * 450)
            elif pos == 'DST':
                sal = int(proj * 500)
            if pos == 'DST':
                if sal < 3000: sal = 3000
                if sal > 5500: sal = 5500
            else:
                if sal < 4000: sal = 4000
                if sal > 11000: sal = 11000
            p['salary'] = sal
        # Boost projections for trending players
        if trending_ids:
            for p in players:
                if p['id'] in trending_ids:
                    p['projection'] = round(p['projection'] + 2.0, 1)
        logging.info(f"Built player pool of {len(players)} players for optimization.")
        return {'players': players, 'team_game_info': team_game_info, 'week': week_number}
