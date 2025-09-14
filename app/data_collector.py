import os
import requests
import json
import logging
from datetime import datetime, timezone

class DataCollector:
    def __init__(self):
        self.season = int(os.getenv('NFL_SEASON', datetime.now().year))
        self.week = int(os.getenv('NFL_WEEK', 0))
        self.odds_api_key = os.getenv('ODDS_API_KEY')
        self.weather_enabled = True
        self._all_players = None
        logging.info(f"DataCollector initialized for season {self.season}, week {self.week or 'current'}")

    def fetch_scoreboard(self):
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"dates": self.season}
        if self.week:
            params.update({"seasontype": 2, "week": self.week})
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"ESPN scoreboard API error: {e}")
            return {"events": []}

    def fetch_odds(self):
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
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logging.warning(f"Odds API error: {resp.status_code}, {resp.text[:200]}")
                return None
            return resp.json()
        except Exception as e:
            logging.error(f"Failed to fetch odds: {e}")
            return None

    def fetch_trending_players(self, hours=24):
        url = "https://api.sleeper.app/v1/players/nfl/trending/add"
        params = {"lookback_hours": hours, "limit": 25}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logging.error(f"Error fetching trending players: {e}")
            return []

    def fetch_all_players(self):
        if self._all_players is None:
            try:
                resp = requests.get("https://api.sleeper.app/v1/players/nfl", timeout=30)
                resp.raise_for_status()
                self._all_players = resp.json()
                logging.info(f"Loaded {len(self._all_players)} players from Sleeper API.")
            except Exception as e:
                logging.error(f"Error fetching Sleeper players: {e}")
                self._all_players = {}
        return self._all_players

    def fetch_weather(self, games):
        weather_info = {}
        NFL_STADIUMS = {
            'GB': {'lat': 44.5013, 'lon': -88.0622},
            'CHI': {'lat': 41.8623, 'lon': -87.6167},
            'BUF': {'lat': 42.7738, 'lon': -78.7870},
        }
        headers = {"User-Agent": "DFS Optimizer Bot (contact@example.com)"}
        for game in games:
            home = game.get('home_team')
            if not home or home not in NFL_STADIUMS:
                continue
            coords = NFL_STADIUMS[home]
            try:
                points_url = f"https://api.weather.gov/points/{coords['lat']},{coords['lon']}"
                resp = requests.get(points_url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    continue
                grid_data = resp.json()
                forecast_url = grid_data.get('properties', {}).get('forecast')
                if forecast_url:
                    forecast_resp = requests.get(forecast_url, headers=headers, timeout=15)
                    if forecast_resp.status_code == 200:
                        forecast = forecast_resp.json()
                        periods = forecast.get('properties', {}).get('periods') or []
                        if periods:
                            weather_info[home] = periods[0].get('shortForecast', '')
            except Exception as e:
                logging.error(f"Weather API error for {home}: {e}")
        return weather_info

    def collect_weekly_data(self):
        scoreboard = self.fetch_scoreboard()
        games = []
        team_name_map = {}
        team_game_info = {}
        current_week = None

        # Parse ESPN data safely
        try:
            current_week = (
                scoreboard.get('week', {}).get('number') or
                scoreboard.get('week', {}).get('weekNumber') or
                None
            )
            for event in scoreboard.get('events', []):
                comp_list = event.get('competitions') or []
                if not comp_list:
                    continue
                comp = comp_list[0]
                teams = comp.get('competitors') or []
                if len(teams) < 2:
                    continue
                home_team = away_team = None
                for t in teams:
                    team_data = t.get('team') or {}
                    abbr = team_data.get('abbreviation')
                    disp_name = team_data.get('displayName')
                    if t.get('homeAway') == 'home':
                        home_team = abbr
                    else:
                        away_team = abbr
                    if disp_name and abbr:
                        team_name_map[disp_name] = abbr
                start = comp.get('date')
                dt = None
                if start:
                    try:
                        dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    except Exception:
                        dt = None
                if home_team and away_team:
                    games.append({'home_team': home_team, 'away_team': away_team, 'start_time': dt})
                    team_game_info.setdefault(home_team, {})['opponent'] = away_team
                    team_game_info[home_team]['start_time'] = dt
                    team_game_info.setdefault(away_team, {})['opponent'] = home_team
                    team_game_info[away_team]['start_time'] = dt
        except Exception as e:
            logging.error(f"Error parsing scoreboard: {e}")

        week_number = self.week or current_week or 0
        logging.info(f"Collecting data for Week {week_number}.")

        # Odds → implied totals
        odds_data = self.fetch_odds()
        if odds_data:
            for game in odds_data:
                try:
                    home_name = game.get('home_team')
                    away_name = game.get('away_team')
                    total = None
                    spread_home = None
                    for book in game.get('bookmakers', []):
                        for market in book.get('markets', []):
                            if market.get('key') == 'totals':
                                outcomes = market.get('outcomes', [])
                                if outcomes:
                                    total = float(outcomes[0].get('point', 0))
                            if market.get('key') == 'spreads':
                                for outcome in market.get('outcomes', []):
                                    if outcome.get('name') == home_name:
                                        spread_home = float(outcome.get('point', 0))
                    if total is None or spread_home is None:
                        continue
                    home_score = (total / 2) - (spread_home / 2)
                    away_score = total - home_score
                    abbr_home = team_name_map.get(home_name)
                    abbr_away = team_name_map.get(away_name)
                    if abbr_home in team_game_info:
                        team_game_info[abbr_home]['implied_total'] = round(home_score, 1)
                    if abbr_away in team_game_info:
                        team_game_info[abbr_away]['implied_total'] = round(away_score, 1)
                except Exception as e:
                    logging.warning(f"Odds parse failed for a game: {e}")

        # Weather
        weather_info = {}
        if self.weather_enabled and games:
            weather_info = self.fetch_weather(games)
            for team, forecast in weather_info.items():
                team_game_info.setdefault(team, {})['weather'] = forecast

        # Sleeper trending
        trending = self.fetch_trending_players(hours=72) or []
        trending_ids = {str(p.get('player_id')) for p in trending}

        # Build player pool (simple heuristic for v1)
        players = []
        all_players = self.fetch_all_players()
        teams_playing = {g['home_team'] for g in games} | {g['away_team'] for g in games}
        if not teams_playing:
            logging.warning("No games parsed; returning empty player list.")
            return {'players': [], 'team_game_info': {}, 'week': week_number}

        # Track per-team position caps
        team_pos_counts = {t: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0} for t in teams_playing}
        for pid, player in (all_players or {}).items():
            try:
                team = player.get('team')
                pos = player.get('position')
                if not team or not pos or team not in teams_playing:
                    continue
                if pos not in ['QB', 'RB', 'WR', 'TE']:
                    continue
                if player.get('injury_status') in ['IR', 'O']:
                    continue
                cap = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}[pos]
                if team_pos_counts[team][pos] >= cap:
                    continue
                team_pos_counts[team][pos] += 1

                # baseline projections
                proj = 0.0
                if pos == 'QB':
                    implied = team_game_info.get(team, {}).get('implied_total')
                    proj = 8 + 0.5 * implied if implied is not None else 18.0
                elif pos == 'RB':
                    proj = 12.0 + 3.0 * (team_pos_counts[team][pos] == 1)
                elif pos == 'WR':
                    proj = {1: 12.0, 2: 10.0, 3: 8.0}.get(team_pos_counts[team][pos], 8.0)
                elif pos == 'TE':
                    proj = 8.0

                entry = {
                    'id': str(pid),
                    'name': player.get('full_name') or player.get('name') or f"Player {pid}",
                    'team': team,
                    'position': pos,
                    'projection': float(proj),
                    'salary': 0
                }
                players.append(entry)
            except Exception:
                continue

        # Add DSTs
        for team in teams_playing:
            try:
                opp = team_game_info.get(team, {}).get('opponent')
                opp_imp = team_game_info.get(opp, {}).get('implied_total') if opp else None
                dst_proj = 6.0 if opp_imp is None else max(3.0, min(10.0, 6 + (20 - opp_imp) / 2))
                players.append({
                    'id': f"{team}_DST",
                    'name': f"{team} DST",
                    'team': team,
                    'position': 'DST',
                    'projection': round(float(dst_proj), 1),
                    'salary': 0
                })
            except Exception:
                continue

        # Salaries (heuristic, bounded)
        for p in players:
            try:
                proj = float(p['projection'])
                pos = p['position']
                if pos == 'QB':
                    sal = int(proj * 400)
                elif pos in ['RB', 'WR']:
                    sal = int(proj * 500)
                elif pos == 'TE':
                    sal = int(proj * 450)
                elif pos == 'DST':
                    sal = int(proj * 500)
                else:
                    sal = 4000
                if pos == 'DST':
                    sal = max(3000, min(5500, sal))
                else:
                    sal = max(4000, min(11000, sal))
                p['salary'] = sal
            except Exception:
                p['salary'] = 4000

        # Trending bump
        if trending_ids:
            for p in players:
                if p['id'] in trending_ids:
                    p['projection'] = round(float(p['projection']) + 2.0, 1)

        logging.info(f"Built player pool of {len(players)} players for optimization.")
        return {'players': players, 'team_game_info': team_game_info, 'week': week_number}
