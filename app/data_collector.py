import os
import requests
import logging
import re
from datetime import datetime, timedelta, timezone

TEAM_WHITELIST = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND",
    "JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WAS"
}

def _norm(s: str) -> str:
    return re.sub(r'[^a-z]', '', (s or '').lower())

class DataCollector:
    def __init__(self):
        now_year = datetime.now().year
        self.season = int(os.getenv("NFL_SEASON", now_year))
        self.week = int(os.getenv("NFL_WEEK", 0))
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        self.weather_enabled = (os.getenv("USE_WEATHER","0") == "1")
        self._espn_team_map = None
        logging.info(f"DataCollector initialized for season {self.season}, week {self.week or 'auto'}")

    # ---------- ESPN team/roster helpers ----------

    def fetch_espn_teams(self):
        if self._espn_team_map is not None:
            return self._espn_team_map
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
            r = requests.get(url, timeout=15); r.raise_for_status()
            teams = r.json().get("sports", [])[0].get("leagues", [])[0].get("teams", [])
            m = {}
            for t in teams:
                info = t.get("team", {})
                abbr = info.get("abbreviation")
                tid = info.get("id")
                if abbr and tid:
                    m[abbr] = tid
            self._espn_team_map = m
        except Exception as e:
            logging.error(f"ESPN team map error: {e}")
            self._espn_team_map = {}
        return self._espn_team_map

    def fetch_espn_roster(self, abbr: str):
        """
        Return a list of {name, position} from ESPN roster for the given team abbr.
        Only positions we care about will be filtered later.
        """
        tid_map = self.fetch_espn_teams()
        tid = tid_map.get(abbr); out = []
        if not tid: return out
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}/roster"
            r = requests.get(url, timeout=15)
            if r.status_code != 200: return out
            for a in r.json().get("athletes", []):
                name = (a.get("fullName") or a.get("displayName") or "").strip()
                pos = (a.get("position", {}) or {}).get("abbreviation") or ""
                if name and pos:
                    out.append({"name": name, "position": pos})
        except Exception as e:
            logging.error(f"ESPN roster error for {abbr}: {e}")
        return out

    # ---------- External APIs ----------

    def fetch_scoreboard(self):
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
        params = {"dates": self.season}
        if self.week:
            params.update({"seasontype": 2, "week": self.week})
        try:
            r = requests.get(url, params=params, timeout=15); r.raise_for_status()
            return r.json()
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
                "oddsFormat": "american",
            }
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                logging.warning(f"Odds API error: {r.status_code} {r.text[:200]}")
                return None
            return r.json()
        except Exception as e:
            logging.error(f"Failed to fetch odds: {e}")
            return None

    def fetch_weather(self, games):
        # Minimal sample stadium mapping; extend as needed.
        weather_info = {}
        NFL_STADIUMS = {
            "GB": {"lat": 44.5013, "lon": -88.0622},
            "CHI": {"lat": 41.8623, "lon": -87.6167},
            "BUF": {"lat": 42.7738, "lon": -78.7870},
        }
        if not games: return weather_info
        headers = {"User-Agent": "DFS Optimizer (contact@example.com)"}
        for g in games:
            home = g.get("home_team")
            if home not in NFL_STADIUMS: continue
            coords = NFL_STADIUMS[home]
            try:
                p = requests.get(f"https://api.weather.gov/points/{coords['lat']},{coords['lon']}", headers=headers, timeout=15)
                if p.status_code != 200: continue
                f_url = (p.json().get("properties") or {}).get("forecast")
                if not f_url: continue
                f = requests.get(f_url, headers=headers, timeout=15)
                if f.status_code == 200:
                    periods = (f.json().get("properties") or {}).get("periods") or []
                    if periods:
                        weather_info[home] = periods[0].get("shortForecast","")
            except Exception as e:
                logging.error(f"Weather error for {home}: {e}")
        return weather_info

    # ---------- Core collection ----------

    def collect_weekly_data(self):
        sb = self.fetch_scoreboard()

        games = []
        team_name_map = {}
        team_game_info = {}
        current_week = (sb.get("week",{}) or {}).get("number") or (sb.get("week",{}) or {}).get("weekNumber")

        # Slate window: Thu→next Sun
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=2)
        window_end   = now + timedelta(days=8)

        try:
            for event in sb.get("events", []):
                comps = event.get("competitions") or []
                if not comps: continue
                comp = comps[0]
                teams = comp.get("competitors") or []
                if len(teams) < 2: continue

                home_abbr = away_abbr = None
                for t in teams:
                    info = t.get("team") or {}
                    abbr = info.get("abbreviation")
                    disp = info.get("displayName")
                    if abbr not in TEAM_WHITELIST: continue
                    if t.get("homeAway") == "home":
                        home_abbr = abbr
                    else:
                        away_abbr = abbr
                    if disp and abbr:
                        team_name_map[disp] = abbr

                if not home_abbr or not away_abbr: continue

                start_iso = (comp.get("date") or "").replace("Z","+00:00")
                try:
                    start_dt = datetime.fromisoformat(start_iso) if start_iso else None
                except Exception:
                    start_dt = None

                if start_dt and not (window_start <= start_dt <= window_end):
                    continue

                games.append({"home_team": home_abbr, "away_team": away_abbr, "start_time": start_dt})
                team_game_info.setdefault(home_abbr, {})["opponent"] = away_abbr
                team_game_info[home_abbr]["start_time"] = start_dt
                team_game_info.setdefault(away_abbr, {})["opponent"] = home_abbr
                team_game_info[away_abbr]["start_time"] = start_dt
        except Exception as e:
            logging.error(f"Scoreboard parse error: {e}")

        week_number = self.week or current_week or 0
        logging.info(f"Collecting data for Week {week_number} with {len(games)} games in slate window.")

        # Odds → implied totals
        odds = self.fetch_odds()
        if odds:
            for g in odds:
                try:
                    home_name = g.get("home_team"); away_name = g.get("away_team")
                    total = None; spread_home = None
                    for bk in g.get("bookmakers", []):
                        for m in bk.get("markets", []):
                            if m.get("key") == "totals":
                                outs = m.get("outcomes", [])
                                if outs: total = float(outs[0].get("point", 0))
                            if m.get("key") == "spreads":
                                for o in m.get("outcomes", []):
                                    if o.get("name") == home_name:
                                        spread_home = float(o.get("point", 0))
                    if total is None or spread_home is None:
                        continue
                    ah = team_name_map.get(home_name); aa = team_name_map.get(away_name)
                    if ah not in TEAM_WHITELIST or aa not in TEAM_WHITELIST: continue
                    home_score = (total/2) - (spread_home/2); away_score = total - home_score
                    if ah in team_game_info: team_game_info[ah]["implied_total"] = round(home_score,1)
                    if aa in team_game_info: team_game_info[aa]["implied_total"] = round(away_score,1)
                except Exception as e:
                    logging.warning(f"Odds parse failed: {e}")

        # Weather (optional)
        if self.weather_enabled and games:
            for team, fc in self.fetch_weather(games).items():
                team_game_info.setdefault(team, {})["weather"] = fc

        # Build pool straight from ESPN rosters for teams in slate
        slate_teams = {g["home_team"] for g in games} | {g["away_team"] for g in games}
        if not slate_teams:
            logging.warning("No valid slate found; returning empty player list.")
            return {"players": [], "team_game_info": {}, "week": week_number}

        players = []
        # per-team caps to keep pool reasonable
        team_pos_caps = {t: {"QB": 1, "RB": 2, "WR": 3, "TE": 1} for t in slate_teams}
        team_pos_counts = {t: {"QB": 0, "RB": 0, "WR": 0, "TE": 0} for t in slate_teams}

        for team in sorted(slate_teams):
            roster = self.fetch_espn_roster(team)
            # Filter to positions of interest and cap per team
            for entry in roster:
                pos = entry["position"]
                if pos not in {"QB","RB","WR","TE"}:
                    continue
                if team_pos_counts[team][pos] >= team_pos_caps[team][pos]:
                    continue
                team_pos_counts[team][pos] += 1

                # baseline projections
                proj = 0.0
                if pos == "QB":
                    implied = team_game_info.get(team, {}).get("implied_total")
                    proj = 8 + 0.5 * implied if implied is not None else 18.0
                elif pos == "RB":
                    proj = 12.0 + 3.0 * (team_pos_counts[team][pos] == 1)
                elif pos == "WR":
                    rank = team_pos_counts[team][pos]
                    proj = {1:12.0, 2:10.0, 3:8.0}.get(rank, 8.0)
                elif pos == "TE":
                    proj = 8.0

                players.append({
                    "id": f"{_norm(entry['name'])}_{team}",
                    "name": entry["name"],
                    "team": team,
                    "position": pos,
                    "projection": float(proj),
                    "salary": 0
                })

            # Add DST for each slate team
            try:
                opp = team_game_info.get(team, {}).get("opponent")
                opp_imp = team_game_info.get(opp, {}).get("implied_total") if opp else None
                dst_proj = 6.0 if opp_imp is None else max(3.0, min(10.0, 6 + (20 - opp_imp)/2))
                players.append({
                    "id": f"{team}_DST",
                    "name": f"{team} DST",
                    "team": team,
                    "position": "DST",
                    "projection": round(float(dst_proj), 1),
                    "salary": 0
                })
            except Exception:
                pass

        # Salary heuristic
        for p in players:
            try:
                proj = float(p["projection"]); pos = p["position"]
                if pos == "QB":
                    sal = int(proj * 400)
                elif pos in {"RB","WR"}:
                    sal = int(proj * 500)
                elif pos == "TE":
                    sal = int(proj * 450)
                elif pos == "DST":
                    sal = int(proj * 500)
                else:
                    sal = 4000
                if pos == "DST":
                    sal = max(3000, min(5500, sal))
                else:
                    sal = max(4000, min(11000, sal))
                p["salary"] = sal
            except Exception:
                p["salary"] = 4000

        logging.info(f"Built ESPN-roster-based pool of {len(players)} from {len(slate_teams)} teams.")
        return {"players": players, "team_game_info": team_game_info, "week": week_number}
