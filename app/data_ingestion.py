import os, time
from typing import Any, Dict, List, Tuple, DefaultDict
import requests
from collections import defaultdict

USER_AGENT = "DFS-Optimizer/1.0 (+https://github.com/bel52/FanDuel_LeaguePicks)"
SLEEPER = "https://api.sleeper.app/v1"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ODDS = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

def _get(url: str, params: Dict[str, Any] | None = None, timeout: int = 15) -> Any:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()

def _sleeper_state() -> Dict[str, Any]:
    return _get(f"{SLEEPER}/state/nfl")

def _sleeper_players_map() -> Dict[str, Dict[str, Any]]:
    cache = "data/input/sleeper_players.json"
    os.makedirs("data/input", exist_ok=True)
    try:
        if os.path.exists(cache) and (time.time() - os.path.getmtime(cache)) < 3600:
            import json
            with open(cache, "r") as f:
                return json.load(f)
    except Exception:
        pass
    data = _get(f"{SLEEPER}/players/nfl")
    try:
        import json
        with open(cache, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
    return data

def _normalize_to_list_of_dicts(raw: Any) -> List[Dict[str, Any]]:
    """
    Sleeper /projections can be:
      - list[dict]
      - dict with 'projections': list
      - dict keyed by player_id -> dict
    Return a clean list[dict].
    """
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        if isinstance(raw.get("projections"), list):
            return [x for x in raw["projections"] if isinstance(x, dict)]
        return [v for v in raw.values() if isinstance(v, dict)]
    return []

def _proj_for(season: int, week: int, pos: str) -> List[Dict[str, Any]]:
    url = f"{SLEEPER}/projections/nfl/{season}/{week}"
    raw = _get(url, {"position": pos, "season_type": "regular", "order_by": "pts_ppr"})
    return _normalize_to_list_of_dicts(raw)

def _ppr_from_stats(p: Dict[str, Any]) -> float:
    s = p.get("stats") or {}
    pts = (
        float(s.get("pass_yd", 0))/25.0 + float(s.get("pass_td", 0))*4.0 - float(s.get("pass_int", 0))*2.0 +
        float(s.get("rush_yd", 0))/10.0 + float(s.get("rush_td", 0))*6.0 +
        float(s.get("rec", 0)) + float(s.get("rec_yd", 0))/10.0 + float(s.get("rec_td", 0))*6.0 -
        float(s.get("fum_lost", 0))*2.0
    )
    pts += float(s.get("fgm", 0))*3 + float(s.get("xpm", 0))*1
    pts += float(s.get("def_td", 0))*6 + float(s.get("def_sack", 0))*1 + float(s.get("def_int", 0))*2 + float(s.get("def_fum_rec", 0))*2
    return round(pts, 2)

def _abbr(name: str) -> str:
    if name and len(name) <= 4 and name.isupper():
        return name
    quick = {
        "New York Giants":"NYG","New York Jets":"NYJ","Los Angeles Rams":"LAR","Los Angeles Chargers":"LAC",
        "San Francisco 49ers":"SF","Tampa Bay Buccaneers":"TB","Washington Commanders":"WSH",
        "Green Bay Packers":"GB","Chicago Bears":"CHI","Buffalo Bills":"BUF","Dallas Cowboys":"DAL",
        "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV","Denver Broncos":"DEN","Seattle Seahawks":"SEA",
        "Cleveland Browns":"CLE","Pittsburgh Steelers":"PIT","Baltimore Ravens":"BAL","Cincinnati Bengals":"CIN",
        "Philadelphia Eagles":"PHI","Miami Dolphins":"MIA","New England Patriots":"NE","Indianapolis Colts":"IND",
        "Tennessee Titans":"TEN","Houston Texans":"HOU","Jacksonville Jaguars":"JAX","Atlanta Falcons":"ATL",
        "Carolina Panthers":"CAR","New Orleans Saints":"NO","Minnesota Vikings":"MIN","Detroit Lions":"DET",
        "Arizona Cardinals":"ARI"
    }
    return quick.get(name, name)

def _build_team_info_and_implied() -> Tuple[Dict[str, Tuple[int, str]], Dict[str, float]]:
    """Return (kickoff/opponent map, team implied points) from free sources."""
    # ESPN schedule
    scoreboard = _get(ESPN_SCOREBOARD)
    team_info: Dict[str, Tuple[int, str]] = {}
    for ev in scoreboard.get("events", []) or []:
        comps = (ev.get("competitions") or [{}])[0].get("competitors", [])
        if len(comps) != 2:
            continue
        kick = ev.get("date", "")
        try:
            kickoff_ts = int(time.mktime(time.strptime(kick[:19], "%Y-%m-%dT%H:%M:%S")))
        except Exception:
            kickoff_ts = None
        t1 = (comps[0].get("team") or {}).get("abbreviation")
        t2 = (comps[1].get("team") or {}).get("abbreviation")
        if t1 and t2:
            team_info[t1] = (kickoff_ts, t2)
            team_info[t2] = (kickoff_ts, t1)

    # Odds (optional)
    implied: Dict[str, float] = {}
    key = os.getenv("ODDS_API_KEY")
    if key:
        try:
            odds = _get(ODDS, {"regions":"us","markets":"totals,spreads","oddsFormat":"american","apiKey":key})
            for g in odds or []:
                home = _abbr(g.get("home_team","")); away = _abbr(g.get("away_team",""))
                bms = g.get("bookmakers") or []
                if not bms:
                    continue
                totals = next((m for m in bms[0].get("markets", []) if m.get("key")=="totals"), None)
                if totals and totals.get("outcomes"):
                    try:
                        ou = float(totals["outcomes"][0].get("point") or 0)
                    except Exception:
                        ou = 0.0
                    if ou > 0:
                        if home: implied[home] = ou/2.0
                        if away: implied[away] = ou/2.0
        except Exception:
            pass
    return team_info, implied

def _fallback_from_depth(
    pmap: Dict[str, Dict[str, Any]],
    team_info: Dict[str, Tuple[int, str]],
    implied: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Build a viable pool using Sleeper depth charts + Vegas implied team totals.
    For each team playing this week we try to add:
      QB1, RB1, WR1, WR2, TE1, DST
    """
    by_team_pos: DefaultDict[Tuple[str, str], List[Tuple[int, str, Dict[str, Any]]]] = defaultdict(list)
    for pid, v in pmap.items():
        team = (v.get("team") or v.get("team_abbr") or "").upper()
        if not team:
            continue
        poss = set()
        if v.get("position"):
            poss.add(v["position"])
        for fp in v.get("fantasy_positions") or []:
            poss.add(fp)
        if not poss:
            continue
        try:
            depth = int(v.get("depth_chart_order") or 99)
        except Exception:
            depth = 99
        rec = (depth, pid, v)
        for pos in poss:
            p = str(pos).upper()
            if p == "DEF":
                p = "DST"
            if p in {"QB","RB","WR","TE","DST"}:
                by_team_pos[(team, p)].append(rec)

    for k in list(by_team_pos.keys()):
        by_team_pos[k].sort(key=lambda t: t[0])

    def _best(team: str, pos: str, nth: int = 1):
        lst = by_team_pos.get((team, pos)) or []
        if len(lst) >= nth:
            depth, pid, v = lst[nth-1]
            return pid, v
        return None

    players: List[Dict[str, Any]] = []
    default_ip = 21.0
    for team, (ko, opp) in team_info.items():
        ip = float(implied.get(team, default_ip))
        opp_ip = float(implied.get(opp, default_ip))

        qb_proj  = round(10 + 0.50*ip, 2)
        rb1_proj = round( 4 + 0.35*ip, 2)
        wr1_proj = round( 3 + 0.30*ip, 2)
        wr2_proj = round( 2 + 0.18*ip, 2)
        te1_proj = round( 2 + 0.18*ip, 2)
        dst_proj = round(max(3.0, 10.0 - 0.25*opp_ip), 2)

        picks = [("QB",1,qb_proj),("RB",1,rb1_proj),("WR",1,wr1_proj),("WR",2,wr2_proj),("TE",1,te1_proj),("DST",1,dst_proj)]
        for pos, nth, proj in picks:
            sel = _best(team, pos, nth)
            if not sel:
                continue
            pid, v = sel
            name = (v.get("full_name") or f"{v.get('first_name','')} {v.get('last_name','')}".strip() or v.get("last_name") or "Unknown").strip()
            players.append({
                "player_id": str(pid),
                "name": name,
                "team": team,
                "pos": pos,
                "proj": float(proj),
                "opponent": opp or "",
                "kickoff": ko
            })
    return players

def weekly_player_pool() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = {"warnings": []}
    state = _sleeper_state()
    season, week = int(state.get("season")), int(state.get("week"))
    if not season or not week:
        raise RuntimeError("Unable to resolve NFL season/week")

    positions = ["QB","RB","WR","TE","DEF"]  # no K; DEF normalized to DST
    projs: List[Dict[str, Any]] = []
    for pos in positions:
        try:
            norm = _proj_for(season, week, pos)
            projs.extend(norm)
        except Exception as e:
            meta["warnings"].append(f"Proj fetch failed for {pos}: {e}")

    projs = [p for p in projs if isinstance(p, dict)]
    pmap = _sleeper_players_map()
    team_info, implied = _build_team_info_and_implied()

    players: List[Dict[str, Any]] = []

    if projs:
        for p in projs:
            pid = str(p.get("player_id") or p.get("id") or (p.get("player") or {}).get("id") or "")
            m = pmap.get(pid) or {}
            name = (m.get("full_name") or f"{m.get('first_name','')} {m.get('last_name','')}".strip() or m.get("last_name") or "Unknown").strip()
            pos  = (m.get("position") or p.get("position") or p.get("pos") or "").upper()
            if pos == "DEF":
                pos = "DST"
            team = (m.get("team") or m.get("team_abbr") or "").upper()
            proj = float(p.get("pts_ppr") or p.get("fantasy_points") or 0.0)
            if proj == 0.0:
                proj = _ppr_from_stats(p)
            kickoff_ts, opp = None, ""
            if team and team in team_info:
                kickoff_ts, opp = team_info[team]
            if team in implied and proj > 0:
                proj = round(proj * (1.0 + min(0.12, implied[team]/50.0)), 2)
            if pos in {"QB","RB","WR","TE","DST"} and team:
                players.append({
                    "player_id": pid, "name": name, "team": team, "pos": pos,
                    "proj": proj, "opponent": opp, "kickoff": kickoff_ts
                })

    if len(players) < 50:
        fb = _fallback_from_depth(pmap, team_info, implied)
        if fb:
            players = fb
            meta["warnings"].append("Using depth-chart + Vegas implied fallback (Sleeper projections empty).")
        else:
            meta["warnings"].append("Fallback build failed (no depth chart matches).")

    uniq = {}
    for x in players:
        k = (x["name"], x["pos"], x["team"])
        if k not in uniq or float(x["proj"]) > float(uniq[k]["proj"]):
            uniq[k] = x
    players = list(uniq.values())

    if len(players) < 20:
        meta["warnings"].append(f"Small pool ({len(players)} players).")
    return players, meta
