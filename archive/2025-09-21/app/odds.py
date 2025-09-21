from __future__ import annotations
import requests
from typing import Dict, Optional
from app.config import ODDS_API_KEY, ODDS_API_REGION, ODDS_API_MARKETS, ODDS_API_SPORT, ODDS_IMPLIED_MULTIPLIER

def _odds_url() -> str:
    return f"https://api.the-odds-api.com/v4/sports/{ODDS_API_SPORT}/odds"

def fetch_odds() -> Optional[Dict]:
    if not ODDS_API_KEY:
        return None
    try:
        resp = requests.get(
            _odds_url(),
            params={
                "apiKey": ODDS_API_KEY,
                "regions": ODDS_API_REGION,
                "markets": ODDS_API_MARKETS,
                "oddsFormat": "american"
            },
            timeout=25,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None

def implied_total_boost(team: str, odds_blob: Optional[Dict]) -> float:
    if not odds_blob:
        return 1.0
    try:
        for game in odds_blob:
            home = game.get("home_team")
            away = game.get("away_team")
            if team not in (home, away):
                continue
            markets = game.get("bookmakers", [])
            est_total = None
            for bm in markets:
                for mk in bm.get("markets", []):
                    if mk.get("key") == "totals" and mk.get("outcomes"):
                        try:
                            est_total = float(mk["outcomes"][0]["point"])
                            break
                        except Exception:
                            pass
                if est_total:
                    break
            if est_total is None:
                return 1.0
            if est_total >= 48:
                return 1.0 + ODDS_IMPLIED_MULTIPLIER
            elif est_total <= 41:
                return 1.0 - (ODDS_IMPLIED_MULTIPLIER * 0.6)
            return 1.0
    except Exception:
        return 1.0
    return 1.0
