"""FantasyPros public API v2 client. Fail-loud schema validation on every pull.

Auth: x-api-key header (key from GOAT/MVP subscription, activated at fantasypros.com/api-data/).
Base: https://api.fantasypros.com/public/v2/json/nfl/
Endpoints used:
  {season}/projections?position={pos}&week={week}&scoring=HALF   (FanDuel = half-PPR base)
  {season}/players?position=...
  {season}/injuries
  news?limit=...
  {season}/points?position=...&week=...&scoring=HALF             (scored points, result logging)

NOTE: We do NOT use FP's `points`/`points_half` — those are generic fantasy scoring. FP returns
the full projected STAT LINE, which scoring.py converts to true FanDuel points (half-PPR +
300/100/100 bonuses + DST points-allowed ladder). Zero scoring ambiguity.
Env: FANTASYPROS_API_KEY
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Optional

from .scoring import score

BASE = "https://api.fantasypros.com/public/v2/json/nfl"
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]


class FantasyProsError(Exception):
    """API/schema failure. Build stops — no fallback projections exist by design."""


@dataclass
class FPProjection:
    player_id: str            # FantasyPros player id
    name: str
    team: str
    position: str
    points: float             # TRUE FanDuel points, computed from stats by scoring.py
    stats: dict               # raw FP projected stat line
    breakdown: dict           # per-category FanDuel point breakdown (auditable)
    fp_points_half: float = 0.0   # FP's own number, kept for comparison only


class FantasyProsClient:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 20,
                 max_retries: int = 3):
        self.api_key = api_key or os.getenv("FANTASYPROS_API_KEY", "")
        if not self.api_key:
            raise FantasyProsError("FANTASYPROS_API_KEY not set")
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        qs = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        url = f"{BASE}/{path}" + (f"?{qs}" if qs else "")
        req = urllib.request.Request(url, headers={"x-api-key": self.api_key,
                                                   "User-Agent": "dfs-v6/1.0"})
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise FantasyProsError(f"auth failed ({e.code}) — check API key activation") from e
                if e.code == 429:
                    time.sleep(2 ** attempt)
                    last = e
                    continue
                raise FantasyProsError(f"HTTP {e.code} on {url}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(1 + attempt)
        raise FantasyProsError(f"request failed after {self.max_retries} tries: {last}")

    def weekly_projections(self, season: int, week: int,
                           positions: list[str] = POSITIONS) -> list[FPProjection]:
        out: list[FPProjection] = []
        for pos_req in positions:
            data = self._get(f"{season}/projections",
                             {"position": pos_req, "week": week})
            players = data.get("players")
            if players is None:
                raise FantasyProsError(
                    f"schema drift: 'players' missing for {pos_req} wk{week}. Keys: {list(data.keys())}")
            for p in players:
                try:
                    stats = p.get("stats", {}) or {}
                    pos = "D" if pos_req == "DST" else pos_req
                    fd_points, breakdown = score(stats, pos)
                    out.append(FPProjection(
                        player_id=str(p.get("fpid") or p["player_id"]),
                        name=p.get("name") or p.get("player_name", ""),
                        team=(p.get("team_id") or p.get("team") or "").upper(),
                        position=pos,
                        points=fd_points,
                        stats=stats,
                        breakdown=breakdown,
                        fp_points_half=float(stats.get("points_half") or 0.0),
                    ))
                except (KeyError, TypeError, ValueError) as e:
                    raise FantasyProsError(f"schema drift in {pos_req} player record: {e}; record={p}") from e
        if len(out) < 100:
            raise FantasyProsError(f"only {len(out)} projections returned — refusing suspicious pull")
        return out

    def injuries(self, season: int) -> list[dict]:
        """Season is a QUERY param here, not a path segment — `{season}/injuries`
        returns a CloudFront MissingAuthenticationToken 403 (route does not exist),
        which reads like an auth failure but is not. Verified live 2026-08-16."""
        data = self._get("injuries", {"season": season})
        return data.get("injuries", data.get("players", []))

    def news(self, limit: int = 50) -> list[dict]:
        data = self._get("news", {"limit": limit})
        return data.get("items", data.get("news", []))

    def scored_points(self, season: int, week: int, position: str) -> list[dict]:
        data = self._get(f"{season}/points", {"position": position, "week": week})
        return data.get("players", [])

    def smoke_test(self, season: int = 2025, week: int = 1) -> str:
        """One cheap call to verify key + schema. Returns a summary line."""
        projs = self.weekly_projections(season, week, positions=["QB", "WR", "DST"])
        top = max(projs, key=lambda p: p.points)
        n_qb = sum(1 for p in projs if p.position == "QB")
        n_d = sum(1 for p in projs if p.position == "D")
        return (f"OK — {len(projs)} projections wk{week} {season} "
                f"({n_qb} QB / {n_d} DST); top FanDuel-scored: {top.name} {top.points:.2f} "
                f"(FP half-PPR says {top.fp_points_half:.2f})")
