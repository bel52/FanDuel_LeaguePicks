import logging
from typing import List, Dict, Any
from utils import http_get
from models import Player, Position

logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.sources = [
            # Keep/restore your real sources here, e.g.:
            # {"name": "primary", "url": "https://..."},
        ]

    # ---------- async/sync context support ----------
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False  # do not suppress exceptions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
    # ------------------------------------------------

    # --------- what main.py expects ----------
    async def collect_all_data(self) -> Dict[str, Any]:
        """
        Async wrapper used by main.py.
        Returns a dict with keys: 'players', 'games' (and optional 'meta').
        """
        players = self.collect()
        # If you have a real games source, populate it here; keep empty list otherwise.
        games: List[Any] = []
        return {"players": players, "games": games, "meta": {"sources": len(self.sources), "player_count": len(players)}}
    # ----------------------------------------

    def _normalize_json(self, data: Any) -> Any:
        """Ensure we work with dict/list; unwrap {'_raw': '...'} if needed."""
        if isinstance(data, dict) and "_raw" in data:
            raw = data["_raw"]
            try:
                import json
                s = raw.strip() if isinstance(raw, str) else ""
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    return json.loads(s)
            except Exception:
                pass
            return {}
        return data

    def _players_from_payload(self, payload: Any) -> List[Player]:
        """Convert payload into Player objects."""
        payload = self._normalize_json(payload)
        players: List[Player] = []

        rows = []
        if isinstance(payload, dict):
            rows = payload.get("players") or payload.get("data") or payload.get("rows") or []
        elif isinstance(payload, list):
            rows = payload

        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                name = row.get("name") or row.get("player") or row.get("Player")
                pos = row.get("position") or row.get("pos") or row.get("Position")
                team = row.get("team") or row.get("Team")
                salary = int(row.get("salary") or row.get("Salary") or 0)
                proj = float(row.get("projection") or row.get("proj") or row.get("ProjectedPoints") or 0.0)
                ownership = row.get("ownership") or row.get("Ownership") or None

                if not name or not pos or salary <= 0:
                    continue

                # Coerce position to enum
                try:
                    position = Position(pos.upper())
                except Exception:
                    continue

                p = Player(
                    id=str(row.get("id") or row.get("Id") or row.get("player_id") or name),
                    name=name,
                    team=team or "",
                    position=position,
                    salary=salary,
                    projected_points=proj,
                    ownership_projection=float(ownership) if ownership not in (None, "") else None
                )
                players.append(p)
            except Exception as e:
                logger.debug(f"Skipping row due to parse error: {e}")
        return players

    def collect(self) -> List[Player]:
        """Collect and merge players from multiple sources with basic de-dup."""
        all_players: Dict[str, Player] = {}

        for idx, src in enumerate(self.sources):
            url = src.get("url")
            if not url:
                continue
            try:
                payload = http_get(url)
                players = self._players_from_payload(payload)
                for p in players:
                    # prefer higher projection for duplicates
                    if p.id not in all_players or p.projected_points > all_players[p.id].projected_points:
                        all_players[p.id] = p
            except Exception as e:
                logger.error(f"Task {idx} failed: {e}")

        players_list = list(all_players.values())

        # Guardrail: if tiny pool, try local sample
        if len(players_list) < 10:
            logger.warning(f"Player pool too small ({len(players_list)}). Falling back to local sample if available.")
            try:
                import json, pathlib
                sample = pathlib.Path("data/sample_players.json")
                if sample.exists():
                    with sample.open() as f:
                        sample_payload = json.load(f)
                    players_list = self._players_from_payload(sample_payload)
            except Exception as e:
                logger.error(f"Fallback sample load failed: {e}")

        logger.info(f"Collected {len(players_list)} players after merge.")
        return players_list

# Back-compat singleton if other modules import `collector`
collector = DataCollector()
