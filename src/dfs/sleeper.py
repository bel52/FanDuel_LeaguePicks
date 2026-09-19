"""Layer 3 of the injury pipeline: official game-status designations via Sleeper.

Sleeper mirrors the NFL's official designations (Questionable / Doubtful / Out / IR /
PUP / Sus) and keeps them current through game day, including the inactive
declarations that post ~90 minutes before each kickoff, which arrive as
`injury_status = "Out"`. It is free, unauthenticated JSON. It is NOT the NFL's own
feed: it can lag the official list by minutes, and it resets game-week designations
early Wednesday, so a Sunday `Out` never survives into the following week's build.

Rollout is deliberate. `--official-inactives log` (the default) fetches, compares
against what the CSV + FantasyPros layers already know, prints every disagreement,
and GATES NOTHING. `gate` merges the Sleeper records into the sweep so a Sleeper
`Out` removes the player from the pool exactly as a FantasyPros `Out` does. `off`
skips the fetch. Promotion from log to gate is one word on the command line after a
Sunday of side-by-side output shows the source agrees with the official list.

Courtesy: Sleeper asks that the players endpoint (~5 MB, every NFL player) be
fetched sparingly. It is cached on disk and reused inside `max_age_hours`, so a
build plus two Sunday swap checks costs at most three fetches.
"""
from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .injuries import InjuryRecord, Status, parse_status
from .matching import norm_team

PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K"}

# Sleeper's `injury_status` vocabulary, observed. Anything not here falls through to
# injuries.parse_status, which already understands the common words.
_STATUS_MAP = {
    "OUT": Status.OUT,
    "DOUBTFUL": Status.DOUBTFUL,
    "QUESTIONABLE": Status.QUESTIONABLE,
    "IR": Status.IR,
    "PUP": Status.IR,
    "NFI": Status.IR,
    "SUS": Status.IR,          # suspended: cannot play, same as reserve for our purposes
    "COV": Status.OUT,         # COVID reserve (historical vocabulary, kept for safety)
    "DNR": Status.OUT,         # did not report
    "NA": Status.IR,           # reserve / not active
}


class SleeperError(RuntimeError):
    pass


class SleeperClient:
    """Fetch the Sleeper NFL players map with an on-disk cache.

    `players()` returns (payload, age_hours, from_cache). A cached copy younger than
    `max_age_hours` is used without a network call.
    """

    def __init__(self, cache_dir: str | Path | None = "data/cache",
                 max_age_hours: float = 1.0, timeout: int = 40):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_age_hours = float(max_age_hours)
        self.timeout = timeout

    # -- cache -------------------------------------------------------------
    def _paths(self) -> tuple[Path, Path] | None:
        if not self.cache_dir:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return (self.cache_dir / "sleeper-players.json",
                self.cache_dir / "sleeper-players.meta.json")

    def _read_cache(self) -> tuple[dict, float] | None:
        paths = self._paths()
        if not paths:
            return None
        body, meta = paths
        if not (body.is_file() and meta.is_file()):
            return None
        try:
            fetched = float(json.loads(meta.read_text()).get("fetched_epoch", 0))
            age_h = (time.time() - fetched) / 3600.0
            if age_h > self.max_age_hours:
                return None
            return json.loads(body.read_text()), age_h
        except (ValueError, OSError):
            return None

    def _write_cache(self, payload: dict) -> None:
        paths = self._paths()
        if not paths:
            return
        body, meta = paths
        tmp = body.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        tmp.replace(body)
        meta.write_text(json.dumps({"fetched_epoch": time.time(),
                                    "fetched_iso": datetime.now(timezone.utc).isoformat(),
                                    "url": PLAYERS_URL}))

    # -- fetch -------------------------------------------------------------
    def players(self) -> tuple[dict, float, bool]:
        cached = self._read_cache()
        if cached is not None:
            payload, age_h = cached
            return payload, age_h, True
        req = urllib.request.Request(PLAYERS_URL, headers={
            "User-Agent": "dfs-v6 inactives layer (contact: leathfam.com)",
            "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raise SleeperError(f"HTTP {e.code} from Sleeper players endpoint") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise SleeperError(f"Sleeper players endpoint unreachable: {e}") from e
        try:
            payload = json.loads(raw)
        except ValueError as e:
            raise SleeperError("Sleeper players payload was not JSON") from e
        if not isinstance(payload, dict) or len(payload) < 1000:
            # The real map has ~10k entries. A tiny or non-dict body is an error page.
            raise SleeperError(f"Sleeper payload looks wrong ({type(payload).__name__}, "
                               f"{len(payload) if hasattr(payload, '__len__') else '?'} "
                               "entries)")
        self._write_cache(payload)
        return payload, 0.0, False


def _iso_from_epoch_ms(v) -> str | None:
    try:
        ms = int(v)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    # Sleeper stamps `news_updated` in epoch milliseconds.
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def records_from_sleeper(payload: dict) -> dict[str, InjuryRecord]:
    """Normalize the Sleeper players map into InjuryRecords keyed by norm_name.

    Only players with a team and a fantasy position are considered; team DEF entries
    and free agents are skipped. Only `injury_status` is read — Sleeper's roster
    `status` field ("Active" / "Inactive" / "Injured Reserve" ...) describes roster
    membership, and its "Inactive" means off-roster, NOT a game-day scratch, so it is
    deliberately ignored to avoid false removals.
    """
    out: dict[str, InjuryRecord] = {}
    now_iso = datetime.now(timezone.utc).isoformat()
    for _pid, p in (payload or {}).items():
        if not isinstance(p, dict):
            continue
        team = p.get("team")
        pos = p.get("position")
        if not team or pos not in FANTASY_POSITIONS:
            continue
        raw = p.get("injury_status")
        if not raw:
            continue
        key = str(raw).strip().upper()
        status = _STATUS_MAP.get(key) or parse_status(str(raw))
        if status is Status.UNKNOWN:
            continue
        name = (p.get("full_name")
                or f"{p.get('first_name', '')} {p.get('last_name', '')}").strip()
        if not name:
            continue
        bits = [str(p.get("injury_body_part") or "").strip(),
                str(p.get("injury_notes") or "").strip()[:60]]
        rec = InjuryRecord(name=name,
                           team=norm_team(str(team)),
                           status=status,
                           detail=" · ".join(b for b in bits if b),
                           source="sleeper",
                           fetched_ts=_iso_from_epoch_ms(p.get("news_updated")) or now_iso)
        out[rec.key] = rec
    return out
