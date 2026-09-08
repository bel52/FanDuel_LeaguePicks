"""At-lock projection snapshots.

Projection revisions are invisible after the fact unless the exact raw payload is
frozen at build time. This module writes one gzip-compressed JSON per build:

  * every raw FantasyPros response + the request params that produced it
  * retrieval timestamp (UTC) and a SHA-256 over the canonical payload
  * a SHA-256 of dfs/scoring.py source, so a later re-score can tell a code change
    from a projection revision (the two are otherwise indistinguishable)

Design rules:
  * Snapshots must NEVER fail a build — any error degrades to a printed warning.
  * Snapshots live under the persistent data dir (bind-mounted in the container),
    and deploy tarballs never touch data/, so they survive deploys.
  * Same-input re-runs overwrite deterministically (name keys on season/week/
    slate_id, not wall-clock), so a Wednesday rebuild does not litter.
"""
from __future__ import annotations
import gzip
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scorer_source_sha() -> str:
    from . import scoring
    return hashlib.sha256(inspect.getsource(scoring).encode()).hexdigest()


def write_snapshot(directory: str | Path, season: int, week: int, slate_id: str,
                   fp_raw: dict, extra: dict | None = None) -> Path | None:
    """Freeze the raw projection pull. Returns the path, or None on any failure."""
    try:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "fp_at_lock",
            "season": season, "week": week, "slate_id": slate_id,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "payload_sha256": _sha256(fp_raw),
            "scorer_source_sha256": scorer_source_sha(),
            "raw": fp_raw,
        }
        if extra:
            payload["extra"] = extra
        path = d / f"fp-{season}-w{week:02d}-{slate_id or 'noslate'}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        return path
    except Exception as e:                                    # noqa: BLE001
        print(f"  WARNING: snapshot not written ({e}) — build continues")
        return None


def write_props_snapshot(directory: str | Path, season: int, week: int,
                         slate_id: str, boards: dict,
                         extra: dict | None = None) -> Path | None:
    """Freeze the raw prop boards, one file per build.

    The FantasyPros payload has been snapshotted since v6; the prop boards were not,
    and they are the more perishable of the two. The disk cache has a six-hour TTL and
    is overwritten by the next pull, and The Odds API has no free historical endpoint
    for player props -- so a week that passes unsnapshotted can never have its market
    projection re-derived. Nothing about props can then be re-fitted retroactively:
    not the blend weight, not the two coarse TD constants, not a corrected parser.

    That makes this the one piece of the learning loop that must exist BEFORE Week 1
    rather than being added once there is something to learn from.

    `boards` is {event_id: raw board payload} exactly as returned.
    """
    try:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "props_at_lock",
            "season": season, "week": week, "slate_id": slate_id,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "payload_sha256": _sha256(boards),
            "scorer_source_sha256": scorer_source_sha(),
            # The derivation constants live in props.py, so a later re-derivation must
            # be able to tell a constant change from a market move.
            "props_source_sha256": props_source_sha(),
            "n_events": len(boards),
            "raw": boards,
        }
        if extra:
            payload["extra"] = extra
        path = d / f"props-{season}-w{week:02d}-{slate_id or 'noslate'}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        return path
    except Exception as e:                                    # noqa: BLE001
        print(f"  WARNING: props snapshot not written ({e}) — build continues")
        return None


def props_source_sha() -> str:
    from . import props
    return hashlib.sha256(inspect.getsource(props).encode()).hexdigest()


def read_snapshot(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
