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


def read_snapshot(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)
