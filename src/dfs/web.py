"""dfs web — the dashboard backend.

One FastAPI app, four jobs: build lineups from an uploaded FanDuel CSV, run the
Sunday swap check, ingest a pasted contest results page, and show the season.

Long operations (build/swap pull FantasyPros + Vegas and simulate ~20k slates, ~60-90s)
run in a background thread with a job record the page polls, so the UI never hangs on a
request. Job output is the same text the CLI prints — one code path, no drift between
what the terminal says and what the page says.

Security model: the app binds to 127.0.0.1 by default and is exposed only through the
Cloudflare tunnel behind CF Access. It has no auth of its own by design — do not
port-forward it.
"""
from __future__ import annotations
import io
import json
import threading
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from . import cli as dfs_cli
from .results import ResultLog
from .objectives import weights_for

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
LINEUPS = DATA / "lineups"
DB = DATA / "results.db"
STATIC = Path(__file__).parent / "static"
SETTINGS_FILE = DATA / "settings.json"
DEFAULT_SETTINGS = {"me": "brettleath", "season": 2026, "field": 12,
                    "weekly_prize": 12.84, "grand_prizes": "135,81,54",
                    "weeks_total": 21, "contest": "Leather League"}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_FILE.read_text())}
        except json.JSONDecodeError:
            pass
    return dict(DEFAULT_SETTINGS)

app = FastAPI(title="dfs-v6", docs_url=None, redoc_url=None)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _run_job(job_id: str, argv: list[str]) -> None:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = dfs_cli.main(argv)
        status = "done" if rc == 0 else "failed"
    except Exception:
        buf.write("\n" + traceback.format_exc())
        status = "failed"
    with _jobs_lock:
        _jobs[job_id].update(status=status, output=buf.getvalue(),
                             finished=datetime.now(timezone.utc).isoformat())


def _start_job(kind: str, argv: list[str]) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "status": "running",
                         "output": "", "argv": argv,
                         "started": datetime.now(timezone.utc).isoformat()}
    threading.Thread(target=_run_job, args=(job_id, argv), daemon=True).start()
    return job_id


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/calendar")
def api_calendar(season: int | None = None) -> dict:
    """What week the system thinks it is. The UI prefills from this; every form
    still accepts an override."""
    from .nflcal import current_week
    wi = current_week(season_hint=season)
    return {"season": wi.season, "week": wi.week, "reason": wi.reason,
            "label": wi.label, "summary": wi.summary(),
            "days_to_kickoff": wi.days_to_kickoff(),
            "preseason": wi.is_preseason, "postseason": wi.is_postseason}


@app.get("/api/settings")
def api_settings_get() -> dict:
    return load_settings()


@app.post("/api/settings")
async def api_settings_post(payload: dict) -> dict:
    s = {**load_settings(), **{k: v for k, v in payload.items() if k in DEFAULT_SETTINGS}}
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=1))
    return s


@app.get("/health")
def health() -> dict:
    dist = DATA / "distributions.json"
    meta = {}
    if dist.exists():
        meta = json.loads(dist.read_text()).get("meta", {})
    calibrated = "actual/FP" in str(meta.get("method", ""))
    return {"ok": True, "calibrated_distributions": calibrated,
            "dist_meta": meta, "db": DB.exists(),
            "time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/build")
async def api_build(csv: UploadFile = File(...), season: int = Form(0),
                    week: int = Form(0), profile: str = Form("friends_league"),
                    leaderboard: str = Form("total_scores"),
                    field: int = Form(12), entry_fee: float = Form(0.0),
                    weekly_prize: float = Form(12.84),
                    grand_prizes: str = Form("135,81,54"),
                    weeks_total: int = Form(21), pool: int = Form(120),
                    show: int = Form(3), contest: str = Form("Leather League"),
                    prize_pool: str = Form(""), strict_injuries: bool = Form(False)):
    if not season or not week:
        from .nflcal import current_week
        wi = current_week()
        season, week = season or wi.season, week or wi.week
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"{season}-w{week:02d}-{uuid.uuid4().hex[:6]}.csv"
    dest.write_bytes(await csv.read())
    slate_id = f"{season}-w{week:02d}"
    argv = ["build", "--csv", str(dest), "--season", str(season), "--week", str(week),
            "--profile", profile, "--leaderboard", leaderboard,
            "--field", str(field), "--entry-fee", str(entry_fee),
            "--weekly-prize", str(weekly_prize), "--grand-prizes", grand_prizes,
            "--weeks-total", str(weeks_total), "--pool", str(pool),
            "--show", str(show), "--contest", contest,
            "--critical-salary", "7000",
            "--log-db", str(DB), "--auto-context", "--me", load_settings()["me"],
            "--export", str(LINEUPS / f"upload-{slate_id}.csv"),
            "--out", str(LINEUPS / f"{slate_id}.json")]
    if prize_pool:
        argv += ["--prize-pool", prize_pool]
    if strict_injuries:
        argv += ["--strict-injuries"]
    return {"job": _start_job("build", argv), "slate_csv": str(dest)}


@app.post("/api/swap")
async def api_swap(csv: UploadFile | None = File(None),
                   slate_csv: str = Form(""), season: int = Form(0),
                   week: int = Form(0), contest: str = Form("Leather League")):
    if not season or not week:
        from .nflcal import current_week
        wi = current_week()
        season, week = season or wi.season, week or wi.week
    if csv is not None:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS / f"swap-{season}-w{week:02d}-{uuid.uuid4().hex[:6]}.csv"
        dest.write_bytes(await csv.read())
        path = str(dest)
    elif slate_csv and Path(slate_csv).exists():
        path = slate_csv
    else:
        # reuse the most recent upload for this week
        cands = sorted(UPLOADS.glob(f"{season}-w{week:02d}-*.csv"))
        if not cands:
            raise HTTPException(400, "No slate CSV for this week — upload one.")
        path = str(cands[-1])
    argv = ["swap", "--csv", path, "--season", str(season), "--week", str(week),
            "--contest", contest, "--critical-salary", "7000",
            "--log-db", str(DB),
            "--export", str(LINEUPS / f"swap-{season}-w{week:02d}.csv")]
    return {"job": _start_job("swap", argv)}


@app.post("/api/capture")
async def api_capture(page: UploadFile | None = File(None),
                      text: str = Form(""), season: int = Form(0),
                      week: int = Form(0), contest: str = Form("Leather League")):
    if not season or not week:
        from .nflcal import current_week
        wi = current_week()
        season, week = season or wi.season, week or wi.week
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / f"capture-{season}-w{week:02d}.txt"
    if page is not None:
        raw = await page.read()
        dest = dest.with_suffix(Path(page.filename or "x.txt").suffix or ".txt")
        dest.write_bytes(raw)
    elif text.strip():
        dest.write_text(text)
    else:
        raise HTTPException(400, "Paste the contest page text or upload the file.")
    argv = ["capture", str(dest), "--season", str(season), "--week", str(week),
            "--contest", contest, "--log-db", str(DB)]
    return {"job": _start_job("capture", argv)}


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/standings")
def api_standings(season: int, me: str = "", weeks_total: int = 21):
    me = me or load_settings()["me"]
    rl = ResultLog(DB)
    st = rl.standings(season)
    ctx = rl.season_context(season, me, weeks_total)
    w = weights_for("friends_league", ctx)
    return {
        "standings": [{"entrant": s.entrant, "total": round(s.total_points, 2),
                       "avg": round(s.avg, 2), "weeks": s.weeks, "wins": s.wins,
                       "best": round(s.best, 2), "me": s.entrant == me}
                      for s in st],
        "context": {"weeks_played": ctx.weeks_played, "weeks_left": ctx.weeks_left,
                    "my_points": ctx.my_points, "leader_points": ctx.leader_points,
                    "deficit": round(ctx.deficit(), 2)},
        "objective": {"w_points": round(w.w_points, 4), "w_win": w.w_win,
                      "rationale": w.rationale},
        "accuracy": rl.projection_accuracy(season),
    }


@app.get("/api/lineups")
def api_lineups():
    out = []
    for f in sorted(LINEUPS.glob("*.json"), reverse=True)[:12]:
        try:
            d = json.loads(f.read_text())
            out.append({"file": f.name, "slate_id": d.get("slate_id"),
                        "week": d.get("week"), "season": d.get("season"),
                        "lineups": d.get("lineups", [])[:3]})
        except (json.JSONDecodeError, OSError):
            continue
    return out


@app.get("/api/download/{name}")
def api_download(name: str):
    p = (LINEUPS / name).resolve()
    if not str(p).startswith(str(LINEUPS.resolve())) or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p, filename=name)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8093)   # container-internal; tunnel handles exposure


if __name__ == "__main__":
    main()
