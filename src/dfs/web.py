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
from .cli import _proposal_path
from .contest_spec import Profile, SlateType, expected_slate_type
from .results import ResultLog
from .objectives import weights_for

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
LINEUPS = DATA / "lineups"
DB = DATA / "results.db"
# Test sandbox: a "Test run" routes ALL writes (entry log, exports, snapshots) here,
# so a test can never touch season records regardless of what was typed in the form.
TEST = DATA / "test"
DB_TEST = TEST / "results.db"
LINEUPS_TEST = TEST / "lineups"
STATIC = Path(__file__).parent / "static"
SETTINGS_FILE = DATA / "settings.json"
DEFAULT_SETTINGS = {"me": "brettleath", "season": 2026, "field": 12,
                    "weekly_prize": 12.84, "grand_prizes": "135,81,54",
                    "weeks_total": 21, "contest": "Leather League"}


def _csv_slate_type(path: Path) -> "SlateType":
    """Detect FULL vs SINGLE_GAME from a stored salary CSV by counting teams —
    cheap header-only read, same rule ingest uses."""
    import csv as _csv
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            teams = {(r.get("Team") or "").strip().upper()
                     for r in _csv.DictReader(f)}
        teams.discard("")
        return SlateType.SINGLE_GAME if len(teams) == 2 else SlateType.FULL
    except Exception:                                          # noqa: BLE001
        return SlateType.FULL


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


@app.get("/api/schedule")
def api_schedule(season: int | None = None, week: int | None = None) -> dict:
    """The week's games grouped into slate windows, for the Week picker."""
    from .nflcal import current_week
    from .kickoffs import KickoffSchedule, ET
    wi = current_week(season_hint=season)
    season, week = season or wi.season, week or wi.week
    try:
        sched = KickoffSchedule.from_nflverse(season, week)
    except Exception as e:
        return {"season": season, "week": week, "windows": [], "error": str(e)}
    seen, games = set(), []
    for g in sched.by_team.values():
        key = tuple(sorted((g.team, g.opponent)))
        if key in seen:
            continue
        seen.add(key)
        et = g.kickoff_utc.astimezone(ET)
        games.append({"away": g.opponent if g.game_id.split("_")[-2:] and
                              g.game_id.endswith(g.team) else g.team,
                      "home": g.team, "teams": list(key),
                      "kickoff_et": et.strftime("%a %b %-d, %-I:%M %p"),
                      "iso": g.kickoff_utc.isoformat(),
                      "dow": et.weekday(), "hour": et.hour,
                      "game_id": g.game_id, "started": g.locked()})
    def window(g):
        # 2026 opens on a WEDNESDAY (Sep 9, NE@SEA) because the Thursday game is in
        # Melbourne — so weekday buckets must cover Wed/Fri too, not fall into "Other".
        d, h = g["dow"], g["hour"]
        if d == 2: return "Wednesday Kickoff"
        if d == 3: return "Thursday Night"
        if d == 4: return "Friday"
        if d == 5: return "Saturday"
        if d == 6 and h < 16: return "Sunday Early"
        if d == 6 and h < 19: return "Sunday Late"
        if d == 6: return "Sunday Night"
        if d == 0: return "Monday Night"
        return "Other"
    order = ["Wednesday Kickoff", "Thursday Night", "Friday", "Saturday",
             "Sunday Early", "Sunday Late", "Sunday Night", "Monday Night", "Other"]
    grouped: dict[str, list] = {}
    for g in sorted(games, key=lambda x: x["iso"]):
        grouped.setdefault(window(g), []).append(g)
    n_sun = sum(len(grouped.get(k, [])) for k in ("Sunday Early", "Sunday Late"))
    return {"season": season, "week": week, "label": f"{season} Week {week}",
            "sunday_main_games": n_sun,
            "windows": [{"name": k, "games": grouped[k]} for k in order if k in grouped]}


@app.get("/api/slates")
def api_slates(season: int, week: int) -> list[dict]:
    """Stored FanDuel CSVs for a week — upload once, reuse for every build/swap."""
    out = []
    for f in sorted(UPLOADS.glob(f"*{season}-w{week:02d}*.csv"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            head = f.open(encoding="utf-8-sig").readline()
            rows = sum(1 for _ in f.open()) - 1
        except OSError:
            continue
        kind = ("entry-history" if "Entry Id" in head else
                "salary" if "Salary" in head else "unknown")
        if kind != "salary":
            continue
        out.append({"file": f.name, "path": str(f), "rows": rows,
                    "uploaded": datetime.fromtimestamp(f.stat().st_mtime,
                                                       timezone.utc).isoformat()})
    return out


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
async def api_build(csv: UploadFile | None = File(None), season: int = Form(0),
                    week: int = Form(0), profile: str = Form("friends_league"),
                    leaderboard: str = Form("total_scores"),
                    field: int = Form(12), entry_fee: float = Form(0.0),
                    weekly_prize: float = Form(12.84),
                    grand_prizes: str = Form("135,81,54"),
                    weeks_total: int = Form(21), pool: int = Form(120),
                    show: int = Form(3), contest: str = Form("Leather League"),
                    prize_pool: str = Form(""), strict_injuries: bool = Form(False),
                    test_mode: bool = Form(False)):
    if not season or not week:
        from .nflcal import current_week
        wi = current_week()
        season, week = season or wi.season, week or wi.week
    UPLOADS.mkdir(parents=True, exist_ok=True)
    if csv is not None and csv.filename:
        dest = UPLOADS / f"{season}-w{week:02d}-{uuid.uuid4().hex[:6]}.csv"
        dest.write_bytes(await csv.read())
    else:
        # Reusing a stored CSV must respect the CONTEST, not just the week. Brett
        # plays single-game more often than league games, so both kinds of file exist
        # for the same week; picking the newest would hand a showdown player list to a
        # league build (which then silently builds a 6-man lineup).
        stored = sorted(UPLOADS.glob(f"*{season}-w{week:02d}*.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        stored = [s for s in stored
                  if "Salary" in s.open(encoding="utf-8-sig").readline()]
        if not stored:
            raise HTTPException(400, f"No stored salary CSV for {season} week {week} — "
                                     "upload one.")
        want = expected_slate_type(Profile(profile))
        if want is not None:
            typed = [s for s in stored if _csv_slate_type(s) == want]
            if not typed:
                raise HTTPException(
                    400, f"No stored {want.value} salary CSV for {season} week {week} "
                         f"— the {len(stored)} stored file(s) are for a different "
                         "contest type. Upload the right player list.")
            stored = typed
        dest = stored[0]
    slate_id = f"{season}-w{week:02d}"
    log_db, lineups_dir = (DB_TEST, LINEUPS_TEST) if test_mode else (DB, LINEUPS)
    lineups_dir.mkdir(parents=True, exist_ok=True)
    argv = ["build", "--csv", str(dest), "--season", str(season), "--week", str(week),
            "--profile", profile, "--leaderboard", leaderboard,
            "--field", str(field), "--entry-fee", str(entry_fee),
            "--weekly-prize", str(weekly_prize), "--grand-prizes", grand_prizes,
            "--weeks-total", str(weeks_total), "--pool", str(pool),
            "--show", str(show), "--contest", contest,
            "--critical-salary", "7000",
            "--log-db", str(log_db), "--auto-context", "--me", load_settings()["me"],
            "--export", str(lineups_dir / f"upload-{slate_id}.csv"),
            "--out", str(lineups_dir / f"{slate_id}.json")]
    if test_mode:
        argv += ["--snapshot-dir", str(TEST / "snapshots")]
    if prize_pool:
        argv += ["--prize-pool", prize_pool]
    if strict_injuries:
        argv += ["--strict-injuries"]
    return {"job": _start_job("build", argv), "slate_csv": str(dest)}


@app.post("/api/swap")
async def api_swap(csv: UploadFile | None = File(None),
                   slate_csv: str = Form(""), season: int = Form(0),
                   week: int = Form(0), contest: str = Form("Leather League"),
                   profile: str = Form("friends_league"),
                   test_mode: bool = Form(False)):
    if not season or not week:
        from .nflcal import current_week
        wi = current_week()
        season, week = season or wi.season, week or wi.week
    if csv is not None:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = UPLOADS / f"swap-{season}-w{week:02d}-{uuid.uuid4().hex[:6]}.csv"
        dest.write_bytes(await csv.read())
        path = str(dest)
    elif slate_csv:
        # Only stored uploads are acceptable — a raw form path would let any caller
        # feed an arbitrary server-readable file into the parser.
        p = Path(slate_csv).resolve()
        if not p.is_file() or UPLOADS.resolve() not in p.parents:
            raise HTTPException(400, "slate_csv must reference a stored upload")
        path = str(p)
    else:
        # reuse the most recent upload for this week
        cands = sorted(UPLOADS.glob(f"{season}-w{week:02d}-*.csv"))
        if not cands:
            raise HTTPException(400, "No slate CSV for this week — upload one.")
        path = str(cands[-1])
    log_db, lineups_dir = (DB_TEST, LINEUPS_TEST) if test_mode else (DB, LINEUPS)
    lineups_dir.mkdir(parents=True, exist_ok=True)
    argv = ["swap", "--csv", path, "--season", str(season), "--week", str(week),
            "--contest", contest, "--profile", profile,
            "--critical-salary", "7000",
            "--log-db", str(log_db),
            "--proposal-out", str(lineups_dir / Path(
                _proposal_path(season, week, contest)).name),
            "--export", str(lineups_dir / f"swap-{season}-w{week:02d}.csv")]
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
            "--contest", contest, "--log-db", str(DB),
            "--me", load_settings()["me"]]
    return {"job": _start_job("capture", argv)}


@app.get("/api/entry")
def api_entry(season: int, week: int, contest: str = "Leather League",
              test: bool = False):
    """The logged entry for a week, plus its shadow arm — the page renders these as
    readable lineup cards so a lineup can be hand-entered into FanDuel without
    reading console text."""
    import sqlite3
    db = DB_TEST if test else DB
    if not Path(db).exists():
        raise HTTPException(404, "no results database yet")
    rl = ResultLog(db)
    def _row_to_dict(r):
        if r is None:
            return None
        d = dict(r)
        d["lineup"] = json.loads(d.pop("lineup_json"))
        arm = "model"
        if (d.get("objective") or "").startswith("arm=max-proj"):
            arm = "max-proj"
        d["arm"] = arm
        d["status"] = d.get("status") or "pending"
        return d

    def _augment(d):
        if d is not None:
            d["season"], d["week"], d["contest"] = season, week, contest
        return d
    with rl._c() as c:
        row = c.execute("""SELECT * FROM entries WHERE season=? AND week=? AND
                           contest=?""", (season, week, contest)).fetchone()
        shadow = c.execute("""SELECT * FROM entries WHERE season=? AND week=? AND
                              contest LIKE ?""",
                           (season, week, f"{contest} [shadow:%")).fetchone()
    if row is None:
        raise HTTPException(404, f"no logged entry for {contest} {season} w{week}")
    return {"entry": _augment(_row_to_dict(row)),
            "shadow": _row_to_dict(shadow)}


@app.get("/api/job-latest/{kind}")
def api_job_latest(kind: str):
    """Most recent job of a kind, so the page can RECONNECT when the POST response
    is lost in transit (observed 2026-08-23: the build POST reached the app and
    returned 200, but Safari never received the body and reported 'Load failed'
    while the build ran on happily). Also lets the page re-attach after a reload,
    a closed laptop, or a dropped tunnel mid-build."""
    with _jobs_lock:
        jobs = [j for j in _jobs.values() if j["kind"] == kind]
    if not jobs:
        raise HTTPException(404, "no jobs of that kind yet")
    return max(jobs, key=lambda j: j["started"])


@app.post("/api/confirm-entry")
def api_confirm_entry(season: int = Form(...), week: int = Form(...),
                      contest: str = Form("Leather League"),
                      test_mode: bool = Form(False)):
    """Promote the build recommendation to the confirmed active entry."""
    log_db = DB_TEST if test_mode else DB
    return {"job": _start_job("confirm-entry",
                              ["confirm-entry", "--season", str(season),
                               "--week", str(week), "--contest", contest,
                               "--log-db", str(log_db)])}


@app.post("/api/swap-accept")
def api_swap_accept(season: int = Form(...), week: int = Form(...),
                    contest: str = Form("Leather League"),
                    test_mode: bool = Form(False)):
    """Record that the last proposed swap was actually entered on FanDuel."""
    log_db, lineups_dir = (DB_TEST, LINEUPS_TEST) if test_mode else (DB, LINEUPS)
    argv = ["swap-accept", "--season", str(season), "--week", str(week),
            "--contest", contest, "--log-db", str(log_db),
            "--proposal", str(lineups_dir / Path(
                _proposal_path(season, week, contest)).name)]
    return {"job": _start_job("swap-accept", argv)}


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/standings")
def api_standings(season: int, me: str = "", weeks_total: int = 21,
                  contest: str = ""):
    s = load_settings()
    me = me or s["me"]
    contest = contest or s["contest"]        # league scope by default
    rl = ResultLog(DB)
    st = rl.standings(season, contest_like=contest)
    ctx = rl.season_context(season, me, weeks_total, contest_like=contest)
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
