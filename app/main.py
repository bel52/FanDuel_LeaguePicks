import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.kickoff_times import (
    get_schedule,
    build_kickoff_map,
    auto_lock_started_players,
    persist_kickoffs,
    weekly_csv_path,
    snapshot_weekly_csv,
)


class Settings:
    def __init__(self) -> None:
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8010"))
        self.data_dir: str = os.getenv("DATA_DIR", "data")
        self.input_dir: str = os.path.join(self.data_dir, "input")
        self.output_dir: str = os.path.join(self.data_dir, "output")
        self.weekly_dir: str = os.path.join(self.data_dir, "weekly")
        self.log_dir: str = os.getenv("LOG_DIR", "logs")
        self.kickoff_api_url: Optional[str] = os.getenv("KICKOFF_API_URL")


settings = Settings()

os.makedirs(settings.log_dir, exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("app.main")

_REQUIRED_INPUTS = ("qb.csv", "rb.csv", "wr.csv", "te.csv", "dst.csv")


def _data_is_present() -> bool:
    try:
        if not os.path.isdir(settings.input_dir):
            return False
        return all(os.path.isfile(os.path.join(settings.input_dir, f)) for f in _REQUIRED_INPUTS)
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting FanDuel DFS Optimizer...")

    for d in (settings.data_dir, settings.input_dir, settings.output_dir, settings.weekly_dir, settings.log_dir):
        os.makedirs(d, exist_ok=True)

    # Warm kickoff cache; if we have kickoff data, persist JSON and (if missing) a weekly CSV snapshot.
    try:
        team_map = build_kickoff_map(settings.kickoff_api_url)
        if team_map:
            persist_kickoffs(team_map, output_path=os.path.join(settings.output_dir, "kickoffs.json"))
            csv_path = weekly_csv_path()
            if not os.path.exists(csv_path):
                # Make initial weekly snapshot if none exists yet
                snapshot_weekly_csv(team_map=team_map)
        else:
            log.warning("No kickoff data available at startup (API and saved file both empty).")
    except Exception as e:
        log.error("Failed to warm kickoff data: %s", e)

    yield
    return


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> JSONResponse:
    data_status = "operational" if _data_is_present() else "no_data"
    try:
        sched = get_schedule(settings.kickoff_api_url)
        kickoff_status = "operational" if sched else "no_kickoffs"
    except Exception as e:
        kickoff_status = f"error: {e}"

    body: Dict[str, Any] = {
        "status": "healthy",
        "components": {"api": "operational", "data": data_status, "kickoffs": kickoff_status},
    }
    return JSONResponse(body)


@app.get("/schedule")
def schedule() -> JSONResponse:
    try:
        games = get_schedule(settings.kickoff_api_url) or []
        # Persist normalized kickoffs.json and ensure weekly CSV exists
        if games:
            team_map: Dict[str, str] = {}
            for g in games:
                iso = str(g.get("kickoff"))
                for t in g.get("teams", []):
                    team_map[str(t).upper()] = iso
            persist_kickoffs(team_map, output_path=os.path.join(settings.output_dir, "kickoffs.json"))
            csv_path = weekly_csv_path()
            if not os.path.exists(csv_path):
                snapshot_weekly_csv(games=games)
        return JSONResponse(games)
    except Exception as e:
        log.error("Schedule fetch failed: %s", e)
        return JSONResponse([])


@app.post("/kickoffs/snapshot")
def kickoffs_snapshot(week: Optional[str] = None) -> JSONResponse:
    """
    Force-create the weekly kickoff CSV (and refresh kickoffs.json) for this week.
    Use when you've just set KICKOFF_API_URL or dropped a saved file.
    """
    games = get_schedule(settings.kickoff_api_url)
    if not games:
        return JSONResponse({"saved": False, "reason": "no schedule available from API/CSV/JSON"}, status_code=400)

    # Persist JSON + CSV
    team_map: Dict[str, str] = {}
    for g in games:
        iso = str(g.get("kickoff"))
        for t in g.get("teams", []):
            team_map[str(t).upper()] = iso
    persist_kickoffs(team_map, output_path=os.path.join(settings.output_dir, "kickoffs.json"))
    saved_to = snapshot_weekly_csv(games=games, week_id=week)

    return JSONResponse({"saved": True, "csv_path": saved_to, "games": len(games)})


@app.get("/optimize_text")
def optimize_text(width: int = Query(110, ge=40, le=160)) -> PlainTextResponse:
    if not _data_is_present():
        return PlainTextResponse("No player data available. Please place CSVs under data/input")
    lines: List[str] = [
        f"Optimizer is wired. Text width={width}.",
        "Hook in your lineup generation where this message is returned.",
    ]
    return PlainTextResponse("\n".join(lines))


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse({"ok": True})
