from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import List
import json

from app.config import APP_PORT, LOG_LEVEL, BASELINE_WEEK
from app.models import Player, PlayersResponse, OptimizeResponse, Lineup
from app.utils import ensure_dirs
from app.data_sources import load_players
from app.odds import fetch_odds, implied_total_boost
from app.weather import weather_adjustment
from app.ai import tweak_projections_with_ai, ai_available
from app.optimizer import select_optimal

app = FastAPI(title="FanDuel LeaguePicks", version="1.1.1")

@app.on_event("startup")
def _startup():
    ensure_dirs()

@app.get("/", response_class=PlainTextResponse)
def root():
    return "FanDuel LeaguePicks is running. Try /players/current or /optimize/text?game_type=gpp"

def _apply_context_adjustments(players: List[Player]) -> List[Player]:
    odds_blob = fetch_odds()
    updated = []
    for p in players:
        mult = implied_total_boost(p.team, odds_blob)
        proj = p.projection * mult
        updated.append(p.copy(update={"projection": proj}))
    return updated

@app.get("/players/current", response_model=PlayersResponse)
def get_players_current():
    base = load_players()
    with_odds = _apply_context_adjustments(base)
    final_players, ai_note = tweak_projections_with_ai(with_odds, context_hint="NFL main slate focus.")
    return PlayersResponse(week=BASELINE_WEEK, count=len(final_players), players=final_players)

@app.get("/optimize/json", response_model=OptimizeResponse)
def optimize_json(game_type: str = Query("gpp")):
    base = load_players()
    adjusted = _apply_context_adjustments(base)
    players_ai, ai_comment = tweak_projections_with_ai(adjusted, context_hint=f"Optimize for {game_type.upper()} strategy.")
    chosen, total_salary, total_proj, notes = select_optimal(players_ai, game_type=game_type)  # type: ignore
    lineup = Lineup(players=chosen, total_salary=total_salary, projected_points=total_proj,
                    game_type=game_type, notes=notes, ai_commentary=ai_comment)
    meta = {"ai_enabled": ai_available()}
    return OptimizeResponse(lineup=lineup, metadata=meta)

@app.get("/optimize/text", response_class=PlainTextResponse)
def optimize_text(game_type: str = Query("gpp")):
    res = optimize_json(game_type=game_type)
    l = res.lineup
    lines = []
    lines.append(f"Week: {BASELINE_WEEK}")
    lines.append(f"Game type: {l.game_type}")
    lines.append(f"Projected points: {l.projected_points:.2f}")
    lines.append(f"Salary: {l.total_salary}")
    lines.append("Roster:")
    for p in l.players:
        lines.append(f" - {p.position}  {p.name} ({p.team})  ${p.salary}  proj {p.projection:.2f}")
    if l.notes:
        lines.append("")
        lines.append(f"Optimizer notes: {l.notes}")
    if l.ai_commentary:
        lines.append("")
        lines.append(f"AI: {l.ai_commentary}")
    return "\n".join(lines)

@app.post("/ingest/players")
async def ingest_players(file: UploadFile = File(...)):
    from app.config import INPUT_DIR
    path = INPUT_DIR / "players.sample.json"
    raw = await file.read()
    data = json.loads(raw.decode("utf-8"))
    _ = [Player(**p) for p in data]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return JSONResponse({"ok": True, "count": len(data), "saved": str(path)})
