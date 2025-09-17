from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Optional, List, Dict, Any

from app.data_ingestion import weekly_player_pool
from app.salary_provider import attach_salaries, FD_CAP
from app.enhanced_optimizer import EnhancedDFSOptimizer

app = FastAPI(title="FanDuel DFS Optimizer")

@app.get("/")
def root():
    return {"ok": True, "version": "live-2025-09-16-c"}

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "live-2025-09-16-c"}

@app.get("/__debug")
def debug_info():
    routes = [r.path for r in app.router.routes]
    return {
        "version": "live-2025-09-16-c",
        "routes": routes,
        "ingestion_has_weekly_player_pool": True,
        "ingestion_defs": ["weekly_player_pool"],
    }

@app.get("/players/current")
def players_current() -> Dict[str, Any]:
    players, meta = weekly_player_pool()
    payload = attach_salaries(players)
    return {
        "count": len(payload["players"]),
        "players": payload["players"],
        "meta": {**meta, "salary_source": payload["salary_source"], "cap": payload["cap"]},
    }

@app.get("/optimize")
def optimize(
    game_type: Optional[str] = "league",
    num_lineups: Optional[int] = 1,  # reserved for portfolio; returns 1 for now
    lock: Optional[List[str]] = None,
    ban: Optional[List[str]] = None
):
    players, meta = weekly_player_pool()
    payload = attach_salaries(players)
    opt = EnhancedDFSOptimizer(payload["players"], cap=FD_CAP, locks=lock, bans=ban)
    lineup, info = opt.optimize_one()

    if not lineup:
        return JSONResponse({
            "game_type": game_type,
            "lineups": [{"players": [], "total_proj": 0, "total_salary": 0}],
            "meta": {**meta, **info}
        })

    total_salary = sum(int(p["salary"]) for p in lineup)
    return JSONResponse({
        "game_type": game_type,
        "lineups": [{
            "players": lineup,
            "total_proj": info["total_proj"],
            "total_salary": total_salary,
        }],
        "meta": {**meta, "cap": FD_CAP, "cap_used": info["cap_used"], "salary_source": "proxy"}
    })

@app.get("/optimize/text")
def optimize_text(
    game_type: Optional[str] = "league",
    lock: Optional[List[str]] = None,
    ban: Optional[List[str]] = None
):
    players, meta = weekly_player_pool()
    payload = attach_salaries(players)
    opt = EnhancedDFSOptimizer(payload["players"], cap=FD_CAP, locks=lock, bans=ban)
    lineup, info = opt.optimize_one()
    if not lineup:
        warns = (meta.get("warnings") or []) + (info.get("warnings") or [])
        msg = "No valid lineup."
        if warns:
            msg += " " + "; ".join(warns)
        return PlainTextResponse(msg, status_code=200)

    order = ["QB", "RB", "WR", "TE", "FLEX", "DST"]
    by_slot: Dict[str, List[Dict[str, Any]]] = {k: [] for k in order}
    for p in lineup:
        by_slot.setdefault(p["pos_out"], []).append(p)

    lines = []
    for slot in order:
        for p in by_slot.get(slot, []):
            opp = p.get("opponent", "") or ""
            opp_str = f" vs {opp}" if opp else ""
            lines.append(f"{slot}: {p['name']} ({p['team']}{opp_str}) — {float(p['proj']):.2f} proj — ${int(p['salary']):,}")

    total_proj = info["total_proj"]
    total_salary = sum(int(p["salary"]) for p in lineup)
    lines.append("")
    lines.append(f"Total Salary: ${total_salary:,} / ${FD_CAP:,}")
    lines.append(f"Projected Total: {total_proj:.2f}")

    warns = (meta.get("warnings") or []) + (info.get("warnings") or [])
    if warns:
        lines.append("")
        for w in warns:
            lines.append(f"NOTE: {w}")

    return PlainTextResponse("\n".join(lines), status_code=200)
