from fastapi import FastAPI, Query
from fastapi.responses import ORJSONResponse, PlainTextResponse
from typing import List
from app.services.optimizer import generate_lineups
from app.services import data

app = FastAPI(title="FanDuel League Picks", default_response_class=ORJSONResponse)

@app.get("/health")
async def health():
    week = await data.try_load_from_espm_current_week() if hasattr(data, "try_load_from_espm_current_week") else 1
    return {"status": "ok", "week": week}

@app.get("/players/current")
async def players_current():
    players = await data.get_players()
    return {"week": await data.get_current_week(), "count": len(players), "players": players}

@app.get("/optimize")
async def optimize(
    game_type: str = Query("gpp", pattern="^(h2h|league|gpp)$"),
    num_lineups: int = Query(1, ge=1, le=50),
    uniqueness: int = Query(3, ge=0, le=9),
):
    players = await data.get_players()
    lineups = generate_lineups(players, mode=game_type, num_lineups=num_lineups, uniqueness=uniqueness)

    def pack(roster: List[dict]):
        total_salary = sum(int(p["salary"]) for p in roster)
        total_proj = round(sum(float(p["projection"]) for p in roster), 2)
        return {
            "mode": game_type,
            "total_salary": total_salary,
            "total_projection": total_proj,
            "players": [{
                "id": p.get("id"),
                "name": p.get("name"),
                "team": p.get("team"),
                "position": p.get("position"),
                "salary": int(p.get("salary")),
                "projection": float(p.get("projection")),
            } for p in roster],
        }

    return {
        "week": await data.get_current_week(),
        "players_considered": len(players),
        "count": len(lineups),
        "lineups": [pack(r) for r in lineups],
    }

@app.get("/optimize/text")
async def optimize_text(
    game_type: str = Query("gpp", pattern="^(h2h|league|gpp)$"),
    num_lineups: int = Query(1, ge=1, le=20),
    uniqueness: int = Query(3, ge=0, le=9),
):
    players = await data.get_players()
    lineups = generate_lineups(players, mode=game_type, num_lineups=num_lineups, uniqueness=uniqueness)

    if not lineups:
        return PlainTextResponse("No valid lineup generated. Check player pool, positions, or salary cap.", status_code=200)

    cards = []
    for i, roster in enumerate(lineups, start=1):
        total_salary = sum(int(p["salary"]) for p in roster)
        total_proj = round(sum(float(p["projection"]) for p in roster), 2)
        by_pos = {}
        for p in roster:
            by_pos.setdefault(p["position"], []).append(f'{p["name"]} ({p["team"]}) ${p["salary"]} proj {p["projection"]:.1f}')
        # FLEX is implied by counts; we still print 9 names total.
        body_lines = []
        for label in ["QB", "RB", "WR", "TE", "DST"]:
            if label in by_pos:
                for row in by_pos[label]:
                    body_lines.append(f"{label}: {row}")
        # Add remaining player as FLEX (RB/WR/TE) if we have 9 players already printed < 9
        if len(body_lines) < 9 and len(roster) == 9:
            flex = [p for p in roster if p["position"] in {"RB","WR","TE"}]
            # Choose one not yet fully accounted for (simple heuristic)
            body_lines.append("FLEX: " + f'{flex[-1]["name"]} ({flex[-1]["team"]}) ${flex[-1]["salary"]} proj {flex[-1]["projection"]:.1f}')

        header = f"[{game_type.upper()}] Lineup #{i} | Salary: ${total_salary} | Proj: {total_proj}"
        # Quick mode note
        if game_type == "h2h":
            note = "Note: Cash build — favors floor and stability."
        elif game_type == "league":
            note = "Note: Small-field — balanced floor/ceiling and value."
        else:
            note = "Note: GPP — ceiling/value biased for upside."
        cards.append(header + "\n" + "\n".join(body_lines) + "\n" + note)

    return PlainTextResponse("\n\n" + ("\n" + "-"*64 + "\n\n").join(cards))
