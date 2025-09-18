from __future__ import annotations
from typing import List, Tuple
import json

from app.models import Player
from app.config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_TIMEOUT_SECS

def ai_available() -> bool:
    return bool(OPENAI_API_KEY)

def _client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECS)

def tweak_projections_with_ai(players: List[Player], context_hint: str = "") -> Tuple[List[Player], str]:
    if not ai_available():
        return players, "AI not configured."

    prompt = f"""
You are an elite DFS analyst. Given players with fields:
id, name, team, opponent, position, salary, projection.

Task:
1) Suggest small projection deltas (-2.0..+2.0) for at most 12 players.
2) Output JSON ONLY:
{{
  "adjustments":[{{"id":"string","delta":number}}...],
  "commentary":"short paragraph"
}}
Constraints: Keep deltas small; most players unchanged.

Context:
{context_hint}
"""
    roster_json = [p.dict() for p in players]
    system = "You produce concise DFS analysis with strict JSON outputs."
    user = f"PLAYERS_JSON:\n{json.dumps(roster_json, ensure_ascii=False)}\n\nINSTRUCTIONS:\n{prompt}"

    try:
        client = _client()
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            response_format={"type":"json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        adjustments = {a["id"]: float(a["delta"]) for a in data.get("adjustments", []) if "id" in a and "delta" in a}
        commentary = str(data.get("commentary", "")).strip()

        updated: List[Player] = []
        for p in players:
            delta = adjustments.get(p.id, 0.0)
            if abs(delta) > 0:
                p = p.copy(update={"projection": max(0.0, p.projection + delta)})
            updated.append(p)
        return updated, (commentary or "AI provided minor adjustments.")
    except Exception as e:
        return players, f"AI adjustment skipped due to error: {e}"
