from __future__ import annotations
from typing import Dict, List, Any

FD_CAP = 60000

MIN_SAL = {"QB": 6000, "RB": 4500, "WR": 4500, "TE": 4500, "DST": 3000}
MAX_SAL = {"QB":10000, "RB":10000, "WR":10000, "TE": 9000, "DST": 5500}
ALPHA   = {"QB": 150.0, "RB": 200.0, "WR": 200.0, "TE": 180.0, "DST": 250.0}
ALLOWED = {"QB", "RB", "WR", "TE", "DEF", "DST"}

def _pos_norm(p: str) -> str:
    p = (p or "").upper()
    return "DST" if p == "DEF" else p

def _est_salary(pos: str, proj: float) -> int:
    pos = _pos_norm(pos)
    if pos not in MIN_SAL:
        pos = "WR"
    base = MIN_SAL[pos]
    alpha = ALPHA[pos]
    raw = base + alpha * max(0.0, float(proj))
    return int(max(MIN_SAL[pos], min(MAX_SAL[pos], round(raw / 100.0) * 100)))

def attach_salaries(players: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = []
    for p in players:
        if not isinstance(p, dict):
            continue
        pos = _pos_norm(p.get("pos"))
        if pos not in ALLOWED:
            continue
        proj = float(p.get("proj") or 0.0)
        sal = int(p.get("salary") or _est_salary(pos, proj))
        pp = dict(p)
        pp["pos"] = pos
        pp["salary"] = sal
        out.append(pp)
    return {"players": out, "salary_source": "proxy", "cap": FD_CAP}
