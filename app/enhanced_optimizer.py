from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional

try:
    import pulp  # ILP solver interface
except ImportError as e:
    raise RuntimeError("pulp is required. Add 'pulp' to requirements.txt") from e

FD_CAP = 60000
ROSTER_REQ = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "DST": 1}
FLEX_ELIGIBLE = {"RB", "WR", "TE"}

class EnhancedDFSOptimizer:
    def __init__(
        self,
        players: List[Dict[str, Any]],
        cap: int = FD_CAP,
        locks: Optional[List[str]] = None,
        bans: Optional[List[str]] = None,
    ):
        self.players = [p for p in players if isinstance(p, dict)]
        self.cap = int(cap)
        self.locks = set((locks or []))
        self.bans = set((bans or []))
        self.locks = {str(x).strip().lower() for x in self.locks}
        self.bans  = {str(x).strip().lower() for x in self.bans}

    @staticmethod
    def _name_key(p: Dict[str, Any]) -> str:
        return str(p.get("name", "")).strip().lower()

    def _filtered(self) -> List[Dict[str, Any]]:
        out = []
        for p in self.players:
            nm = self._name_key(p)
            if nm in self.bans:
                continue
            out.append(p)
        missing = [x for x in self.locks if x not in {self._name_key(p) for p in out}]
        if missing:
            raise RuntimeError(f"Locked players not in pool: {missing}")
        return out

    def optimize_one(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        pool = self._filtered()
        if not pool:
            return [], {"warnings": ["Empty player pool after filters."]}

        n = len(pool)
        x = pulp.LpVariable.dicts("pick", list(range(n)), lowBound=0, upBound=1, cat=pulp.LpBinary)

        model = pulp.LpProblem("FD_NFL_Optimizer", pulp.LpMaximize)
        eps = 0.0005  # tiny incentive to use available salary
        model += pulp.lpSum([x[i] * (float(pool[i]["proj"]) + eps * float(pool[i]["salary"])) for i in range(n)])

        # Cap
        model += pulp.lpSum([x[i] * int(pool[i]["salary"]) for i in range(n)]) <= self.cap

        # Exact counts for fixed positions
        for pos, cnt in ROSTER_REQ.items():
            model += pulp.lpSum([x[i] for i in range(n) if (pool[i]["pos"] == pos)]) == cnt

        # Aggregate for FLEX (RB/WR/TE total = 7 == 2+3+1 + 1 flex)
        model += pulp.lpSum([x[i] for i in range(n) if (pool[i]["pos"] in FLEX_ELIGIBLE)]) == 7

        # Global roster size = 9
        model += pulp.lpSum([x[i] for i in range(n)]) == 9

        # Locks
        for i in range(n):
            if self._name_key(pool[i]) in self.locks:
                model += x[i] == 1

        # Solve (CBC from coin-or installed in Dockerfile)
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != "Optimal":
            return [], {"warnings": [f"Solver status: {pulp.LpStatus[status]} (may be infeasible)."]}

        picked = [pool[i] for i in range(n) if x[i].value() == 1]
        if len(picked) != 9:
            return [], {"warnings": [f"Unexpected roster size {len(picked)} after solve."]}

        # Assign output slots
        roster = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": [], "FLEX": []}
        by_pos: Dict[str, List[Dict[str, Any]]] = {"QB": [], "RB": [], "WR": [], "TE": [], "DST": []}
        for p in picked:
            by_pos.setdefault(p["pos"], []).append(p)
        for pos in by_pos:
            by_pos[pos].sort(key=lambda r: (-float(r.get("proj", 0.0)), r.get("name","")))

        roster["QB"] = by_pos.get("QB", [])[:1]
        roster["DST"] = by_pos.get("DST", [])[:1]
        roster["RB"] = by_pos.get("RB", [])[:2]
        roster["WR"] = by_pos.get("WR", [])[:3]
        roster["TE"] = by_pos.get("TE", [])[:1]

        used_ids = set(id(r) for grp in ["QB","DST","RB","WR","TE"] for r in roster.get(grp, []))
        flex_pool = [p for pos in FLEX_ELIGIBLE for p in by_pos.get(pos, []) if id(p) not in used_ids]
        flex_pool.sort(key=lambda r: (-float(r.get("proj", 0.0)), r.get("name","")))
        roster["FLEX"] = flex_pool[:1]

        final = roster["QB"] + roster["RB"] + roster["WR"] + roster["TE"] + roster["FLEX"] + roster["DST"]
        total_proj = round(sum(float(p["proj"]) for p in final), 2)
        total_salary = sum(int(p["salary"]) for p in final)

        def tag(slot: str, arr: List[Dict[str, Any]]):
            for p in arr:
                p["pos_out"] = slot

        tag("QB", roster["QB"])
        tag("RB", roster["RB"])
        tag("WR", roster["WR"])
        tag("TE", roster["TE"])
        tag("FLEX", roster["FLEX"])
        tag("DST", roster["DST"])

        meta = {"warnings": [], "cap_used": total_salary, "cap": self.cap, "total_proj": total_proj}
        return final, meta
