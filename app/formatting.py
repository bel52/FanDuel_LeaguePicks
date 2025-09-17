from typing import List, Dict

def lineup_to_text(players: List[Dict[str, any]], total: float) -> str:
    out = []
    # exact slots
    def pick(pos, count=1, label=None):
        picked = [p for p in players if (p.get("pos_out") or p["pos"]) == pos][:count]
        for i, p in enumerate(picked, 1):
            tag = f"{pos}{i}" if count>1 else (label or pos)
            out.append(f"{tag}: {p['name']} ({p['team']} vs {p.get('opponent','')}) ~ {p['proj']:.2f}")
    pick("QB"); pick("RB",2); pick("WR",2); pick("TE"); pick("K")
    dst = next((p for p in players if (p.get("pos_out") or p["pos"]) in ("DEF","DST")), None)
    if dst: out.append(f"DST: {dst['name']} ({dst['team']}) ~ {dst['proj']:.2f}")
    flex = next((p for p in players if (p.get("pos_out") or p["pos"]) in ("RB","WR","TE")), None)
    if flex: out.append(f"FLEX: {flex['name']} ({flex['team']}) ~ {flex['proj']:.2f}")
    out.append(f"\nProjected Total: {total:.2f}")
    return "\n".join(out)
