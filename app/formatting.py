def build_text_report(result: dict, width: int = 100) -> str:
    gt = result.get("game_type","").upper()
    title = f"FANDUEL NFL DFS LINEUP - {gt} STRATEGY" if gt else "FANDUEL NFL DFS LINEUP"
    line = "=" * min(len(title), width)
    rows = []
    rows.append(title)
    rows.append(line)
    rows.append("OPTIMIZED LINEUP:")
    rows.append("-" * min(57, width))
    rows.append(f"{'POS':<4} {'PLAYER':<22} {'TEAM':<4} {'OPP':<4} {'SALARY':>7} {'PROJ':>6} {'OWN%':>6} {'VALUE':>6}")
    rows.append("-" * min(57, width))
    total_proj = 0.0
    for p in result.get("lineup", []):
        pos = p.get("position","")
        name = p.get("name","")[:22]
        team = p.get("team","")
        opp = p.get("opponent","") or "N/A"
        sal = p.get("salary",0)
        proj = float(p.get("proj_points",0.0))
        own = p.get("own_pct", None)
        val = 0.0 if sal == 0 else proj / (sal/1000)
        total_proj += proj
        own_str = f"{own:.1f}%" if own is not None else "--"
        rows.append(f"{pos:<4} {name:<22} {team:<4} {opp:<4} ${sal:>6} {proj:>6.1f} {own_str:>6} {val:>6.2f}")
    rows.append("-" * min(57, width))
    cap = result.get("cap_usage", {})
    rows.append(f"TOTALS:{'':<28} ${cap.get('total_salary',0):>6}  {total_proj:>6.1f}")
    rows.append(f"SALARY REMAINING: ${cap.get('remaining',0)}")
    rows.append("-" * min(57, width))
    return "\n".join(rows)
