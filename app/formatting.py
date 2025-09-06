import pandas as pd

def build_text_report(result: dict, width: int = 100) -> str:
    """
    Builds a clean, formatted text report for the optimized lineup.
    """
    game_type = result.get("game_type", "DFS").upper()
    title = f"FANDUEL NFL DFS LINEUP - {game_type} STRATEGY"
    line = "=" * len(title)
    
    header = f"{'POS':<4} {'PLAYER':<22} {'TEAM':<4} {'OPP':<4} {'SALARY':>7} {'PROJ':>6} {'OWN%':>6} {'VALUE':>6}"
    separator = "-" * len(header)
    
    rows = [title, line, "OPTIMIZED LINEUP:", separator, header, separator]
    
    lineup = result.get("lineup", [])
    if not lineup:
        rows.append("No lineup generated.")
        return "\n".join(rows)

    total_proj = 0.0
    for p in lineup:
        pos = p.get("POS", "N/A")
        name = str(p.get("PLAYER NAME", "N/A"))[:22]
        team = p.get("TEAM", "N/A")
        opp = p.get("OPP", "N/A")
        salary = p.get("SALARY", 0)
        proj = float(p.get("PROJ PTS", 0.0))
        own = p.get("OWN_PCT")
        
        total_proj += proj
        
        own_str = f"{own:.1f}%" if pd.notna(own) else "--"
        value = (proj / (salary / 1000)) if salary > 0 else 0
        
        rows.append(
            f"{pos:<4} {name:<22} {team:<4} {opp:<4} ${salary:>6} {proj:>6.1f} {own_str:>6} {value:>6.2f}"
        )

    rows.append(separator)
    cap_usage = result.get("cap_usage", {})
    total_salary = cap_usage.get("total_salary", 0)
    remaining_salary = cap_usage.get("remaining", 0)
    
    rows.append(f"TOTALS:{'':<28} ${total_salary:>6}  {total_proj:>6.1f}")
    rows.append(f"SALARY REMAINING: ${remaining_salary}")
    rows.append(separator)
    
    return "\n".join(rows)
