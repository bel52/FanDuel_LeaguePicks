import pandas as pd
from typing import Dict, Any, List

def build_text_report(result: Dict[str, Any], width: int = 100) -> str:
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
        pos = p.get("POS", p.get("position", "N/A"))
        name = str(p.get("PLAYER NAME", p.get("player_name", "N/A")))[:22]
        team = p.get("TEAM", p.get("team", "N/A"))
        opp = p.get("OPP", p.get("opponent", "N/A"))
        salary = p.get("SALARY", p.get("salary", 0))
        proj = float(p.get("PROJ PTS", p.get("projection", 0.0)))
        own = p.get("OWN_PCT", p.get("ownership"))
        
        total_proj += proj
        
        own_str = f"{own:.1f}%" if own is not None and pd.notna(own) else "--"
        value = (proj / (salary / 1000)) if salary > 0 else 0
        
        rows.append(
            f"{pos:<4} {name:<22} {team:<4} {opp:<4} ${salary:>6} {proj:>6.1f} {own_str:>6} {value:>6.2f}"
        )

    rows.append(separator)
    cap_usage = result.get("cap_usage", {})
    total_salary = cap_usage.get("total_salary", sum(p.get("SALARY", p.get("salary", 0)) for p in lineup))
    remaining_salary = cap_usage.get("remaining", 60000 - total_salary)
    
    rows.append(f"TOTALS:{'':<28} ${total_salary:>6}  {total_proj:>6.1f}")
    rows.append(f"SALARY REMAINING: ${remaining_salary}")
    rows.append(separator)
    
    return "\n".join(rows)
