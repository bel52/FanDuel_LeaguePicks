def calculate_max_exposure(num_lineups: int, position: str) -> int:
    """
    Calculate max player appearances based on position and total lineups
    
    Target exposure rates (% of lineups):
    - QB/DEF: 50% max (more concentrated, fewer good options)
    - RB: 60% max (moderate scarcity)
    - WR: 65% max (deep position, more options)
    - TE: 55% max (shallow but need variety)
    
    Ensures no player dominates unless lineup count is tiny
    """
    target_pct = {
        'QB': 0.50,
        'RB': 0.60,
        'WR': 0.65,
        'TE': 0.55,
        'D': 0.50,
    }.get(position, 0.60)
    
    # Calculate max appearances, minimum 1
    max_uses = max(1, int(num_lineups * target_pct))
    
    return max_uses


# Example outputs for different lineup counts:
for n in [3, 5, 10, 20, 50]:
    print(f"\n{n} lineups:")
    for pos in ['QB', 'RB', 'WR', 'TE', 'D']:
        max_apps = calculate_max_exposure(n, pos)
        pct = (max_apps / n) * 100
        print(f"  {pos}: max {max_apps} uses ({pct:.0f}%)")
