# Add this to your prepare_players function in optimizer.py

# Better QB filtering - add after line where you check salary > 6000 and projection == 0
if position == 'QB':
    # Filter out backup QBs more aggressively
    if salary < 7000 and projection < 15:
        logger.warning(f"Filtering backup QB: {player_name} (${salary}, {projection:.1f} proj)")
        continue
    
    # Also filter QBs with no FPPG data regardless of salary
    fppg_source = data.get('fppg_source', 'unknown')
    if fppg_source == 'estimated' and projection < 18:
        logger.warning(f"Filtering estimated QB: {player_name} (no real FPPG data)")
        continue

