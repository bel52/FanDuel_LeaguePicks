# Emergency QB filter for optimizer.py prepare_players method
if position == 'QB':
    # Only keep actual starting QBs
    starter_qbs = [
        'Josh Allen', 'Lamar Jackson', 'Jalen Hurts', 'Justin Herbert',
        'Patrick Mahomes', 'Baker Mayfield', 'Jared Goff', 'Caleb Williams',
        'Drake Maye', 'Matthew Stafford', 'Russell Wilson', 'Bryce Young',
        'Trevor Lawrence', 'C.J. Stroud', 'Jayden Daniels'
    ]
    
    is_starter = any(starter.lower() in player_name.lower() for starter in starter_qbs)
    if not is_starter and salary < 7500:
        logger.info(f"FILTERING non-starter QB: {player_name}")
        continue
