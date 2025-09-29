import asyncio
from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups

async def diagnose():
    print("=== DIAGNOSTIC TEST ===\n")
    
    # Get data
    data = await get_fresh_data()
    players = data['players']
    
    # Find Wright
    wright = [p for p in players if 'wright' in p['name'].lower() and p['position'] == 'TE']
    if wright:
        print("WRIGHT FOUND:")
        for w in wright:
            print(f"  {w['name']}: ${w['salary']}, {w['projected_points']:.1f} FPPG")
    else:
        print("Wright filtered out - GOOD")
    
    # Generate test lineup
    print("\n=== GENERATING TEST GPP LINEUP ===")
    lineups = optimize_dfs_lineups(
        player_data=players[:100],  # Sample
        num_lineups=1,
        contest_type='gpp',
        use_monte_carlo=True,
        mc_simulations=1000
    )
    
    if lineups:
        lu = lineups[0]
        print(f"\nProjection: {lu.projected_points:.1f}")
        print(f"Ceiling 90: {lu.ceiling_90:.1f}")
        print(f"Boom Rate: {lu.boom_probability:.1%}")
        print(f"Ownership: {lu.ownership_total:.1f}%")
        print(f"Risk: {lu.risk_level}")
        
        if lu.boom_probability < 0.10:
            print("\n❌ BOOM RATE TOO LOW FOR GPP")
        if lu.ownership_total > 150:
            print("❌ OWNERSHIP TOO HIGH (TOO CHALKY)")
        if lu.ceiling_90 < lu.projected_points * 1.3:
            print("❌ CEILING TOO LOW")

asyncio.run(diagnose())
