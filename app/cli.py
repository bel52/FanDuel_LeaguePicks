import argparse
import pandas as pd
from . import data_ingestion, optimization, analysis, openai_utils
from .formatting import build_text_report
from .player_match import match_names_to_indices

def main():
    ap = argparse.ArgumentParser(description="FanDuel DFS Optimizer (Readable CLI Output)")
    ap.add_argument("--salary-cap", type=int, default=60000)
    ap.add_argument("--enforce-stack", action="store_true", help="Require QB to have >= min-stack-receivers WR/TE from same team")
    ap.add_argument("--min-stack-receivers", type=int, default=1)
    ap.add_argument("--width", type=int, default=100, help="Output width (70-160)")
    ap.add_argument("--lock", action="append", default=[], help='Lock a player by name (can repeat). Example: --lock "Ja\'Marr Chase"')
    ap.add_argument("--ban",  action="append", default=[], help='Ban a player by name (can repeat). Example: --ban "Trey McBride"')
    args = ap.parse_args()

    df = data_ingestion.load_weekly_data()
    if df is None or df.empty:
        print("ERROR: No data available for optimization")
        return 2

    lock_idx, lock_nf = match_names_to_indices(args.lock, df)
    ban_idx,  ban_nf  = match_names_to_indices(args.ban,  df)

    lineup = optimization.optimize_lineup(
        df,
        salary_cap=args.salary_cap,
        enforce_stack=args.enforce_stack,
        min_stack_receivers=args.min_stack_receivers,
        lock_indices=lock_idx,
        ban_indices=ban_idx
    )
    if not lineup:
        print("ERROR: No feasible lineup found with given constraints")
        if args.lock or args.ban:
            print(f"Locks: {', '.join(args.lock)}")
            print(f"Bans: {', '.join(args.ban)}")
            nf = list(set(lock_nf + ban_nf))
            if nf:
                print(f"Not found: {', '.join(nf)}")
        return 3

    # Monte Carlo
    sim = analysis.MonteCarloSimulator(num_simulations=10000)
    pdata = {}
    for pid in lineup:
        proj = float(df.loc[pid, 'PROJ PTS'])
        pdata[pid] = {'projected_points': proj, 'historical_std_dev': max(proj * 0.15, 1.0)}
    sim_results = sim.simulate_lineup_performance(lineup, pdata)

    # AI analysis (best-effort; cached)
    try:
        prompt = analysis.build_lineup_analysis_prompt(lineup, df, sim_results)
        ai_text = openai_utils.analyze_prompt_with_gpt(prompt)
    except Exception:
        ai_text = "Analysis not available"

    total_proj = 0.0
    total_salary = 0
    lineup_players = []
    for pid in lineup:
        r = df.loc[pid]
        total_proj += float(r['PROJ PTS'])
        total_salary += int(r['SALARY'])
        own_pct = None
        if 'OWN_PCT' in r and not pd.isna(r['OWN_PCT']):
            own_pct = float(r['OWN_PCT'])
        lineup_players.append({
            'name': r['PLAYER NAME'],
            'position': r['POS'],
            'team': r['TEAM'],
            'opponent': r.get('OPP', ''),
            'proj_points': float(r['PROJ PTS']),
            'salary': int(r['SALARY']),
            'proj_roster_pct_raw': r.get('PROJ ROSTER %', None),
            'own_pct': own_pct,
        })

    result = {
        "params": {
            "salary_cap": args.salary_cap,
            "enforce_stack": args.enforce_stack,
            "min_stack_receivers": args.min_stack_receivers
        },
        "constraints": {
            "locks": [df.loc[i,'PLAYER NAME'] for i in lock_idx],
            "bans":  [df.loc[i,'PLAYER NAME'] for i in ban_idx],
            "not_found": list(set(lock_nf + ban_nf))
        },
        "cap_usage": {
            "total_salary": total_salary,
            "remaining": args.salary_cap - total_salary
        },
        "lineup": lineup_players,
        "total_projected_points": round(total_proj, 2),
        "simulation": sim_results,
        "analysis": ai_text
    }

    print(build_text_report(result, width=max(70, min(160, args.width))))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
