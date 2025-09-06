import os
import json
import requests
import pandas as pd
from dateutil import parser as dateparser
import pytz

# Make imports work whether run as `python -m app.cli` or `python app/cli.py`
try:
    from app import data_ingestion
    from app import optimization
    from app.formatting import build_text_report
    from app.kickoff_times import load_last_lineup, save_last_lineup
except Exception:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app import data_ingestion
    from app import optimization
    from app.formatting import build_text_report
    from app.kickoff_times import load_last_lineup, save_last_lineup


def _check_openai_key():
    """
    Non-fatal verification for OPENAI_API_KEY.
    Compatible with openai>=1.0.0. If verification fails, we proceed without AI.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        print("WARNING: OPENAI_API_KEY not set. AI analysis disabled.")
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        _ = client.models.list()  # lightweight call; may still fail offline
        return key
    except Exception as e:
        print(f"WARNING: Could not verify OpenAI key ({e}). Proceeding without AI.")
        return None


def _load_strategy_weights():
    cfg_path = os.path.join(os.path.dirname(__file__), "strategy_weights.json")
    default_cfg = {
        "h2h":   {"projection_weight": 0.3, "leverage_weight": 0.7},
        "league":{"projection_weight": 0.7, "leverage_weight": 0.3}
    }
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        for k in ["h2h","league"]:
            if k not in cfg:
                cfg[k] = default_cfg[k]
        return cfg
    except FileNotFoundError:
        print("NOTE: strategy_weights.json not found. Using defaults.")
        return default_cfg
    except Exception as e:
        print(f"WARNING: Failed to load strategy_weights.json: {e}. Using defaults.")
        return default_cfg


def _fetch_games():
    """
    Returns list like:
      [{'teams': None, 'label': 'All Games (Full Slate)'},
       {'teams': ['AWY','HOM'], 'label': 'AWY vs HOM - Sun 01:00 PM ET'}, ...]
    with index 0 = Full Slate.
    """
    options = [{"teams": None, "label": "All Games (Full Slate)"}]
    try:
        r = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"WARNING: Could not retrieve ESPN games: {e}. Defaulting to Full Slate only.")
        return options

    eastern = pytz.timezone("America/New_York")
    for ev in data.get("events", []):
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        teams = []
        for c in comp.get("competitors", []):
            abbr = c.get("team", {}).get("abbreviation") or ""
            side = c.get("homeAway", "")
            teams.append((side, abbr))
        if not teams:
            continue
        away = next((a for s,a in teams if s == "away"), teams[0][1] if len(teams)>=1 else "TBD")
        home = next((a for s,a in teams if s == "home"), teams[-1][1] if len(teams)>=1 else "TBD")

        dt_str = comp.get("date")
        label_time = "TBA"
        if dt_str:
            try:
                dt = dateparser.parse(dt_str)
                dt_et = dt.astimezone(eastern)
                label_time = dt_et.strftime("%a %I:%M %p ET")
            except Exception:
                pass
        options.append({"teams":[away, home], "label": f"{away} vs {home} - {label_time}"})
    return options


def _prompt_select(options):
    print("\nSelect a game/slate:")
    for i, opt in enumerate(options):
        print(f"{i}. {opt['label']}")
    while True:
        choice = input("Enter number (or 'q' to quit): ").strip().lower()
        if choice == 'q':
            return None
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid selection.")


def _prompt_mode():
    while True:
        m = input("Choose game type - (L)eague or (H)ead-to-Head: ").strip().lower()
        if m in ("l","league"):
            return "league"
        if m in ("h","h2h","head"):
            return "h2h"
        print("Please enter L or H.")


def main():
    # 1) AI key check (non-fatal)
    _check_openai_key()

    # 2) Load player data + warnings
    df, data_warnings = data_ingestion.load_weekly_data_with_warnings()
    if data_warnings:
        for w in data_warnings:
            print(w)

    if df is None or df.empty:
        print("ERROR: No player data loaded. Ensure CSVs exist under data/input/.")
        return 2

    required_positions = ["QB","RB","WR","TE","DST"]
    missing = [p for p in required_positions if p not in set(df["POS"].unique())]
    if missing:
        for p in missing:
            print(f"ERROR: Missing {p} data (no rows found). Confirm data/input/{p.lower()}.csv exists.")
        return 2

    # 3) Prompt for game/slate
    options = _fetch_games()
    selected = _prompt_select(options)
    if selected is None:
        print("Exiting.")
        return 0

    # 4) Prompt for mode
    game_type = _prompt_mode()

    # 5) Filter by teams if single game chosen
    teams = selected["teams"]
    if teams:
        # Try to match team names (handle potential mismatches)
        df2 = df[df["TEAM"].isin(teams)].copy()
        if df2.empty or len(df2) < 20:  # Need enough players for a lineup
            print(f"WARNING: Insufficient data for teams {teams}. Using full slate instead.")
            df2 = df.copy()
    else:
        df2 = df.copy()

    # 6) Load strategy weights
    strat = _load_strategy_weights()
    leverage_weight = strat[game_type]["leverage_weight"]

    # 7) Optimize - use original indices
    print(f"\nOptimizing {game_type} lineup...")
    print(f"Players available: {len(df2)}")
    print(f"Salary cap: $60,000")
    
    lineup_idxs = optimization.optimize_lineup(
        df2,
        salary_cap=60000,
        enforce_stack=True,
        min_stack_receivers=1,
        lock_indices=[],
        ban_indices=[],
        leverage_weight=leverage_weight
    )
    
    if not lineup_idxs:
        print("ERROR: No feasible lineup found.")
        return 3

    # 8) Build lineup dict
    lineup = []
    total_salary = 0
    for idx in lineup_idxs:
        row = df2.loc[idx]
        total_salary += int(row["SALARY"])
        own = None
        if "OWN_PCT" in row and pd.notna(row["OWN_PCT"]):
            own = float(row["OWN_PCT"])
        lineup.append({
            "name": row["PLAYER NAME"],
            "position": row["POS"],
            "team": row["TEAM"],
            "opponent": row.get("OPP","N/A"),
            "proj_points": float(row["PROJ PTS"]),
            "salary": int(row["SALARY"]),
            "own_pct": own
        })

    result = {
        "game_type": f"{game_type}" + (f" ({' vs '.join(teams)})" if teams else ""),
        "lineup": lineup,
        "cap_usage": {"total_salary": total_salary, "remaining": 60000 - total_salary}
    }

    # 9) League compare/save
    if game_type == "league":
        print("\n*** League Mode Selected ***")
        prev = load_last_lineup()
        if prev:
            print("Current Saved League Lineup:")
            for p in prev:
                print(f" - {p['position']}: {p['name']} ({p['team']} vs {p.get('opponent','N/A')}) – {p.get('proj_points',0):.1f} pts")
        else:
            print("No saved league lineup yet.")

        if prev:
            prev_names = {p["name"] for p in prev}
            new_names  = {p["name"] for p in lineup}
            out = [p for p in prev if p["name"] not in new_names]
            inn = [p for p in lineup if p["name"] not in prev_names]
            if not out and not inn:
                print("\nNo lineup changes recommended.")
            else:
                print("\nRecommended Changes:")
                for p in out:
                    print(f" - Remove {p['name']} ({p['position']}, {p['team']})")
                for p in inn:
                    print(f" + Add {p['name']} ({p['position']}, {p['team']})")
        save_last_lineup(lineup)

    # 10) Print report
    print("\nOptimized Lineup:")
    print(build_text_report(result, width=100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
