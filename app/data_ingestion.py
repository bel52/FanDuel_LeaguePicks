import os
import pandas as pd
import re

INPUT_DIR = "data/input"

def _safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def _parse_own(x):
    # Accept "20-30%-" -> 25.0
    if x is None: return None
    s = str(x).strip().replace("%","").replace("-"," ").replace("–"," ")
    nums = [float(n) for n in re.findall(r"[0-9]+\.?[0-9]*", s)]
    if not nums: return None
    return sum(nums)/len(nums)

def load_weekly_data():
    files = {
        "QB": os.path.join(INPUT_DIR, "qb.csv"),
        "RB": os.path.join(INPUT_DIR, "rb.csv"),
        "WR": os.path.join(INPUT_DIR, "wr.csv"),
        "TE": os.path.join(INPUT_DIR, "te.csv"),
        "DST": os.path.join(INPUT_DIR, "dst.csv"),
    }
    frames = []
    for pos, path in files.items():
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        # Common column mapping
        cols = {c.lower(): c for c in df.columns}
        def pick(*cands):
            for c in cands:
                for k in df.columns:
                    if k.strip().lower() == c.lower():
                        return k
            return None

        name_c = pick("player", "player name", "name")
        team_c = pick("team")
        opp_c  = pick("opp","opponent")
        proj_c = pick("proj pts","projection","proj")
        sal_c  = pick("salary")
        own_c  = pick("proj roster %","own%","ownership","proj roster pct")

        df2 = pd.DataFrame()
        df2["PLAYER NAME"] = df[name_c] if name_c else df.iloc[:,0]
        df2["POS"] = pos
        df2["TEAM"] = df[team_c] if team_c else ""
        df2["OPP"]  = df[opp_c] if opp_c else ""
        df2["PROJ PTS"] = df[proj_c].apply(_safe_float) if proj_c else 0.0
        df2["SALARY"] = df[sal_c].apply(lambda x: int(float(x))) if sal_c else 0
        df2["PROJ ROSTER %"] = df[own_c] if own_c else ""
        df2["OWN_PCT"] = df2["PROJ ROSTER %"].apply(_parse_own)
        frames.append(df2)

    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    # basic sanity: drop rows without salary or projection
    all_df = all_df[(all_df["SALARY"] > 0) & (all_df["PROJ PTS"] >= 0)]
    all_df.reset_index(drop=True, inplace=True)
    return all_df
