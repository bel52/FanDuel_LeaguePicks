import os
import logging
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "PLAYER NAME",  # str
    "POS",          # str (QB/RB/WR/TE/DST)
    "TEAM",         # str team code (optional in files; we keep column present)
    "OPP",          # str opponent (optional)
    "SALARY",       # int (can be missing; we coerce/fill)
    "PROJ PTS",     # float (can be missing; we coerce/fill)
]

COLUMN_ALIASES: Dict[str, str] = {
    "player": "PLAYER NAME",
    "player_name": "PLAYER NAME",
    "name": "PLAYER NAME",
    "player name": "PLAYER NAME",
    "playername": "PLAYER NAME",

    "position": "POS",
    "pos": "POS",

    "team": "TEAM",
    "team_id": "TEAM",

    "opp": "OPP",
    "opponent": "OPP",

    "salary": "SALARY",
    "sal": "SALARY",
    "cost": "SALARY",

    "proj": "PROJ PTS",
    "projection": "PROJ PTS",
    "projected": "PROJ PTS",
    "projected points": "PROJ PTS",
    "proj pts": "PROJ PTS",
    "fpts": "PROJ PTS",
    "fantasy points": "PROJ PTS",

    "ownership": "PROJ ROSTER %",
    "projected roster %": "PROJ ROSTER %",
    "proj roster %": "PROJ ROSTER %",
    "own%": "PROJ ROSTER %",
    "own_pct": "OWN_PCT",
}

POSITION_FROM_FILENAME = {
    "qb": "QB",
    "rb": "RB",
    "wr": "WR",
    "te": "TE",
    "dst": "DST",
    "def": "DST",
    "d/st": "DST",
    "d_st": "DST",
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    for c in df.columns:
        key = str(c).strip().lower()
        if key in COLUMN_ALIASES:
            colmap[c] = COLUMN_ALIASES[key]
    if colmap:
        df = df.rename(columns=colmap)
    return df

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["PLAYER NAME", "POS", "TEAM", "OPP"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "POS" in df.columns:
        df["POS"] = df["POS"].str.upper().replace({"D/ST": "DST", "DEF": "DST"})

    if "SALARY" in df.columns:
        df["SALARY"] = pd.to_numeric(df["SALARY"], errors="coerce")
    else:
        df["SALARY"] = pd.NA

    if "PROJ PTS" in df.columns:
        df["PROJ PTS"] = pd.to_numeric(df["PROJ PTS"], errors="coerce")
    else:
        df["PROJ PTS"] = pd.NA

    if "OWN_PCT" in df.columns:
        df["OWN_PCT"] = pd.to_numeric(df["OWN_PCT"], errors="coerce")
    elif "PROJ ROSTER %" in df.columns:
        own = df["PROJ ROSTER %"].astype(str).str.replace("%", "", regex=False)
        df["OWN_PCT"] = pd.to_numeric(own, errors="coerce")

    return df

def _infer_pos_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path).lower()
    name = os.path.splitext(base)[0]
    return POSITION_FROM_FILENAME.get(name)

def _load_one_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        logger.warning(f"[data_ingestion] Could not read {path}: {e}")
        return pd.DataFrame()

    df = _normalize_columns(df)

    if "POS" not in df.columns:
        pos_inferred = _infer_pos_from_filename(path)
        if pos_inferred:
            df["POS"] = pos_inferred

    df = _coerce_types(df)

    # Ensure required columns exist (fill if missing)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only relevant columns
    keep_cols = list(dict.fromkeys(REQUIRED_COLS + ["PROJ ROSTER %", "OWN_PCT"]))
    df = df[[c for c in keep_cols if c in df.columns]]

    # Minimal drop: require name + pos (we keep rows even if salary/proj is NaN)
    df = df.dropna(subset=["PLAYER NAME", "POS"], how="any")
    df = df[df["POS"].isin(["QB", "RB", "WR", "TE", "DST"])]

    return df

def _find_input_files(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    return sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".csv")
    )

def _merge_and_finalize(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame(columns=REQUIRED_COLS + ["PROJ ROSTER %", "OWN_PCT"])

    df = pd.concat(dfs, ignore_index=True)

    # Dedup by player+pos+team when available
    keys = [k for k in ["PLAYER NAME", "POS", "TEAM"] if k in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="first")

    df = _coerce_types(df)

    # Order columns
    ordered = [c for c in REQUIRED_COLS if c in df.columns]
    for opt in ["PROJ ROSTER %", "OWN_PCT"]:
        if opt in df.columns:
            ordered.append(opt)
    df = df[ordered]

    # Coerce types + sensible fills (don’t drop rows because of NaNs)
    df["SALARY"] = pd.to_numeric(df["SALARY"], errors="coerce")
    df["PROJ PTS"] = pd.to_numeric(df["PROJ PTS"], errors="coerce")

    # Leave NaNs; optimizer will handle defaults.
    df = df.reset_index(drop=True)
    return df

def load_weekly_data(input_dir: str = "data/input") -> Optional[pd.DataFrame]:
    files = _find_input_files(input_dir)
    if not files:
        logger.error(f"[data_ingestion] No CSV files found under {input_dir}/")
        return None

    frames: List[pd.DataFrame] = []
    for p in files:
        dfi = _load_one_csv(p)
        if not dfi.empty:
            frames.append(dfi)
        else:
            logger.warning(f"[data_ingestion] {os.path.basename(p)} produced 0 rows after cleaning")

    merged = _merge_and_finalize(frames)
    if merged is None or merged.empty:
        logger.error("[data_ingestion] Combined dataset is empty after processing")
        return None

    try:
        pos_counts = merged["POS"].value_counts().to_dict()
        logger.info(f"[data_ingestion] Loaded {len(merged)} players from {len(files)} files — {pos_counts}")
    except Exception:
        pass

    return merged
