import os
import time
import re
import pandas as pd
from typing import Tuple, List, Optional, Dict


# Canonical columns we use
REQ = ["PLAYER NAME", "POS", "TEAM", "SALARY", "PROJ PTS"]
OPT = ["OPP", "OWN_PCT"]

# Header aliases -> canonical (includes your "PROJ ROSTER %")
ALIASES: Dict[str, str] = {
    "PLAYER": "PLAYER NAME",
    "NAME": "PLAYER NAME",
    "PLAYERNAME": "PLAYER NAME",
    "POSITION": "POS",
    "TEAM_ABBR": "TEAM",
    "DST_TEAM": "TEAM",
    "SAL": "SALARY",
    "PRICE": "SALARY",
    "PROJ": "PROJ PTS",
    "PROJECTION": "PROJ PTS",
    "PROJECTED POINTS": "PROJ PTS",
    "FPTS": "PROJ PTS",
    "OWNERSHIP": "OWN_PCT",
    "OWNERSHIP%": "OWN_PCT",
    "OWN%": "OWN_PCT",
    "PROJ ROSTER %": "OWN_PCT",
    "OPPONENT": "OPP",
    "MATCHUP": "OPP",
}

POS_EXPECTED = {"QB","RB","WR","TE","DST"}


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _clean_money(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace("$","").replace(",","")
    if s == "":
        return pd.NA
    try:
        return float(s)
    except Exception:
        return pd.NA


def _clean_percent(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace("%","")
    if s == "":
        return pd.NA
    try:
        return float(s)
    except Exception:
        return pd.NA


def _upper_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip().upper() for c in df.columns})


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    # apply aliases only if canonical not already present
    for c in list(df.columns):
        canon = ALIASES.get(c, c)
        if canon != c and canon not in df.columns:
            df.rename(columns={c: canon}, inplace=True)
    return df


def _normalize_opp(val: Optional[str]) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper()
    s = s.replace("VS", "").replace("@", "").strip()
    m = re.search(r"([A-Z]{2,4})", s)
    return m.group(1) if m else (s or None)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQ + OPT:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def _normalize(df: pd.DataFrame, pos_hint: Optional[str], warnings: List[str]) -> pd.DataFrame:
    df = _upper_strip_cols(df)
    df = _apply_aliases(df)
    df = _ensure_columns(df)

    # PLAYER NAME
    df["PLAYER NAME"] = df["PLAYER NAME"].astype(str).str.strip()

    # POS (fill from hint if missing)
    if pos_hint:
        df["POS"] = df["POS"].fillna(pos_hint).replace("", pos_hint)
    df["POS"] = df["POS"].astype(str).str.upper().str.strip()

    # TEAM — force UNK if missing/blank BEFORE any string ops so we don't keep <NA>
    df["TEAM"] = df["TEAM"].fillna("UNK")
    df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip()
    df.loc[df["TEAM"].isin(["", "NAN", "NONE"]), "TEAM"] = "UNK"

    # OPP normalize
    if "OPP" in df.columns:
        df["OPP"] = df["OPP"].apply(_normalize_opp)

    # numeric coercions
    df["SALARY"] = df["SALARY"].apply(_clean_money)
    df["PROJ PTS"] = df["PROJ PTS"].apply(_clean_money)
    if "OWN_PCT" in df.columns:
        df["OWN_PCT"] = df["OWN_PCT"].apply(_clean_percent)

    # Diagnostics before drop
    total = len(df)
    drop_reasons = []

    # Required fields (TEAM not strict; allow UNK)
    need = ["PLAYER NAME","POS","SALARY","PROJ PTS"]
    df = df.dropna(subset=need)
    drop_missing = total - len(df)
    if drop_missing > 0:
        drop_reasons.append(f"missing required fields: {drop_missing}")

    # POS sanity
    before_pos = len(df)
    df = df[df["POS"].isin(POS_EXPECTED)]
    drop_pos = before_pos - len(df)
    if drop_pos > 0:
        drop_reasons.append(f"invalid POS: {drop_pos}")

    if drop_reasons:
        warnings.append("NOTICE: Dropped rows (" + "; ".join(drop_reasons) + ").")

    return df


def load_weekly_data() -> Optional[pd.DataFrame]:
    df, _ = load_weekly_data_with_warnings()
    return df


def load_weekly_data_with_warnings() -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Reads per-position CSVs from data/input and merges them.
    Expected files: qb.csv, rb.csv, wr.csv, te.csv, dst.csv (case-insensitive)
    Returns (df, warnings) and prints per-file column discovery.
    """
    warnings: List[str] = []
    base = os.path.join("data", "input")
    files = {
        "QB": os.path.join(base, "qb.csv"),
        "RB": os.path.join(base, "rb.csv"),
        "WR": os.path.join(base, "wr.csv"),
        "TE": os.path.join(base, "te.csv"),
        "DST": os.path.join(base, "dst.csv"),
    }

    now = time.time()
    frames = []

    for pos, path in files.items():
        if not os.path.exists(path):
            warnings.append(f"ERROR: Missing file: {path}")
            continue

        try:
            mtime = os.path.getmtime(path)
            if now - mtime > 7 * 24 * 3600:
                warnings.append(f"WARNING: {path} is older than 7 days—consider updating.")
        except Exception:
            pass

        raw = _read_csv(path)
        if raw is None:
            warnings.append(f"ERROR: Could not read or file is empty: {path}")
            continue

        warnings.append(f"INFO: {os.path.basename(path)} columns: {list(raw.columns)}")

        before_rows = len(raw)
        dfp = _normalize(raw, pos_hint=pos, warnings=warnings)
        after_rows = len(dfp)
        warnings.append(f"INFO: {pos}.csv usable rows: {after_rows}/{before_rows}")

        # Force POS to file's position
        dfp["POS"] = pos
        frames.append(dfp)

    if not frames:
        return None, warnings

    df = pd.concat(frames, ignore_index=True)

    if df.empty:
        warnings.append("ERROR: Combined data has no valid rows after cleaning.")
        return None, warnings

    # ensure numeric
    df["SALARY"] = df["SALARY"].astype(int)
    df["PROJ PTS"] = df["PROJ PTS"].astype(float)

    # simple index
    df = df.reset_index(drop=True)
    return df, warnings
