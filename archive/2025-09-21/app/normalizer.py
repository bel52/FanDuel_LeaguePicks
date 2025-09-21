import re
import pandas as pd

COMMON_MAPS = {
    'PLAYER NAME': ['PLAYER NAME','Player','Name','PLAYER'],
    'TEAM': ['TEAM','Tm','Team'],
    'POS': ['POS','Position','Pos'],
    'OPP': ['OPP','Opponent','Opp'],
    'SALARY': ['SALARY','Salary','FD Salary','FanDuel Salary','Fanduel Salary','FANDUEL SALARY'],
    'PROJ PTS': ['PROJ PTS','Proj Pts','Proj. Pts','FPTS','FPts','Projected Pts','Projected Points'],
    'PROJ ROSTER %': ['PROJ ROSTER %','Proj Roster %','Own %','Ownership %','Proj. Own%','Proj Own %'],
    'O/U': ['O/U','OU','Over/Under','Total'],
    'SPREAD': ['Spread','SPREAD']
}

def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    for target, alts in COMMON_MAPS.items():
        if target in df.columns:
            colmap[target] = target
            continue
        for c in df.columns:
            if c in alts:
                colmap[c] = target
    return df.rename(columns=colmap)

def _extract_from_player_name(df: pd.DataFrame):
    if 'PLAYER NAME' not in df.columns:
        return df
    if 'TEAM' not in df.columns or df['TEAM'].isna().all():
        team = df['PLAYER NAME'].str.extract(r'\(([^()\-]+?)[\s\-]+[A-Z]{2,3}\)')[0]
        df['TEAM'] = team
    if 'POS' not in df.columns or df['POS'].isna().all():
        pos = df['PLAYER NAME'].str.extract(r'\(([A-Z]{2,3})\)$')[0]
        pos = pos.fillna(df['PLAYER NAME'].str.extract(r'\-\s*([A-Z]{2,3})\)')[0])
        df['POS'] = pos
    return df

def _clean_salary(df: pd.DataFrame):
    if 'SALARY' in df.columns:
        df['SALARY'] = (
            df['SALARY'].astype(str)
            .str.replace(r'[\$,]', '', regex=True)
            .str.extract(r'(\d+)', expand=False)
        )
        df['SALARY'] = pd.to_numeric(df['SALARY'], errors='coerce').astype('Int64')
    return df

def _clean_proj_pts(df: pd.DataFrame):
    if 'PROJ PTS' in df.columns:
        df['PROJ PTS'] = pd.to_numeric(df['PROJ PTS'], errors='coerce')
    return df

def _standardize_opponent(df: pd.DataFrame):
    if 'OPP' in df.columns:
        df['OPP'] = df['OPP'].astype(str).str.strip()
    return df

def _clean_ownership(df: pd.DataFrame):
    """Create OWN_PCT as a numeric estimate from strings like '20-30%-', '5%' etc."""
    if 'PROJ ROSTER %' not in df.columns:
        df['OWN_PCT'] = pd.NA
        return df
    raw = df['PROJ ROSTER %'].astype(str).str.replace('%', '', regex=False).str.strip()
    # Remove trailing dashes or odd chars
    raw = raw.str.replace(r'[^0-9\.\-]', '', regex=True)
    own_est = []
    for val in raw.fillna(''):
        if not val:
            own_est.append(pd.NA)
            continue
        # ranges like "20-30" -> midpoint 25
        nums = re.findall(r'\d+\.?\d*', val)
        if len(nums) == 0:
            own_est.append(pd.NA)
        elif len(nums) == 1:
            own_est.append(float(nums[0]))
        else:
            try:
                lo, hi = float(nums[0]), float(nums[1])
                own_est.append((lo + hi) / 2.0)
            except:
                own_est.append(pd.NA)
    df['OWN_PCT'] = pd.to_numeric(pd.Series(own_est), errors='coerce')
    return df

def normalize(df: pd.DataFrame, pos_hint: str|None=None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = _rename_columns(df)
    df = _extract_from_player_name(df)
    if pos_hint and ('POS' not in df.columns or df['POS'].isna().any()):
        df['POS'] = df.get('POS', pd.Series([None]*len(df))).fillna(pos_hint.upper())
    df = _clean_salary(df)
    df = _clean_proj_pts(df)
    df = _standardize_opponent(df)
    df = _clean_ownership(df)

    required = ['PLAYER NAME','TEAM','POS','PROJ PTS','SALARY']
    for m in required:
        if m not in df.columns:
            df[m] = pd.NA

    df = df.dropna(subset=['PROJ PTS','SALARY'])
    df['PROJ PTS'] = df['PROJ PTS'].astype(float)
    df['SALARY'] = df['SALARY'].astype(int)
    df = df[(df['SALARY'] >= 3000) & (df['SALARY'] <= 15000)]
    return df
