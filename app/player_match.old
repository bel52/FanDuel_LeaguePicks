import re

def _clean(s: str) -> str:
    if not s: return ""
    s = re.sub(r'\s*\([^)]+\)\s*', '', str(s))
    s = re.sub(r'\s+(Q|O|D|T|GTD|P)$', '', s)
    return re.sub(r'\s+', ' ', s).strip().lower()

def match_names_to_indices(names, players_df):
    names = [n for n in (names or []) if n]
    wanted = set(_clean(n) for n in names)
    found_idx = []
    not_found = []
    if players_df is None or players_df.empty or not wanted:
        return [], names
    for idx, row in players_df.iterrows():
        nm = _clean(row.get("PLAYER NAME", ""))
        if nm in wanted:
            found_idx.append(idx)
    for n in names:
        if _clean(n) not in set(_clean(players_df.loc[i, "PLAYER NAME"]) for i in found_idx):
            not_found.append(n)
    return found_idx, not_found
