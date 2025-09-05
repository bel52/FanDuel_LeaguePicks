import re

def clean_name(full_name: str) -> str:
    """Strip '(TEAM - POS)' and trailing injury tags like 'Q', 'O', 'GTD' etc."""
    if not full_name:
        return ""
    s = str(full_name)
    s = re.sub(r'\s*\([^)]+\)\s*', '', s).strip()
    s = re.sub(r'\s+(Q|O|D|T|GTD|P)$', '', s).strip()
    return s

def _build_name_index(df):
    idx = {}
    for i, row in df.iterrows():
        nm = clean_name(row.get('PLAYER NAME', '')).casefold()
        if not nm:
            continue
        idx.setdefault(nm, []).append(i)
    return idx

def match_names_to_indices(names, df):
    """Return (indices, not_found) given a list of raw names."""
    if not names:
        return [], []
    name_index = _build_name_index(df)
    used = set()
    indices = []
    not_found = []
    for raw in names:
        if not raw:
            continue
        nm = clean_name(raw).casefold()
        # exact
        if nm in name_index and name_index[nm]:
            cand = next((i for i in name_index[nm] if i not in used), None)
            if cand is not None:
                used.add(cand)
                indices.append(cand)
                continue
        # partial (substring)
        cands = []
        for key, ilist in name_index.items():
            if nm and nm in key:
                for i in ilist:
                    if i not in used:
                        cands.append(i)
        if cands:
            used.add(cands[0])
            indices.append(cands[0])
        else:
            not_found.append(raw)
    return indices, not_found
