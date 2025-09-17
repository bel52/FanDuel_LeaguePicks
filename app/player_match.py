import difflib

def match_player_name(name: str, choices: list, cutoff: float = 0.8):
    """Fuzzy match `name` against a list of choice names. Return best match or None."""
    matches = difflib.get_close_matches(name, choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None
