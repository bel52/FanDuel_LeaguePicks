import os
import json
from typing import List, Dict, Optional

_LAST = os.path.join("data", "output", "last_lineup.json")


def save_last_lineup(lineup: List[Dict]) -> None:
    """Persist the latest optimized lineup for league comparison."""
    os.makedirs(os.path.dirname(_LAST), exist_ok=True)
    try:
        with open(_LAST, "w") as f:
            json.dump(lineup, f, indent=2)
    except Exception as e:
        print(f"WARNING: Could not save last lineup: {e}")


def load_last_lineup() -> Optional[List[Dict]]:
    """Load previously saved lineup, or None if not available."""
    if not os.path.exists(_LAST):
        return None
    try:
        with open(_LAST, "r") as f:
            return json.load(f)
    except Exception:
        return None
