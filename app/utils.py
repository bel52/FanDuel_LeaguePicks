from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
from app.config import DATA_DIR, INPUT_DIR, EXPORTS_DIR, LOG_DIR

def ensure_dirs():
    for d in (DATA_DIR, INPUT_DIR, EXPORTS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

def pretty(obj: Dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)

def save_csv(path: Path, header: List[str], rows: List[List[Any]]):
    lines = []
    lines.append(",".join(header))
    for r in rows:
        def csv_cell(x):
            s = str(x)
            if any(ch in s for ch in [",", '"', "\n"]):
                s = '"' + s.replace('"', '""') + '"'
            return s
        lines.append(",".join(csv_cell(c) for c in r))
    path.write_text("\n".join(lines), encoding="utf-8")
