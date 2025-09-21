from __future__ import annotations
from typing import List
from pathlib import Path
import json

from app.models import Player
from app.config import INPUT_DIR

_SAMPLE_FILE = INPUT_DIR / "players.sample.json"

def load_players() -> List[Player]:
    if _SAMPLE_FILE.exists():
        data = json.loads(_SAMPLE_FILE.read_text(encoding="utf-8"))
        return [Player(**p) for p in data]
    return _fallback_players()

def _fallback_players() -> List[Player]:
    base = [
        {"id":"1001","name":"Elite QB","team":"KC","position":"QB","salary":8500,"projection":22.5,"opponent":"BUF","home":True},
        {"id":"2001","name":"RB One","team":"SF","position":"RB","salary":8800,"projection":20.1,"opponent":"LAR"},
        {"id":"2002","name":"RB Two","team":"DET","position":"RB","salary":7000,"projection":15.4,"opponent":"CHI"},
        {"id":"3001","name":"WR Alpha","team":"DAL","position":"WR","salary":8600,"projection":19.0,"opponent":"PHI"},
        {"id":"3002","name":"WR Beta","team":"MIA","position":"WR","salary":7900,"projection":17.4,"opponent":"NYJ"},
        {"id":"3003","name":"WR Gamma","team":"JAX","position":"WR","salary":6400,"projection":13.0,"opponent":"IND"},
        {"id":"4001","name":"TE Prime","team":"KC","position":"TE","salary":7800,"projection":16.2,"opponent":"BUF"},
        {"id":"5001","name":"DEF Unit","team":"NE","position":"DEF","salary":4200,"projection":7.0,"opponent":"NYJ"}
    ]
    return [Player(**p) for p in base]
