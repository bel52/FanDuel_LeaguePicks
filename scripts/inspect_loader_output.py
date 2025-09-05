#!/usr/bin/env python3
from collections import defaultdict
from src.data_fetch import load_projections

P = load_projections()
by = defaultdict(int)
for p in P:
    by[p['Pos']] += 1

print("TOTAL usable rows:", len(P))
for pos in ('QB','RB','WR','TE','DST'):
    print(pos, by[pos])

print("\nFirst 5 players:")
for x in P[:5]:
    print(x)
