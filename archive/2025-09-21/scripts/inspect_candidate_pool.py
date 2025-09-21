#!/usr/bin/env python3
from collections import defaultdict
from src.data_fetch import load_projections
from src import lineup_rules as rules

P = load_projections()
pool = [p for p in P if rules.meets_value(p)]
by = defaultdict(int)
for p in pool: by[p['Pos']] += 1
print("Pool size after value gate:", len(pool))
for pos in ('QB','RB','WR','TE','DST'):
    print(pos, by[pos])
