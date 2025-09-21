#!/usr/bin/env python3
import os, csv, json
week = 1
# auto-detect latest week from executed
try:
    with open("data/executed/fd_executed.csv", newline='') as f:
        rows = list(csv.DictReader(f))
        ws = [int(r['Week']) for r in rows if r.get('Week')]
        week = max(ws) if ws else 1
except: pass
log = f"logs/late_swap_week{week}.log"
if not os.path.exists(log):
    print("No late-swap log yet:", log); raise SystemExit(0)
lines = [l.strip() for l in open(log)]
moves = [l for l in lines if "] OUT " in l and "-> IN " in l]
print("\nLate Swap Summary (week", week, ")\n")
for m in moves: print(m)
# write a machine-readable copy too
outp = os.path.join("data","weekly",f"2025_w{week:02d}","late_suggestions.json")
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump({"moves": moves}, open(outp,"w"), indent=2)
print("\nSaved:", outp)
