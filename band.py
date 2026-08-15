import sys; sys.path.insert(0,"src")
from dfs.ingest_fanduel import ingest_csv
from dfs.fantasypros import FantasyProsClient
from dfs.matching import match_slate, ProjectionIndex, norm_name

slate,_ = ingest_csv("/home/brett/fanduel/data/fanduel_salaries_manual.csv","b",2025,14,strict=False)
fp = FantasyProsClient().weekly_projections(2025,14)
mapping,_ = match_slate(slate.players, fp)

print("MATCH RATE BY SALARY BAND")
bands=[(9000,99999),(8000,9000),(7000,8000),(6500,7000),(6000,6500),(5000,6000),(0,5000)]
for lo,hi in bands:
    grp=[p for p in slate.players if lo<=p.salary<hi]
    if not grp: continue
    hit=sum(1 for p in grp if p.fd_id in mapping)
    print(f"  ${lo:5d}-{hi:<6d} {hit:3d}/{len(grp):3d} = {hit/len(grp):5.1%}")

print("\nUNMATCHED >= $6000 by position:")
for p in sorted([p for p in slate.players if p.fd_id not in mapping and p.salary>=6000], key=lambda x:-x.salary):
    print(f"  ${p.salary} {p.position:3s} {p.name}")

print("\nOdunze in FP at all?", [f"{x.name}|{x.team}|{x.position}" for x in fp if "odunze" in norm_name(x.name)])
