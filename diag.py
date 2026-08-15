import sys; sys.path.insert(0,"src")
from dfs.fantasypros import FantasyProsClient
from dfs.ingest_fanduel import ingest_csv
from dfs.blend import _match_key

fp = FantasyProsClient().weekly_projections(2025, 14)
print("FP PHI/JAC/SEA players:")
for p in fp:
    if p.team in ("PHI","JAC","JAX","SEA") and p.position in ("WR","RB"):
        print(f"  {p.team:4s} {p.position:3s} {p.name!r:28s} key={_match_key(p.name,p.team)!r}")

slate,_ = ingest_csv("/home/brett/fanduel/data/fanduel_salaries_manual.csv","d",2025,14,strict=False)
print("\nCSV counterparts:")
for s in slate.players:
    if s.name.split()[-1] in ("Brown","Etienne","Smith-Njigba","Jr."):
        print(f"  {s.team:4s} {s.position:3s} {s.name!r:28s} key={_match_key(s.name,s.team)!r}")
