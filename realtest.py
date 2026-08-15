import sys; sys.path.insert(0, "src")
from dfs.ingest_fanduel import ingest_csv
from dfs.fantasypros import FantasyProsClient
from dfs.matching import match_slate

CSV = "/home/brett/fanduel/data/fanduel_salaries_manual.csv"
WEEK = int(sys.argv[1]) if len(sys.argv) > 1 else 14

slate, rep = ingest_csv(CSV, "real", 2025, WEEK, strict=False)
print(rep.summary()); print()
fp = FantasyProsClient().weekly_projections(2025, WEEK)
print(f"FP projections: {len(fp)}\n")
mapping, mrep = match_slate(slate.players, fp)
print(mrep.summary())
