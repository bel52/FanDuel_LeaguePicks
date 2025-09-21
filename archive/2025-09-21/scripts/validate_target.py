#!/usr/bin/env python3
import csv, sys

path = "data/targets/fd_target.csv"
try:
    rows = list(csv.DictReader(open(path, newline='')))
except FileNotFoundError:
    print("No target file found:", path); sys.exit(1)

slots_needed = {"QB","RB1","RB2","WR1","WR2","WR3","TE","FLEX","DST"}
have_slots = {r["Position"] for r in rows}
missing = slots_needed - have_slots
if missing:
    print("ERROR: missing slots:", sorted(missing)); sys.exit(2)

salary = sum(int(r["Salary"]) for r in rows)
if salary > 60000:
    print("ERROR: Salary cap exceeded:", salary); sys.exit(3)

by_pos = {r["Position"]: r for r in rows}
qb = by_pos["QB"]; qb_team = qb["Team"]; qb_opp = qb["Opp"]

# Single-stack required
stack_mates = [r for r in rows if r["Team"]==qb_team and r["Position"] in ("WR1","WR2","WR3","TE")]
if len(stack_mates) < 1:
    print("ERROR: No single-stack mate with QB found."); sys.exit(4)

# ≤2 non-QB teammates
non_qb_same_team = [r for r in rows if r["Team"]==qb_team and r["Position"]!="QB"]
if len(non_qb_same_team) > 2:
    print("ERROR: More than 2 non-QB teammates with QB team:", [r["Position"] for r in non_qb_same_team]); sys.exit(5)

# DST not vs QB opp
if by_pos["DST"]["Team"] == qb_opp:
    print("ERROR: DST opposes QB opponent."); sys.exit(6)

# FLEX must be RB/WR/TE by actual Pos column
flex_pos = by_pos["FLEX"].get("Pos","")
if flex_pos not in ("RB","WR","TE"):
    print("ERROR: FLEX Pos invalid:", flex_pos); sys.exit(7)

print("VALID ✅  Salary =", salary, " Stack mate(s) =", [r["Player"] for r in stack_mates], "DST ok =", by_pos["DST"]["Team"]!=qb_opp)
