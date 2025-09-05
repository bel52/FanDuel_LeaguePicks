#!/usr/bin/env python3
import os, csv, re, collections

BASE = os.path.join('data','fantasypros')
FILES = ['qb.csv','rb.csv','wr.csv','te.csv','dst.csv']

def parse_pos_from_name(s):
    if not s: return ''
    m = re.search(r'\(([^)]+)\)$', s)
    if not m: return ''
    inside = m.group(1)
    parts = [p.strip() for p in inside.split('-')]
    return (parts[-1] if parts else '').upper()

def main():
    total_lines = 0
    kickoff_counts = collections.Counter()
    pos_counts = collections.Counter()
    header_set = set()

    for fn in FILES:
        p = os.path.join(BASE, fn)
        with open(p, newline='') as f:
            rd = csv.reader(f)
            header = next(rd)
            header_set.add(tuple(header))
            rows = list(rd)
            total_lines += len(rows)
            # sample
            sample = rows[:3]

        print(f"\n=== {fn} ===")
        print("Columns:", header)
        print("Sample rows (first 2):", sample[:2])

        # DictReader for field lookups
        with open(p, newline='') as f:
            rd = csv.DictReader(f)
            for r in rd:
                name = (r.get('PLAYER NAME') or r.get('Player') or '').strip()
                if not name: continue
                pos = parse_pos_from_name(name)
                pos_counts[pos] += 1
                ko = (r.get('KICKOFF') or '').strip().lower()
                if ko:
                    kickoff_counts[ko] += 1

    print("\nAll files had identical header sets?" , len(header_set)==1)
    print("Total non-header rows across files:", total_lines)
    print("\nKickoff buckets (top 12):")
    for k,v in kickoff_counts.most_common(12):
        print(f"  {k}: {v}")
    print("\nDetected positions in PLAYER NAME:")
    for k in sorted(pos_counts):
        print(f"  {k}: {pos_counts[k]}")

if __name__ == "__main__":
    main()
