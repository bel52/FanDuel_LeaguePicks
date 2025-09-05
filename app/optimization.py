import itertools

# Simple, fast search over top candidates. Not perfect but stable & deterministic.
def optimize_lineup(df, salary_cap=60000, enforce_stack=False, min_stack_receivers=1, lock_ids=None, ban_ids=None):
    lock_ids = set(lock_ids or [])
    ban_ids = set(ban_ids or [])

    use = df[~df.index.isin(ban_ids)].copy()

    # Keep only necessary columns
    cols = ["PLAYER NAME","POS","TEAM","OPP","PROJ PTS","SALARY","PROJ ROSTER %","OWN_PCT"]
    for c in cols:
        if c not in use.columns:
            use[c] = None

    # Index we will use to refer back
    use = use.reset_index().rename(columns={"index":"RID"})

    # Split by pos and take top N by projection to constrain the search space
    def topn(pos, n):
        subset = use[use["POS"]==pos].sort_values("PROJ PTS", ascending=False).head(n)
        return subset

    Q = topn("QB", 5)
    R = topn("RB", 7)
    W = topn("WR", 8)
    T = topn("TE", 6)
    D = topn("DST", 5)

    best = None
    best_pts = -1

    # Precompute flex pool
    F = use[use["POS"].isin(["RB","WR","TE"])].sort_values("PROJ PTS", ascending=False).head(20)

    # lock feasibility: must include all locks
    lock_set = set(lock_ids)
    lock_rows = use[use["RID"].isin(lock_set)]
    # If any lock is banned/absent, return empty
    if len(lock_rows) != len(lock_set):
        return []

    # Utility
    def total(items, col): return sum(float(x[col]) for _,x in items)
    def names(items): return [x["RID"] for _,x in items]

    for q in Q.iterrows():
        for r2 in itertools.combinations(R.iterrows(), 2):
            for w3 in itertools.combinations(W.iterrows(), 3):
                for t in T.iterrows():
                    for d in D.iterrows():
                        chosen = list(r2) + list(w3) + [q, t, d]
                        chosen_ids = set(names(chosen))
                        # add FLEX best fit not already chosen
                        remaining = salary_cap - int(total(chosen, "SALARY"))
                        if remaining <= 0:
                            continue
                        flex = None
                        for f in F.iterrows():
                            frid = f[1]["RID"]
                            if frid in chosen_ids: continue
                            if int(f[1]["SALARY"]) <= remaining:
                                flex = f
                                break
                        if not flex:
                            continue

                        final = chosen + [flex]
                        final_ids = set(names(final))

                        # must include all locks
                        if not lock_set.issubset(final_ids):
                            continue

                        # stack rule: if enforce_stack, QB team must have >= min_stack_receivers among WR/TE
                        if enforce_stack:
                            qb_team = str(q[1]["TEAM"])
                            recv = 0
                            for it in final:
                                pos = it[1]["POS"]
                                tm = str(it[1]["TEAM"])
                                if pos in ("WR","TE") and tm == qb_team:
                                    recv += 1
                            if recv < int(min_stack_receivers):
                                continue

                        pts = total(final, "PROJ PTS")
                        sal = int(total(final, "SALARY"))
                        if sal <= salary_cap and pts > best_pts:
                            best = final
                            best_pts = pts

    if not best:
        return []

    # Return original df row indices (RID is original index)
    return [int(it[1]["RID"]) for it in best]
