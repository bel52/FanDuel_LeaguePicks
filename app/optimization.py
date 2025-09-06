import itertools
import logging
from typing import List, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_lineup(
    df: pd.DataFrame,
    salary_cap: int = 60000,
    enforce_stack: bool = True,
    min_stack_receivers: int = 1,
    lock_indices: Optional[List[int]] = None,
    ban_indices: Optional[List[int]] = None,
    leverage_weight: float = 0.2
) -> List[int]:
    """
    Lineup optimization. Tries LP if PuLP is present; otherwise a heuristic.
    If TEAM data is unreliable (mostly 'UNK' / single unique team), stacking is auto-disabled.
    Includes a cheapest-lineup fallback to ensure we detect impossible caps.
    """
    lock_indices = set(lock_indices or [])
    ban_indices  = set(ban_indices or [])

    use_df = df[~df.index.isin(ban_indices)].copy()
    if use_df.empty:
        return []

    # Auto-disable stacking if teams look unreliable
    teams_nonunk = set([t for t in use_df["TEAM"].astype(str) if t and t != "UNK"])
    if (not teams_nonunk) or (use_df["TEAM"].nunique() <= 1):
        enforce_stack = False

    # value & adj proj
    use_df['value']    = use_df['PROJ PTS'] / (use_df['SALARY'] / 1000)
    use_df['adj_proj'] = use_df['PROJ PTS']

    if 'OWN_PCT' in use_df.columns:
        # small warning fix for future pandas changes by infer_objects:
        own = use_df['OWN_PCT'].fillna(0)
        try:
            own = own.infer_objects(copy=False)
        except Exception:
            pass
        use_df['leverage'] = use_df['PROJ PTS'] / (own + 1)
        use_df['adj_proj'] = use_df['PROJ PTS'] * (1 - leverage_weight) + use_df['leverage'] * leverage_weight

    # Try LP first
    try:
        return _opt_lp(use_df, salary_cap, enforce_stack, min_stack_receivers, lock_indices)
    except ImportError:
        logger.warning("PuLP not available; using heuristic optimizer.")
        lineup = _opt_heuristic(use_df, salary_cap, enforce_stack, min_stack_receivers, lock_indices)
        if lineup:
            return lineup
        # Fallback to cheapest valid lineup (if none exists under cap, explain why)
        cheap = _cheapest_valid_lineup(use_df)
        if not cheap:
            return []
        if _total_salary(use_df, cheap) > salary_cap:
            min_sal = _total_salary(use_df, cheap)
            print(f"ERROR: Even the cheapest valid lineup costs ${min_sal}, which exceeds the salary cap ${salary_cap}.")
            print("Hint: Your salary scale may be from a different site/scoring. Adjust the cap in CLI or config.")
            return []
        return cheap


def _opt_lp(df: pd.DataFrame, salary_cap: int, enforce_stack: bool, min_stack_receivers: int, lock_indices: Set[int]) -> List[int]:
    import pulp

    df = df.reset_index(drop=True)

    prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"p_{i}", cat="Binary") for i in df.index}

    prob += pulp.lpSum(df.loc[i, 'adj_proj'] * x[i] for i in df.index)
    prob += pulp.lpSum(df.loc[i, 'SALARY'] * x[i] for i in df.index) <= salary_cap

    pos = df['POS']
    def _sum_pos(p): return pulp.lpSum(x[i] for i in df.index if pos[i] == p)

    prob += _sum_pos('QB') == 1
    prob += _sum_pos('DST') == 1
    prob += _sum_pos('RB') >= 2
    prob += _sum_pos('RB') <= 3
    prob += _sum_pos('WR') >= 3
    prob += _sum_pos('WR') <= 4
    prob += _sum_pos('TE') >= 1
    prob += _sum_pos('TE') <= 2
    prob += pulp.lpSum(x[i] for i in df.index) == 9

    for i in lock_indices:
        if i in x:
            prob += x[i] == 1

    if enforce_stack:
        for i in df.index:
            if df.loc[i, 'POS'] != 'QB':
                continue
            team = df.loc[i, 'TEAM']
            if pd.isna(team) or team == "UNK":
                continue
            mates = [j for j in df.index if df.loc[j, 'TEAM'] == team and df.loc[j, 'POS'] in ('WR','TE')]
            if mates:
                prob += pulp.lpSum(x[j] for j in mates) >= min_stack_receivers * x[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    try:
        status = pulp.LpStatus[prob.status]
    except Exception:
        status = "Undefined"

    if status != 'Optimal':
        raise ImportError("LP unavailable or failed")

    return [i for i in df.index if x[i].value() == 1]


def _opt_heuristic(df: pd.DataFrame, salary_cap: int, enforce_stack: bool, min_stack_receivers: int, lock_indices: Set[int]) -> List[int]:
    df = df.reset_index(drop=True)

    # Wider pools to allow cheaper combos
    def pool(p, n):
        # Mix top by adj_proj and by value to get both ceiling and efficiency
        top_proj  = df[df['POS'] == p].nlargest(max(1, n//2), 'adj_proj')
        top_value = df[df['POS'] == p].nlargest(max(1, n - len(top_proj)), 'value')
        return pd.concat([top_proj, top_value]).drop_duplicates().nlargest(n, 'adj_proj')

    qbs = pool('QB', 8)
    rbs = pool('RB', 24)
    wrs = pool('WR', 32)
    tes = pool('TE', 12)
    dsts = pool('DST', 10)

    # As another angle, try some cheap subsets
    cheap_rbs = df[df['POS']=='RB'].nsmallest(24, 'SALARY')
    cheap_wrs = df[df['POS']=='WR'].nsmallest(32, 'SALARY')
    cheap_tes = df[df['POS']=='TE'].nsmallest(12, 'SALARY')
    cheap_dsts= df[df['POS']=='DST'].nsmallest(10, 'SALARY')

    rbs = pd.concat([rbs, cheap_rbs]).drop_duplicates()
    wrs = pd.concat([wrs, cheap_wrs]).drop_duplicates()
    tes = pd.concat([tes, cheap_tes]).drop_duplicates()
    dsts= pd.concat([dsts, cheap_dsts]).drop_duplicates()

    best_score = -1.0
    best = []

    for qb_i in qbs.index:
        qb_team = df.loc[qb_i, 'TEAM']
        # Stack candidates or just best WR/TEs if stacking off
        if enforce_stack and qb_team != "UNK":
            stack_pool = df[(df['TEAM'] == qb_team) & (df['POS'].isin(['WR','TE']))].nlargest(8, 'adj_proj')
        else:
            stack_pool = pd.concat([wrs, tes]).nlargest(8, 'adj_proj')

        stack_sizes = [min_stack_receivers] if enforce_stack and min_stack_receivers > 0 else [0]
        for req_stack in stack_sizes:
            stacks = [tuple()] if req_stack == 0 else itertools.combinations(stack_pool.index, min(req_stack, len(stack_pool)))

            for stack in stacks:
                chosen = set([qb_i, *stack])
                if _total_salary(df, chosen) > salary_cap:
                    continue

                # Try a few DSTs across both strong & cheap
                for dst_i in dsts.index[:6]:
                    if dst_i in chosen:
                        continue
                    s1 = chosen | {dst_i}
                    if _total_salary(df, s1) > salary_cap:
                        continue

                    # RBs: try 2 first
                    rb_pool = [i for i in rbs.index if i not in s1]
                    for rb_combo in itertools.combinations(rb_pool[:16], 2):
                        s2 = s1 | set(rb_combo)
                        if _total_salary(df, s2) > salary_cap:
                            continue

                        # Ensure >=3 WR total
                        wr_pool = [i for i in wrs.index if i not in s2]
                        have_wr = sum(df.loc[list(s2), 'POS'].eq('WR'))
                        wr_needed = max(3 - have_wr, 0)
                        if wr_needed > 3:
                            wr_needed = 3
                        wr_sets = [tuple()] if wr_needed == 0 else itertools.combinations(wr_pool[:20], wr_needed)

                        for wr_combo in wr_sets:
                            s3 = s2 | set(wr_combo)
                            if _total_salary(df, s3) > salary_cap:
                                continue

                            # ensure at least 1 TE
                            have_te = any(df.loc[list(s3), 'POS'].eq('TE'))
                            s4 = set(s3)
                            if not have_te:
                                te_pool = [i for i in tes.index if i not in s3]
                                if not te_pool:
                                    continue
                                s4.add(te_pool[0])
                                if _total_salary(df, s4) > salary_cap:
                                    continue

                            # Fill to 9 with a FLEX
                            current = set(s4)
                            if len(current) < 9:
                                flex_pool = df[(df['POS'].isin(['RB','WR','TE'])) & (~df.index.isin(current))]
                                # try value-first then proj-first
                                flex_try = pd.concat([
                                    flex_pool.nlargest(30, 'value'),
                                    flex_pool.nsmallest(30, 'SALARY'),
                                    flex_pool.nlargest(30, 'adj_proj')
                                ]).drop_duplicates()
                                added = False
                                for f in flex_try.index:
                                    if len(current) >= 9:
                                        break
                                    if _total_salary(df, current | {f}) <= salary_cap:
                                        current.add(f)
                                        added = True
                                        break
                                if not added or len(current) != 9:
                                    continue

                            # lock check
                            if not lock_indices.issubset(current):
                                continue

                            score = _total_adjproj(df, current)
                            if score > best_score:
                                best_score = score
                                best = list(current)

    return best


def _cheapest_valid_lineup(df: pd.DataFrame) -> List[int]:
    """
    Build the absolute cheapest lineup that satisfies roster constraints.
    Useful to diagnose impossible caps.
    """
    df = df.reset_index(drop=True)
    picks: Set[int] = set()

    def pick_n(pos, n):
        nonlocal picks
        pool = df[(df['POS'] == pos) & (~df.index.isin(picks))].nsmallest(max(1,n), 'SALARY').index.tolist()
        for i in pool[:n]:
            picks.add(i)

    # Required slots
    pick_n('QB', 1)
    pick_n('DST', 1)
    pick_n('RB', 2)
    pick_n('WR', 3)
    pick_n('TE', 1)

    # One more as FLEX from cheapest across RB/WR/TE
    flex_pool = df[(df['POS'].isin(['RB','WR','TE'])) & (~df.index.isin(picks))].nsmallest(50, 'SALARY').index
    for f in flex_pool:
        if len(picks) >= 9:
            break
        picks.add(f)

    return list(picks) if len(picks) == 9 else []


def _total_salary(df: pd.DataFrame, ids: Set[int]) -> int:
    return int(df.loc[list(ids), 'SALARY'].sum())


def _total_adjproj(df: pd.DataFrame, ids: Set[int]) -> float:
    return float(df.loc[list(ids), 'adj_proj'].sum())
