from ortools.sat.python import cp_model
import logging

def optimize_lineup(players_df, salary_cap=60000, enforce_stack=False, min_stack_receivers=1,
                    lock_indices=None, ban_indices=None):
    """FanDuel NFL: QB(1), RB(2), WR(3), TE(1), FLEX(1 from RB/WR/TE), DST(1), total 9, cap=60k."""
    model = cp_model.CpModel()
    idxs = list(players_df.index)
    x = {i: model.NewBoolVar(f"x{i}") for i in idxs}

    # Salary cap
    model.Add(sum(int(players_df.loc[i,'SALARY']) * x[i] for i in idxs) <= salary_cap)

    # Pos groups
    qb_idx = [i for i in idxs if players_df.loc[i,'POS'] == 'QB']
    rb_idx = [i for i in idxs if players_df.loc[i,'POS'] == 'RB']
    wr_idx = [i for i in idxs if players_df.loc[i,'POS'] == 'WR']
    te_idx = [i for i in idxs if players_df.loc[i,'POS'] == 'TE']
    dst_idx = [i for i in idxs if players_df.loc[i,'POS'] == 'DST']

    if qb_idx:  model.Add(sum(x[i] for i in qb_idx) == 1)
    if dst_idx: model.Add(sum(x[i] for i in dst_idx) == 1)
    if rb_idx:  model.Add(sum(x[i] for i in rb_idx) >= 2)
    if wr_idx:  model.Add(sum(x[i] for i in wr_idx) >= 3)
    if te_idx:  model.Add(sum(x[i] for i in te_idx) >= 1)

    # RB/WR/TE = 7 (2 RB + 3 WR + 1 TE + 1 FLEX)
    model.Add(sum(x[i] for i in (rb_idx + wr_idx + te_idx)) == 7)

    # Total players
    model.Add(sum(x[i] for i in idxs) == 9)

    # Late-swap locks/bans
    lock_indices = lock_indices or []
    ban_indices = ban_indices or []
    for i in lock_indices:
        if i in x:
            model.Add(x[i] == 1)
    for i in ban_indices:
        if i in x:
            model.Add(x[i] == 0)

    # Optional stacking
    if enforce_stack and qb_idx:
        for q in qb_idx:
            q_team = str(players_df.loc[q,'TEAM'])
            receivers_same_team = [j for j in (wr_idx + te_idx) if str(players_df.loc[j,'TEAM']) == q_team]
            if receivers_same_team:
                model.Add(sum(x[j] for j in receivers_same_team) >= min_stack_receivers * x[q])

    # Objective
    model.Maximize(sum(int(float(players_df.loc[i,'PROJ PTS']) * 10) * x[i] for i in idxs))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logging.error("No feasible lineup found.")
        return []
    chosen = [i for i in idxs if solver.Value(x[i]) == 1]
    logging.info(f"Selected {len(chosen)} players. Objective={solver.ObjectiveValue()/10:.2f}")
    return chosen
