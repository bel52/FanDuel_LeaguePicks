import re
import textwrap

def _clean_player_name(full_name: str) -> str:
    if not full_name:
        return ""
    base = re.sub(r'\s*\([^)]+\)\s*', '', str(full_name)).strip()
    base = re.sub(r'\s+(Q|O|D|T|GTD|P)$', '', base).strip()
    return base

def _fmt_money(n) -> str:
    try:
        return f"${int(n):,}"
    except Exception:
        return "-"

def _pad(s: str, width: int) -> str:
    s = str(s) if s is not None else ""
    return (s[:width]).ljust(width)

def format_lineup_table(lineup_players, width=100) -> str:
    # POS | PLAYER | TEAM | OPP | SALARY | PROJ | OWN%
    pos_w = 3
    team_w = 4
    opp_w  = 4
    sal_w  = 8
    proj_w = 6
    own_w  = 6
    fixed = pos_w + team_w + opp_w + sal_w + proj_w + own_w + 6*3
    player_w = max(18, min(40, width - fixed))

    header = (
        f"{'POS':<{pos_w}} | "
        f"{'PLAYER':<{player_w}} | "
        f"{'TEAM':<{team_w}} | "
        f"{'OPP':<{opp_w}} | "
        f"{'SALARY':>{sal_w}} | "
        f"{'PROJ':>{proj_w}} | "
        f"{'OWN%':>{own_w}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for p in lineup_players:
        pos = _pad(p.get('position',''), pos_w)
        name = _clean_player_name(p.get('name',''))
        team = _pad(p.get('team',''), team_w)
        opp  = _pad(p.get('opponent',''), opp_w)
        sal  = _pad(_fmt_money(p.get('salary',0)), sal_w)
        proj = _pad(f"{float(p.get('proj_points',0.0)):.1f}", proj_w)
        own_pct = p.get('own_pct')
        if own_pct is None:
            raw = str(p.get('proj_roster_pct_raw') or "").strip("-").strip()
            own_str = raw or ""
        else:
            own_str = f"{float(own_pct):.1f}%"
        own  = _pad(own_str if own_str else " - ", own_w)

        lines.append(
            f"{pos} | "
            f"{_pad(name, player_w)} | "
            f"{team} | "
            f"{opp} | "
            f"{sal:>{sal_w}} | "
            f"{proj:>{proj_w}} | "
            f"{own:>{own_w}}"
        )
    return "\n".join(lines)

def build_text_report(result, width=100) -> str:
    width = max(70, min(160, int(width)))

    title = "FANDUEL OPTIMIZED LINEUP"
    header = f"{title}\n{'='*len(title)}"

    cap = result.get('cap_usage', {}) or {}
    cap_line = f"Cap Used: {_fmt_money(cap.get('total_salary',0))}  |  Remaining: {_fmt_money(cap.get('remaining',0))}"
    proj_line = f"Total Projection: {float(result.get('total_projected_points',0.0)):.2f} pts"

    cons = result.get("constraints", {}) or {}
    auto_locked = [*cons.get("auto_locked", [])]
    locked      = [*cons.get("locks", [])]
    banned      = [*cons.get("bans", [])]
    not_found   = [*cons.get("not_found", [])]
    cons_block = ""
    if auto_locked or locked or banned or not_found:
        cons_lines = ["CONSTRAINTS", "-"*len("CONSTRAINTS")]
        if auto_locked: cons_lines.append(f"Auto-locked: {', '.join(auto_locked)}")
        if locked:      cons_lines.append(f"Locked: {', '.join(locked)}")
        if banned:      cons_lines.append(f"Banned: {', '.join(banned)}")
        if not_found:   cons_lines.append(f"Not found: {', '.join(not_found)}")
        cons_lines.append("")
        cons_block = "\n".join(cons_lines)

    table = format_lineup_table(result.get('lineup', []), width=width)

    sim = result.get('simulation', {}) or {}
    mean = float(sim.get('mean_score', 0.0))
    std  = float(sim.get('std_dev', 0.0)) if sim.get('std_dev') is not None else 0.0
    pcts = sim.get('percentiles', {}) or {}
    p50  = float(pcts.get('50th', 0.0))
    p90  = float(pcts.get('90th', 0.0))
    p95  = float(pcts.get('95th', 0.0))
    sharpe = float(sim.get('sharpe_ratio', 0.0))
    sim_block = "\n".join([
        "SIMULATION SUMMARY",
        "------------------",
        f"Mean: {mean:.2f}   StdDev: {std:.2f}   P50: {p50:.2f}   P90: {p90:.2f}   P95: {p95:.2f}   Sharpe: {sharpe:.3f}",
        ""
    ])

    analysis_text = str(result.get("analysis") or "Analysis not available").strip()
    if analysis_text and not analysis_text.lower().startswith("analysis not available"):
        txt = analysis_text
        for sect in ["Correlation", "Leverage", "Risk", "Risk/Ceiling", "Strengths", "Weaknesses", "Swap Ideas", "Suggested Swaps"]:
            txt = re.sub(rf"(?i)\b{re.escape(sect)}\b\s*:?", sect.upper() + ":", txt)
        blocks = ["ANALYSIS", "--------"]
        for para in txt.split("\n"):
            para = re.sub(r"\s+", " ", para).strip()
            if para:
                blocks.append(textwrap.fill(para, width=width))
        analysis_block = "\n".join(blocks) + "\n"
    else:
        analysis_block = "ANALYSIS\n--------\nAnalysis not available\n"

    parts = [
        header,
        cap_line,
        proj_line,
        "",
        cons_block if cons_block else "",
        table,
        "",
        sim_block,
        analysis_block
    ]
    return "\n".join([p for p in parts if p is not None and p != ""])
