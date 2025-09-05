import re
import textwrap

def _clean_player_name(full_name: str) -> str:
    if not full_name:
        return ""
    base = re.sub(r'\s*\([^)]+\)\s*', '', str(full_name)).strip()
    base = re.sub(r'\s+(Q|O|D|T|GTD|P)$', '', base).strip()
    return base

def _fmt_money(n: int) -> str:
    try:
        return f"${int(n):,}"
    except Exception:
        return "-"

def _fmt_pct(x) -> str:
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "-"

def _pad(s: str, width: int) -> str:
    s = str(s) if s is not None else ""
    return (s[:width]).ljust(width)

def format_lineup_table(lineup_players, width=100) -> str:
    pos_w = 4   # a touch wider for 'FLEX'
    team_w = 4
    opp_w = 4
    sal_w = 9
    proj_w = 6
    own_w = 6
    fixed = pos_w + team_w + opp_w + sal_w + proj_w + own_w + 6*3
    player_w = max(18, min(44, width - fixed))

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
        pos = _pad(p.get("position",""), pos_w)
        name = _pad(_clean_player_name(p.get("name","")), player_w)
        team = _pad(p.get("team",""), team_w)
        opp = _pad(p.get("opponent",""), opp_w)
        sal = _fmt_money(p.get("salary"))
        proj = f"{float(p.get('proj_points',0.0)):.1f}"
        own = p.get("own_pct")
        if own is None:
            raw = str(p.get("proj_roster_pct_raw","")).replace("%","")
            m = re.findall(r"\d+\.?\d*", raw)
            if len(m) == 1:
                own = float(m[0])
            elif len(m) >= 2:
                own = (float(m[0]) + float(m[1]))/2.0
        own_txt = _fmt_pct(own) if own is not None else "-"

        line = (
            f"{pos} | "
            f"{name} | "
            f"{team} | "
            f"{opp} | "
            f"{sal:>{sal_w}} | "
            f"{proj:>{proj_w}} | "
            f"{own_txt:>{own_w}}"
        )
        lines.append(line)

    return "\n".join(lines)

def _wrap_markdown_console(text: str, width: int) -> str:
    """Wrap paragraphs but preserve headings and bullets with proper indentation."""
    out = []
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        lines = para.splitlines()
        if not lines:
            out.append("")
            continue
        # Is this a list or heading block?
        is_list = all(l.strip().startswith(("-", "*")) or re.match(r"^\d+\.\s", l.strip() or "") for l in lines if l.strip()) \
                  or any(l.strip().startswith(("#","##","###")) for l in lines)
        if is_list:
            for l in lines:
                s = l.rstrip()
                if not s:
                    out.append("")
                    continue
                # Headings: print as-is
                if s.lstrip().startswith(("#","##","###")):
                    out.append(s.strip())
                    continue
                # Bullets/numbers: wrap with hanging indent
                stripped = s.lstrip()
                indent = " " * (len(s) - len(stripped))
                if stripped.startswith(("-", "*")):
                    content = stripped[1:].strip()
                    wrapped = textwrap.fill(content, width=width, initial_indent=indent + "- ",
                                            subsequent_indent=indent + "  ")
                    out.append(wrapped)
                else:
                    m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
                    if m:
                        num, content = m.group(1), m.group(2)
                        prefix = f"{num}. "
                        wrapped = textwrap.fill(content, width=width,
                                                initial_indent=indent + prefix,
                                                subsequent_indent=indent + " " * len(prefix))
                        out.append(wrapped)
                    else:
                        out.append(textwrap.fill(s, width=width))
            out.append("")
        else:
            out.append(textwrap.fill(para.strip(), width=width))
            out.append("")
    return "\n".join(out).rstrip() + "\n"

def build_text_report(result: dict, width=100) -> str:
    cap = result.get("cap_usage", {})
    lineup = result.get("lineup", [])
    total_proj = result.get("total_projected_points", 0.0)
    sim = result.get("simulation", {}) or {}
    analysis = result.get("analysis", "")

    title = "FANDUEL OPTIMIZED LINEUP"
    bar = "=" * min(len(title), width)

    header = [
        title,
        bar,
        f"Cap Used: {_fmt_money(cap.get('total_salary',0))}  |  Remaining: {_fmt_money(cap.get('remaining',0))}",
        f"Total Projection: {total_proj:.2f} pts",
        ""
    ]
    table = format_lineup_table(lineup, width=width)

    # Render constraints section if provided
    cons = result.get('constraints', {})
    locks = cons.get('locks', []) or []
    bans = cons.get('bans', []) or []
    not_found = cons.get('not_found', []) or []
    cons_lines = []
    if locks or bans or not_found:
        cons_lines.append('CONSTRAINTS')
        cons_lines.append('-----------')
        if locks:
            cons_lines.append('Locked: ' + ', '.join(locks))
        if bans:
            cons_lines.append('Banned: ' + ', '.join(bans))
        if not_found:
            cons_lines.append('Not found: ' + ', '.join(not_found))
        cons_lines.append('')
        cons_block='\n'.join(cons_lines)
    else:
        cons_block=''


    sim_lines = [
        "",
        "SIMULATION SUMMARY",
        "------------------",
        f"Mean: {sim.get('mean_score',0):.2f}   "
        f"StdDev: {sim.get('std_dev',0):.2f}   "
        f"P50: {sim.get('percentiles',{}).get('50th',0):.2f}   "
        f"P90: {sim.get('percentiles',{}).get('90th',0):.2f}   "
        f"P95: {sim.get('percentiles',{}).get('95th',0):.2f}   "
        f"Sharpe: {sim.get('sharpe_ratio',0):.3f}",
        ""
    ]

    if analysis and isinstance(analysis, str):
        pretty = _wrap_markdown_console(analysis.strip(), width=width)
        analysis_lines = ["ANALYSIS", "--------", pretty.rstrip()]
    else:
        analysis_lines = ["ANALYSIS", "--------", "Analysis not available"]

    blocks = header + ([cons_block] if cons_block else []) + [table] + sim_lines + analysis_lines
    return "\n".join(blocks) + "\n"
