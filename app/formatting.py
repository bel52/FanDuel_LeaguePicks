from typing import Dict, List, Any

def build_text_report(result: Dict[str, Any], width: int = 100) -> str:
    """Build formatted text report for lineup results"""
    
    divider = "=" * width
    sub_divider = "-" * width
    
    # Header
    game_type = result.get('game_type', 'league').upper()
    report = f"\n{divider}\n"
    report += f"FANDUEL NFL DFS LINEUP - {game_type} STRATEGY\n"
    report += f"{divider}\n\n"
    
    # Lineup details
    report += "OPTIMIZED LINEUP:\n"
    report += f"{sub_divider}\n"
    report += f"{'POS':<4} {'PLAYER':<25} {'TEAM':<5} {'OPP':<5} {'SALARY':<8} {'PROJ':<7} {'OWN%':<6} {'VALUE':<6}\n"
    report += f"{sub_divider}\n"
    
    total_salary = 0
    total_proj = 0
    
    for player in result.get('lineup', []):
        name = player['name'][:24]  # Truncate long names
        pos = player['position']
        team = player['team']
        opp = player.get('opponent', 'N/A')
        salary = player['salary']
        proj = player['proj_points']
        own = player.get('own_pct', 0)
        value = proj / (salary / 1000)
        
        total_salary += salary
        total_proj += proj
        
        report += f"{pos:<4} {name:<25} {team:<5} {opp:<5} ${salary:<7,} {proj:<7.1f} {own:<5.1f}% {value:<6.2f}\n"
    
    report += f"{sub_divider}\n"
    report += f"{'TOTALS:':<42} ${total_salary:<7,} {total_proj:<7.1f}\n"
    
    # Cap usage
    cap_info = result.get('cap_usage', {})
    salary_remaining = cap_info.get('remaining', 60000 - total_salary)
    report += f"SALARY REMAINING: ${salary_remaining:,}\n"
    report += f"{sub_divider}\n\n"
    
    # Simulation results
    sim = result.get('simulation', {})
    if sim:
        report += "SIMULATION ANALYSIS:\n"
        report += f"{sub_divider}\n"
        report += f"Expected Score: {sim.get('mean_score', total_proj):.1f} points\n"
        report += f"Standard Deviation: {sim.get('std_dev', 0):.1f} points\n"
        
        percentiles = sim.get('percentiles', {})
        if percentiles:
            report += f"90th Percentile: {percentiles.get('90th', 0):.1f} points\n"
            report += f"95th Percentile: {percentiles.get('95th', 0):.1f} points\n"
        
        sharpe = sim.get('sharpe_ratio', 0)
        report += f"Sharpe Ratio: {sharpe:.3f}\n"
        report += f"{sub_divider}\n\n"
    
    # AI Analysis
    ai_analysis = result.get('analysis', '')
    if ai_analysis and ai_analysis != 'Analysis not available':
        report += "AI ANALYSIS:\n"
        report += f"{sub_divider}\n"
        report += format_text_block(ai_analysis, width - 4)
        report += f"\n{sub_divider}\n\n"
    
    # Constraints applied
    constraints = result.get('constraints', {})
    if any([constraints.get('locks'), constraints.get('bans'), constraints.get('auto_locked')]):
        report += "CONSTRAINTS APPLIED:\n"
        report += f"{sub_divider}\n"
        
        if constraints.get('auto_locked'):
            report += f"Auto-locked (started): {', '.join(constraints['auto_locked'])}\n"
        if constraints.get('locks'):
            report += f"Manual locks: {', '.join(constraints['locks'])}\n"
        if constraints.get('bans'):
            report += f"Banned players: {', '.join(constraints['bans'])}\n"
        if constraints.get('not_found'):
            report += f"Players not found: {', '.join(constraints['not_found'])}\n"
        
        report += f"{sub_divider}\n\n"
    
    # Optimization details
    opt_details = result.get('optimization_details', {})
    if opt_details:
        report += "OPTIMIZATION INFO:\n"
        report += f"{sub_divider}\n"
        report += f"Method: {opt_details.get('method', 'unknown')}\n"
        report += f"AI Enhanced: {opt_details.get('ai_enhanced', False)}\n"
        report += f"Game Type: {opt_details.get('game_type', 'league')}\n"
        
        if opt_details.get('objective_value'):
            report += f"Objective Value: {opt_details['objective_value']:.2f}\n"
        
        report += f"{sub_divider}\n"
    
    report += f"\n{divider}\n"
    
    return report

def format_text_block(text: str, width: int) -> str:
    """Format a text block to fit within specified width"""
    lines = []
    current_line = ""
    
    # Split by existing newlines first
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        words = paragraph.split()
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append("  " + current_line)
                current_line = word
        
        if current_line:
            lines.append("  " + current_line)
        
        # Add blank line between paragraphs
        if paragraph != paragraphs[-1]:
            lines.append("")
    
    return '\n'.join(lines)

def format_player_summary(player: Dict[str, Any]) -> str:
    """Format a single player summary"""
    return (
        f"{player['position']} - {player['name']} "
        f"({player['team']} vs {player.get('opponent', 'N/A')}) "
        f"${player['salary']:,} - {player['proj_points']:.1f} pts"
    )

def format_swap_summary(swap: Dict[str, Any]) -> str:
    """Format a swap summary"""
    out_player = swap['player_out']
    in_player = swap['player_in']
    trigger = swap.get('trigger', {})
    
    return (
        f"SWAP: {out_player['name']} → {in_player['name']}\n"
        f"  Position: {out_player['position']}\n"
        f"  Salary: ${out_player['salary']:,} → ${in_player['salary']:,}\n"
        f"  Projection: {out_player.get('proj_points', 0):.1f} → {in_player.get('projection', 0):.1f}\n"
        f"  Reason: {trigger.get('description', 'Manual swap')}"
    )

def format_money(amount: float) -> str:
    """Format money values"""
    return f"${amount:,.0f}" if amount >= 0 else f"-${abs(amount):,.0f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage values"""
    return f"{value:.{decimals}f}%"
