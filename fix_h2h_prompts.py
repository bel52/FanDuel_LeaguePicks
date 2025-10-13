# In ai_analyzer.py, find _get_openai_insights (around line 395-450)
# Replace the prompt with this:

def _get_openai_insights(self, data: Dict, contest_type: str) -> str:
    """Get strategic insights from OpenAI with contest-aware prompting"""
    
    # H2H NEEDS DIFFERENT STRATEGY THAN GPP
    if contest_type == 'h2h':
        prompt = f"""You are analyzing NFL DFS for HEAD-TO-HEAD (1v1) format. This is NOT a tournament.

🎯 H2H GOAL: Beat ONE person with the highest-scoring lineup possible.

CRITICAL H2H RULES:
1. MAXIMIZE POINTS - Forget leverage, forget ownership, forget being clever
2. CORRELATION STACKING - QB MUST be stacked with TE or 2+ WRs from same team
3. USE FULL SALARY - Every dollar matters. Aim for $59,500+ of $60,000
4. NO PUNT PLAYS - Avoid players under $3,000. They're desperation moves.
5. CEILING > FLOOR - In 1v1, you need upside, not consistency

GAME ENVIRONMENT:
High Total Game: {data.get('vegas_high_total_games', [{}])[0].get('game_id', 'N/A')} 
  Total: {data.get('vegas_high_total_games', [{}])[0].get('total', 0)} points

TOP PLAYERS:
QBs: {[(p['name'], f"${p['salary']}", f"{p.get('projection', 0):.1f}pts") for p in data['top_players']['QB'][:3]]}
RBs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['RB'][:4]]}
WRs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['WR'][:4]]}
TEs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['TE'][:3]]}

PROVIDE:
1. MUST-PLAY QB (highest ceiling in high-total game)
2. STACK-WITH (which TE/WRs to pair with that QB)
3. BRING-BACK (opposing team's best RB/WR)
4. AVOID (punt plays under $3K, low-ceiling options)

Be direct. No "leverage" talk. Just tell me the highest-ceiling lineup core."""
        
    else:  # GPP/Cash/Contrarian
        prompt = f"""Analyze this NFL DFS slate for {contest_type.upper()} format.

Contest: {contest_type.upper()} (12-person friends league)
Players: {data['slate_size']}
Average Salary: ${data['avg_salary']:.0f}

HIGH-TOTAL GAMES (47+ points):
{[(g.get('game_id'), f"{g.get('total')}pts") for g in data.get('vegas_high_total_games', [])[:3]]}

TOP PLAYERS BY POSITION:
QBs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['QB'][:3]]}
RBs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['RB'][:4]]}
WRs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['WR'][:4]]}
TEs: {[(p['name'], f"${p['salary']}") for p in data['top_players']['TE'][:3]]}

Provide leverage plays for {contest_type} differentiation."""

    return self._call_openai(prompt)
