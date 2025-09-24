import numpy as np
from typing import List, Dict, Any

def validate_tournament_lineup(lineup_data: Dict, all_players: List[Dict]) -> Dict[str, Any]:
    """Validate if lineup is truly optimized for tournaments"""
    
    validation = {
        'is_optimal': True,
        'warnings': [],
        'suggestions': []
    }
    
    # Check for tournament-specific issues
    qb_salary = next((p['salary'] for p in lineup_data['players'] if p['position'] == 'QB'), 0)
    if qb_salary < 8000:
        validation['warnings'].append(f"Low QB salary (${qb_salary:,}) - tournaments typically need elite QBs")
        validation['is_optimal'] = False
    
    # Check for correlation stacking
    teams = [p['team'] for p in lineup_data['players']]
    team_counts = {team: teams.count(team) for team in set(teams)}
    max_stack = max(team_counts.values()) if team_counts else 0
    
    if max_stack < 2:
        validation['warnings'].append("No team stacks - tournaments benefit from correlation")
        validation['is_optimal'] = False
    
    # Check ceiling vs projection ratio
    ceiling_ratio = lineup_data['ceiling_score'] / lineup_data['projected_points']
    if ceiling_ratio < 1.4:
        validation['warnings'].append(f"Low ceiling multiplier ({ceiling_ratio:.1f}x) - tournaments need upside")
    
    return validation
