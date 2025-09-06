import logging
from typing import List, Tuple, Set
import pandas as pd
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

def match_names_to_indices(
    names: List[str], 
    df: pd.DataFrame,
    name_column: str = 'PLAYER NAME',
    threshold: float = 0.8
) -> Tuple[List[int], List[str]]:
    """
    Match player names to DataFrame indices with fuzzy matching
    
    Returns:
        - List of matched indices
        - List of names that couldn't be matched
    """
    if not names:
        return [], []
    
    matched_indices = []
    not_found = []
    
    for name in names:
        if not name:
            continue
        
        # Try exact match first (case-insensitive)
        exact_match = df[df[name_column].str.lower() == name.lower()]
        
        if not exact_match.empty:
            matched_indices.append(exact_match.index[0])
        else:
            # Try fuzzy matching
            best_match_idx, best_score = find_best_match(name, df[name_column], threshold)
            
            if best_match_idx is not None:
                matched_indices.append(best_match_idx)
                logger.info(f"Fuzzy matched '{name}' to '{df.loc[best_match_idx, name_column]}' (score: {best_score:.2f})")
            else:
                not_found.append(name)
                logger.warning(f"Could not find player: {name}")
    
    return matched_indices, not_found

def find_best_match(
    target: str, 
    candidates: pd.Series,
    threshold: float = 0.8
) -> Tuple[int, float]:
    """Find best fuzzy match for a player name"""
    
    target_lower = target.lower().strip()
    best_idx = None
    best_score = 0
    
    for idx, candidate in candidates.items():
        if pd.isna(candidate):
            continue
        
        candidate_lower = str(candidate).lower().strip()
        
        # Calculate similarity score
        score = calculate_similarity(target_lower, candidate_lower)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_idx = idx
    
    return best_idx, best_score

def calculate_similarity(str1: str, str2: str) -> float:
    """Calculate similarity score between two strings"""
    
    # Handle exact match
    if str1 == str2:
        return 1.0
    
    # Handle last name only matching
    if ' ' in str2:
        last_name = str2.split()[-1]
        if str1 == last_name:
            return 0.85
    
    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, str1, str2).ratio()

def normalize_team_names(team_name: str) -> str:
    """Normalize team name abbreviations"""
    
    team_mapping = {
        # Handle common variations
        'JAC': 'JAX',
        'JAGS': 'JAX',
        'JAGUARS': 'JAX',
        'WFT': 'WAS',
        'WASHINGTON': 'WAS',
        'COMMANDERS': 'WAS',
        'CARDS': 'ARI',
        'CARDINALS': 'ARI',
        'NINERS': 'SF',
        '49ERS': 'SF',
        'PATS': 'NE',
        'PATRIOTS': 'NE',
        'BUCS': 'TB',
        'BUCCANEERS': 'TB',
        'PACK': 'GB',
        'PACKERS': 'GB'
    }
    
    normalized = team_name.upper().strip()
    return team_mapping.get(normalized, normalized)

def find_players_by_team(
    df: pd.DataFrame,
    team: str,
    positions: List[str] = None
) -> pd.DataFrame:
    """Find all players from a specific team"""
    
    normalized_team = normalize_team_names(team)
    team_mask = df['TEAM'] == normalized_team
    
    if positions:
        pos_mask = df['POS'].isin(positions)
        return df[team_mask & pos_mask]
    
    return df[team_mask]

def find_opposing_players(
    df: pd.DataFrame,
    team: str,
    positions: List[str] = None
) -> pd.DataFrame:
    """Find players facing a specific team"""
    
    normalized_team = normalize_team_names(team)
    opp_mask = df['OPP'] == normalized_team
    
    if positions:
        pos_mask = df['POS'].isin(positions)
        return df[opp_mask & pos_mask]
    
    return df[opp_mask]

def validate_lineup_players(
    lineup_indices: List[int],
    df: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """Validate that lineup meets position requirements"""
    
    errors = []
    
    if len(lineup_indices) != 9:
        errors.append(f"Lineup has {len(lineup_indices)} players, need exactly 9")
    
    # Count positions
    positions = df.loc[lineup_indices, 'POS'].value_counts()
    
    # Check QB
    if positions.get('QB', 0) != 1:
        errors.append(f"Need exactly 1 QB, have {positions.get('QB', 0)}")
    
    # Check RB
    rb_count = positions.get('RB', 0)
    if rb_count < 2 or rb_count > 3:
        errors.append(f"Need 2-3 RBs, have {rb_count}")
    
    # Check WR
    wr_count = positions.get('WR', 0)
    if wr_count < 3 or wr_count > 4:
        errors.append(f"Need 3-4 WRs, have {wr_count}")
    
    # Check TE
    te_count = positions.get('TE', 0)
    if te_count < 1 or te_count > 2:
        errors.append(f"Need 1-2 TEs, have {te_count}")
    
    # Check DST
    if positions.get('DST', 0) != 1:
        errors.append(f"Need exactly 1 DST, have {positions.get('DST', 0)}")
    
    # Check FLEX (RB + WR + TE should equal 7)
    flex_total = positions.get('RB', 0) + positions.get('WR', 0) + positions.get('TE', 0)
    if flex_total != 7:
        errors.append(f"FLEX positions total {flex_total}, need exactly 7")
    
    return len(errors) == 0, errors

def get_stack_info(
    lineup_indices: List[int],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """Get stacking information for a lineup"""
    
    lineup_df = df.loc[lineup_indices]
    
    # Find QB
    qb_mask = lineup_df['POS'] == 'QB'
    if not qb_mask.any():
        return {'has_stack': False, 'stack_count': 0}
    
    qb = lineup_df[qb_mask].iloc[0]
    qb_team = qb['TEAM']
    
    # Count teammates
    teammates = lineup_df[(lineup_df['TEAM'] == qb_team) & (lineup_df['POS'] != 'QB')]
    receivers = lineup_df[(lineup_df['TEAM'] == qb_team) & (lineup_df['POS'].isin(['WR', 'TE']))]
    
    return {
        'has_stack': len(receivers) > 0,
        'stack_count': len(receivers),
        'qb_name': qb['PLAYER NAME'],
        'qb_team': qb_team,
        'stack_players': receivers['PLAYER NAME'].tolist(),
        'total_teammates': len(teammates)
    }
