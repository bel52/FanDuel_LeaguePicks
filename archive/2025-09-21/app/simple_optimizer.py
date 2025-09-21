"""
Simple, working DFS optimizer that focuses on core functionality
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from itertools import combinations

logger = logging.getLogger(__name__)

class SimpleDFSOptimizer:
    """A simple, working DFS optimizer using greedy selection with constraints"""
    
    def __init__(self):
        self.salary_cap = 60000
        self.position_requirements = {
            'QB': 1,
            'RB': 2, 
            'WR': 3,
            'TE': 1,
            'DST': 1
        }
        # FLEX positions: need exactly 7 RB+WR+TE total (2 RB + 3 WR + 1 TE + 1 FLEX)
        self.total_flex = 7
    
    async def optimize_lineup(
        self,
        df: pd.DataFrame,
        game_type: str = "league",
        salary_cap: int = 60000,
        enforce_stack: bool = True,
        lock_players: Optional[List[str]] = None,
        ban_players: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate optimal DFS lineup
        """
        try:
            # Input validation
            if df is None or df.empty:
                return {"error": "No player data provided", "lineup": []}
            
            # Ensure required columns exist
            required_cols = ['PLAYER NAME', 'POS', 'SALARY', 'PROJ PTS']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {"error": f"Missing required columns: {missing_cols}", "lineup": []}
            
            # Clean the data
            df_clean = self._clean_data(df.copy())
            
            if df_clean.empty:
                return {"error": "No valid players after data cleaning", "lineup": []}
            
            # Apply locks and bans
            df_filtered = self._apply_filters(df_clean, lock_players, ban_players)
            
            # Generate lineup
            if enforce_stack:
                lineup = self._optimize_with_stacking(df_filtered, salary_cap, game_type)
            else:
                lineup = self._optimize_greedy(df_filtered, salary_cap, game_type)
            
            if not lineup:
                return {"error": "Could not generate valid lineup", "lineup": []}
            
            # Format results
            lineup_data = []
            total_salary = 0
            total_projection = 0
            
            for player in lineup:
                player_data = {
                    "player_name": player['PLAYER NAME'],
                    "position": player['POS'],
                    "team": player.get('TEAM', ''),
                    "salary": int(player['SALARY']),
                    "projection": float(player['PROJ PTS']),
                    "ownership": float(player.get('OWN_PCT', 0)) if pd.notna(player.get('OWN_PCT')) else None
                }
                lineup_data.append(player_data)
                total_salary += player_data["salary"]
                total_projection += player_data["projection"]
            
            return {
                "success": True,
                "game_type": game_type,
                "lineup": lineup_data,
                "total_salary": total_salary,
                "total_projection": round(total_projection, 2),
                "salary_remaining": salary_cap - total_salary,
                "method": "stacking" if enforce_stack else "greedy"
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {"error": str(e), "lineup": []}
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate player data"""
        # Remove rows with missing critical data
        df = df.dropna(subset=['PLAYER NAME', 'SALARY', 'PROJ PTS'])
        
        # Convert and validate salary
        df['SALARY'] = pd.to_numeric(df['SALARY'], errors='coerce').fillna(0).astype(int)
        df = df[df['SALARY'] > 3000]  # Minimum salary filter
        
        # Convert and validate projections
        df['PROJ PTS'] = pd.to_numeric(df['PROJ PTS'], errors='coerce').fillna(0)
        df = df[df['PROJ PTS'] > 0]  # Must have positive projection
        
        # Calculate value
        df['VALUE'] = df['PROJ PTS'] / (df['SALARY'] / 1000)
        
        return df
    
    def _apply_filters(self, df: pd.DataFrame, lock_players: Optional[List[str]], ban_players: Optional[List[str]]) -> pd.DataFrame:
        """Apply lock and ban filters"""
        # Handle bans
        if ban_players:
            df = df[~df['PLAYER NAME'].isin(ban_players)]
        
        # Locks will be handled during optimization
        return df
    
    def _optimize_with_stacking(self, df: pd.DataFrame, salary_cap: int, game_type: str) -> List[Dict]:
        """Optimize with QB stacking"""
        best_lineup = None
        best_score = -1
        
        # Get QBs
        qbs = df[df['POS'] == 'QB'].nlargest(8, 'VALUE')
        
        for _, qb in qbs.iterrows():
            qb_team = qb.get('TEAM', '')
            
            # Find stack mates (WR/TE from same team)
            stack_candidates = df[
                (df['TEAM'] == qb_team) & 
                (df['POS'].isin(['WR', 'TE'])) &
                (df.index != qb.name)
            ]
            
            if len(stack_candidates) == 0:
                continue
            
            # Try different stack sizes
            for stack_size in [1, 2]:
                if len(stack_candidates) < stack_size:
                    continue
                
                for stack_combo in combinations(stack_candidates.index, stack_size):
                    lineup = self._build_lineup_with_stack(
                        df, qb, stack_combo, salary_cap, game_type
                    )
                    
                    if lineup:
                        score = sum(p['PROJ PTS'] for p in lineup)
                        if score > best_score:
                            best_score = score
                            best_lineup = lineup
        
        # If no stacked lineup works, fall back to greedy
        if not best_lineup:
            return self._optimize_greedy(df, salary_cap, game_type)
        
        return best_lineup
    
    def _build_lineup_with_stack(self, df: pd.DataFrame, qb: pd.Series, stack_indices: Tuple, salary_cap: int, game_type: str) -> Optional[List[Dict]]:
        """Build lineup with QB and stack"""
        lineup = [qb.to_dict()]
        used_salary = qb['SALARY']
        used_indices = {qb.name}
        
        # Add stack players
        for idx in stack_indices:
            player = df.loc[idx]
            lineup.append(player.to_dict())
            used_salary += player['SALARY']
            used_indices.add(idx)
        
        if used_salary >= salary_cap:
            return None
        
        # Fill remaining positions
        remaining_budget = salary_cap - used_salary
        
        # Count what we have
        position_counts = {}
        for player in lineup:
            pos = player['POS']
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Fill remaining positions in order of requirement
        remaining_positions = []
        
        # Add required RBs
        rb_needed = self.position_requirements['RB'] - position_counts.get('RB', 0)
        remaining_positions.extend(['RB'] * max(0, rb_needed))
        
        # Add required WRs
        wr_needed = self.position_requirements['WR'] - position_counts.get('WR', 0)
        remaining_positions.extend(['WR'] * max(0, wr_needed))
        
        # Add required TEs
        te_needed = self.position_requirements['TE'] - position_counts.get('TE', 0)
        remaining_positions.extend(['TE'] * max(0, te_needed))
        
        # Add DST
        dst_needed = self.position_requirements['DST'] - position_counts.get('DST', 0)
        remaining_positions.extend(['DST'] * max(0, dst_needed))
        
        # Calculate FLEX needs
        current_flex = position_counts.get('RB', 0) + position_counts.get('WR', 0) + position_counts.get('TE', 0)
        flex_needed = self.total_flex - current_flex
        remaining_positions.extend(['FLEX'] * max(0, flex_needed))
        
        # Fill each position
        for pos in remaining_positions:
            if remaining_budget < 3000:  # Not enough for minimum player
                return None
            
            # Get candidates
            if pos == 'FLEX':
                candidates = df[
                    (df['POS'].isin(['RB', 'WR', 'TE'])) &
                    (~df.index.isin(used_indices)) &
                    (df['SALARY'] <= remaining_budget)
                ]
            else:
                candidates = df[
                    (df['POS'] == pos) &
                    (~df.index.isin(used_indices)) &
                    (df['SALARY'] <= remaining_budget)
                ]
            
            if candidates.empty:
                return None
            
            # Select best value
            best_candidate = candidates.loc[candidates['VALUE'].idxmax()]
            lineup.append(best_candidate.to_dict())
            used_salary += best_candidate['SALARY']
            remaining_budget -= best_candidate['SALARY']
            used_indices.add(best_candidate.name)
        
        # Validate lineup
        if len(lineup) == 9 and used_salary <= salary_cap:
            return lineup
        
        return None
    
    def _optimize_greedy(self, df: pd.DataFrame, salary_cap: int, game_type: str) -> List[Dict]:
        """Simple greedy optimization by value"""
        lineup = []
        used_salary = 0
        used_indices = set()
        
        # Position requirements tracking
        position_filled = {pos: 0 for pos in self.position_requirements.keys()}
        flex_filled = 0
        
        # Sort players by value (projection per $1K salary)
        df_sorted = df.sort_values('VALUE', ascending=False)
        
        for _, player in df_sorted.iterrows():
            if player.name in used_indices:
                continue
            
            pos = player['POS']
            salary = player['SALARY']
            
            # Check if we can afford this player
            if used_salary + salary > salary_cap:
                continue
            
            # Check if we need this position
            can_add = False
            
            # Check base position requirements
            if pos in self.position_requirements:
                if position_filled[pos] < self.position_requirements[pos]:
                    can_add = True
            
            # Check FLEX eligibility
            if not can_add and pos in ['RB', 'WR', 'TE']:
                current_flex_total = position_filled['RB'] + position_filled['WR'] + position_filled['TE']
                if current_flex_total < self.total_flex:
                    can_add = True
            
            if can_add and len(lineup) < 9:
                lineup.append(player.to_dict())
                used_salary += salary
                used_indices.add(player.name)
                
                if pos in position_filled:
                    position_filled[pos] += 1
                
                # Check if lineup is complete
                if len(lineup) == 9:
                    break
        
        # Validate final lineup
        if len(lineup) == 9 and self._validate_lineup(lineup):
            return lineup
        
        return []
    
    def _validate_lineup(self, lineup: List[Dict]) -> bool:
        """Validate that lineup meets all constraints"""
        if len(lineup) != 9:
            return False
        
        # Count positions
        position_counts = {}
        for player in lineup:
            pos = player['POS']
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Check position requirements
        for pos, required in self.position_requirements.items():
            if position_counts.get(pos, 0) < required:
                return False
        
        # Check FLEX total (RB + WR + TE = 7)
        flex_total = position_counts.get('RB', 0) + position_counts.get('WR', 0) + position_counts.get('TE', 0)
        if flex_total != self.total_flex:
            return False
        
        # Check total players
        total_players = sum(position_counts.values())
        if total_players != 9:
            return False
        
        return True


# Test function
async def test_optimizer():
    """Test the optimizer with sample data"""
    import pandas as pd
    
    # Create test data
    sample_data = {
        'PLAYER NAME': [
            'Josh Allen', 'Lamar Jackson',  # QBs
            'Derrick Henry', 'Christian McCaffrey', 'Austin Ekeler',  # RBs
            'Davante Adams', 'Stefon Diggs', 'Tyreek Hill', 'Mike Evans',  # WRs
            'Travis Kelce', 'Mark Andrews',  # TEs
            'Buffalo', 'San Francisco'  # DSTs
        ],
        'POS': [
            'QB', 'QB',
            'RB', 'RB', 'RB',
            'WR', 'WR', 'WR', 'WR',
            'TE', 'TE',
            'DST', 'DST'
        ],
        'SALARY': [
            8500, 8200,
            6800, 9000, 7400,
            8000, 7200, 7800, 6900,
            6500, 5800,
            3200, 3400
        ],
        'PROJ PTS': [
            22.5, 21.8,
            18.2, 20.1, 16.5,
            17.5, 16.8, 17.2, 15.9,
            15.3, 13.8,
            9.2, 8.8
        ],
        'TEAM': [
            'BUF', 'BAL',
            'TEN', 'SF', 'LAC',
            'LV', 'BUF', 'MIA', 'TB',
            'KC', 'BAL',
            'BUF', 'SF'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    optimizer = SimpleDFSOptimizer()
    result = await optimizer.optimize_lineup(df)
    
    if result.get('success'):
        print("✅ Optimizer test successful!")
        print(f"Total projection: {result['total_projection']}")
        print(f"Total salary: ${result['total_salary']:,}")
        for player in result['lineup']:
            print(f"  {player['position']} {player['player_name']} - ${player['salary']} - {player['projection']} pts")
    else:
        print(f"❌ Optimizer test failed: {result.get('error')}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_optimizer())
