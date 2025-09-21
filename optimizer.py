"""
Advanced DFS lineup optimizer with correlation modeling
"""
import pulp
from ortools.linear_solver import pywraplp
import numpy as np
from typing import List, Dict, Optional, Tuple
import pandas as pd
import polars as pl
from loguru import logger
from config import config
import itertools
import random

class AdvancedLineupOptimizer:
    """Production-grade lineup optimizer with advanced features"""
    
    def __init__(self):
        self.salary_cap = config.SALARY_CAP
        self.position_requirements = config.POSITION_REQUIREMENTS
        self.solver_type = 'CBC'  # Can switch to 'GLPK' or 'OR-Tools'
        
    def optimize_single_lineup(self, players_df: pl.DataFrame, 
                              constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize a single lineup using integer linear programming
        
        Args:
            players_df: Polars DataFrame with player data
            constraints: Additional constraints (stacking, exclusions, etc)
            
        Returns:
            Optimal lineup dictionary
        """
        try:
            logger.info("Starting single lineup optimization")
            
            # Convert to pandas for PuLP compatibility
            players_pd = players_df.to_pandas()
            
            # Create optimization problem
            prob = pulp.LpProblem("DFS_Lineup_Optimization", pulp.LpMaximize)
            
            # Decision variables (binary: player selected or not)
            player_vars = {}
            for idx in players_pd.index:
                player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
            
            # Objective function: maximize projected points
            prob += pulp.lpSum([
                players_pd.loc[idx, 'adjusted_projection'] * player_vars[idx]
                for idx in players_pd.index
            ]), "Total_Projected_Points"
            
            # Constraint: Salary cap
            prob += pulp.lpSum([
                players_pd.loc[idx, 'Salary'] * player_vars[idx]
                for idx in players_pd.index
            ]) <= self.salary_cap, "Salary_Cap"
            
            # Position constraints
            for position, count in self.position_requirements.items():
                if position == 'FLEX':
                    # FLEX can be RB, WR, or TE
                    flex_positions = ['RB', 'WR', 'TE']
                    eligible = players_pd[players_pd['Position'].isin(flex_positions)].index
                    
                    # Account for players already used in RB/WR/TE slots
                    rb_used = pulp.lpSum([
                        player_vars[idx] for idx in players_pd[players_pd['Position'] == 'RB'].index
                    ])
                    wr_used = pulp.lpSum([
                        player_vars[idx] for idx in players_pd[players_pd['Position'] == 'WR'].index
                    ])
                    te_used = pulp.lpSum([
                        player_vars[idx] for idx in players_pd[players_pd['Position'] == 'TE'].index
                    ])
                    
                    # Total FLEX-eligible players must account for position requirements
                    prob += pulp.lpSum([player_vars[idx] for idx in eligible]) >= \
                           self.position_requirements['RB'] + \
                           self.position_requirements['WR'] + \
                           self.position_requirements['TE'] + 1, f"FLEX_Min"
                else:
                    eligible = players_pd[players_pd['Position'] == position].index
                    prob += pulp.lpSum([
                        player_vars[idx] for idx in eligible
                    ]) == count, f"Position_{position}"
            
            # Total players constraint
            prob += pulp.lpSum([
                player_vars[idx] for idx in players_pd.index
            ]) == 9, "Total_Players"
            
            # Apply additional constraints
            if constraints:
                prob = self._apply_constraints(prob, players_pd, player_vars, constraints)
            
            # Solve
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            # Extract lineup
            if prob.status == pulp.LpStatusOptimal:
                lineup = self._extract_lineup(players_pd, player_vars)
                logger.info(f"Optimization successful. Score: {pulp.value(prob.objective):.2f}")
                return lineup
            else:
                logger.error(f"Optimization failed with status: {pulp.LpStatus[prob.status]}")
                return {}
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {}
    
    def optimize_multiple_lineups(self, players_df: pl.DataFrame, 
                                 num_lineups: int,
                                 max_overlap: int = 6,
                                 ownership_limits: Optional[Dict] = None) -> List[Dict]:
        """
        Generate multiple diverse lineups for tournaments
        
        Args:
            players_df: Player data
            num_lineups: Number of lineups to generate
            max_overlap: Maximum players shared between lineups
            ownership_limits: Max exposure limits per player
            
        Returns:
            List of optimized lineups
        """
        lineups = []
        used_combinations = set()
        
        logger.info(f"Generating {num_lineups} lineups with max overlap of {max_overlap}")
        
        for i in range(num_lineups):
            # Add diversity constraints based on previous lineups
            constraints = {
                'exclude_players': [],
                'max_exposure': ownership_limits or {},
                'force_different': i > 0
            }
            
            # Calculate player exposure
            if lineups:
                player_exposure = self._calculate_exposure(lineups)
                for player_name, exposure in player_exposure.items():
                    if ownership_limits and player_name in ownership_limits:
                        if exposure >= ownership_limits[player_name] * len(lineups):
                            constraints['exclude_players'].append(player_name)
            
            # Add overlap constraints
            if lineups and i > 0:
                constraints['previous_lineups'] = lineups
                constraints['max_overlap'] = max_overlap
            
            # Generate lineup
            lineup = self.optimize_single_lineup(players_df, constraints)
            
            if lineup:
                # Check uniqueness
                lineup_key = tuple(sorted([p['name'] for p in lineup['players']]))
                if lineup_key not in used_combinations:
                    lineups.append(lineup)
                    used_combinations.add(lineup_key)
                    logger.info(f"Generated lineup {i+1}/{num_lineups}")
                else:
                    logger.warning(f"Duplicate lineup generated, retrying...")
                    # Retry with more constraints
                    continue
        
        return lineups
    
    def optimize_with_correlation(self, players_df: pl.DataFrame,
                                 correlation_matrix: Dict) -> Dict:
        """
        Optimize lineup with correlation constraints for stacking
        
        Args:
            players_df: Player data
            correlation_matrix: Player correlation values
            
        Returns:
            Correlation-optimized lineup
        """
        # Find optimal stacks first
        stacks = self._find_game_stacks(players_df)
        
        best_lineup = None
        best_score = 0
        
        for stack in stacks[:5]:  # Try top 5 stacks
            constraints = {
                'force_stack': stack,
                'correlation_bonus': True
            }
            
            lineup = self.optimize_single_lineup(players_df, constraints)
            
            if lineup:
                # Calculate combined score with correlation
                base_score = lineup['total_projection']
                correlation_score = self._calculate_correlation_score(
                    lineup['players'], 
                    correlation_matrix
                )
                combined_score = base_score * (1 + correlation_score * 0.1)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_lineup = lineup
                    best_lineup['correlation_score'] = correlation_score
        
        return best_lineup or {}
    
    def _apply_constraints(self, prob, players_df, player_vars, constraints):
        """Apply additional constraints to optimization problem"""
        
        # Exclude specific players
        if 'exclude_players' in constraints:
            for player_name in constraints['exclude_players']:
                player_idx = players_df[players_df['Name'] == player_name].index
                for idx in player_idx:
                    prob += player_vars[idx] == 0, f"Exclude_{player_name}"
        
        # Force include specific players
        if 'include_players' in constraints:
            for player_name in constraints['include_players']:
                player_idx = players_df[players_df['Name'] == player_name].index
                for idx in player_idx:
                    prob += player_vars[idx] == 1, f"Include_{player_name}"
        
        # Stacking constraints
        if 'force_stack' in constraints:
            stack = constraints['force_stack']
            
            # Force QB-WR stack from same team
            if 'qb' in stack and 'receivers' in stack:
                qb_idx = players_df[players_df['Name'] == stack['qb']].index
                if len(qb_idx) > 0:
                    prob += player_vars[qb_idx[0]] == 1, "Force_QB"
                    
                    # Force at least one receiver from same team
                    receiver_indices = []
                    for receiver in stack['receivers']:
                        rec_idx = players_df[players_df['Name'] == receiver].index
                        if len(rec_idx) > 0:
                            receiver_indices.extend(rec_idx)
                    
                    if receiver_indices:
                        prob += pulp.lpSum([
                            player_vars[idx] for idx in receiver_indices
                        ]) >= 1, "Force_Stack_Receiver"
        
        # Overlap constraints with previous lineups
        if 'previous_lineups' in constraints and 'max_overlap' in constraints:
            for i, prev_lineup in enumerate(constraints['previous_lineups']):
                prev_players = [p['name'] for p in prev_lineup['players']]
                prev_indices = players_df[players_df['Name'].isin(prev_players)].index
                
                prob += pulp.lpSum([
                    player_vars[idx] for idx in prev_indices
                ]) <= constraints['max_overlap'], f"Max_Overlap_{i}"
        
        return prob
    
    def _extract_lineup(self, players_df, player_vars) -> Dict:
        """Extract lineup from solved problem"""
        selected_players = []
        total_salary = 0
        total_projection = 0
        
        for idx, var in player_vars.items():
            if var.varValue == 1:
                player = players_df.loc[idx].to_dict()
                selected_players.append({
                    'name': player['Name'],
                    'position': player['Position'],
                    'team': player['Team'],
                    'salary': player['Salary'],
                    'projection': player.get('adjusted_projection', 0)
                })
                total_salary += player['Salary']
                total_projection += player.get('adjusted_projection', 0)
        
        return {
            'players': selected_players,
            'total_salary': total_salary,
            'salary_remaining': self.salary_cap - total_salary,
            'total_projection': total_projection
        }
    
    def _find_game_stacks(self, players_df: pl.DataFrame) -> List[Dict]:
        """Find optimal game stacking opportunities"""
        stacks = []
        
        # Convert to pandas for easier manipulation
        players_pd = players_df.to_pandas()
        
        # Find QBs with high projections
        qbs = players_pd[players_pd['Position'] == 'QB'].nlargest(10, 'adjusted_projection')
        
        for _, qb in qbs.iterrows():
            qb_team = qb['Team']
            
            # Find receivers from same team
            team_receivers = players_pd[
                (players_pd['Team'] == qb_team) & 
                (players_pd['Position'].isin(['WR', 'TE']))
            ].nlargest(3, 'adjusted_projection')
            
            if len(team_receivers) >= 1:
                stack = {
                    'qb': qb['Name'],
                    'receivers': team_receivers['Name'].tolist(),
                    'team': qb_team,
                    'total_projection': qb['adjusted_projection'] + 
                                      team_receivers['adjusted_projection'].sum()
                }
                stacks.append(stack)
        
        # Sort by total projection
        stacks.sort(key=lambda x: x['total_projection'], reverse=True)
        
        return stacks
    
    def _calculate_exposure(self, lineups: List[Dict]) -> Dict[str, float]:
        """Calculate player exposure across lineups"""
        player_counts = {}
        
        for lineup in lineups:
            for player in lineup['players']:
                name = player['name']
                player_counts[name] = player_counts.get(name, 0) + 1
        
        # Convert to percentages
        total_lineups = len(lineups)
        exposure = {
            name: count / total_lineups 
            for name, count in player_counts.items()
        }
        
        return exposure
    
    def _calculate_correlation_score(self, players: List[Dict], 
                                    correlation_matrix: Dict) -> float:
        """Calculate correlation score for a lineup"""
        total_correlation = 0
        pairs_checked = 0
        
        # Find QB
        qb = next((p for p in players if p['position'] == 'QB'), None)
        if not qb:
            return 0.0
        
        qb_team = qb['team']
        
        # Check each player pair
        for player in players:
            if player['name'] == qb['name']:
                continue
            
            # Same team stacking
            if player['team'] == qb_team:
                if player['position'] == 'WR':
                    total_correlation += correlation_matrix.get('QB-WR1', 0.5)
                    pairs_checked += 1
                elif player['position'] == 'TE':
                    total_correlation += correlation_matrix.get('QB-TE', 0.3)
                    pairs_checked += 1
        
        if pairs_checked > 0:
            return total_correlation / pairs_checked
        return 0.0


class MonteCarloSimulator:
    """Monte Carlo simulation for variance analysis"""
    
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
        
    def simulate_tournament(self, lineups: List[Dict], 
                          field_size: int = 1000,
                          payout_structure: Dict = None) -> Dict:
        """
        Simulate tournament outcomes
        
        Args:
            lineups: Your lineups to test
            field_size: Total tournament entrants
            payout_structure: Prize distribution
            
        Returns:
            Simulation results
        """
        if not payout_structure:
            payout_structure = self._get_default_payouts()
        
        results = {
            'roi': [],
            'min_cash_rate': 0,
            'top_10_rate': 0,
            'win_rate': 0
        }
        
        for sim in range(self.num_simulations):
            # Generate field scores
            field_scores = np.random.normal(150, 25, field_size)
            
            # Simulate your lineup scores with correlation
            your_scores = []
            for lineup in lineups:
                base_score = lineup['total_projection']
                # Add variance
                actual_score = np.random.normal(base_score, base_score * 0.15)
                your_scores.append(max(0, actual_score))
            
            # Combine and rank
            all_scores = list(field_scores) + your_scores
            all_scores.sort(reverse=True)
            
            # Calculate payouts
            total_payout = 0
            for score in your_scores:
                rank = all_scores.index(score) + 1
                payout = self._get_payout(rank, field_size, payout_structure)
                total_payout += payout
            
            # Track results
            entry_fees = len(lineups) * 5  # Assuming $5 entry
            roi = (total_payout - entry_fees) / entry_fees if entry_fees > 0 else 0
            results['roi'].append(roi)
            
            # Track placement rates
            for score in your_scores:
                rank = all_scores.index(score) + 1
                if rank <= field_size * 0.2:  # Top 20% cash
                    results['min_cash_rate'] += 1
                if rank <= 10:
                    results['top_10_rate'] += 1
                if rank == 1:
                    results['win_rate'] += 1
# Calculate final statistics
        total_entries = self.num_simulations * len(lineups)
        results['min_cash_rate'] /= total_entries
        results['top_10_rate'] /= total_entries
        results['win_rate'] /= total_entries
        results['avg_roi'] = np.mean(results['roi'])
        results['roi_std'] = np.std(results['roi'])
        results['sharpe_ratio'] = results['avg_roi'] / results['roi_std'] if results['roi_std'] > 0 else 0
        
        return results
    
    def _get_default_payouts(self) -> Dict:
        """Default GPP payout structure"""
        return {
            1: 1000,
            2: 500,
            3: 300,
            4: 200,
            5: 150,
            10: 100,
            20: 50,
            50: 25,
            100: 15,
            200: 10
        }
    
    def _get_payout(self, rank: int, field_size: int, payout_structure: Dict) -> float:
        """Calculate payout for a given rank"""
        for rank_threshold, payout in sorted(payout_structure.items()):
            if rank <= rank_threshold:
                return payout
        
        # Min cash line (typically top 20%)
        if rank <= field_size * 0.2:
            return 10  # 2x entry fee
        
        return 0
