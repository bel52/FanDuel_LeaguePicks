import pulp
import numpy as np
from typing import List, Dict, Optional
from scipy import stats
from itertools import combinations
import logging
from models import Player, Lineup, OptimizationSettings
from config import config

logger = logging.getLogger(__name__)

class DFSOptimizer:
    def __init__(self):
        self.position_limits = {
            'QB': {'min': 1, 'max': 1},
            'RB': {'min': 2, 'max': 3},
            'WR': {'min': 3, 'max': 4},
            'TE': {'min': 1, 'max': 2},
            'DST': {'min': 1, 'max': 1}
        }
        self.correlations = config.CORRELATIONS
        
    def optimize(self, players: List[Player], settings: OptimizationSettings) -> List[Lineup]:
        """Main optimization function"""
        logger.info(f"Optimizing {settings.num_lineups} lineups...")
        
        lineups = []
        used_players = set()
        
        for i in range(settings.num_lineups):
            # Apply exposure limits
            available_players = self._apply_exposure_limits(players, lineups, settings.max_exposure)
            
            # Generate lineup based on type
            if settings.lineup_type == "cash":
                lineup = self._optimize_cash_lineup(available_players, settings)
            elif settings.lineup_type == "gpp":
                lineup = self._optimize_gpp_lineup(available_players, settings, used_players)
            else:
                lineup = self._optimize_balanced_lineup(available_players, settings)
            
            if lineup:
                lineups.append(lineup)
                # Track used players for diversity
                for player in lineup.players:
                    used_players.add(player.id)
        
        return lineups
    
    def _optimize_cash_lineup(self, players: List[Player], settings: OptimizationSettings) -> Optional[Lineup]:
        """Optimize for cash games (50/50s, H2H) - focus on floor"""
        prob = pulp.LpProblem("DFS_Cash_Optimization", pulp.LpMaximize)
        
        # Decision variables
        player_vars = {}
        for i, player in enumerate(players):
            player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
        
        # Objective: maximize floor (use projected points if floor not available)
        prob += pulp.lpSum([
            (players[i].floor if players[i].floor else players[i].projected_points * 0.85) * player_vars[i]
            for i in range(len(players))
        ])
        
        # Add constraints
        prob = self._add_base_constraints(prob, players, player_vars, settings)
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            return self._extract_lineup(prob, players, player_vars)
        return None
    
    def _optimize_gpp_lineup(self, players: List[Player], settings: OptimizationSettings, 
                            used_players: set) -> Optional[Lineup]:
        """Optimize for GPPs (tournaments) - focus on ceiling and differentiation"""
        prob = pulp.LpProblem("DFS_GPP_Optimization", pulp.LpMaximize)
        
        player_vars = {}
        for i, player in enumerate(players):
            player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
        
        # Objective: maximize ceiling with ownership leverage
        prob += pulp.lpSum([
            self._calculate_gpp_score(players[i], used_players) * player_vars[i]
            for i in range(len(players))
        ])
        
        # Add constraints
        prob = self._add_base_constraints(prob, players, player_vars, settings)
        
        # Add stacking constraints for GPP
        if settings.stack_rules:
            prob = self._add_stacking_constraints(prob, players, player_vars, settings.stack_rules)
        
        # Add correlation constraints
        if settings.correlation_rules:
            prob = self._add_correlation_constraints(prob, players, player_vars)
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            return self._extract_lineup(prob, players, player_vars)
        return None
    
    def _optimize_balanced_lineup(self, players: List[Player], settings: OptimizationSettings) -> Optional[Lineup]:
        """Balanced approach for mixed contests"""
        prob = pulp.LpProblem("DFS_Balanced_Optimization", pulp.LpMaximize)
        
        player_vars = {}
        for i, player in enumerate(players):
            player_vars[i] = pulp.LpVariable(f"player_{i}", cat='Binary')
        
        # Objective: balance between floor and ceiling
        prob += pulp.lpSum([
            (players[i].projected_points * 0.7 + 
             (players[i].ceiling if players[i].ceiling else players[i].projected_points * 1.15) * 0.3) * player_vars[i]
            for i in range(len(players))
        ])
        
        prob = self._add_base_constraints(prob, players, player_vars, settings)
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if prob.status == pulp.LpStatusOptimal:
            return self._extract_lineup(prob, players, player_vars)
        return None
    
    def _add_base_constraints(self, prob, players: List[Player], player_vars: dict, 
                             settings: OptimizationSettings):
        """Add basic constraints (salary, positions)"""
        # Salary constraint
        prob += pulp.lpSum([
            players[i].salary * player_vars[i] for i in range(len(players))
        ]) <= config.SALARY_CAP
        
        # Minimum salary constraint
        prob += pulp.lpSum([
            players[i].salary * player_vars[i] for i in range(len(players))
        ]) >= settings.min_salary
        
        # Position constraints
        for position, limits in self.position_limits.items():
            position_players = [i for i, p in enumerate(players) if p.position.value == position]
            
            if position == 'RB' or position == 'WR' or position == 'TE':
                # Flex position handling
                flex_eligible = [i for i, p in enumerate(players) 
                               if p.position.value in ['RB', 'WR', 'TE']]
                
                # Ensure minimum positions filled
                prob += pulp.lpSum([player_vars[i] for i in position_players]) >= limits['min']
                
                # Total flex-eligible players equals required spots
                if position == 'RB':
                    total_rb_wr_te = 7  # 2 RB + 3 WR + 1 TE + 1 FLEX = 7
                    prob += pulp.lpSum([player_vars[i] for i in flex_eligible]) == total_rb_wr_te
            else:
                # Fixed positions (QB, DST)
                prob += pulp.lpSum([player_vars[i] for i in position_players]) == limits['min']
        
        # Exactly 9 players
        prob += pulp.lpSum([player_vars[i] for i in range(len(players))]) == 9
        
        # No duplicate teams for DST
        dst_players = [i for i, p in enumerate(players) if p.position.value == 'DST']
        for team in set(p.team for p in players if p.position.value != 'DST'):
            team_players = [i for i, p in enumerate(players) 
                          if p.team == team and p.position.value != 'DST']
            team_dst = [i for i, p in enumerate(players) 
                       if p.team == team and p.position.value == 'DST']
            
            if team_players and team_dst:
                # If DST is selected, no players from that team
                for dst_idx in team_dst:
                    for player_idx in team_players:
                        prob += player_vars[dst_idx] + player_vars[player_idx] <= 1
        
        return prob
    
    def _add_stacking_constraints(self, prob, players: List[Player], player_vars: dict, 
                                 stack_rules: dict):
        """Add stacking constraints for correlation plays"""
        if stack_rules.get('qb_stack'):
            # QB must have at least one pass catcher from same team
            qbs = [i for i, p in enumerate(players) if p.position.value == 'QB']
            
            for qb_idx in qbs:
                qb_team = players[qb_idx].team
                team_catchers = [i for i, p in enumerate(players) 
                               if p.team == qb_team and p.position.value in ['WR', 'TE']]
                
                if team_catchers:
                    # If QB selected, at least 1 catcher from same team
                    prob += pulp.lpSum([player_vars[i] for i in team_catchers]) >= player_vars[qb_idx]
        
        if stack_rules.get('game_stack'):
            # Bring-back correlation from opposing team
            for qb_idx in [i for i, p in enumerate(players) if p.position.value == 'QB']:
                qb = players[qb_idx]
                opp_players = [i for i, p in enumerate(players)
                             if p.team == qb.opponent and p.position.value in ['WR', 'TE', 'RB']]
                
                if opp_players:
                    # If QB selected, consider opponent correlation
                    prob += pulp.lpSum([player_vars[i] for i in opp_players]) >= player_vars[qb_idx] * 0.5
        
        return prob
    
    def _add_correlation_constraints(self, prob, players: List[Player], player_vars: dict):
        """Add correlation-based constraints"""
        # Negative correlation: QB-DST
        for qb_idx in [i for i, p in enumerate(players) if p.position.value == 'QB']:
            qb = players[qb_idx]
            opp_dst = [i for i, p in enumerate(players)
                      if p.position.value == 'DST' and p.team == qb.opponent]
            
            # Avoid QB facing opposing DST
            for dst_idx in opp_dst:
                prob += player_vars[qb_idx] + player_vars[dst_idx] <= 1
        
        return prob
    
    def _calculate_gpp_score(self, player: Player, used_players: set) -> float:
        """Calculate GPP-optimized score with ownership leverage"""
        base_score = player.ceiling if player.ceiling else player.projected_points * 1.2
        
        # Ownership leverage
        if player.ownership_projection:
            if player.ownership_projection < 10:  # Low owned
                base_score *= 1.15
            elif player.ownership_projection > 30:  # High owned
                base_score *= 0.9
        
        # Differentiation bonus
        if player.id not in used_players:
            base_score *= 1.05
        
        # Weather adjustment
        base_score *= player.weather_impact
        
        return base_score
    
    def _apply_exposure_limits(self, players: List[Player], existing_lineups: List[Lineup], 
                              max_exposure: float) -> List[Player]:
        """Filter players based on exposure limits"""
        if not existing_lineups:
            return players
        
        player_counts = {}
        for lineup in existing_lineups:
            for player in lineup.players:
                player_counts[player.id] = player_counts.get(player.id, 0) + 1
        
        max_count = int(len(existing_lineups) * max_exposure)
        
        return [p for p in players if player_counts.get(p.id, 0) < max_count]
    
    def _extract_lineup(self, prob, players: List[Player], player_vars: dict) -> Lineup:
        """Extract lineup from solved problem"""
        selected_players = []
        for i in range(len(players)):
            if player_vars[i].varValue == 1:
                selected_players.append(players[i])
        
        total_salary = sum(p.salary for p in selected_players)
        total_projected = sum(p.projected_points for p in selected_players)
        
        return Lineup(
            players=selected_players,
            total_salary=total_salary,
            total_projected=total_projected,
            stack_score=self._calculate_stack_score(selected_players),
            ownership_sum=sum(p.ownership_projection or 20 for p in selected_players)
        )
    
    def _calculate_stack_score(self, players: List[Player]) -> float:
        """Calculate correlation score for lineup"""
        score = 0.0
        
        # Check for QB stacks
        qbs = [p for p in players if p.position.value == 'QB']
        if qbs:
            qb = qbs[0]
            # Same team receivers
            team_catchers = [p for p in players 
                           if p.team == qb.team and p.position.value in ['WR', 'TE']]
            score += len(team_catchers) * 10
            
            # Game stack (opposing players)
            opp_players = [p for p in players if p.team == qb.opponent]
            score += len(opp_players) * 5
        
        return score
    
    def run_monte_carlo_simulation(self, lineup: Lineup, iterations: int = 10000) -> Dict:
        """Run Monte Carlo simulation for lineup variance analysis"""
        scores = []
        
        for _ in range(iterations):
            sim_score = 0
            for player in lineup.players:
                # Use normal distribution with historical variance
                std_dev = player.projected_points * 0.3  # 30% standard deviation
                if player.position.value == 'DST':
                    std_dev *= 1.5  # Higher variance for DST
                
                player_score = np.random.normal(player.projected_points, std_dev)
                player_score = max(0, player_score)  # Floor at 0
                sim_score += player_score
            
            scores.append(sim_score)
        
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std_dev': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'percentiles': {
                '25th': np.percentile(scores, 25),
                '50th': np.percentile(scores, 50),
                '75th': np.percentile(scores, 75),
                '90th': np.percentile(scores, 90),
                '95th': np.percentile(scores, 95)
            },
            'probability_150plus': sum(1 for s in scores if s >= 150) / iterations,
            'probability_175plus': sum(1 for s in scores if s >= 175) / iterations,
            'probability_200plus': sum(1 for s in scores if s >= 200) / iterations
        }

optimizer = DFSOptimizer()
