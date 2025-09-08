import logging
import asyncio
import itertools
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime
import time

from app.config import settings
from app.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class OptimizationEngine:
    """Advanced DFS optimization engine with multiple algorithms and strategies"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.solver_available = self._check_solver_availability()
        
        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "average_time": 0.0,
            "cache_hits": 0
        }
    
    def _check_solver_availability(self) -> bool:
        """Check if PuLP linear programming solver is available"""
        try:
            import pulp
            return True
        except ImportError:
            logger.warning("PuLP not available, will use heuristic optimization")
            return False
    
    async def optimize_lineup(
        self,
        player_data: pd.DataFrame,
        game_type: str = "league",
        salary_cap: int = None,
        enforce_stack: bool = None,
        lock_players: List[str] = None,
        ban_players: List[str] = None,
        use_ai: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Main optimization entry point"""
        
        start_time = time.time()
        self.optimization_stats["total_optimizations"] += 1
        
        try:
            # Use settings defaults if not provided
            salary_cap = salary_cap or settings.salary_cap
            enforce_stack = enforce_stack if enforce_stack is not None else settings.enforce_qb_stack
            lock_players = lock_players or []
            ban_players = ban_players or []
            
            # Check cache first
            cache_key = self._generate_cache_key(
                player_data, game_type, salary_cap, enforce_stack, 
                lock_players, ban_players, use_ai
            )
            
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                self.optimization_stats["cache_hits"] += 1
                logger.info("Using cached optimization result")
                return cached_result
            
            # Prepare player data
            prepared_data = await self._prepare_player_data(
                player_data, game_type, lock_players, ban_players, use_ai
            )
            
            if prepared_data is None or prepared_data.empty:
                logger.error("No valid players after preparation")
                return None
            
            # Run optimization
            if self.solver_available:
                result = await self._optimize_with_linear_programming(
                    prepared_data, game_type, salary_cap, enforce_stack
                )
            else:
                result = await self._optimize_with_heuristic(
                    prepared_data, game_type, salary_cap, enforce_stack
                )
            
            if result:
                # Add metadata
                optimization_time = time.time() - start_time
                result["optimization_metadata"] = {
                    "method": "linear_programming" if self.solver_available else "heuristic",
                    "optimization_time": round(optimization_time, 2),
                    "game_type": game_type,
                    "salary_cap": salary_cap,
                    "enforce_stack": enforce_stack,
                    "ai_enhanced": use_ai,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Cache result
                await self.cache_manager.set(cache_key, result, ttl=settings.cache_ttl)
                
                # Update stats
                self.optimization_stats["successful_optimizations"] += 1
                self.optimization_stats["average_time"] = (
                    (self.optimization_stats["average_time"] * (self.optimization_stats["successful_optimizations"] - 1) + optimization_time) /
                    self.optimization_stats["successful_optimizations"]
                )
                
                logger.info(f"Optimization completed in {optimization_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return None
    
    async def _prepare_player_data(
        self,
        player_data: pd.DataFrame,
        game_type: str,
        lock_players: List[str],
        ban_players: List[str],
        use_ai: bool
    ) -> Optional[pd.DataFrame]:
        """Prepare and enhance player data for optimization"""
        
        try:
            df = player_data.copy()
            
            # Remove banned players
            if ban_players:
                ban_mask = df['PLAYER NAME'].isin(ban_players)
                df = df[~ban_mask]
                logger.info(f"Removed {ban_mask.sum()} banned players")
            
            # Mark locked players
            df['LOCKED'] = df['PLAYER NAME'].isin(lock_players)
            locked_count = df['LOCKED'].sum()
            if locked_count > 0:
                logger.info(f"Locked {locked_count} players")
            
            # Add game-type specific scoring
            strategy_weights = settings.get_strategy_weights(game_type)
            
            # Calculate enhanced scores
            df['CEILING_SCORE'] = df['CEILING'] if 'CEILING' in df.columns else df['PROJ PTS'] * 1.4
            df['FLOOR_SCORE'] = df['FLOOR'] if 'FLOOR' in df.columns else df['PROJ PTS'] * 0.6
            
            # Leverage score (inverse of ownership)
            if 'OWN_PCT' in df.columns:
                df['LEVERAGE_SCORE'] = df['PROJ PTS'] / (df['OWN_PCT'] + 1)
            else:
                df['LEVERAGE_SCORE'] = df['PROJ PTS']
            
            # Correlation score (position-specific multipliers)
            correlation_multipliers = {
                'QB': 1.2,
                'WR': 1.1,
                'TE': 1.05,
                'RB': 0.95,
                'DST': 0.9
            }
            df['CORRELATION_SCORE'] = df.apply(
                lambda row: row['PROJ PTS'] * correlation_multipliers.get(row['POS'], 1.0),
                axis=1
            )
            
            # Combined optimization score
            df['OPT_SCORE'] = (
                df['CEILING_SCORE'] * strategy_weights['ceiling_weight'] +
                df['FLOOR_SCORE'] * strategy_weights['floor_weight'] +
                df['LEVERAGE_SCORE'] * strategy_weights['leverage_weight'] +
                df['CORRELATION_SCORE'] * strategy_weights['correlation_weight']
            )
            
            # Filter by value threshold
            value_filter = df['VALUE'] >= settings.min_value_threshold
            df = df[value_filter]
            
            if df.empty:
                logger.error("No players meet value threshold")
                return None
            
            logger.info(f"Prepared {len(df)} players for optimization")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing player data: {e}")
            return None
    
    async def _optimize_with_linear_programming(
        self,
        player_data: pd.DataFrame,
        game_type: str,
        salary_cap: int,
        enforce_stack: bool
    ) -> Optional[Dict[str, Any]]:
        """Optimize using linear programming with PuLP"""
        
        try:
            import pulp
            
            # Create the problem
            prob = pulp.LpProblem("DFS_Optimization", pulp.LpMaximize)
            
            # Decision variables
            player_vars = {}
            for idx in player_data.index:
                player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
            
            # Objective function: maximize optimization score
            prob += pulp.lpSum([
                player_data.loc[idx, 'OPT_SCORE'] * player_vars[idx] 
                for idx in player_data.index
            ])
            
            # Constraints
            await self._add_basic_constraints(prob, player_data, player_vars, salary_cap)
            
            if enforce_stack:
                await self._add_stacking_constraints(prob, player_data, player_vars)
            
            # Lock constraints
            locked_players = player_data[player_data['LOCKED'] == True]
            for idx in locked_players.index:
                prob += player_vars[idx] == 1
            
            # Solve the problem
            prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=settings.max_optimization_time))
            
            if prob.status == 1:  # Optimal solution found
                lineup_indices = [idx for idx in player_data.index if player_vars[idx].varValue == 1]
                return await self._build_result(player_data, lineup_indices, "linear_programming")
            else:
                logger.warning(f"LP solver status: {prob.status}")
                return None
                
        except Exception as e:
            logger.error(f"Linear programming optimization failed: {e}")
            return None
    
    async def _optimize_with_heuristic(
        self,
        player_data: pd.DataFrame,
        game_type: str,
        salary_cap: int,
        enforce_stack: bool
    ) -> Optional[Dict[str, Any]]:
        """Optimize using advanced heuristic approach"""
        
        try:
            # Get locked players first
            locked_players = player_data[player_data['LOCKED'] == True]
            available_players = player_data[player_data['LOCKED'] == False]
            
            # Start with locked players
            lineup = {}
            used_salary = 0
            positions_filled = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'DST': 0}
            
            # Add locked players
            for _, player in locked_players.iterrows():
                pos = player['POS']
                if pos in positions_filled:
                    if pos == 'QB' and positions_filled[pos] < 1:
                        lineup[f'{pos}'] = player
                        positions_filled[pos] += 1
                        used_salary += player['SALARY']
                    elif pos in ['RB', 'WR'] and positions_filled[pos] < 3:
                        lineup[f'{pos}{positions_filled[pos] + 1}'] = player
                        positions_filled[pos] += 1
                        used_salary += player['SALARY']
                    elif pos == 'TE' and positions_filled[pos] < 2:
                        lineup[f'{pos}'] = player
                        positions_filled[pos] += 1
                        used_salary += player['SALARY']
                    elif pos == 'DST' and positions_filled[pos] < 1:
                        lineup[f'{pos}'] = player
                        positions_filled[pos] += 1
                        used_salary += player['SALARY']
            
            # Fill remaining positions using greedy approach
            remaining_budget = salary_cap - used_salary
            
            # Sort by optimization score
            available_players = available_players.sort_values('OPT_SCORE', ascending=False)
            
            # Fill required positions
            position_requirements = settings.position_requirements
            
            # QB (if not filled)
            if positions_filled['QB'] == 0:
                qb_candidates = available_players[available_players['POS'] == 'QB']
                for _, qb in qb_candidates.iterrows():
                    if qb['SALARY'] <= remaining_budget:
                        lineup['QB'] = qb
                        positions_filled['QB'] = 1
                        remaining_budget -= qb['SALARY']
                        break
            
            # Get QB for stacking
            qb = lineup.get('QB')
            
            # RBs
            rb_needed = max(0, position_requirements['RB']['min'] - positions_filled['RB'])
            if rb_needed > 0:
                rb_candidates = available_players[
                    (available_players['POS'] == 'RB') & 
                    (available_players['SALARY'] <= remaining_budget)
                ]
                
                added_rbs = 0
                for _, rb in rb_candidates.iterrows():
                    if added_rbs >= rb_needed:
                        break
                    if rb['SALARY'] <= remaining_budget:
                        lineup[f'RB{positions_filled["RB"] + 1}'] = rb
                        positions_filled['RB'] += 1
                        remaining_budget -= rb['SALARY']
                        added_rbs += 1
            
            # WRs (with stacking consideration)
            wr_needed = max(0, position_requirements['WR']['min'] - positions_filled['WR'])
            if wr_needed > 0:
                wr_candidates = available_players[
                    (available_players['POS'] == 'WR') & 
                    (available_players['SALARY'] <= remaining_budget)
                ]
                
                # Prioritize same team as QB for stacking
                if enforce_stack and qb is not None:
                    stack_wrs = wr_candidates[wr_candidates['TEAM'] == qb['TEAM']]
                    if not stack_wrs.empty:
                        # Add at least one stack mate
                        stack_wr = stack_wrs.iloc[0]
                        if stack_wr['SALARY'] <= remaining_budget:
                            lineup[f'WR{positions_filled["WR"] + 1}'] = stack_wr
                            positions_filled['WR'] += 1
                            remaining_budget -= stack_wr['SALARY']
                            wr_needed -= 1
                            # Remove from candidates
                            wr_candidates = wr_candidates[wr_candidates.index != stack_wr.name]
                
                # Fill remaining WR spots
                added_wrs = 0
                for _, wr in wr_candidates.iterrows():
                    if added_wrs >= wr_needed:
                        break
                    if wr['SALARY'] <= remaining_budget:
                        lineup[f'WR{positions_filled["WR"] + 1}'] = wr
                        positions_filled['WR'] += 1
                        remaining_budget -= wr['SALARY']
                        added_wrs += 1
            
            # TE
            te_needed = max(0, position_requirements['TE']['min'] - positions_filled['TE'])
            if te_needed > 0:
                te_candidates = available_players[
                    (available_players['POS'] == 'TE') & 
                    (available_players['SALARY'] <= remaining_budget)
                ]
                
                # Consider stacking with QB
                if enforce_stack and qb is not None:
                    stack_tes = te_candidates[te_candidates['TEAM'] == qb['TEAM']]
                    if not stack_tes.empty:
                        stack_te = stack_tes.iloc[0]
                        if stack_te['SALARY'] <= remaining_budget:
                            lineup['TE'] = stack_te
                            positions_filled['TE'] = 1
                            remaining_budget -= stack_te['SALARY']
                            te_needed = 0
                
                if te_needed > 0:
                    for _, te in te_candidates.iterrows():
                        if te['SALARY'] <= remaining_budget:
                            lineup['TE'] = te
                            positions_filled['TE'] = 1
                            remaining_budget -= te['SALARY']
                            break
            
            # DST (avoid opposing QB's team)
            if positions_filled['DST'] == 0:
                dst_candidates = available_players[available_players['POS'] == 'DST']
                
                # Filter out DST opposing QB
                if qb is not None and 'OPP' in qb and pd.notna(qb['OPP']):
                    dst_candidates = dst_candidates[dst_candidates['TEAM'] != qb['OPP']]
                
                for _, dst in dst_candidates.iterrows():
                    if dst['SALARY'] <= remaining_budget:
                        lineup['DST'] = dst
                        positions_filled['DST'] = 1
                        remaining_budget -= dst['SALARY']
                        break
            
            # Fill FLEX position
            total_positions = sum(positions_filled.values())
            if total_positions < 9:
                flex_candidates = available_players[
                    (available_players['POS'].isin(['RB', 'WR', 'TE'])) & 
                    (available_players['SALARY'] <= remaining_budget)
                ]
                
                for _, flex in flex_candidates.iterrows():
                    if flex['SALARY'] <= remaining_budget:
                        lineup['FLEX'] = flex
                        break
            
            # Validate lineup
            if len(lineup) == 9:
                lineup_indices = [player.name for player in lineup.values()]
                return await self._build_result(player_data, lineup_indices, "heuristic")
            else:
                logger.warning(f"Incomplete lineup: {len(lineup)} players")
                return None
                
        except Exception as e:
            logger.error(f"Heuristic optimization failed: {e}")
            return None
    
    async def _add_basic_constraints(self, prob, player_data, player_vars, salary_cap):
        """Add basic optimization constraints"""
        import pulp
        
        # Salary constraint
        prob += pulp.lpSum([
            player_data.loc[idx, 'SALARY'] * player_vars[idx] 
            for idx in player_data.index
        ]) <= salary_cap
        
        # Total players = 9
        prob += pulp.lpSum([player_vars[idx] for idx in player_data.index]) == 9
        
        # Position constraints
        position_requirements = settings.position_requirements
        
        # QB: exactly 1
        qb_indices = player_data[player_data['POS'] == 'QB'].index
        prob += pulp.lpSum([player_vars[idx] for idx in qb_indices]) == 1
        
        # RB: 2-3
        rb_indices = player_data[player_data['POS'] == 'RB'].index
        prob += pulp.lpSum([player_vars[idx] for idx in rb_indices]) >= 2
        prob += pulp.lpSum([player_vars[idx] for idx in rb_indices]) <= 3
        
        # WR: 3-4
        wr_indices = player_data[player_data['POS'] == 'WR'].index
        prob += pulp.lpSum([player_vars[idx] for idx in wr_indices]) >= 3
        prob += pulp.lpSum([player_vars[idx] for idx in wr_indices]) <= 4
        
        # TE: 1-2
        te_indices = player_data[player_data['POS'] == 'TE'].index
        prob += pulp.lpSum([player_vars[idx] for idx in te_indices]) >= 1
        prob += pulp.lpSum([player_vars[idx] for idx in te_indices]) <= 2
        
        # DST: exactly 1
        dst_indices = player_data[player_data['POS'] == 'DST'].index
        prob += pulp.lpSum([player_vars[idx] for idx in dst_indices]) == 1
        
        # FLEX constraint (RB + WR + TE = 7)
        flex_indices = player_data[player_data['POS'].isin(['RB', 'WR', 'TE'])].index
        prob += pulp.lpSum([player_vars[idx] for idx in flex_indices]) == 7
    
    async def _add_stacking_constraints(self, prob, player_data, player_vars):
        """Add stacking constraints"""
        import pulp
        
        # QB stacking: if QB selected, must have at least 1 receiver from same team
        qb_indices = player_data[player_data['POS'] == 'QB'].index
        
        for qb_idx in qb_indices:
            qb_team = player_data.loc[qb_idx, 'TEAM']
            
            # Find receivers from same team
            stack_receivers = player_data[
                (player_data['TEAM'] == qb_team) & 
                (player_data['POS'].isin(['WR', 'TE'])) & 
                (player_data.index != qb_idx)
            ].index
            
            if len(stack_receivers) > 0:
                # If this QB is selected, must have at least 1 receiver from same team
                prob += pulp.lpSum([player_vars[idx] for idx in stack_receivers]) >= \
                        settings.min_stack_receivers * player_vars[qb_idx]
        
        # Team exposure limits
        for team in player_data['TEAM'].unique():
            team_players = player_data[player_data['TEAM'] == team].index
            if len(team_players) > settings.max_team_exposure:
                prob += pulp.lpSum([player_vars[idx] for idx in team_players]) <= settings.max_team_exposure
    
    async def _build_result(self, player_data: pd.DataFrame, lineup_indices: List, method: str) -> Dict[str, Any]:
        """Build the optimization result dictionary"""
        
        try:
            lineup_players = []
            total_salary = 0
            total_projection = 0.0
            total_ownership = 0.0
            
            for idx in lineup_indices:
                player = player_data.loc[idx]
                
                player_dict = {
                    'player_name': player['PLAYER NAME'],
                    'position': player['POS'],
                    'team': player['TEAM'],
                    'opponent': player.get('OPP', ''),
                    'salary': int(player['SALARY']),
                    'projection': float(player['PROJ PTS']),
                    'value': float(player['VALUE']),
                    'ownership': float(player.get('OWN_PCT', 0.0)),
                    'ceiling': float(player.get('CEILING_SCORE', player['PROJ PTS'] * 1.4)),
                    'floor': float(player.get('FLOOR_SCORE', player['PROJ PTS'] * 0.6))
                }
                
                lineup_players.append(player_dict)
                total_salary += player_dict['salary']
                total_projection += player_dict['projection']
                total_ownership += player_dict['ownership']
            
            # Calculate lineup statistics
            avg_ownership = total_ownership / len(lineup_players) if lineup_players else 0
            
            # Analyze stacking
            stack_analysis = await self._analyze_stacking(lineup_players)
            
            # Monte Carlo simulation
            simulation_results = await self._run_monte_carlo_simulation(lineup_players)
            
            result = {
                "success": True,
                "lineup": lineup_players,
                "summary": {
                    "total_salary": total_salary,
                    "salary_remaining": settings.salary_cap - total_salary,
                    "total_projection": round(total_projection, 2),
                    "average_projection": round(total_projection / len(lineup_players), 2),
                    "average_ownership": round(avg_ownership, 2),
                    "total_ceiling": round(sum(p['ceiling'] for p in lineup_players), 2),
                    "total_floor": round(sum(p['floor'] for p in lineup_players), 2)
                },
                "stack_analysis": stack_analysis,
                "simulation_results": simulation_results,
                "optimization_method": method
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error building result: {e}")
            return None
    
    async def _analyze_stacking(self, lineup_players: List[Dict]) -> Dict[str, Any]:
        """Analyze stacking in the lineup"""
        
        try:
            qb = next((p for p in lineup_players if p['position'] == 'QB'), None)
            if not qb:
                return {"has_stack": False, "stack_count": 0}
            
            qb_team = qb['team']
            stack_mates = [
                p for p in lineup_players 
                if p['team'] == qb_team and p['position'] in ['WR', 'TE'] and p != qb
            ]
            
            return {
                "has_stack": len(stack_mates) > 0,
                "stack_count": len(stack_mates),
                "qb_name": qb['player_name'],
                "qb_team": qb_team,
                "stack_players": [p['player_name'] for p in stack_mates],
                "stack_projection": round(qb['projection'] + sum(p['projection'] for p in stack_mates), 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stacking: {e}")
            return {"has_stack": False, "error": str(e)}
    
    async def _run_monte_carlo_simulation(self, lineup_players: List[Dict], iterations: int = 10000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for variance analysis"""
        
        try:
            scores = []
            
            for _ in range(iterations):
                lineup_score = 0
                for player in lineup_players:
                    # Use normal distribution with mean = projection, std = 25% of projection
                    std_dev = player['projection'] * 0.25
                    simulated_score = max(0, np.random.normal(player['projection'], std_dev))
                    lineup_score += simulated_score
                
                scores.append(lineup_score)
            
            scores = np.array(scores)
            
            return {
                "mean_score": round(float(np.mean(scores)), 2),
                "std_dev": round(float(np.std(scores)), 2),
                "percentiles": {
                    "10th": round(float(np.percentile(scores, 10)), 2),
                    "25th": round(float(np.percentile(scores, 25)), 2),
                    "50th": round(float(np.percentile(scores, 50)), 2),
                    "75th": round(float(np.percentile(scores, 75)), 2),
                    "90th": round(float(np.percentile(scores, 90)), 2),
                    "95th": round(float(np.percentile(scores, 95)), 2)
                },
                "sharpe_ratio": round(float(np.mean(scores) / np.std(scores)), 3) if np.std(scores) > 0 else 0,
                "iterations": iterations
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {"error": str(e)}
    
    def _generate_cache_key(self, player_data, game_type, salary_cap, enforce_stack, lock_players, ban_players, use_ai) -> str:
        """Generate cache key for optimization results"""
        
        # Create a hash of the key parameters
        key_components = [
            f"game_type:{game_type}",
            f"salary_cap:{salary_cap}",
            f"enforce_stack:{enforce_stack}",
            f"lock_players:{sorted(lock_players)}",
            f"ban_players:{sorted(ban_players)}",
            f"use_ai:{use_ai}",
            f"data_hash:{hash(tuple(player_data['PLAYER NAME'].tolist()))}",
            f"hour:{datetime.now().strftime('%Y-%m-%d-%H')}"  # Refresh every hour
        ]
        
        cache_key = "optimization:" + "|".join(key_components)
        return str(hash(cache_key))
    
    def health_check(self) -> str:
        """Check health of optimization engine"""
        try:
            if self.solver_available:
                return "healthy_lp"
            else:
                return "healthy_heuristic"
        except Exception as e:
            return f"error: {e}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization engine statistics"""
        return self.optimization_stats.copy()
