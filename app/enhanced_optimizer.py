import itertools
import logging
import asyncio
from typing import List, Optional, Set, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.ai_integration import AIAnalyzer
from app.data_monitor import RealTimeDataMonitor
from app.cache_manager import CacheManager

logger = logging.getLogger(__name__)

class EnhancedDFSOptimizer:
    """AI-powered DFS optimizer with real-time data integration"""
    
    def __init__(self):
        self.ai_analyzer = AIAnalyzer()
        self.data_monitor = RealTimeDataMonitor()
        self.cache_manager = CacheManager()
        
        # Strategy weights for different game types
        self.strategy_weights = {
            'h2h': {
                'ceiling_weight': 0.7,
                'floor_weight': 0.3,
                'leverage_weight': 0.4,
                'correlation_weight': 0.6
            },
            'league': {
                'ceiling_weight': 0.5,
                'floor_weight': 0.5,
                'leverage_weight': 0.3,
                'correlation_weight': 0.7
            }
        }
    
    async def optimize_lineup(
        self,
        df: pd.DataFrame,
        game_type: str = "league",  # "league" or "h2h"
        salary_cap: int = 60000,
        enforce_stack: bool = True,
        min_stack_receivers: int = 1,
        lock_indices: Optional[List[int]] = None,
        ban_indices: Optional[List[int]] = None,
        auto_swap_enabled: bool = True
    ) -> Tuple[List[int], Dict[str, Any]]:
        """
        Enhanced lineup optimization with AI integration
        """
        
        lock_indices = set(lock_indices or [])
        ban_indices = set(ban_indices or [])
        
        # Get real-time updates and apply them
        updated_df = await self._apply_real_time_updates(df)
        
        # Remove banned players
        use_df = updated_df[~updated_df.index.isin(ban_indices)].copy()
        
        # Enhanced player scoring with AI insights
        use_df = await self._enhance_player_scoring(use_df, game_type)
        
        # Try advanced optimization strategies
        lineup, metadata = await self._optimize_with_strategy(
            use_df, game_type, salary_cap, enforce_stack,
            min_stack_receivers, lock_indices
        )
        
        if not lineup:
            logger.warning("Advanced optimization failed, falling back to basic")
            lineup = self._fallback_optimization(
                use_df, salary_cap, enforce_stack, 
                min_stack_receivers, lock_indices
            )
            metadata = {"method": "fallback", "ai_enhanced": False}
        
        # Generate AI analysis of the lineup
        if lineup:
            lineup_analysis = await self._analyze_final_lineup(
                lineup, use_df, game_type, metadata
            )
            metadata.update(lineup_analysis)
        
        return lineup, metadata
    
    async def _apply_real_time_updates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply real-time player updates to projections"""
        updated_df = df.copy()
        
        try:
            # Get recent updates from the last 4 hours
            recent_updates = await self.data_monitor.get_recent_updates(hours=4)
            
            for update in recent_updates:
                player_name = update['player_name']
                update_type = update['update_type']
                severity = update['severity']
                
                # Find matching players in dataframe
                player_mask = updated_df['PLAYER NAME'].str.contains(
                    player_name, case=False, na=False
                )
                
                if player_mask.any():
                    # Apply update based on type and severity
                    adjustment_factor = self._calculate_adjustment_factor(
                        update_type, severity
                    )
                    
                    # Update projections
                    updated_df.loc[player_mask, 'PROJ PTS'] *= adjustment_factor
                    
                    # For injuries, also adjust ownership
                    if update_type == 'injury' and severity > 0.5:
                        if 'OWN_PCT' in updated_df.columns:
                            updated_df.loc[player_mask, 'OWN_PCT'] *= 0.7  # Reduce ownership
                    
                    logger.info(f"Applied update to {player_name}: {adjustment_factor:.3f}x projection")
        
        except Exception as e:
            logger.error(f"Error applying real-time updates: {e}")
        
        return updated_df
    
    async def _enhance_player_scoring(self, df: pd.DataFrame, game_type: str) -> pd.DataFrame:
        """Enhance player scoring with AI-driven insights"""
        enhanced_df = df.copy()
        
        # Base value calculation
        enhanced_df['value'] = enhanced_df['PROJ PTS'] / (enhanced_df['SALARY'] / 1000)
        enhanced_df['base_score'] = enhanced_df['PROJ PTS']
        
        # Strategy-specific adjustments
        weights = self.strategy_weights[game_type]
        
        # Ceiling score (90th percentile potential)
        enhanced_df['ceiling_score'] = enhanced_df['PROJ PTS'] * 1.3  # Simplified ceiling calc
        
        # Floor score (10th percentile potential)  
        enhanced_df['floor_score'] = enhanced_df['PROJ PTS'] * 0.7  # Simplified floor calc
        
        # Leverage score (based on ownership)
        if 'OWN_PCT' in enhanced_df.columns:
            enhanced_df['leverage_score'] = enhanced_df['PROJ PTS'] / (enhanced_df['OWN_PCT'] + 1)
        else:
            enhanced_df['leverage_score'] = enhanced_df['PROJ PTS']
        
        # Correlation potential (position-specific)
        enhanced_df['correlation_score'] = enhanced_df.apply(
            lambda row: self._calculate_correlation_score(row), axis=1
        )
        
        # Combined AI-enhanced score
        enhanced_df['ai_score'] = (
            enhanced_df['ceiling_score'] * weights['ceiling_weight'] +
            enhanced_df['floor_score'] * weights['floor_weight'] +
            enhanced_df['leverage_score'] * weights['leverage_weight'] +
            enhanced_df['correlation_score'] * weights['correlation_weight']
        )
        
        return enhanced_df
    
    async def _optimize_with_strategy(
        self,
        df: pd.DataFrame,
        game_type: str,
        salary_cap: int,
        enforce_stack: bool,
        min_stack_receivers: int,
        lock_indices: Set[int]
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Advanced optimization using AI-enhanced scoring"""
        
        # Try linear programming first
        try:
            import pulp
            return await self._optimize_with_lp_enhanced(
                df, salary_cap, enforce_stack, min_stack_receivers, 
                lock_indices, game_type
            )
        except ImportError:
            logger.info("PuLP not available, using enhanced heuristic")
            return await self._optimize_heuristic_enhanced(
                df, salary_cap, enforce_stack, min_stack_receivers, 
                lock_indices, game_type
            )
    
    async def _optimize_with_lp_enhanced(
        self,
        df: pd.DataFrame,
        salary_cap: int,
        enforce_stack: bool,
        min_stack_receivers: int,
        lock_indices: Set[int],
        game_type: str
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Enhanced linear programming optimization"""
        import pulp
        
        # Create problem
        prob = pulp.LpProblem("Enhanced_DFS_Optimization", pulp.LpMaximize)
        
        # Decision variables
        player_vars = {}
        for idx in df.index:
            player_vars[idx] = pulp.LpVariable(f"player_{idx}", cat='Binary')
        
        # Objective: maximize AI-enhanced score
        prob += pulp.lpSum([
            df.loc[idx, 'ai_score'] * player_vars[idx] 
            for idx in df.index
        ])
        
        # Standard constraints
        # Salary constraint
        prob += pulp.lpSum([
            df.loc[idx, 'SALARY'] * player_vars[idx] 
            for idx in df.index
        ]) <= salary_cap
        
        # Position constraints
        position_limits = {
            'QB': (1, 1),
            'RB': (2, 3),
            'WR': (3, 4),
            'TE': (1, 2),
            'DST': (1, 1)
        }
        
        for pos, (min_count, max_count) in position_limits.items():
            pos_players = df[df['POS'] == pos].index
            if pos in ['QB', 'DST']:
                prob += pulp.lpSum([player_vars[idx] for idx in pos_players]) == min_count
            else:
                prob += pulp.lpSum([player_vars[idx] for idx in pos_players]) >= min_count
                prob += pulp.lpSum([player_vars[idx] for idx in pos_players]) <= max_count
        
        # Total players constraint
        prob += pulp.lpSum([player_vars[idx] for idx in df.index]) == 9
        
        # FLEX constraint
        flex_positions = df[df['POS'].isin(['RB', 'WR', 'TE'])].index
        prob += pulp.lpSum([player_vars[idx] for idx in flex_positions]) == 7
        
        # Lock constraints
        for idx in lock_indices:
            if idx in player_vars:
                prob += player_vars[idx] == 1
        
        # Enhanced stacking constraints
        if enforce_stack:
            for qb_idx in df[df['POS'] == 'QB'].index:
                qb_team = df.loc[qb_idx, 'TEAM']
                teammates = df[(df['TEAM'] == qb_team) & 
                              (df['POS'].isin(['WR', 'TE']))].index
                
                # If QB is selected, must have at least min_stack_receivers teammates
                prob += pulp.lpSum([player_vars[tm_idx] for tm_idx in teammates]) >= \
                       min_stack_receivers * player_vars[qb_idx]
                
                # Enhanced: Limit total teammates for balanced exposure
                all_teammates = df[(df['TEAM'] == qb_team) & (df['POS'] != 'QB')].index
                prob += pulp.lpSum([player_vars[tm_idx] for tm_idx in all_teammates]) <= \
                       3 * player_vars[qb_idx]  # Max 3 teammates per QB
        
        # Game-type specific constraints
        if game_type == "h2h":
            # For head-to-head, prefer higher ceiling players
            high_ceiling_players = df[df['ceiling_score'] > df['ceiling_score'].quantile(0.7)].index
            prob += pulp.lpSum([player_vars[idx] for idx in high_ceiling_players]) >= 4
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if pulp.LpStatus[prob.status] == 'Optimal':
            lineup = []
            for idx in df.index:
                if player_vars[idx].varValue == 1:
                    lineup.append(idx)
            
            metadata = {
                "method": "linear_programming_enhanced",
                "ai_enhanced": True,
                "objective_value": pulp.value(prob.objective),
                "game_type": game_type
            }
            
            return lineup, metadata
        else:
            logger.warning("Enhanced LP optimization failed")
            return [], {}
    
    async def _optimize_heuristic_enhanced(
        self,
        df: pd.DataFrame,
        salary_cap: int,
        enforce_stack: bool,
        min_stack_receivers: int,
        lock_indices: Set[int],
        game_type: str
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Enhanced heuristic optimization with AI scoring"""
        
        # Sort by position and AI-enhanced score
        def get_top_players(pos: str, n: int, score_col: str = 'ai_score'):
            pos_df = df[df['POS'] == pos]
            if game_type == "h2h":
                # For H2H, weight ceiling more heavily
                combined_score = pos_df['ceiling_score'] * 0.6 + pos_df[score_col] * 0.4
                pos_df = pos_df.assign(combined_score=combined_score)
                return pos_df.nlargest(n, 'combined_score')
            else:
                return pos_df.nlargest(n, score_col)
        
        qbs = get_top_players('QB', 8)
        rbs = get_top_players('RB', 15)
        wrs = get_top_players('WR', 20)
        tes = get_top_players('TE', 12)
        dsts = get_top_players('DST', 8)
        
        best_lineup = []
        best_score = -1
        attempts = 0
        max_attempts = 1000
        
        # Enhanced search with AI-guided exploration
        for qb_idx in qbs.index:
            if attempts >= max_attempts:
                break
            
            qb = df.loc[qb_idx]
            qb_team = qb['TEAM']
            
            # Get stack mates with AI-enhanced correlation scoring
            stack_mates = df[(df['TEAM'] == qb_team) & 
                            (df['POS'].isin(['WR', 'TE']))].copy()
            
            if len(stack_mates) == 0:
                continue
            
            # Sort stack mates by correlation potential
            stack_mates['stack_synergy'] = stack_mates.apply(
                lambda row: self._calculate_stack_synergy(qb, row), axis=1
            )
            stack_mates = stack_mates.nlargest(6, 'stack_synergy')
            
            if enforce_stack and len(stack_mates) < min_stack_receivers:
                continue
            
            # Try different stack combinations
            for stack_size in range(min_stack_receivers, min(len(stack_mates) + 1, 4)):
                for stack_combo in itertools.combinations(stack_mates.index, stack_size):
                    attempts += 1
                    
                    used_indices = {qb_idx} | set(stack_combo)
                    remaining_salary = salary_cap - qb['SALARY'] - \
                                     sum(df.loc[idx, 'SALARY'] for idx in stack_combo)
                    
                    if remaining_salary < 20000:  # Not enough for remaining players
                        continue
                    
                    # Build rest of lineup with AI-enhanced selection
                    lineup_result = await self._build_remaining_lineup(
                        df, used_indices, remaining_salary, qb, 
                        stack_combo, rbs, wrs, tes, dsts, game_type
                    )
                    
                    if lineup_result:
                        lineup_indices, total_score = lineup_result
                        
                        # Check constraints
                        if self._validate_lineup_constraints(df, lineup_indices, salary_cap, lock_indices):
                            if total_score > best_score:
                                best_score = total_score
                                best_lineup = lineup_indices
        
        metadata = {
            "method": "heuristic_enhanced",
            "ai_enhanced": True,
            "attempts": attempts,
            "best_score": best_score,
            "game_type": game_type
        }
        
        return best_lineup, metadata
    
    async def _build_remaining_lineup(
        self,
        df: pd.DataFrame,
        used_indices: Set[int],
        remaining_salary: int,
        qb: pd.Series,
        stack_combo: Tuple[int, ...],
        rbs: pd.DataFrame,
        wrs: pd.DataFrame,
        tes: pd.DataFrame,
        dsts: pd.DataFrame,
        game_type: str
    ) -> Optional[Tuple[List[int], float]]:
        """Build the remaining lineup positions with AI guidance"""
        
        # Count positions already filled
        stack_wr_count = sum(1 for idx in stack_combo if df.loc[idx, 'POS'] == 'WR')
        stack_te_count = sum(1 for idx in stack_combo if df.loc[idx, 'POS'] == 'TE')
        
        positions_needed = {
            'RB': 2,
            'WR': 3 - stack_wr_count,
            'TE': 1 - stack_te_count,
            'DST': 1,
            'FLEX': 1  # Will be filled from RB/WR/TE
        }
        
        # Remove already used players
        available_rbs = rbs[~rbs.index.isin(used_indices)]
        available_wrs = wrs[~wrs.index.isin(used_indices)]
        available_tes = tes[~tes.index.isin(used_indices)]
        available_dsts = dsts[~dsts.index.isin(used_indices)]
        
        # Use AI-enhanced selection for remaining positions
        best_combination = None
        best_score = -1
        
        # Try different RB combinations
        for rb_combo in itertools.combinations(available_rbs.index[:8], 2):
            rb_cost = sum(df.loc[idx, 'SALARY'] for idx in rb_combo)
            if rb_cost > remaining_salary * 0.5:  # Don't overspend on RBs
                continue
            
            current_used = used_indices | set(rb_combo)
            current_salary = rb_cost
            
            # Add required WRs
            wr_combo = []
            if positions_needed['WR'] > 0:
                available_wrs_filtered = available_wrs[~available_wrs.index.isin(current_used)]
                for wr_count in range(positions_needed['WR']):
                    if len(available_wrs_filtered) == 0:
                        break
                    
                    # Select best available WR within budget
                    for wr_idx in available_wrs_filtered.index:
                        if df.loc[wr_idx, 'SALARY'] + current_salary <= remaining_salary * 0.8:
                            wr_combo.append(wr_idx)
                            current_salary += df.loc[wr_idx, 'SALARY']
                            current_used.add(wr_idx)
                            available_wrs_filtered = available_wrs_filtered[~available_wrs_filtered.index.isin({wr_idx})]
                            break
            
            # Add TE if needed
            te_idx = None
            if positions_needed['TE'] > 0:
                available_tes_filtered = available_tes[~available_tes.index.isin(current_used)]
                for te_candidate in available_tes_filtered.index:
                    if df.loc[te_candidate, 'SALARY'] + current_salary <= remaining_salary * 0.9:
                        te_idx = te_candidate
                        current_salary += df.loc[te_candidate, 'SALARY']
                        current_used.add(te_candidate)
                        break
            
            # Add DST
            dst_idx = None
            available_dsts_filtered = available_dsts[~available_dsts.index.isin(current_used)]
            # Avoid DST opposing QB
            qb_opp = qb.get('OPP', '')
            for dst_candidate in available_dsts_filtered.index:
                dst_team = df.loc[dst_candidate, 'TEAM']
                if dst_team != qb_opp and df.loc[dst_candidate, 'SALARY'] + current_salary <= remaining_salary:
                    dst_idx = dst_candidate
                    current_salary += df.loc[dst_candidate, 'SALARY']
                    current_used.add(dst_candidate)
                    break
            
            # Add FLEX
            flex_idx = None
            remaining_budget = remaining_salary - current_salary
            flex_candidates = []
            
            # Add remaining RBs, WRs, TEs as FLEX candidates
            for pos_df in [available_rbs, available_wrs, available_tes]:
                candidates = pos_df[~pos_df.index.isin(current_used)]
                affordable = candidates[candidates['SALARY'] <= remaining_budget]
                flex_candidates.extend(affordable.index.tolist())
            
            if flex_candidates:
                # Select best FLEX by AI score
                flex_scores = [(idx, df.loc[idx, 'ai_score']) for idx in flex_candidates]
                flex_scores.sort(key=lambda x: x[1], reverse=True)
                flex_idx = flex_scores[0][0]
            
            # Validate complete lineup
            complete_lineup = list(used_indices) + list(rb_combo) + wr_combo
            if te_idx:
                complete_lineup.append(te_idx)
            if dst_idx:
                complete_lineup.append(dst_idx)
            if flex_idx:
                complete_lineup.append(flex_idx)
            
            if len(complete_lineup) == 9:  # Complete lineup
                total_score = sum(df.loc[idx, 'ai_score'] for idx in complete_lineup)
                if total_score > best_score:
                    best_score = total_score
                    best_combination = complete_lineup
        
        return (best_combination, best_score) if best_combination else None
    
    def _calculate_adjustment_factor(self, update_type: str, severity: float) -> float:
        """Calculate projection adjustment factor based on update"""
        if update_type == 'injury':
            return max(0.1, 1.0 - severity)  # Reduce projection
        elif update_type == 'weather':
            if severity > 0.7:
                return 0.85  # Significant weather impact
            elif severity > 0.4:
                return 0.95  # Moderate weather impact
            else:
                return 1.0  # Minimal impact
        elif update_type == 'news':
            return max(0.3, 1.0 - (severity * 0.5))  # Moderate adjustment for news
        else:
            return 1.0  # No adjustment for unknown types
    
    def _calculate_correlation_score(self, player_row: pd.Series) -> float:
        """Calculate position-specific correlation potential"""
        position = player_row['POS']
        base_score = player_row['PROJ PTS']
        
        if position == 'QB':
            return base_score * 1.2  # QBs have high correlation potential
        elif position in ['WR', 'TE']:
            return base_score * 1.1  # Pass catchers have good correlation
        elif position == 'RB':
            return base_score * 0.9  # RBs have moderate correlation
        else:
            return base_score * 0.8  # DST has lower correlation
    
    def _calculate_stack_synergy(self, qb: pd.Series, receiver: pd.Series) -> float:
        """Calculate synergy score between QB and receiver"""
        base_synergy = qb['PROJ PTS'] + receiver['PROJ PTS']
        
        # Bonus for high-projection QB-receiver combinations
        if qb['PROJ PTS'] > 20 and receiver['PROJ PTS'] > 12:
            base_synergy *= 1.15
        
        # Bonus for value combinations
        qb_value = qb['PROJ PTS'] / (qb['SALARY'] / 1000)
        rec_value = receiver['PROJ PTS'] / (receiver['SALARY'] / 1000)
        
        if qb_value > 2.5 and rec_value > 2.5:
            base_synergy *= 1.1
        
        return base_synergy
    
    def _validate_lineup_constraints(
        self, 
        df: pd.DataFrame, 
        lineup_indices: List[int], 
        salary_cap: int,
        lock_indices: Set[int]
    ) -> bool:
        """Validate lineup meets all constraints"""
        
        if len(lineup_indices) != 9:
            return False
        
        # Salary constraint
        total_salary = sum(df.loc[idx, 'SALARY'] for idx in lineup_indices)
        if total_salary > salary_cap:
            return False
        
        # Position constraints
        positions = df.loc[lineup_indices, 'POS'].value_counts()
        
        if positions.get('QB', 0) != 1:
            return False
        if positions.get('DST', 0) != 1:
            return False
        if positions.get('RB', 0) < 2 or positions.get('RB', 0) > 3:
            return False
        if positions.get('WR', 0) < 3 or positions.get('WR', 0) > 4:
            return False
        if positions.get('TE', 0) < 1 or positions.get('TE', 0) > 2:
            return False
        
        # Lock constraints
        if not lock_indices.issubset(set(lineup_indices)):
            return False
        
        return True
    
    def _fallback_optimization(
        self,
        df: pd.DataFrame,
        salary_cap: int,
        enforce_stack: bool,
        min_stack_receivers: int,
        lock_indices: Set[int]
    ) -> List[int]:
        """Fallback to basic optimization if enhanced methods fail"""
        
        # Simple greedy selection by value
        df_sorted = df.sort_values('value', ascending=False)
        
        lineup = []
        total_salary = 0
        positions_needed = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'DST': 1, 'FLEX': 1}
        
        # Add locked players first
        for idx in lock_indices:
            if idx in df.index:
                lineup.append(idx)
                total_salary += df.loc[idx, 'SALARY']
                pos = df.loc[idx, 'POS']
                if pos in positions_needed and positions_needed[pos] > 0:
                    positions_needed[pos] -= 1
        
        # Fill remaining positions
        for idx in df_sorted.index:
            if idx in lineup:
                continue
            
            pos = df.loc[idx, 'POS']
            salary = df.loc[idx, 'SALARY']
            
            # Check if position is needed and salary fits
            if positions_needed.get(pos, 0) > 0 and total_salary + salary <= salary_cap:
                lineup.append(idx)
                total_salary += salary
                positions_needed[pos] -= 1
            elif positions_needed.get('FLEX', 0) > 0 and pos in ['RB', 'WR', 'TE'] and total_salary + salary <= salary_cap:
                lineup.append(idx)
                total_salary += salary
                positions_needed['FLEX'] -= 1
            
            # Check if lineup is complete
            if sum(positions_needed.values()) == 0:
                break
        
        return lineup if len(lineup) == 9 else []
    
    async def _analyze_final_lineup(
        self,
        lineup_indices: List[int],
        df: pd.DataFrame,
        game_type: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the final lineup with AI insights"""
        
        lineup_players = []
        for idx in lineup_indices:
            player = df.loc[idx]
            lineup_players.append({
                'name': player['PLAYER NAME'],
                'position': player['POS'],
                'team': player['TEAM'],
                'opponent': player.get('OPP', ''),
                'proj_points': player['PROJ PTS'],
                'salary': player['SALARY'],
                'own_pct': player.get('OWN_PCT', 0)
            })
        
        # Simulation results (simplified)
        total_proj = sum(p['proj_points'] for p in lineup_players)
        sim_results = {
            'mean_score': total_proj,
            'std_dev': total_proj * 0.15,
            'percentiles': {
                '90th': total_proj * 1.25,
                '95th': total_proj * 1.35
            },
            'sharpe_ratio': total_proj / (total_proj * 0.15) if total_proj > 0 else 0
        }
        
        # Get AI analysis
        try:
            ai_analysis = await self.ai_analyzer.analyze_lineup(
                lineup_players, sim_results, game_type
            )
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            ai_analysis = "AI analysis unavailable"
        
        return {
            'lineup_players': lineup_players,
            'total_projection': total_proj,
            'simulation_results': sim_results,
            'ai_analysis': ai_analysis,
            'optimization_metadata': metadata
        }
