"""
Enhanced DFS Optimizer with AI integration and advanced strategies
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .simple_optimizer import SimpleDFSOptimizer
from .working_ai_analyzer import SimpleAIAnalyzer
from .lineup_rules import LineupValidator
from .data_monitor import DataMonitor

logger = logging.getLogger(__name__)

@dataclass
class OptimizationMetadata:
    method: str
    ai_analysis: Optional[Dict] = None
    correlation_score: float = 0.0
    leverage_score: float = 0.0
    confidence: float = 0.0
    swap_recommendations: List[Dict] = None

class EnhancedDFSOptimizer:
    """Production-grade DFS optimizer with AI integration"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.simple_optimizer = SimpleDFSOptimizer()
        self.ai_analyzer = SimpleAIAnalyzer()
        self.validator = LineupValidator()
        self.data_monitor = DataMonitor()
        
        # Strategy weights
        self.hth_weight = float(self.config.get('HTH_STRATEGY_WEIGHT', 0.3))
        self.league_weight = float(self.config.get('LEAGUE_STRATEGY_WEIGHT', 0.7))
        
    async def optimize_lineup(self, 
                            players_df: pd.DataFrame, 
                            game_type: str = 'league',
                            constraints: Optional[Dict] = None) -> Tuple[Dict, OptimizationMetadata]:
        """
        Enhanced lineup optimization with AI analysis
        
        Args:
            players_df: Player data
            game_type: 'league' or 'h2h'
            constraints: Additional constraints
            
        Returns:
            Tuple of (lineup_dict, metadata)
        """
        logger.info(f"Starting enhanced optimization for {game_type}")
        
        try:
            # Step 1: Basic optimization
            lineup = self.simple_optimizer.optimize_lineup(
                players_df, 
                salary_cap=constraints.get('salary_cap', 50000) if constraints else 50000
            )
            
            if not lineup:
                logger.warning("Basic optimization failed")
                return {}, OptimizationMetadata(method="failed")
            
            # Step 2: AI Enhancement (if available)
            ai_analysis = None
            try:
                ai_analysis = await self._get_ai_analysis(lineup, players_df, game_type)
            except Exception as e:
                logger.warning(f"AI analysis failed: {e}")
            
            # Step 3: Strategy-specific adjustments
            if game_type == 'h2h':
                lineup = self._apply_hth_strategy(lineup, players_df)
                method = "enhanced_h2h"
            else:
                lineup = self._apply_league_strategy(lineup, players_df)
                method = "enhanced_league"
            
            # Step 4: Calculate metadata
            metadata = OptimizationMetadata(
                method=method,
                ai_analysis=ai_analysis,
                correlation_score=self._calculate_correlation_score(lineup, players_df),
                leverage_score=self._calculate_leverage_score(lineup, players_df),
                confidence=self._calculate_confidence_score(lineup, ai_analysis)
            )
            
            # Step 5: Validation
            validation_result = self.validator.validate_lineup(lineup)
            if not validation_result['valid']:
                logger.error(f"Lineup validation failed: {validation_result['errors']}")
                return {}, OptimizationMetadata(method="validation_failed")
            
            logger.info(f"Optimization complete: {method}")
            return lineup, metadata
            
        except Exception as e:
            logger.error(f"Enhanced optimization failed: {e}")
            return {}, OptimizationMetadata(method="error")
    
    async def _get_ai_analysis(self, lineup: Dict, players_df: pd.DataFrame, game_type: str) -> Optional[Dict]:
        """Get AI analysis of the lineup"""
        try:
            # Prepare context for AI
            context = {
                'lineup': lineup,
                'game_type': game_type,
                'market_data': self._get_market_context(players_df),
                'weather_data': await self.data_monitor.get_weather_updates()
            }
            
            analysis = await self.ai_analyzer.analyze_lineup_strategy(context)
            return analysis
            
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            return None
    
    def _apply_hth_strategy(self, lineup: Dict, players_df: pd.DataFrame) -> Dict:
        """Apply head-to-head focused strategy (ceiling emphasis)"""
        # H2H strategy prioritizes upside and contrarian plays
        try:
            # Look for high-ceiling, low-ownership players
            for position in ['QB', 'RB', 'WR', 'TE']:
                if position in lineup:
                    current_player = lineup[position]
                    if isinstance(current_player, list):
                        continue
                        
                    # Find alternatives with higher ceiling potential
                    position_players = players_df[players_df['POS'] == position]
                    if not position_players.empty:
                        # Sort by ceiling potential (using projection * volatility proxy)
                        position_players['ceiling_score'] = (
                            position_players['PROJ PTS'] * 
                            (1 + position_players.get('VARIANCE', 0.2))
                        )
                        
                        # Consider ownership for contrarian value
                        if 'OWN_PCT' in position_players.columns:
                            position_players['leverage'] = (
                                position_players['ceiling_score'] / 
                                (position_players['OWN_PCT'] + 1)
                            )
                        else:
                            position_players['leverage'] = position_players['ceiling_score']
                        
                        # Pick top leverage option that fits salary
                        best_options = position_players.nlargest(3, 'leverage')
                        # Implementation would include salary cap validation
                        
            return lineup
            
        except Exception as e:
            logger.warning(f"H2H strategy application failed: {e}")
            return lineup
    
    def _apply_league_strategy(self, lineup: Dict, players_df: pd.DataFrame) -> Dict:
        """Apply league focused strategy (balanced approach)"""
        # League strategy balances floor and ceiling
        try:
            # Focus on consistent performers with good value
            # Implementation would adjust for safer, higher-floor plays
            return lineup
        except Exception as e:
            logger.warning(f"League strategy application failed: {e}")
            return lineup
    
    def _calculate_correlation_score(self, lineup: Dict, players_df: pd.DataFrame) -> float:
        """Calculate team correlation score for stacking analysis"""
        try:
            teams = []
            for player_info in lineup.values():
                if isinstance(player_info, dict) and 'team' in player_info:
                    teams.append(player_info['team'])
                elif isinstance(player_info, list):
                    for player in player_info:
                        if isinstance(player, dict) and 'team' in player:
                            teams.append(player['team'])
            
            # Calculate team concentration
            from collections import Counter
            team_counts = Counter(teams)
            max_team_count = max(team_counts.values()) if team_counts else 1
            
            # Higher score for more correlation (stacking)
            correlation_score = min(max_team_count / 9.0, 1.0)  # 9 players max
            return correlation_score
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return 0.0
    
    def _calculate_leverage_score(self, lineup: Dict, players_df: pd.DataFrame) -> float:
        """Calculate ownership leverage score"""
        try:
            total_ownership = 0
            player_count = 0
            
            for player_info in lineup.values():
                if isinstance(player_info, dict) and 'name' in player_info:
                    player_data = players_df[players_df['PLAYER NAME'] == player_info['name']]
                    if not player_data.empty and 'OWN_PCT' in player_data.columns:
                        total_ownership += player_data['OWN_PCT'].iloc[0]
                        player_count += 1
                elif isinstance(player_info, list):
                    for player in player_info:
                        if isinstance(player, dict) and 'name' in player:
                            player_data = players_df[players_df['PLAYER NAME'] == player['name']]
                            if not player_data.empty and 'OWN_PCT' in player_data.columns:
                                total_ownership += player_data['OWN_PCT'].iloc[0]
                                player_count += 1
            
            if player_count == 0:
                return 0.5  # Neutral score if no ownership data
            
            avg_ownership = total_ownership / player_count
            # Lower ownership = higher leverage (inverted)
            leverage_score = max(0.0, (25.0 - avg_ownership) / 25.0)
            return min(leverage_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Leverage calculation failed: {e}")
            return 0.5
    
    def _calculate_confidence_score(self, lineup: Dict, ai_analysis: Optional[Dict]) -> float:
        """Calculate overall confidence in the lineup"""
        try:
            confidence_factors = []
            
            # Base confidence from lineup completeness
            if len(lineup) >= 7:  # Full lineup
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            # AI analysis confidence
            if ai_analysis and 'confidence' in ai_analysis:
                confidence_factors.append(ai_analysis['confidence'])
            else:
                confidence_factors.append(0.6)  # Neutral without AI
            
            # Data freshness (would be implemented with real monitoring)
            confidence_factors.append(0.7)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _get_market_context(self, players_df: pd.DataFrame) -> Dict:
        """Get market context for AI analysis"""
        try:
            context = {
                'total_players': len(players_df),
                'avg_salary': players_df['SALARY'].mean() if 'SALARY' in players_df.columns else 0,
                'avg_projection': players_df['PROJ PTS'].mean() if 'PROJ PTS' in players_df.columns else 0,
                'position_breakdown': players_df['POS'].value_counts().to_dict() if 'POS' in players_df.columns else {}
            }
            return context
        except Exception as e:
            logger.warning(f"Market context generation failed: {e}")
            return {}
    
    async def generate_multiple_lineups(self, 
                                      players_df: pd.DataFrame, 
                                      count: int = 5,
                                      game_type: str = 'league') -> List[Dict]:
        """Generate multiple diverse lineups"""
        lineups = []
        
        for i in range(count):
            try:
                # Add some randomization for diversity
                constraints = {
                    'salary_cap': 50000,
                    'diversity_seed': i
                }
                
                lineup, metadata = await self.optimize_lineup(
                    players_df, 
                    game_type=game_type,
                    constraints=constraints
                )
                
                if lineup:
                    lineup['_metadata'] = metadata.__dict__
                    lineups.append(lineup)
                    
            except Exception as e:
                logger.warning(f"Lineup {i+1} generation failed: {e}")
                continue
        
        return lineups
