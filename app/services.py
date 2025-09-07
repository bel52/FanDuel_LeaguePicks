import logging
import asyncio
from typing import Optional, List, Dict, Any
import pandas as pd

from app.data_ingestion import load_data_from_input_dir
from app.enhanced_optimizer import EnhancedDFSOptimizer
from app.state_manager import state_manager
from app.config import SALARY_CAP
from app.news_fetcher import fetch_latest_news
from app.ai_analyzer import (
    analyze_news_and_adjust_projections,
    get_weather_impact_analysis,
    analyze_lineup_correlation,
    get_game_script_predictions,
    optimize_late_swaps
)
from app.weather_analyzer import WeatherAnalyzer
from app.player_match import match_names_to_indices

logger = logging.getLogger(__name__)

async def generate_and_save_lineup(
    game_mode: str,
    teams: Optional[List[str]] = None,
    use_ai: bool = True,
    use_weather: bool = True,
    use_game_scripts: bool = True,
    lock_players: Optional[List[str]] = None,
    ban_players: Optional[List[str]] = None,
    salary_cap: int = SALARY_CAP,
    enforce_stack: bool = True,
    min_stack_receivers: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Generate an AI-enhanced lineup with comprehensive analysis.
    """
    
    # Load base player data
    df, warnings = load_data_from_input_dir()
    for w in warnings:
        logger.info(w)

    if df is None or df.empty:
        logger.error("No player data available to generate a lineup.")
        return None

    # Initialize player pool
    player_pool = df.copy()
    
    # Step 1: Apply AI news analysis if enabled
    if use_ai:
        logger.info("Fetching and analyzing latest news...")
        latest_news = fetch_latest_news()
        
        if latest_news:
            player_pool = analyze_news_and_adjust_projections(player_pool, latest_news)
            adjusted_count = player_pool['AI_ADJUSTED'].sum() if 'AI_ADJUSTED' in player_pool.columns else 0
            logger.info(f"Applied AI adjustments to {adjusted_count} players")
        else:
            logger.info("No new news to analyze.")
    
    # Step 2: Apply weather analysis if enabled
    if use_weather:
        logger.info("Analyzing weather conditions...")
        weather_analyzer = WeatherAnalyzer()
        
        # Get games for weather analysis
        games = _extract_games_from_players(player_pool)
        
        # Run weather fetch
        weather_data = await weather_analyzer.get_all_game_weather(games)
        
        if weather_data:
            player_pool = weather_analyzer.apply_weather_to_projections(player_pool, weather_data)
            logger.info(f"Applied weather adjustments to {len(weather_data)} teams")
    
    # Step 3: Apply game script predictions if enabled
    if use_game_scripts and use_ai:
        logger.info("Generating game script predictions...")
        games_for_scripts = _extract_games_with_totals(player_pool)
        game_scripts = get_game_script_predictions(games_for_scripts)
        
        if game_scripts:
            player_pool = _apply_game_script_adjustments(player_pool, game_scripts)
            logger.info(f"Applied game script adjustments for {len(game_scripts)} games")
    
    # Step 4: Filter by teams if specified
    if teams:
        filtered_pool = player_pool[player_pool['TEAM'].isin(teams)]
        if len(filtered_pool) < 50:  # Need enough players for a lineup
            logger.warning(f"Insufficient players for teams {teams}. Using full slate.")
        else:
            player_pool = filtered_pool
    
    # Step 5: Handle lock and ban players
    lock_indices = []
    ban_indices = []
    
    if lock_players:
        lock_indices, not_found = match_names_to_indices(lock_players, player_pool)
        if not_found:
            logger.warning(f"Could not find players to lock: {not_found}")
    
    if ban_players:
        ban_indices, not_found = match_names_to_indices(ban_players, player_pool)
        if not_found:
            logger.warning(f"Could not find players to ban: {not_found}")
    
    # Step 6: Run optimization
    logger.info(f"Optimizing {game_mode} lineup with {len(player_pool)} players.")
    
    optimizer = EnhancedDFSOptimizer()
    lineup_indices, metadata = await optimizer.optimize_lineup(
        df=player_pool,
        game_type=game_mode,
        salary_cap=salary_cap,
        enforce_stack=enforce_stack,
        min_stack_receivers=min_stack_receivers,
        lock_indices=lock_indices,
        ban_indices=ban_indices,
        auto_swap_enabled=True
    )
    
    if not lineup_indices:
        logger.error("Optimization failed to produce a lineup")
        return None
    
    # Step 7: Build lineup dataframe
    lineup_df = player_pool.loc[lineup_indices].copy()
    
    # Step 8: Analyze lineup correlation if AI enabled
    if use_ai:
        correlation_analysis = analyze_lineup_correlation(lineup_df)
        logger.info(f"Lineup correlation score: {correlation_analysis.get('correlation_score', 0)}/10")
    else:
        correlation_analysis = {}
    
    # Step 9: Prepare result
    lineup_data = []
    total_salary = 0
    total_projection = 0
    
    for idx in lineup_indices:
        player = player_pool.loc[idx]
        lineup_data.append({
            'PLAYER NAME': player['PLAYER NAME'],
            'POS': player['POS'],
            'TEAM': player['TEAM'],
            'OPP': player.get('OPP', ''),
            'SALARY': int(player['SALARY']),
            'PROJ PTS': float(player['PROJ PTS']),
            'OWN_PCT': float(player.get('OWN_PCT', 0)) if pd.notna(player.get('OWN_PCT')) else None,
            'AI_ADJUSTED': player.get('AI_ADJUSTED', False),
            'AI_REASON': player.get('AI_REASON', ''),
            'WEATHER_IMPACT': player.get('WEATHER_IMPACT', 1.0),
            'WEATHER_NOTE': player.get('WEATHER_NOTE', '')
        })
        total_salary += int(player['SALARY'])
        total_projection += float(player['PROJ PTS'])
    
    # Step 10: Get late swap suggestions if available
    late_swap_suggestions = []
    if use_ai:
        # Simulate being behind for aggressive swaps
        late_swap_suggestions = await optimize_late_swaps(
            lineup_df, 
            {"status": "BEHIND"},  # Simulated live scores
            player_pool[~player_pool.index.isin(lineup_indices)]
        )
    
    result = {
        "game_type": game_mode,
        "lineup": lineup_data,
        "total_projected_points": round(total_projection, 2),
        "cap_usage": {
            "total_salary": total_salary,
            "remaining": salary_cap - total_salary
        },
        "optimization_metadata": metadata,
        "correlation_analysis": correlation_analysis,
        "late_swap_suggestions": late_swap_suggestions[:3] if late_swap_suggestions else [],
        "ai_enhanced": use_ai,
        "weather_adjusted": use_weather,
        "game_scripts_applied": use_game_scripts
    }
    
    # Save to state manager
    state_manager.save_lineup(result, game_mode)
    logger.info(f"Successfully generated and saved {game_mode} lineup")
    
    return result

def _extract_games_from_players(df: pd.DataFrame) -> List[Dict]:
    """Extract unique games from player data"""
    games = []
    teams_seen = set()
    
    for _, player in df.iterrows():
        home_team = player.get('TEAM')
        away_team = player.get('OPP')
        
        if home_team and away_team:
            game_key = tuple(sorted([home_team, away_team]))
            if game_key not in teams_seen:
                teams_seen.add(game_key)
                games.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'game_time': None  # Would need schedule data
                })
    
    return games

def _extract_games_with_totals(df: pd.DataFrame) -> List[Dict]:
    """Extract games with betting totals for game script analysis"""
    games = _extract_games_from_players(df)
    
    # Add mock totals for demonstration (would come from odds API)
    for game in games:
        game['total'] = 47.5  # Mock total
        game['spread'] = -3.5  # Mock spread
        game['home_implied'] = 25.5
        game['away_implied'] = 22.0
    
    return games

def _apply_game_script_adjustments(
    df: pd.DataFrame, 
    game_scripts: Dict[str, Dict]
) -> pd.DataFrame:
    """Apply game script predictions to player projections"""
    
    adjusted_df = df.copy()
    
    for game_key, script in game_scripts.items():
        teams = game_key.split('_vs_')
        if len(teams) != 2:
            continue
        
        script_type = script.get('script_type', 'balanced')
        passing_grade = script.get('passing_game_grade', 'C')
        rushing_grade = script.get('rushing_game_grade', 'C')
        
        for team in teams:
            team_mask = adjusted_df['TEAM'] == team
            
            # Adjust based on script type
            if script_type == 'shootout':
                # Boost passing game
                adjusted_df.loc[team_mask & (adjusted_df['POS'].isin(['QB', 'WR', 'TE'])), 'PROJ PTS'] *= 1.1
                adjusted_df.loc[team_mask & (adjusted_df['POS'] == 'RB'), 'PROJ PTS'] *= 0.95
            elif script_type == 'grind':
                # Boost running game
                adjusted_df.loc[team_mask & (adjusted_df['POS'] == 'RB'), 'PROJ PTS'] *= 1.1
                adjusted_df.loc[team_mask & (adjusted_df['POS'].isin(['QB', 'WR'])), 'PROJ PTS'] *= 0.95
            elif script_type == 'blowout':
                # Varies by team (winner vs loser)
                # This would need more context to determine
                pass
    
    return adjusted_df
