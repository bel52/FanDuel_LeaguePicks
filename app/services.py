import logging
import pandas as pd
import asyncio
from typing import Optional, List, Dict, Any

from app.data_ingestion import load_data_from_input_dir
from app.optimization import run_optimization
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

logger = logging.getLogger(__name__)

async def generate_and_save_lineup(
    game_mode: str, 
    teams: Optional[List[str]] = None,
    use_ai: bool = True,
    use_weather: bool = True,
    use_game_scripts: bool = True
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
    
    # Step 1: Apply AI news analysis
    if use_ai:
        logger.info("Fetching and analyzing latest news...")
        latest_news = fetch_latest_news()
        
        if latest_news:
            player_pool = analyze_news_and_adjust_projections(player_pool, latest_news)
            logger.info(f"Applied AI adjustments to {player_pool['AI_ADJUSTED'].sum()} players")
        else:
            logger.info("No new news to analyze.")
    
    # Step 2: Apply weather analysis
    if use_weather:
        logger.info("Analyzing weather conditions...")
        weather_analyzer = WeatherAnalyzer()
        
        # Get games for weather analysis
        games = _extract_games_from_players(player_pool)
        
        # Run async weather fetch
        weather_data = asyncio.run(weather_analyzer.get_all_game_weather(games))
        
        if weather_data:
            player_pool = weather_analyzer.apply_weather_to_projections(player_pool, weather_data)
            logger.info(f"Applied weather adjustments to {len(weather_data)} teams")
    
    # Step 3: Apply game script predictions
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
        if len(filtered_pool) < 18:
            logger.warning(f"Insufficient players for teams {teams}. Using full slate.")
        else:
            player_pool = filtered_pool
    
    # Step 5: Run optimization
    logger.info(f"Optimizing {game_mode} lineup with {len(player_pool)} players.")
    lineups, errors = run_optimization(player_df=player_pool, game_mode=game_mode)

    if errors or not lineups:
        logger.error(f"Optimization failed: {errors[0] if errors else 'No lineup returned.'}")
        return None

    lineup_indices = lineups[0]['players']
    lineup_df = player_pool.loc[lineup_indices].copy()
    
    # Step 6: Analyze lineup correlation
    if use_ai:
        correlation_analysis = analyze_lineup_correlation(lineup_df)
        logger.info(f"Lineup correlation score: {correlation_analysis.get('correlation_score', 0)}/10")
    else:
        correlation_analysis = {}
    
    # Clean up data types for JSON serialization
    lineup_df['OWN_PCT'] = lineup_df['OWN_PCT'].astype(object).where(pd.notna(lineup_df['OWN_PCT']), None)
    total_salary = int(lineup_df['SALARY'].sum
