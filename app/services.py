import logging
import pandas as pd
from typing import Optional, List, Dict, Any

from app.data_ingestion import load_data_from_input_dir
from app.optimization import run_optimization
from app.state_manager import state_manager
from app.config import SALARY_CAP
from app.news_fetcher import fetch_latest_news
from app.ai_analyzer import analyze_news_and_adjust_projections

logger = logging.getLogger(__name__)

def generate_and_save_lineup(game_mode: str, teams: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    df, warnings = load_data_from_input_dir()
    for w in warnings:
        logger.info(w)

    if df is None or df.empty:
        logger.error("No player data available to generate a lineup.")
        return None

    latest_news = fetch_latest_news()
    
    if latest_news:
        player_pool = analyze_news_and_adjust_projections(df, latest_news)
    else:
        logger.info("No new news to analyze. Using base projections.")
        player_pool = df.copy()

    if teams:
        filtered_pool = player_pool[player_pool['TEAM'].isin(teams)]
        if len(filtered_pool) < 18:
            logger.warning(f"Insufficient players for teams {teams}. Using full slate.")
        else:
            player_pool = filtered_pool
            
    logger.info(f"Optimizing {game_mode} lineup with {len(player_pool)} players.")
    lineups, errors = run_optimization(player_df=player_pool, game_mode=game_mode)

    if errors or not lineups:
        logger.error(f"Optimization failed: {errors[0] if errors else 'No lineup returned.'}")
        return None

    lineup_indices = lineups[0]['players']
    lineup_df = player_pool.loc[lineup_indices].copy()
    
    lineup_df['OWN_PCT'] = lineup_df['OWN_PCT'].astype(object).where(pd.notna(lineup_df['OWN_PCT']), None)
    total_salary = int(lineup_df['SALARY'].sum())

    result = {
        "game_type": game_mode.upper(),
        "slate": " vs ".join(teams) if teams else "Full Slate",
        "lineup": lineup_df.to_dict(orient='records'),
        "cap_usage": {
            "total_salary": total_salary,
            "remaining": SALARY_CAP - total_salary
        }
    }
    
    state_manager.save_lineup(result, game_mode)
    return result
