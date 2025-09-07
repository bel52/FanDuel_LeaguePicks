import os
import logging
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
import openai

from app.config import OPENAI_API_KEY
from app.state_manager import state_manager

logger = logging.getLogger(__name__)

# --- Global OpenAI Client ---
AI_CLIENT = None
try:
    if OPENAI_API_KEY:
        # Fixed: Use proper OpenAI client initialization
        AI_CLIENT = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Test connection
        AI_CLIENT.models.list()
        logger.info("OpenAI client initialized successfully.")
    else:
        logger.warning("OPENAI_API_KEY not set. AI analysis will be disabled.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    AI_CLIENT = None


def get_ai_projection_adjustment(player_name: str, headline: str, current_proj: float) -> Tuple[float, str]:
    """Calls OpenAI to get a numerical adjustment for a player's projection."""
    if not AI_CLIENT:
        return 0.0, "AI client not available."

    system_prompt = (
        "You are a quantitative fantasy football analyst. Your task is to adjust a player's "
        "DFS point projection based on a news headline. Respond ONLY with a JSON object with two keys: "
        "'adjustment' (a number, can be positive, negative, or 0) and 'reason' (a concise explanation)."
    )
    user_prompt = (
        f"Player: {player_name}\n"
        f"Current Projection: {current_proj:.2f}\n"
        f"News Headline: {headline}\n\n"
        "Based ONLY on this headline, what is the specific numerical point adjustment for the player's projection?"
    )

    try:
        response = AI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        result = json.loads(response.choices[0].message.content)
        adjustment = float(result.get('adjustment', 0.0))
        reason = result.get('reason', 'No specific reason provided.')
        return adjustment, reason
    except Exception as e:
        logger.error(f"Could not get AI adjustment for {player_name}: {e}")
        return 0.0, "AI analysis failed."


def analyze_news_and_adjust_projections(player_df: pd.DataFrame, news_items: List[Dict]) -> pd.DataFrame:
    """Analyzes news for all relevant players and adjusts their projections."""
    if not AI_CLIENT:
        logger.warning("Skipping projection adjustments as AI client is not available.")
        return player_df

    adjusted_df = player_df.copy()
    if 'AI_REASON' not in adjusted_df.columns:
        adjusted_df['AI_REASON'] = None
    if 'AI_ADJUSTED' not in adjusted_df.columns:
        adjusted_df['AI_ADJUSTED'] = False

    logger.info(f"Analyzing {len(news_items)} news items for projection adjustments...")

    for item in news_items:
        headline = item['headline']
        for index, player in adjusted_df.iterrows():
            player_name = player['PLAYER NAME']
            last_name = player_name.split(' ')[-1] if ' ' in player_name else player_name

            # Check if player is mentioned in headline
            if len(last_name) > 2 and last_name.lower() in headline.lower():
                # Check if this exact player/headline combo is cached
                cache_key = f"ai_proj_adj:{headline}:{player_name}"
                cached_data = state_manager.load_lineup(cache_key)

                if cached_data:
                    adjustment = cached_data.get('adjustment', 0.0)
                    reason = cached_data.get('reason', 'Cached result')
                else:
                    adjustment, reason = get_ai_projection_adjustment(
                        player_name, headline, player['PROJ PTS']
                    )
                    # Cache the result for 1 hour to prevent repeat API calls
                    state_manager.save_lineup(
                        {"adjustment": adjustment, "reason": reason}, 
                        cache_key
                    )
                
                if adjustment != 0:
                    logger.info(f"AI adjustment for {player_name}: {adjustment:+.2f} pts. Reason: {reason}")
                    adjusted_df.loc[index, 'PROJ PTS'] += adjustment
                    adjusted_df.loc[index, 'AI_REASON'] = reason
                    adjusted_df.loc[index, 'AI_ADJUSTED'] = True
                
                break  # Matched this headline to a player, move to next headline
    
    return adjusted_df


def get_weather_impact_analysis(games_weather: List[Dict]) -> Dict[str, float]:
    """Analyze weather conditions and return impact factors for each team."""
    if not AI_CLIENT:
        return {}
    
    weather_impacts = {}
    
    for game in games_weather:
        if not game.get('weather_data'):
            continue
            
        weather = game['weather_data']
        home_team = game['home_team']
        away_team = game['away_team']
        
        system_prompt = (
            "You are a DFS weather impact analyst. Given weather conditions for an NFL game, "
            "provide a multiplier (0.8 to 1.2) for fantasy scoring. "
            "Respond with JSON: {'passing_impact': float, 'rushing_impact': float, 'kicking_impact': float, 'reason': str}"
        )
        
        user_prompt = (
            f"Game: {home_team} vs {away_team}\n"
            f"Temperature: {weather.get('temperature', 'N/A')}°F\n"
            f"Wind: {weather.get('wind_speed', 'N/A')} mph\n"
            f"Precipitation: {weather.get('precipitation', 'None')}\n"
            f"Conditions: {weather.get('conditions', 'Clear')}\n\n"
            "What are the fantasy scoring impact multipliers?"
        )
        
        try:
            response = AI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Store impacts for both teams
            for team in [home_team, away_team]:
                weather_impacts[team] = {
                    'passing': float(result.get('passing_impact', 1.0)),
                    'rushing': float(result.get('rushing_impact', 1.0)),
                    'kicking': float(result.get('kicking_impact', 1.0)),
                    'reason': result.get('reason', '')
                }
                
        except Exception as e:
            logger.error(f"Weather analysis failed for {home_team} vs {away_team}: {e}")
    
    return weather_impacts


def analyze_lineup_correlation(lineup_df: pd.DataFrame) -> Dict[str, any]:
    """Analyze lineup correlation and provide strategic insights."""
    if not AI_CLIENT:
        return {"correlation_score": 0, "insights": "AI not available"}
    
    # Build lineup summary
    lineup_summary = []
    for _, player in lineup_df.iterrows():
        lineup_summary.append(
            f"{player['POS']}: {player['PLAYER NAME']} ({player['TEAM']} vs {player['OPP']}) - ${player['SALARY']} - {player['PROJ PTS']:.1f} pts"
        )
    
    system_prompt = (
        "You are a DFS correlation expert. Analyze this lineup for correlation plays, stacking quality, "
        "and leverage opportunities. Provide a JSON response with: "
        "{'correlation_score': 1-10, 'stack_quality': 'weak/moderate/strong', "
        "'leverage_plays': [list], 'concerns': [list], 'improvements': [list]}"
    )
    
    user_prompt = (
        "Analyze this FanDuel lineup for correlation and strategic positioning:\n\n" +
        "\n".join(lineup_summary) +
        "\n\nProvide correlation analysis and strategic insights."
    )
    
    try:
        response = AI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        logger.error(f"Lineup correlation analysis failed: {e}")
        return {"correlation_score": 0, "insights": "Analysis failed"}


def get_game_script_predictions(games: List[Dict]) -> Dict[str, Dict]:
    """Predict game scripts and their fantasy implications."""
    if not AI_CLIENT:
        return {}
    
    game_scripts = {}
    
    for game in games:
        system_prompt = (
            "You are an NFL game script predictor. Based on team matchup and implied totals, "
            "predict the game flow and its fantasy implications. "
            "Respond with JSON: {'script_type': 'shootout/grind/blowout', "
            "'passing_game_grade': 'A-F', 'rushing_game_grade': 'A-F', "
            "'fantasy_stacks': [recommended stacks], 'avoid': [players to avoid]}"
        )
        
        user_prompt = (
            f"Game: {game.get('home_team')} vs {game.get('away_team')}\n"
            f"Total: {game.get('total', 'N/A')}\n"
            f"Spread: {game.get('spread', 'N/A')}\n"
            f"Home Implied: {game.get('home_implied', 'N/A')}\n"
            f"Away Implied: {game.get('away_implied', 'N/A')}\n\n"
            "Predict the game script and fantasy implications."
        )
        
        try:
            response = AI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
            
            result = json.loads(response.choices[0].message.content)
            game_key = f"{game.get('home_team')}_vs_{game.get('away_team')}"
            game_scripts[game_key] = result
            
        except Exception as e:
            logger.error(f"Game script prediction failed: {e}")
    
    return game_scripts


def optimize_late_swaps(current_lineup: pd.DataFrame, live_scores: Dict, available_players: pd.DataFrame) -> List[Dict]:
    """Suggest optimal late swaps based on live game flow."""
    if not AI_CLIENT:
        return []
    
    system_prompt = (
        "You are a late swap optimizer for DFS. Based on current scores and game flow, "
        "suggest player swaps to maximize upside or protect floor. "
        "Respond with JSON: {'swaps': [{'out': player_name, 'in': player_name, 'reason': str}], "
        "'strategy': 'aggressive/balanced/conservative'}"
    )
    
    # Build context
    lineup_info = current_lineup[['PLAYER NAME', 'POS', 'TEAM', 'SALARY', 'PROJ PTS']].to_dict('records')
    
    user_prompt = (
        f"Current lineup:\n{json.dumps(lineup_info, indent=2)}\n\n"
        f"Live scores:\n{json.dumps(live_scores, indent=2)}\n\n"
        f"Available players count: {len(available_players)}\n\n"
        "Suggest optimal late swaps to maximize winning probability."
    )
    
    try:
        response = AI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('swaps', [])
        
    except Exception as e:
        logger.error(f"Late swap optimization failed: {e}")
        return []
