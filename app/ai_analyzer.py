import os
import logging
import json
import pandas as pd
from typing import Dict, List, Tuple

import openai
from app.config import OPENAI_API_KEY
from app.state_manager import state_manager

logger = logging.getLogger(__name__)

# --- Global OpenAI Client ---
AI_CLIENT = None
try:
    if OPENAI_API_KEY:
        AI_CLIENT = openai.OpenAI(api_key=OPENAI_API_KEY)
        AI_CLIENT.models.list()  # Test connection
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

    logger.info(f"Analyzing {len(news_items)} news items for projection adjustments...")

    for item in news_items:
        headline = item['headline']
        for index, player in adjusted_df.iterrows():
            player_name = player['PLAYER NAME']
            last_name = player_name.split(' ')[-1]

            if len(last_name) > 2 and last_name.lower() in headline.lower():
                # Check if this exact player/headline combo is cached
                cache_key = f"ai_proj_adj:{headline}:{player_name}"
                cached_data = state_manager.load_lineup(cache_key)

                if cached_data:
                    adjustment = cached_data.get('adjustment', 0.0)
                    reason = cached_data.get('reason', 'Cached result')
                else:
                    adjustment, reason = get_ai_projection_adjustment(player_name, headline, player['PROJ PTS'])
                    # Cache the result for 1 hour to prevent repeat API calls
                    state_manager.save_lineup({"adjustment": adjustment, "reason": reason}, cache_key)
                
                if adjustment != 0:
                    logger.info(f"AI adjustment for {player_name}: {adjustment:+.2f} pts. Reason: {reason}")
                    adjusted_df.loc[index, 'PROJ PTS'] += adjustment
                    adjusted_df.loc[index, 'AI_REASON'] = reason
                
                break # Matched this headline to a player, move to next headline
    
    return adjusted_df
