import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from app.services import generate_and_save_lineup

logger = logging.getLogger(__name__)

async def generate_hth_lineup(
    teams: List[str],
    use_ai: bool = True,
    use_weather: bool = True,
    use_game_scripts: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Generates a Head-to-Head lineup focused on specific teams.
    """
    if not teams or len(teams) > 4: # Limit to a reasonable number of teams for H2H
        logger.error("Please provide 1 to 4 teams for H2H lineup generation.")
        return None

    logger.info(f"Generating H2H lineup for teams: {teams}")

    # For H2H, we still use the 'main' game mode but filter by teams.
    # The optimization constraints for H2H are generally the same as for main slates.
    lineup_data = await generate_and_save_lineup(
        game_mode='main',  # Use main slate optimization
        teams=teams,
        use_ai=use_ai,
        use_weather=use_weather,
        use_game_scripts=use_game_scripts
    )

    if lineup_data:
        logger.info("Successfully generated H2H lineup.")
        # Optionally, you could save this to a different state key
        # from app.state_manager import state_manager
        # state_manager.save_lineup(lineup_data, f"generated_lineup_hth_{'_'.join(teams)}")
    
    return lineup_data
