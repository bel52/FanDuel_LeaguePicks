from dotenv import load_dotenv
load_dotenv()
"""
FIXED: Real Vegas lines integration for tournament-winning DFS
High-total games (47+ points) produce 70%+ of tournament winners
SCALABLE: Works for any week of any season without hardcoded matchups
"""
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime
import os


class VegasDataCollector:
    """Collect real Vegas lines that drive DFS success"""

    def __init__(self):
        # Get API key from environment
        self.api_key = os.getenv('ODDS_API_KEY', '')
        self.base_url = "https://api.the-odds-api.com/v4"

        # Free tier: 500 calls/month (perfect for weekly DFS)
        self.requests_per_week = 20  # Very conservative usage

        if not self.api_key:
            logger.warning("⚠️ No ODDS_API_KEY found - will use graceful fallback")

    async def get_nfl_odds_data(self) -> Dict:
        """Get NFL odds with proper team deduplication"""
        api_key = os.getenv("ODDS_API_KEY", "")

        if not api_key:
            logger.info("📊 No Vegas API key - using team multiplier defaults")
            return self._get_scalable_fallback()

        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,totals",
            "oddsFormat": "american",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.warning(f"Vegas API returned {response.status} - using fallback")
                        return self._get_scalable_fallback()

                    data = await response.json()

                    if not data or len(data) == 0:
                        logger.warning("Vegas API returned empty data - using fallback")
                        return self._get_scalable_fallback()

                    return self._process_vegas_data(data)

        except Exception as e:
            logger.warning(f"Vegas API unavailable: {e}")
            return self._get_scalable_fallback()

    def _process_vegas_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Process raw Vegas data with team deduplication"""

        processed_games = {}
        high_total_games = []
        scheduled_teams = set()  # Track teams already scheduled in games

        for game in raw_data:
            try:
                home_team = self._normalize_team_name(game.get('home_team', ''))
                away_team = self._normalize_team_name(game.get('away_team', ''))

                if not home_team or not away_team:
                    continue

                game_id = f"{away_team}@{home_team}"

                # Skip if we already processed this exact game
                if game_id in processed_games:
                    logger.debug(f"Skipping duplicate game: {game_id}")
                    continue

                commence_time = game.get('commence_time', '')

                # Extract betting data from bookmakers
                game_data = {
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'commence_time': commence_time,
                    'total_points': None,
                    'spread': None,
                    'home_moneyline': None,
                    'away_moneyline': None
                }

                # Get data from reputable sportsbooks
                for bookmaker in game.get('bookmakers', []):
                    book_key = bookmaker.get('key', '')

                    if book_key in ['fanduel', 'draftkings', 'betmgm', 'caesars', 'pointsbet']:
                        markets = bookmaker.get('markets', [])

                        for market in markets:
                            market_key = market.get('key')
                            outcomes = market.get('outcomes', [])

                            # TOTALS - Critical for DFS success
                            if market_key == 'totals' and outcomes:
                                total_point = outcomes[0].get('point')
                                if total_point:
                                    game_data['total_points'] = float(total_point)

                            # SPREADS
                            elif market_key == 'spreads' and len(outcomes) >= 2:
                                for outcome in outcomes:
                                    if home_team.upper() in outcome.get('name', '').upper():
                                        spread = outcome.get('point')
                                        if spread:
                                            game_data['spread'] = float(spread)
                                            break

                            # MONEYLINES
                            elif market_key == 'h2h' and len(outcomes) >= 2:
                                for outcome in outcomes:
                                    team_name = outcome.get('name', '')
                                    price = outcome.get('price')

                                    if home_team.upper() in team_name.upper():
                                        game_data['home_moneyline'] = price
                                    elif away_team.upper() in team_name.upper():
                                        game_data['away_moneyline'] = price

                        # Use first valid bookmaker
                        if game_data['total_points']:
                            break

                # Only process games with valid betting data
                if game_data['total_points']:
                    # TEAM DEDUPLICATION: Check for impossible scheduling conflicts
                    if home_team in scheduled_teams or away_team in scheduled_teams:
                        conflicting_team = home_team if home_team in scheduled_teams else away_team
                        logger.warning(f"🚫 IMPOSSIBLE SCHEDULE: Skipping {away_team}@{home_team} - {conflicting_team} already scheduled")
                        continue

                    # Mark teams as scheduled
                    scheduled_teams.add(home_team)
                    scheduled_teams.add(away_team)

                    processed_games[game_id] = game_data

                    # Identify high-total games (DFS tournament gold)
                    if game_data['total_points'] >= 47.0:
                        high_total_games.append({
                            'game_id': game_id,
                            'total': game_data['total_points'],
                            'teams': [home_team, away_team]
                        })
                        logger.info(f"🔥 HIGH-TOTAL GAME: {game_id} ({game_data['total_points']} pts)")

            except Exception as e:
                logger.error(f"Error processing game data: {e}")
                continue

        # Sort high-total games by total (highest first)
        high_total_games.sort(key=lambda x: x['total'], reverse=True)

        # Calculate implied team scores for player weighting
        for game_id, game_data in processed_games.items():
            self._calculate_implied_scores(game_data)

        logger.info(f"✅ Processed {len(processed_games)} valid games with {len(high_total_games)} high-total games")

        return {
            'games': processed_games,
            'high_total_games': high_total_games,
            'avg_total': sum(g['total_points'] for g in processed_games.values()) / len(processed_games) if processed_games else 45.5,
            'total_games': len(processed_games),
            'data_source': 'real_vegas_api'
        }

    def _calculate_implied_scores(self, game_data: Dict):
        """Calculate implied team scores from total and spread"""
        total = game_data.get('total_points', 45.5)
        spread = game_data.get('spread', 0)

        # Handle None spread
        if spread is None:
            spread = 0

        try:
            spread = float(spread)
            total = float(total)
        except (ValueError, TypeError):
            spread = 0
            total = 45.5

        # Calculate implied scores
        home_implied = (total - spread) / 2
        away_implied = (total + spread) / 2

        game_data['home_implied_score'] = round(home_implied, 1)
        game_data['away_implied_score'] = round(away_implied, 1)

    def _normalize_team_name(self, team_name: str) -> str:
        """Convert full team names to standard abbreviations"""
        team_mapping = {
            # AFC East
            'Buffalo Bills': 'BUF', 'Miami Dolphins': 'MIA',
            'New England Patriots': 'NE', 'New York Jets': 'NYJ',

            # AFC North
            'Baltimore Ravens': 'BAL', 'Cincinnati Bengals': 'CIN',
            'Cleveland Browns': 'CLE', 'Pittsburgh Steelers': 'PIT',

            # AFC South
            'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND',
            'Jacksonville Jaguars': 'JAX', 'Tennessee Titans': 'TEN',

            # AFC West
            'Denver Broncos': 'DEN', 'Kansas City Chiefs': 'KC',
            'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',

            # NFC East
            'Dallas Cowboys': 'DAL', 'New York Giants': 'NYG',
            'Philadelphia Eagles': 'PHI', 'Washington Commanders': 'WAS',

            # NFC North
            'Chicago Bears': 'CHI', 'Detroit Lions': 'DET',
            'Green Bay Packers': 'GB', 'Minnesota Vikings': 'MIN',

            # NFC South
            'Atlanta Falcons': 'ATL', 'Carolina Panthers': 'CAR',
            'New Orleans Saints': 'NO', 'Tampa Bay Buccaneers': 'TB',

            # NFC West
            'Arizona Cardinals': 'ARI', 'Los Angeles Rams': 'LAR',
            'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA'
        }

        return team_mapping.get(team_name, team_name)

    def _get_scalable_fallback(self) -> Dict[str, Any]:
        """SCALABLE: No hardcoded games - just return reasonable defaults"""
        logger.info("📊 Using scalable fallback - no hardcoded matchups")

        # Return empty games but with reasonable league averages for multipliers
        return {
            'games': {},  # No hardcoded games
            'high_total_games': [],  # No fake high-total games
            'avg_total': 45.5,  # Reasonable NFL average
            'total_games': 0,
            'data_source': 'scalable_fallback'
        }

    def get_game_environment_factors(self, vegas_data: Dict) -> Dict[str, float]:
        """Calculate game environment multipliers - gracefully handles no Vegas data"""

        game_factors = {}
        games = vegas_data.get('games', {})

        # If no Vegas data, return neutral multipliers (1.0x for all teams)
        if not games:
            logger.info("📊 No Vegas data available - using neutral team multipliers (1.0x)")
            return {}  # Empty dict means no multipliers applied

        avg_total = vegas_data.get('avg_total', 45.5)

        for game_id, game_data in games.items():
            total = game_data.get('total_points', avg_total)
            spread = abs(game_data.get('spread', 0))

            # TOTAL FACTOR: Higher totals = higher DFS scoring potential
            total_factor = 1.0
            if total >= 50:
                total_factor = 1.35  # Elite scoring environment
            elif total >= 47:
                total_factor = 1.25  # High scoring
            elif total >= 44:
                total_factor = 1.10  # Above average
            elif total <= 40:
                total_factor = 0.85  # Low scoring

            # SPREAD FACTOR: Close games = more passing = higher DFS upside
            spread_factor = 1.0
            if spread <= 3:
                spread_factor = 1.15  # Competitive games
            elif spread >= 10:
                spread_factor = 0.90  # Potential blowouts

            # COMBINED FACTOR
            combined_factor = total_factor * spread_factor

            # Apply to both teams in the game
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')

            if home_team:
                game_factors[home_team] = combined_factor
            if away_team:
                game_factors[away_team] = combined_factor

            logger.info(f"🎯 {game_id}: {total} total, {spread} spread → {combined_factor:.2f}x factor")

        return game_factors


# Integration functions for data_collector.py
async def get_vegas_odds_data() -> Dict[str, Any]:
    """Get Vegas odds data for DFS optimization"""
    collector = VegasDataCollector()
    return await collector.get_nfl_odds_data()


async def get_game_environment_multipliers() -> Dict[str, float]:
    """Get team multipliers based on game environment"""
    collector = VegasDataCollector()
    vegas_data = await collector.get_nfl_odds_data()
    return collector.get_game_environment_factors(vegas_data)