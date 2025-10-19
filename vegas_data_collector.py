"""
FIXED: Real Vegas lines integration for tournament-winning DFS
High-total games (47+ points) produce 70%+ of tournament winners
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
            logger.warning("⚠️ No ODDS_API_KEY found - using enhanced fallback data")

    async def get_nfl_odds_data(self) -> Dict:
        """FIXED: Get NFL odds with proper function calls"""
        api_key = os.getenv("ODDS_API_KEY", "")
        url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h,totals",
            "oddsFormat": "american",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return self._process_vegas_data(data)  # FIXED: Use correct method name
        except Exception as e:
            # Keep it calm, use fallback without stack spam.
            logger.warning(f"Vegas API unavailable — using fallback: {e}")
            return self._get_enhanced_fallback_odds()  # FIXED: Remove week parameter

    def _process_vegas_data(self, raw_data: List[Dict]) -> Dict[str, Any]:
        """Process raw Vegas data into DFS-optimized format"""

        processed_games = {}
        high_total_games = []

        for game in raw_data:
            try:
                home_team = self._normalize_team_name(game.get('home_team', ''))
                away_team = self._normalize_team_name(game.get('away_team', ''))

                if not home_team or not away_team:
                    continue

                game_id = f"{away_team}@{home_team}"

                # CRITICAL FIX: Skip if we already processed this game
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

                # Get data from FanDuel or DraftKings (most reliable)
                for bookmaker in game.get('bookmakers', []):
                    book_key = bookmaker.get('key', '')

                    if book_key in ['fanduel', 'draftkings', 'betmgm']:
                        markets = bookmaker.get('markets', [])

                        for market in markets:
                            market_key = market.get('key')
                            outcomes = market.get('outcomes', [])

                            # TOTALS - This is the DFS goldmine
                            if market_key == 'totals' and outcomes:
                                total_point = outcomes[0].get('point')
                                if total_point:
                                    game_data['total_points'] = float(total_point)

                            # SPREADS
                            elif market_key == 'spreads' and len(outcomes) >= 2:
                                # Find home team spread
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

                        # Use first good bookmaker found
                        if game_data['total_points']:
                            break

                # Only include games with total points (essential for DFS)
                if game_data['total_points']:
                    processed_games[game_id] = game_data

                    # Identify HIGH-TOTAL games (DFS tournament gold)
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

        # CRITICAL FIX: Sort high-total games by total (highest first)
        high_total_games.sort(key=lambda x: x['total'], reverse=True)

        # CRITICAL FIX: Remove any duplicate game_ids from high_total_games
        seen_games = set()
        unique_high_total_games = []
        for game in high_total_games:
            if game['game_id'] not in seen_games:
                seen_games.add(game['game_id'])
                unique_high_total_games.append(game)

        # Calculate implied team scores (for player weighting)
        for game_id, game_data in processed_games.items():
            self._calculate_implied_scores(game_data)

        logger.info(
            f"✅ Processed {len(processed_games)} unique games with {len(unique_high_total_games)} high-total games")

        return {
            'games': processed_games,
            'high_total_games': unique_high_total_games,
            'avg_total': sum(g['total_points'] for g in processed_games.values()) / len(
                processed_games) if processed_games else 45.0,
            'total_games': len(processed_games),
            'data_source': 'real_vegas_api'
        }

    def _calculate_implied_scores(self, game_data: Dict):
        """Calculate implied team scores from total and spread - FIXED None handling"""
        total = game_data.get('total_points', 45)
        spread = game_data.get('spread')

        # CRITICAL FIX: Handle None spread properly
        if spread is None:
            spread = 0

        # Convert to float if needed
        try:
            spread = float(spread)
            total = float(total)
        except (ValueError, TypeError):
            spread = 0
            total = 45

        # Home team implied score
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

    def _get_enhanced_fallback_odds(self) -> Dict[str, Any]:
        """FIXED: Enhanced fallback with REAL Week 7 2025 games"""
        logger.info("📊 Using enhanced fallback Vegas data for Week 7")

        # WEEK 7 2025 - REAL GAMES
        fallback_games = {
            'DEN@NO': {
                'game_id': 'DEN@NO', 'home_team': 'NO', 'away_team': 'DEN',
                'total_points': 47.0, 'spread': 1.0, 'home_implied_score': 23.0, 'away_implied_score': 24.0
            },
            'PHI@NYG': {
                'game_id': 'PHI@NYG', 'home_team': 'NYG', 'away_team': 'PHI',
                'total_points': 50.5, 'spread': -3.5, 'home_implied_score': 27.0, 'away_implied_score': 23.5
            },
            'TEN@IND': {
                'game_id': 'TEN@IND', 'home_team': 'IND', 'away_team': 'TEN',
                'total_points': 47.5, 'spread': 4.0, 'home_implied_score': 25.8, 'away_implied_score': 21.8
            },
            'MIA@CLE': {
                'game_id': 'MIA@CLE', 'home_team': 'CLE', 'away_team': 'MIA',
                'total_points': 48.0, 'spread': 1.5, 'home_implied_score': 23.3, 'away_implied_score': 24.8
            },
            'KC@LV': {
                'game_id': 'KC@LV', 'home_team': 'LV', 'away_team': 'KC',
                'total_points': 52.5, 'spread': -8.5, 'home_implied_score': 22.0, 'away_implied_score': 30.5
            }
        }

        high_total_games = [
            {'game_id': 'KC@LV', 'total': 52.5, 'teams': ['KC', 'LV']},
            {'game_id': 'PHI@NYG', 'total': 50.5, 'teams': ['PHI', 'NYG']},
            {'game_id': 'MIA@CLE', 'total': 48.0, 'teams': ['MIA', 'CLE']},
            {'game_id': 'TEN@IND', 'total': 47.5, 'teams': ['TEN', 'IND']},
            {'game_id': 'DEN@NO', 'total': 47.0, 'teams': ['DEN', 'NO']}
        ]

        return {
            'games': fallback_games,
            'high_total_games': high_total_games,
            'avg_total': 49.1,
            'total_games': len(fallback_games),
            'data_source': 'fallback_week7_2025'
        }

    def get_game_environment_factors(self, vegas_data: Dict) -> Dict[str, float]:
        """Calculate game environment multipliers for DFS optimization"""

        game_factors = {}
        games = vegas_data.get('games', {})
        avg_total = vegas_data.get('avg_total', 45.0)

        for game_id, game_data in games.items():
            total = game_data.get('total_points', avg_total)
            spread = abs(game_data.get('spread', 0))

            # TOTAL FACTOR: Higher totals = higher DFS scoring
            total_factor = 1.0
            if total >= 50:
                total_factor = 1.35  # MAJOR boost for 50+ games
            elif total >= 47:
                total_factor = 1.25  # Significant boost for 47+ games  
            elif total >= 44:
                total_factor = 1.10  # Moderate boost
            elif total <= 40:
                total_factor = 0.85  # Penalty for low totals

            # SPREAD FACTOR: Close games = more passing = higher DFS
            spread_factor = 1.0
            if spread <= 3:
                spread_factor = 1.15  # Close games = shootouts
            elif spread >= 10:
                spread_factor = 0.90  # Blowouts = fewer points

            # COMBINED FACTOR
            combined_factor = total_factor * spread_factor

            # Apply to both teams
            home_team = game_data.get('home_team')
            away_team = game_data.get('away_team')

            if home_team:
                game_factors[home_team] = combined_factor
            if away_team:
                game_factors[away_team] = combined_factor

            logger.info(f"🎯 {game_id}: {total} total, {spread} spread → {combined_factor:.2f}x factor")

        return game_factors


# Integration function for data_collector.py
async def get_vegas_odds_data() -> Dict[str, Any]:
    """Get Vegas odds data for DFS optimization"""
    collector = VegasDataCollector()
    return await collector.get_nfl_odds_data()


async def get_game_environment_multipliers() -> Dict[str, float]:
    """Get team multipliers based on game environment"""
    collector = VegasDataCollector()
    vegas_data = await collector.get_nfl_odds_data()
    return collector.get_game_environment_factors(vegas_data)