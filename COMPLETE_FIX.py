# ============================================================================
# STEP 1: Add these two methods to EnhancedDataCollector class
# Add after line ~450 (after get_vegas_odds_data method)
# ============================================================================

    async def filter_vegas_to_csv_games(self, vegas_data: Dict, csv_games: List[str]) -> Dict:
        """Filter Vegas data to ONLY games that exist in the CSV"""
        
        if not csv_games:
            logger.warning("No CSV games provided for Vegas filtering")
            return vegas_data
        
        # Normalize CSV games (both directions: DAL@CAR and CAR@DAL)
        normalized_csv_games = set()
        for game in csv_games:
            if '@' in game:
                away, home = game.split('@')
                normalized_csv_games.add(f"{away}@{home}")
                normalized_csv_games.add(f"{home}@{away}")  # Reverse too
        
        logger.info(f"🎯 CSV has {len(csv_games)} games: {sorted(csv_games)}")
        
        # Filter Vegas games
        filtered_games = {}
        filtered_high_total = []
        
        original_games = vegas_data.get('games', {})
        original_high_total = vegas_data.get('high_total_games', [])
        
        for game_id, game_data in original_games.items():
            if game_id in normalized_csv_games:
                filtered_games[game_id] = game_data
                logger.info(f"✅ Matched Vegas game: {game_id}")
        
        for high_game in original_high_total:
            game_id = high_game.get('game_id', '')
            if game_id in normalized_csv_games:
                filtered_high_total.append(high_game)
                logger.info(f"🔥 Matched high-total game: {game_id} ({high_game.get('total')} pts)")
        
        # If NO matches, create estimates from CSV games
        if not filtered_games:
            logger.warning("⚠️ No Vegas games matched CSV - creating estimates")
            return self._create_estimated_vegas_from_csv(csv_games)
        
        # Sort high-total games by total (highest first)
        filtered_high_total.sort(key=lambda x: x.get('total', 0), reverse=True)
        
        logger.info(f"✅ Filtered to {len(filtered_games)} Vegas games matching CSV")
        logger.info(f"🔥 {len(filtered_high_total)} high-total games (47+)")
        
        return {
            'games': filtered_games,
            'high_total_games': filtered_high_total,
            'avg_total': sum(g.get('total_points', 45) for g in filtered_games.values()) / len(filtered_games) if filtered_games else 45.0,
            'total_games': len(filtered_games),
            'data_source': 'filtered_to_csv'
        }

    def _create_estimated_vegas_from_csv(self, csv_games: List[str]) -> Dict:
        """Create estimated Vegas lines when API doesn't match CSV games"""
        
        logger.info("📊 Creating estimated Vegas lines for CSV games")
        
        estimated_games = {}
        high_total_games = []
        
        # Reasonable defaults based on typical NFL scoring
        for game in csv_games:
            if '@' not in game:
                continue
            
            away, home = game.split('@')
            
            # Default: 45 total, -3 home favorite
            estimated_games[game] = {
                'game_id': game,
                'home_team': home,
                'away_team': away,
                'total_points': 45.0,
                'spread': -3.0,
                'home_implied_score': 24.0,
                'away_implied_score': 21.0
            }
            
            # Mark high-scoring teams with higher totals
            high_scoring_teams = ['KC', 'BUF', 'DAL', 'DET', 'SF', 'MIA', 'LAR', 'CIN']
            if home in high_scoring_teams or away in high_scoring_teams:
                estimated_games[game]['total_points'] = 48.0
                estimated_games[game]['home_implied_score'] = 25.5
                estimated_games[game]['away_implied_score'] = 22.5
                
                high_total_games.append({
                    'game_id': game,
                    'total': 48.0,
                    'teams': [away, home]
                })
                logger.info(f"🔥 Estimated high-total: {game} (48.0 pts)")
        
        # Sort high-total by total
        high_total_games.sort(key=lambda x: x['total'], reverse=True)
        
        return {
            'games': estimated_games,
            'high_total_games': high_total_games,
            'avg_total': 45.0,
            'total_games': len(estimated_games),
            'data_source': 'csv_estimated'
        }


# ============================================================================
# STEP 2: Update get_fresh_data() function (around line 853)
# REPLACE these lines:
#     vegas_data = await collector.get_vegas_odds_data()
#     vegas_multipliers = collector.calculate_vegas_multipliers(vegas_data)
# 
# WITH:
# ============================================================================

        # Get Vegas data
        vegas_data = await collector.get_vegas_odds_data()
        
        # Extract unique games from CSV players
        csv_games = list(set(p.get('game', '') for p in players if p.get('game')))
        logger.info(f"📋 CSV games from players: {sorted(csv_games)}")
        
        # Filter Vegas data to match CSV games
        vegas_data = await collector.filter_vegas_to_csv_games(vegas_data, csv_games)
        
        # Calculate multipliers from filtered data
        vegas_multipliers = collector.calculate_vegas_multipliers(vegas_data)

