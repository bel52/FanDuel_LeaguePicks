"""
FIXED: Injury Opportunity Detection with proper team mapping and conservative boosts
Only identifies REAL opportunities that can win tournaments
"""
from typing import Dict, List, Any
from loguru import logger

class InjuryOpportunityDetector:
    """Detects REAL DFS value opportunities created by injuries"""

    def __init__(self):
        # Position hierarchies for NFL teams
        self.position_hierarchies = {
            'RB': ['RB1', 'RB2', 'RB3'],
            'WR': ['WR1', 'WR2', 'WR3', 'WR4'],
            'TE': ['TE1', 'TE2']
        }

        # Only count these as creating opportunities
        self.out_statuses = ['IR', 'OUT', 'SUSP', 'DOUBTFUL']
        self.questionable_statuses = ['Q', 'QUESTIONABLE']

    def analyze_injury_opportunities(self, players: List[Dict]) -> List[Dict]:
        """Find players with REAL increased opportunity due to injuries"""

        opportunities = []

        # Group players by team and position
        team_positions = {}
        for player in players:
            team = player.get('team', '').strip().upper()
            position = player.get('position', '')

            if team not in team_positions:
                team_positions[team] = {}
            if position not in team_positions[team]:
                team_positions[team][position] = []

            team_positions[team][position].append(player)

        # ONLY analyze skill positions that matter
        for team, positions in team_positions.items():
            for position in ['RB', 'WR', 'TE']:
                if position in positions:
                    opps = self._find_positional_opportunities(team, position, positions[position])
                    opportunities.extend(opps)

        logger.info(f"Found {len(opportunities)} REAL injury opportunities")
        return opportunities

    def _find_positional_opportunities(self, team: str, position: str,
                                     players: List[Dict]) -> List[Dict]:
        """Find opportunities within a position group - CONSERVATIVE approach"""

        # Sort by salary (proxy for depth chart position)
        sorted_players = sorted(players, key=lambda p: p.get('salary', 0), reverse=True)

        opportunities = []
        injured_starters = []

        # ONLY top 2 players by salary count as "starters"
        for i, player in enumerate(sorted_players[:2]):
            injury_status = player.get('injury_status', '').strip().upper()
            salary = player.get('salary', 0)

            # Must be high-salary AND definitively out
            if any(status in injury_status for status in self.out_statuses) and salary >= 6000:
                injured_starters.append({
                    'player': player,
                    'depth_position': i + 1,
                    'status': 'OUT',
                    'team': team  # FIXED: Track team properly
                })
                logger.info(f"REAL injured starter: {player.get('name')} ({team} {position}) - {injury_status}")

        # Find backups who benefit
        if injured_starters:
            opportunities.extend(self._identify_beneficiaries(
                team, position, sorted_players, injured_starters
            ))

        return opportunities

    def _identify_beneficiaries(self, team: str, position: str,
                              all_players: List[Dict],
                              injured_starters: List[Dict]) -> List[Dict]:
        """Identify players who ACTUALLY benefit from starter injuries"""

        beneficiaries = []

        for injured in injured_starters:
            injured_player = injured['player']
            injured_salary = injured_player.get('salary', 0)
            injured_team = injured.get('team', '')

            # Look for SAME TEAM players who benefit
            for player in all_players:
                player_name = player.get('name', '')
                player_team = player.get('team', '').strip().upper()
                player_salary = player.get('salary', 0)

                # Skip the injured player
                if player_name == injured_player.get('name'):
                    continue

                # CRITICAL: Must be same team
                if player_team != injured_team:
                    continue

                # Must be significantly cheaper (backup tier)
                salary_ratio = player_salary / injured_salary if injured_salary > 0 else 0

                # CONSERVATIVE: 30-70% of injured player's salary
                if 0.3 <= salary_ratio <= 0.7:
                    opportunity_score = self._calculate_opportunity_score(
                        player, injured, position, team
                    )

                    # HIGHER threshold - only obvious opportunities
                    if opportunity_score > 0.8:
                        projected_boost = self._calculate_projection_boost(
                            player, injured, position
                        )

                        # ONLY boost if meaningful (>2 points)
                        if projected_boost >= 2.0:
                            beneficiaries.append({
                                'player': player,
                                'injured_starter': injured_player,
                                'opportunity_score': opportunity_score,
                                'salary_discount': 1 - salary_ratio,
                                'projected_boost': projected_boost,
                                'reason': f"Backup to injured {injured_player.get('name')}"
                            })

                            logger.info(f"REAL opportunity: {player_name} ({team} {position}) "
                                      f"- {opportunity_score:.2f} score, "
                                      f"+{projected_boost:.1f} pts boost")

        return beneficiaries

    def _calculate_opportunity_score(self, backup_player: Dict,
                                   injured_starter: Dict,
                                   position: str, team: str) -> float:
        """Calculate opportunity score - CONSERVATIVE"""

        score = 0.0

        # Base score by position (lower than before)
        position_base_scores = {
            'RB': 0.6,  # RBs still get good boost
            'WR': 0.4,  # WRs share targets more
            'TE': 0.5   # TEs moderate boost
        }
        score += position_base_scores.get(position, 0.3)

        # Injury severity boost
        injured_status = injured_starter.get('status', '')
        if injured_status == 'OUT':
            score += 0.4  # Higher boost for definite outs
        elif injured_status == 'QUESTIONABLE':
            score += 0.1  # Minimal boost for questionable

        # Salary efficiency (backup must be cheap)
        backup_salary = backup_player.get('salary', 5000)
        if backup_salary < 4500:  # Very cheap
            score += 0.3
        elif backup_salary < 5500:  # Moderately cheap
            score += 0.1

        return min(1.0, score)

    def _calculate_projection_boost(self, backup_player: Dict,
                                  injured_starter: Dict,
                                  position: str) -> float:
        """Calculate CONSERVATIVE projection boost"""

        base_projection = backup_player.get('projected_points', 0)
        injured_projection = injured_starter['player'].get('projected_points', 0)

        # CONSERVATIVE boost factors
        boost_factors = {
            'RB': 0.4,  # Backup RB gets 40% of starter's projection
            'WR': 0.25, # WR gets 25% boost (targets shared)
            'TE': 0.3   # TE gets 30% boost
        }

        boost_factor = boost_factors.get(position, 0.2)
        projected_boost = injured_projection * boost_factor

        # Cap boost at reasonable levels
        max_boost = {
            'RB': 8.0,   # Max 8 point boost for RB
            'WR': 6.0,   # Max 6 point boost for WR
            'TE': 5.0    # Max 5 point boost for TE
        }

        return min(projected_boost, max_boost.get(position, 4.0))

    def apply_injury_boosts(self, players: List[Dict]) -> List[Dict]:
        """Apply CONSERVATIVE injury opportunity boosts"""

        opportunities = self.analyze_injury_opportunities(players)

        # Create lookup for quick access
        boost_lookup = {}
        for opp in opportunities:
            player_name = opp['player'].get('name', '')
            boost_lookup[player_name] = opp

        # Apply boosts
        boosted_players = []
        boost_count = 0

        for player in players:
            player_name = player.get('name', '')

            if player_name in boost_lookup:
                opp = boost_lookup[player_name]

                # Apply projection boost
                original_projection = player.get('projected_points', 0)
                boost_amount = opp['projected_boost']

                player['projected_points'] = original_projection + boost_amount
                player['projection'] = original_projection + boost_amount

                # Add opportunity metadata
                player['injury_opportunity'] = True
                player['opportunity_score'] = opp['opportunity_score']
                player['injured_starter'] = opp['injured_starter'].get('name', '')
                player['boost_reason'] = opp['reason']

                boost_count += 1
                logger.info(f"BOOSTED: {player_name} "
                          f"{original_projection:.1f} → {player['projected_points']:.1f} pts "
                          f"(+{boost_amount:.1f}) - {opp['reason']}")

            boosted_players.append(player)

        logger.info(f"Applied {boost_count} CONSERVATIVE injury boosts")
        return boosted_players

# Integration function for data_collector.py
def enhance_players_with_injury_opportunities(players: List[Dict]) -> List[Dict]:
    """Enhance player data with CONSERVATIVE injury opportunity analysis"""

    detector = InjuryOpportunityDetector()
    enhanced_players = detector.apply_injury_boosts(players)

    # Log summary
    opportunity_count = sum(1 for p in enhanced_players if p.get('injury_opportunity', False))
    logger.info(f"Enhanced {opportunity_count} players with REAL injury opportunities")

    return enhanced_players