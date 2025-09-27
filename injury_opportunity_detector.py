"""
Injury Opportunity Detection for DFS Optimization
Identifies high-value players whose role increases due to injuries
"""
from typing import Dict, List, Any
from loguru import logger

class InjuryOpportunityDetector:
    """Detects DFS value opportunities created by injuries"""

    def __init__(self):
        # Define positional hierarchies for NFL teams
        self.position_hierarchies = {
            'RB': ['RB1', 'RB2', 'RB3'],
            'WR': ['WR1', 'WR2', 'WR3', 'WR4'],
            'TE': ['TE1', 'TE2']
        }

        # Injury status that creates opportunities
        self.out_statuses = ['OUT', 'IR', 'SUSP', 'DOUBTFUL']
        self.questionable_statuses = ['Q', 'QUESTIONABLE']

    def analyze_injury_opportunities(self, players: List[Dict]) -> List[Dict]:
        """Find players with increased opportunity due to injuries"""

        opportunities = []

        # Group players by team and position
        team_positions = {}
        for player in players:
            team = player.get('team', '')
            position = player.get('position', '')

            if team not in team_positions:
                team_positions[team] = {}
            if position not in team_positions[team]:
                team_positions[team][position] = []

            team_positions[team][position].append(player)

        # Analyze each team's depth chart
        for team, positions in team_positions.items():
            for position, team_players in positions.items():
                if position in ['RB', 'WR', 'TE']:
                    opps = self._find_positional_opportunities(team, position, team_players)
                    opportunities.extend(opps)

        logger.info(f"Found {len(opportunities)} injury-driven opportunities")
        return opportunities

    def _find_positional_opportunities(self, team: str, position: str,
                                     players: List[Dict]) -> List[Dict]:
        """Find opportunities within a position group"""

        # Sort by salary (proxy for depth chart position)
        sorted_players = sorted(players, key=lambda p: p.get('salary', 0), reverse=True)

        opportunities = []
        injured_starters = []

        # Identify injured high-salary players
        for i, player in enumerate(sorted_players[:3]):  # Top 3 by salary
            injury_status = player.get('injury_status', '').upper()
            salary = player.get('salary', 0)

            # High-salary player who's out/questionable
            if any(status in injury_status for status in self.out_statuses):
                injured_starters.append({
                    'player': player,
                    'depth_position': i + 1,
                    'status': 'OUT'
                })
                logger.info(f"Injured starter: {player.get('name')} ({team} {position}) - {injury_status}")

            elif any(status in injury_status for status in self.questionable_statuses) and salary > 7000:
                injured_starters.append({
                    'player': player,
                    'depth_position': i + 1,
                    'status': 'QUESTIONABLE'
                })

        # Find backups who benefit
        if injured_starters:
            opportunities.extend(self._identify_beneficiaries(
                team, position, sorted_players, injured_starters
            ))

        return opportunities

    def _identify_beneficiaries(self, team: str, position: str,
                              all_players: List[Dict],
                              injured_starters: List[Dict]) -> List[Dict]:
        """Identify players who benefit from starter injuries"""

        beneficiaries = []

        for injured in injured_starters:
            injured_player = injured['player']
            injured_salary = injured_player.get('salary', 0)

            # Look for players 2-4 salary tiers below the injured player
            for player in all_players:
                player_salary = player.get('salary', 0)
                player_name = player.get('name', '')

                # Skip the injured player
                if player_name == injured_player.get('name'):
                    continue

                # Backup criteria: significantly cheaper but in same position
                salary_ratio = player_salary / injured_salary if injured_salary > 0 else 0

                if 0.3 <= salary_ratio <= 0.8:  # 30-80% of injured player's salary
                    opportunity_score = self._calculate_opportunity_score(
                        player, injured, position, team
                    )

                    if opportunity_score > 0.6:  # High opportunity threshold
                        beneficiaries.append({
                            'player': player,
                            'injured_starter': injured_player,
                            'opportunity_score': opportunity_score,
                            'salary_discount': 1 - salary_ratio,
                            'projected_boost': self._calculate_projection_boost(
                                player, injured, position
                            ),
                            'reason': f"Backup to injured {injured_player.get('name')}"
                        })

                        logger.info(f"Opportunity: {player_name} ({team} {position}) "
                                  f"- {opportunity_score:.2f} score, "
                                  f"{(1-salary_ratio)*100:.0f}% salary discount")

        return beneficiaries

    def _calculate_opportunity_score(self, backup_player: Dict,
                                   injured_starter: Dict,
                                   position: str, team: str) -> float:
        """Calculate opportunity score for a backup player"""

        score = 0.0

        # Base score by position
        position_base_scores = {
            'RB': 0.8,  # RBs get highest boost when starter is out
            'WR': 0.6,  # WRs share targets
            'TE': 0.7   # TEs often step into bigger role
        }
        score += position_base_scores.get(position, 0.5)

        # Injury severity boost
        injured_status = injured_starter.get('status', '')
        if injured_status == 'OUT':
            score += 0.3
        elif injured_status == 'QUESTIONABLE':
            score += 0.1

        # Salary efficiency boost
        backup_salary = backup_player.get('salary', 5000)
        if backup_salary < 5000:  # Very cheap
            score += 0.2
        elif backup_salary < 6500:  # Moderately cheap
            score += 0.1

        return min(1.0, score)

    def _calculate_projection_boost(self, backup_player: Dict,
                                  injured_starter: Dict,
                                  position: str) -> float:
        """Calculate expected projection boost"""

        base_projection = backup_player.get('projected_points', 0)
        injured_projection = injured_starter['player'].get('projected_points', 0)

        # Estimate boost based on position
        boost_factors = {
            'RB': 0.6,  # Backup RB gets 60% of starter's projection
            'WR': 0.4,  # WR gets 40% boost due to target share
            'TE': 0.5   # TE gets moderate boost
        }

        boost_factor = boost_factors.get(position, 0.3)
        projected_boost = injured_projection * boost_factor

        return min(projected_boost, base_projection * 2.0)  # Cap at 2x current projection

    def apply_injury_boosts(self, players: List[Dict]) -> List[Dict]:
        """Apply injury opportunity boosts to player projections"""

        opportunities = self.analyze_injury_opportunities(players)

        # Create lookup for quick access
        boost_lookup = {}
        for opp in opportunities:
            player_name = opp['player'].get('name', '')
            boost_lookup[player_name] = opp

        # Apply boosts
        boosted_players = []
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

                logger.info(f"BOOSTED: {player_name} "
                          f"{original_projection:.1f} → {player['projected_points']:.1f} pts "
                          f"(+{boost_amount:.1f}) - {opp['reason']}")

            boosted_players.append(player)

        return boosted_players

# Integration function for data_collector.py
def enhance_players_with_injury_opportunities(players: List[Dict]) -> List[Dict]:
    """Enhance player data with injury opportunity analysis"""

    detector = InjuryOpportunityDetector()
    enhanced_players = detector.apply_injury_boosts(players)

    # Log summary
    opportunity_count = sum(1 for p in enhanced_players if p.get('injury_opportunity', False))
    logger.info(f"Enhanced {opportunity_count} players with injury opportunities")

    return enhanced_players