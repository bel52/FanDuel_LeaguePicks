#!/usr/bin/env python3
"""
SIMPLIFIED: On-demand DFS optimization for friends league
No scheduling - you control when it runs
"""
import asyncio
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from config import API_PORT, LOGS_DIR
from data_collector import get_fresh_data
from optimizer import optimize_dfs_lineups


def setup_logging():
    """Simple logging setup"""
    logger.remove()
    logger.add(
        LOGS_DIR / "dfs_optimizer_{time:YYYY-MM-DD}.log",
        rotation="7 days",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    logger.add(sys.stderr, level="INFO")


async def generate_lineups(contest_type: str = 'gpp', num_lineups: int = 10):
    """Generate lineups on-demand with fresh data"""

    logger.info(f"🏈 Generating {num_lineups} {contest_type.upper()} lineups...")

    # Step 1: Collect ALL fresh data
    logger.info("📡 Collecting fresh data...")
    data = await get_fresh_data()

    if not data or not data.get('players'):
        logger.error("❌ No player data available. Make sure data/fanduel_salaries_manual.csv exists!")
        return None

    logger.info(f"✅ Loaded {len(data['players'])} players")

    # Step 2: Show data quality summary
    quality = data.get('data_quality', {})
    logger.info(f"📊 Data Quality:")
    logger.info(f"   • Week: {quality.get('current_week', 'Unknown')}")
    logger.info(f"   • Games: {quality.get('main_slate_games', 0)}")
    logger.info(f"   • Real projections: {quality.get('real_projections', 0)}")
    logger.info(f"   • Teams: {len(quality.get('teams_in_slate', []))}")

    # Step 3: Generate optimized lineups
    logger.info(f"🧠 Optimizing {contest_type} lineups...")
    lineups = optimize_dfs_lineups(
        player_data=data['players'],
        weather_data=data.get('weather', {}),
        vegas_multipliers=data.get('vegas_multipliers', {}),
        vegas_data=data.get('vegas_odds', {}),
        num_lineups=num_lineups,
        contest_type=contest_type
    )

    if not lineups:
        logger.error("❌ Optimization failed!")
        return None

    # Step 4: Display results with ENHANCED output for Step 8
    logger.info(f"✅ Generated {len(lineups)} {contest_type.upper()} lineups!")
    logger.info("=" * 80)

    for i, lineup in enumerate(lineups[:3]):  # Show first 3
        # Calculate average weather impact
        weather_avg = sum(p.weather_factor for p in lineup.players) / len(lineup.players)
        
        # Find stacked game (most common team pairing)
        from collections import Counter
        team_counts = Counter(p.team for p in lineup.players)
        top_teams = team_counts.most_common(2)
        if len(top_teams) >= 2 and top_teams[0][1] >= 2:
            stacked_game = f"{top_teams[0][0]}+{top_teams[1][0]}" if top_teams[1][1] >= 2 else top_teams[0][0]
        else:
            stacked_game = top_teams[0][0] if top_teams else "None"
        
        # Identify boom players (high ceiling or boom rate)
        boom_players = []
        for p in lineup.players:
            is_boom = False
            if p.monte_carlo_analyzed:
                ceiling_ratio = p.ceiling_90 / p.projection if p.projection > 0 else 1
                is_boom = (ceiling_ratio >= 1.15 and p.salary >= 7000) or (p.boom_rate >= 0.20)
            else:
                is_boom = (p.salary >= 8500 and p.projection >= 18)
            
            if is_boom:
                boom_players.append(p)
        
        logger.info(f"")
        logger.info(f"🏆 LINEUP {i + 1} SUMMARY:")
        logger.info(f"   Salary: ${lineup.total_salary:,} / $60,000")
        logger.info(f"   Projection: {lineup.projected_points:.1f} pts")
        logger.info(f"   Ceiling 90%: {lineup.ceiling_90:.1f} pts (+{lineup.ceiling_90 - lineup.projected_points:.1f})")
        logger.info(f"   Weather Impact: {weather_avg:.2f}x avg")
        logger.info(f"   Stacked Game: {stacked_game}")
        logger.info(f"   Boom Players ({len(boom_players)}): {', '.join(p.name for p in boom_players)}")
        logger.info(f"   Ownership: {lineup.ownership_total/9:.1f}% avg | Risk: {lineup.risk_level}")
        logger.info(f"")

        # Show players in FanDuel order
        logger.info("   ROSTER:")
        for j, player in enumerate(lineup.players):
            pos_label = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF'][j]
            
            # Mark boom players with emoji
            boom_marker = " 💥" if player in boom_players else ""
            
            # Show Monte Carlo data if available
            if player.monte_carlo_analyzed and player.ceiling_90 > 0:
                logger.info(f"   {pos_label:5} {player.name:20} ${player.salary:5,} | "
                           f"{player.projection:4.1f}→{player.ceiling_90:4.1f}pts{boom_marker}")
            else:
                logger.info(f"   {pos_label:5} {player.name:20} ${player.salary:5,} | "
                           f"{player.projection:4.1f}pts{boom_marker}")
        
        logger.info("=" * 80)

    # Step 5: Export to organized CSV files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create organized directory structure
    lineup_dir = Path("data/lineups")
    week_dir = lineup_dir / f"week_{quality.get('current_week', 'unknown')}"
    week_dir.mkdir(parents=True, exist_ok=True)

    csv_file = week_dir / f"{contest_type}_lineups_{timestamp}.csv"

    # Create CSV export
    lineup_data = []
    for i, lineup in enumerate(lineups):
        lineup_row = {
            'Lineup': i + 1,
            'QB': f"{lineup.players[0].name}",
            'RB1': f"{lineup.players[1].name}",
            'RB2': f"{lineup.players[2].name}",
            'WR1': f"{lineup.players[3].name}",
            'WR2': f"{lineup.players[4].name}",
            'WR3': f"{lineup.players[5].name}",
            'TE': f"{lineup.players[6].name}",
            'FLEX': f"{lineup.players[7].name}",
            'DEF': f"{lineup.players[8].name}",
            'Salary': lineup.total_salary,
            'Projected': round(lineup.projected_points, 1),
            'Ceiling90': round(lineup.ceiling_90, 1),
            'Uniqueness': round(max(0, 100 - (lineup.ownership_total / 3.6)), 0),
            'Avg_Ownership': round(lineup.ownership_total / 9, 1)
        }
        lineup_data.append(lineup_row)

    import pandas as pd
    df = pd.DataFrame(lineup_data)
    df.to_csv(csv_file, index=False)

    logger.info(f"💾 Exported to: {csv_file}")
    logger.info(f"📁 Organized in: {week_dir}")

    return lineups


def run_web_interface():
    """Run the web interface"""
    try:
        from app import app
        import uvicorn

        logger.info(f"🌐 Starting web interface on http://localhost:{API_PORT}")
        logger.info("💡 Use this for interactive lineup generation")

        uvicorn.run("app:app", host="0.0.0.0", port=API_PORT, reload=False, log_level="info")

    except Exception as e:
        logger.error(f"❌ Web interface failed: {e}")
        return False

    return True


async def test_system():
    """Test the system end-to-end with constraint verification"""
    logger.info("🧪 Testing DFS system...")

    try:
        # Test data collection
        logger.info("Testing data collection...")
        data = await get_fresh_data()

        if data and data.get('players'):
            player_count = len(data['players'])
            week = data.get('data_quality', {}).get('current_week', 'Unknown')
            logger.info(f"✅ Data collection: {player_count} players, Week {week}")
        else:
            logger.error("❌ Data collection failed")
            return False

        # Test optimization with constraint verification
        logger.info("Testing lineup optimization with friends_league constraints...")
        lineups = optimize_dfs_lineups(
            player_data=data['players'],
            weather_data=data.get('weather', {}),
            vegas_multipliers=data.get('vegas_multipliers', {}),
            vegas_data=data.get('vegas_odds', {}),
            num_lineups=2,
            contest_type='friends_league'
        )

        if lineups:
            logger.info(f"✅ Optimization: Generated {len(lineups)} test lineups")
            
            # VERIFY STEP 6 CONSTRAINTS
            logger.info("")
            logger.info("🔍 VERIFYING FRIENDS LEAGUE CONSTRAINTS:")
            
            for i, lineup in enumerate(lineups):
                logger.info(f"")
                logger.info(f"Lineup {i+1}:")
                
                # Get Vegas data
                vegas_data = data.get('vegas_odds', {})
                high_total_games = vegas_data.get('high_total_games', [])
                
                if high_total_games:
                    top_game = high_total_games[0]
                    top_game_teams = top_game.get('teams', [])
                    
                    # Count players from top Vegas game
                    players_from_top_game = [p for p in lineup.players if p.team in top_game_teams]
                    logger.info(f"   ✓ Vegas constraint: {len(players_from_top_game)} players from {top_game['game_id']} "
                               f"(total {top_game['total']} pts) - {'PASS' if 3 <= len(players_from_top_game) <= 4 else 'FAIL'}")
                
                # Count boom players
                boom_count = 0
                for p in lineup.players:
                    if p.monte_carlo_analyzed:
                        ceiling_ratio = p.ceiling_90 / p.projection if p.projection > 0 else 1
                        is_boom = (ceiling_ratio >= 1.15 and p.salary >= 7000) or (p.boom_rate >= 0.20)
                    else:
                        is_boom = (p.salary >= 8500 and p.projection >= 18)
                    
                    if is_boom:
                        boom_count += 1
                
                logger.info(f"   ✓ Boom constraint: {boom_count} boom candidates - {'PASS' if boom_count >= 3 else 'FAIL'}")
                
                # Count studs ($9K+)
                stud_count = sum(1 for p in lineup.players if p.salary >= 9000)
                logger.info(f"   ✓ Stud constraint: {stud_count} players ≥$9K - {'PASS' if stud_count >= 1 else 'FAIL'}")
                
                logger.info(f"   Total salary: ${lineup.total_salary:,}")
                logger.info(f"   Projected: {lineup.projected_points:.1f} pts")
        else:
            logger.error("❌ Optimization failed")
            return False

        logger.info("")
        logger.info("✅ All systems working!")
        return True

    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Simplified main function"""
    parser = argparse.ArgumentParser(description="FanDuel DFS Optimizer for Friends League")
    parser.add_argument('mode', choices=['gpp', 'cash', 'contrarian', 'friends_league', 'web', 'test'],
                        help='Contest type or special mode')
    parser.add_argument('-n', '--num-lineups', type=int, default=10,
                        help='Number of lineups to generate (default: 10)')
    parser.add_argument('--no-ai', action='store_true',
                        help='Disable AI analysis to save API costs')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Ensure directories exist
    Path('data/lineups').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    # ⚡ SET AI FLAG FIRST - BEFORE ANY MODE LOGIC
    if args.no_ai:
        os.environ['AI_ENABLED'] = 'false'
        print("🚫 AI analysis disabled (--no-ai flag)")
    else:
        os.environ['AI_ENABLED'] = 'true'
    
    print(f"""
🏈 FanDuel Friends League Optimizer
====================================
Mode: {args.mode.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

📋 SIMPLE WORKFLOW:
1. Download FanDuel salary CSV manually
2. Save as: data/fanduel_salaries_manual.csv  
3. Run: python main.py {args.mode}
4. Use generated lineups manually or export CSV
""")
    
    try:
        if args.mode == 'web':
            success = run_web_interface()
        elif args.mode == 'test':
            success = asyncio.run(test_system())
        else:
            # Generate lineups for contest type
            lineups = asyncio.run(generate_lineups(args.mode, args.num_lineups))
            success = lineups is not None
        
        if success:
            logger.info("✅ Operation completed successfully!")
            print("\n🎯 Next steps:")
            print("1. Review generated lineups")
            print("2. Enter lineups manually or upload CSV")
            print("3. Dominate your friends! 🏆")
        else:
            logger.error("❌ Operation failed!")

    except KeyboardInterrupt:
        logger.info("👋 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
