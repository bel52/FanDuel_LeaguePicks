#!/usr/bin/env python3
"""
SIMPLIFIED: On-demand DFS optimization for friends league
Removes complex scheduling - you control when it runs
"""
import asyncio
import sys
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
        num_lineups=num_lineups,
        contest_type=contest_type
    )

    if not lineups:
        logger.error("❌ Optimization failed!")
        return None

    # Step 4: Display results
    logger.info(f"✅ Generated {len(lineups)} {contest_type.upper()} lineups!")

    for i, lineup in enumerate(lineups[:3]):  # Show first 3
        logger.info(
            f"Lineup {i + 1}: ${lineup.total_salary:,} | {lineup.projected_points:.1f} pts | {lineup.ownership_total:.1f}% owned")

        # Show players in FanDuel order
        for j, player in enumerate(lineup.players):
            pos_label = ['QB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE', 'FLEX', 'DEF'][j]
            logger.info(f"  {pos_label}: {player.name} (${player.salary:,})")

    # Step 5: Export to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = f"data/lineups/{contest_type}_lineups_{timestamp}.csv"

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
            'Ownership': round(lineup.ownership_total, 1)
        }
        lineup_data.append(lineup_row)

    import pandas as pd
    df = pd.DataFrame(lineup_data)
    df.to_csv(csv_file, index=False)

    logger.info(f"💾 Exported to: {csv_file}")
    logger.info(f"📁 Ready to upload to FanDuel!")

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
    """Test the system end-to-end"""
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

        # Test optimization
        logger.info("Testing lineup optimization...")
        lineups = optimize_dfs_lineups(
            player_data=data['players'][:50],  # Use subset for speed
            weather_data={},
            vegas_multipliers={},
            num_lineups=2,
            contest_type='gpp'
        )

        if lineups:
            logger.info(f"✅ Optimization: Generated {len(lineups)} test lineups")
        else:
            logger.error("❌ Optimization failed")
            return False

        logger.info("✅ All systems working!")
        return True

    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False


def main():
    """Simplified main function"""
    parser = argparse.ArgumentParser(description="FanDuel DFS Optimizer for Friends League")
    parser.add_argument('mode', choices=['gpp', 'cash', 'contrarian', 'web', 'test'],
                        help='Operation mode')
    parser.add_argument('-n', '--num-lineups', type=int, default=10,
                        help='Number of lineups to generate (default: 10)')

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Ensure directories exist
    Path('data/lineups').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)

    print(f"""
🏈 FanDuel Friends League Optimizer
====================================
Mode: {args.mode.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

📋 USAGE WORKFLOW:
1. Download FanDuel salary CSV manually
2. Save as: data/fanduel_salaries_manual.csv  
3. Run: python main.py {args.mode}
4. Upload generated CSV to FanDuel
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
            print("2. Upload CSV to FanDuel")
            print("3. Dominate your friends! 🏆")
        else:
            logger.error("❌ Operation failed!")

    except KeyboardInterrupt:
        logger.info("👋 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()