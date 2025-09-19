#!/usr/bin/env python3
import asyncio
import argparse
import sys
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from config import config
from data_collector import DataCollector
from optimizer import DFSOptimizer, OptimizationSettings
from ai_analyzer import AIAnalyzer
from agents import AgentOrchestrator
from models import Player, Lineup
from database import db
from utils import format_currency

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dfs_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DFSOptimization:
    def __init__(self):
        self.collector = DataCollector()
        self.optimizer = DFSOptimizer()
        self.ai_analyzer = AIAnalyzer()
        self.orchestrator = AgentOrchestrator()
        
    async def run_single_optimization(self, contest_type: str = "balanced", num_lineups: int = 5):
        """Run a single optimization cycle"""
        print("\n" + "="*50)
        print(f"DFS Lineup Optimization - {contest_type.upper()}")
        print("="*50 + "\n")
        
        # Collect latest data
        print("📊 Collecting data from multiple sources...")
        async with self.collector as collector:
            data = await collector.collect_all_data()
        
        # Get DFS salaries (implement based on your source)
        print("💰 Fetching current DFS salaries...")
        async with self.collector as collector:
            players = await collector.fetch_dfs_salaries()
        
        if not players:
            print("❌ No players found. Please check data sources.")
            return
        
        print(f"✅ Found {len(players)} players\n")
        
        # Get AI analysis if available
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            print("🤖 Running AI analysis...")
            games = data.get('vegas', {}).get('games', [])
            ai_insights = await self.ai_analyzer.analyze_slate(players, games)
            print("✅ AI analysis complete\n")
        else:
            print("ℹ️ AI analysis skipped (no API key configured)\n")
            ai_insights = {}
        
        # Configure optimization settings
        settings = OptimizationSettings(
            lineup_type=contest_type,
            num_lineups=num_lineups,
            max_exposure=0.4 if contest_type == "gpp" else 0.6,
            min_salary=58000,
            correlation_rules=True,
            weather_adjustments=True
        )
        
        if contest_type == "gpp" and ai_insights:
            settings.stack_rules = {
                'qb_stack': True,
                'game_stack': True
            }
        
        # Optimize lineups
        print(f"🔧 Optimizing {num_lineups} lineups...")
        lineups = self.optimizer.optimize(players, settings)
        
        if not lineups:
            print("❌ Optimization failed. Please check constraints.")
            return
        
        print(f"✅ Generated {len(lineups)} optimized lineups\n")
        
        # Display lineups
        for i, lineup in enumerate(lineups, 1):
            self.display_lineup(lineup, i)
            
            # Run Monte Carlo for variance analysis
            if contest_type == "gpp":
                print("  📈 Running Monte Carlo simulation...")
                simulation = self.optimizer.run_monte_carlo_simulation(lineup, iterations=10000)
                print(f"  Expected: {simulation['mean']:.1f} ± {simulation['std_dev']:.1f}")
                print(f"  90th percentile: {simulation['percentiles']['90th']:.1f}")
                print(f"  Prob 150+: {simulation['probability_150plus']*100:.1f}%")
                print(f"  Prob 175+: {simulation['probability_175plus']*100:.1f}%\n")
            
            # Save to database
            self.save_lineup(lineup, contest_type)
        
        print("✅ All lineups saved to database")
        
        # Display AI insights
        if ai_insights:
            self.display_ai_insights(ai_insights)
    
    async def run_continuous(self):
        """Run continuous agent-based optimization"""
        print("\n" + "="*50)
        print("DFS Agent System - Continuous Mode")
        print("="*50 + "\n")
        
        print("🚀 Starting agent orchestrator...")
        print("   - Data collection every 5 minutes")
        print("   - Lineup optimization every 15 minutes")
        print("   - News monitoring every 2 minutes")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            await self.orchestrator.start()
        except KeyboardInterrupt:
            print("\n⛔ Stopping agents...")
            await self.orchestrator.stop()
            print("✅ Shutdown complete")
    
    def display_lineup(self, lineup: Lineup, number: int):
        """Display a lineup in readable format"""
        print(f"\n{'='*30}")
        print(f"LINEUP #{number}")
        print(f"{'='*30}")
        
        positions_order = ['QB', 'RB', 'WR', 'TE', 'DST']
        position_counts = {'RB': 0, 'WR': 0, 'TE': 0}
        
        for position in positions_order:
            position_players = [p for p in lineup.players if p.position.value == position]
            
            for player in position_players:
                # Handle FLEX positions
                if position in ['RB', 'WR', 'TE']:
                    position_counts[position] += 1
                    if position == 'RB' and position_counts[position] > 2:
                        display_pos = 'FLEX'
                    elif position == 'WR' and position_counts[position] > 3:
                        display_pos = 'FLEX'
                    elif position == 'TE' and position_counts[position] > 1:
                        display_pos = 'FLEX'
                    else:
                        display_pos = position
                else:
                    display_pos = position
                
                print(f"  {display_pos:4} {player.name:20} {player.team:3} "
                      f"${player.salary:,}  {player.projected_points:.1f}pts")
        
        print(f"\n  Total Salary: ${lineup.total_salary:,} / $60,000")
        print(f"  Projected: {lineup.total_projected:.1f} points")
        if lineup.stack_score:
            print(f"  Stack Score: {lineup.stack_score:.1f}")
        if lineup.ownership_sum:
            print(f"  Total Ownership: {lineup.ownership_sum:.1f}%")
    
    def display_ai_insights(self, insights: dict):
        """Display AI analysis insights"""
        print("\n" + "="*50)
        print("AI ANALYSIS INSIGHTS")
        print("="*50 + "\n")
        
        if 'game_stacks' in insights:
            print("📊 RECOMMENDED GAME STACKS:")
            for stack in insights['game_stacks'][:3]:
                print(f"  • {stack.get('game', 'N/A')} (O/U: {stack.get('total', 'N/A')})")
                if 'reasoning' in stack:
                    print(f"    {stack['reasoning']}")
        
        if 'leverage_plays' in insights:
            print("\n💎 LEVERAGE PLAYS:")
            for play in insights['leverage_plays'][:5]:
                print(f"  • {play.get('player', 'N/A')} "
                      f"({play.get('projected_ownership', 'N/A')}% owned)")
        
        if 'fade_candidates' in insights:
            print("\n⚠️ CONSIDER FADING:")
            for fade in insights['fade_candidates'][:3]:
                print(f"  • {fade.get('player', 'N/A')} - {fade.get('concerns', 'N/A')}")
    
    def save_lineup(self, lineup: Lineup, contest_type: str):
        """Save lineup to database"""
        lineup_data = {
            'slate_id': datetime.now().strftime('%Y%m%d'),
            'players': [p.dict() for p in lineup.players],
            'total_salary': lineup.total_salary,
            'projected_points': lineup.total_projected,
            'lineup_type': contest_type,
            'created_at': datetime.now()
        }
        db.save_lineup(lineup_data)

async def main():
    parser = argparse.ArgumentParser(description='FanDuel DFS Lineup Optimizer')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single',
                      help='Optimization mode')
    parser.add_argument('--type', choices=['cash', 'gpp', 'balanced'], default='balanced',
                      help='Contest type')
    parser.add_argument('--lineups', type=int, default=5,
                      help='Number of lineups to generate')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    optimizer = DFSOptimization()
    
    try:
        if args.mode == 'continuous':
            await optimizer.run_continuous()
        else:
            await optimizer.run_single_optimization(args.type, args.lineups)
    except KeyboardInterrupt:
        print("\n✅ Optimization stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
