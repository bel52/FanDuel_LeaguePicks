"""
Main DFS Optimizer Application with Automated FanDuel Salaries
"""
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import polars as pl
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger
import json
import sys

from config import config
from data_collector import NFLDataCollector, DataProcessor
from ai_analyzer import DFSAIAnalyzer, CorrelationAnalyzer
from optimizer import AdvancedLineupOptimizer, MonteCarloSimulator
from fanduel_scraper import FanDuelScraper

class DFSOptimizerApp:
    """Main application orchestrator with automated salary updates"""
    
    def __init__(self):
        self.data_collector = NFLDataCollector()
        self.data_processor = DataProcessor()
        self.ai_analyzer = DFSAIAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.optimizer = AdvancedLineupOptimizer()
        self.simulator = MonteCarloSimulator()
        self.fanduel_scraper = FanDuelScraper()
        self.scheduler = AsyncIOScheduler()
        self.current_week = self._get_current_nfl_week()
        self.current_season = 2024
        
    def _get_current_nfl_week(self) -> int:
        """Determine current NFL week"""
        season_start = datetime(2024, 9, 5)  # 2024 season start
        current_date = datetime.now()
        
        if current_date < season_start:
            return 1
        
        weeks_elapsed = (current_date - season_start).days // 7
        return min(18, weeks_elapsed + 1)  # Max 18 regular season weeks
    
    async def initialize(self):
        """Initialize the application"""
        logger.info("Initializing DFS Optimizer with automated salary updates")
        
        # Load or fetch initial data
        await self.update_all_data()
        
        # Schedule regular updates
        self.scheduler.add_job(
            self.update_all_data,
            'interval',
            minutes=config.UPDATE_INTERVAL_MINUTES,
            id='update_data'
        )
        
        # Schedule salary updates (every 6 hours)
        self.scheduler.add_job(
            self.update_salaries,
            'interval',
            hours=config.SALARY_UPDATE_HOURS,
            id='update_salaries'
        )
        
        # Schedule injury report updates (more frequent)
        self.scheduler.add_job(
            self.update_injury_reports,
            'interval',
            minutes=config.INJURY_UPDATE_MINUTES,
            id='update_injuries'
        )
        
        # Schedule late swap monitoring (Sunday only)
        self.scheduler.add_job(
            self.monitor_late_swaps,
            'cron',
            day_of_week='sun',
            hour='13-20',
            minute='*/5',
            id='late_swaps'
        )
        
        self.scheduler.start()
        logger.info("Scheduler started with automated salary updates")
    
    async def update_salaries(self):
        """Update FanDuel salaries automatically"""
        try:
            logger.info("Updating FanDuel salaries automatically")
            
            # Fetch latest salaries
            salaries = await self.fanduel_scraper.get_nfl_salaries(self.current_week)
            
            if not salaries.empty:
                # Update processed data with new salaries
                processed_data = self.load_processed_data()
                if processed_data is not None:
                    # Merge new salaries with existing projections
                    updated_data = self._merge_salary_updates(processed_data, salaries)
                    self.save_processed_data(updated_data)
                    logger.info(f"Updated {len(salaries)} player salaries")
            else:
                logger.warning("Failed to update salaries")
                
        except Exception as e:
            logger.error(f"Error updating salaries: {e}")
    
    async def update_all_data(self):
        """Update all data sources including automated FanDuel salaries"""
        try:
            logger.info(f"Updating all data for Week {self.current_week}")
            
            # Collect data from all sources (includes FanDuel salaries)
            async with self.data_collector as collector:
                raw_data = await collector.collect_all_data(
                    self.current_week, 
                    self.current_season
                )
            
            # Process and combine data (FanDuel salaries are in raw_data)
            processed_data = self.data_processor.process_all_data(raw_data)
            
            # Save processed data
            self.save_processed_data(processed_data)
            
            # Run AI analysis with both models
            await self.run_ai_analysis(processed_data)
            
            # Log AI spending
            spend_report = self.ai_analyzer.get_weekly_spend_report()
            logger.info(f"AI Spending - OpenAI: ${spend_report['openai_spend']:.2f}, "
                       f"Claude: ${spend_report['claude_spend']:.2f}, "
                       f"Total: ${spend_report['total_spend']:.2f}/${spend_report['budget']:.2f}")
            
            logger.info("Data update completed successfully")
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
    
    async def run_ai_analysis(self, data: pl.DataFrame):
        """Run AI analysis using both OpenAI and Claude"""
        try:
            logger.info("Running AI analysis with both models")
            
            # Prepare data for AI
            slate_data = data.head(100).to_pandas().to_json(orient='records')
            
            # Get AI insights (will use best model based on config)
            analysis = self.ai_analyzer.analyze_slate(slate_data)
            
            # Save analysis
            analysis_path = config.DATA_DIR / f"ai_analysis_week{self.current_week}.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info("AI analysis completed and saved")
            
            # Get ownership projections (may use both models for consensus)
            players = data.to_pandas().to_dict('records')[:50]
            ownership = self.ai_analyzer.get_ownership_projections(players)
            
            # Save ownership projections
            ownership_path = config.DATA_DIR / f"ownership_week{self.current_week}.json"
            with open(ownership_path, 'w') as f:
                json.dump(ownership, f, indent=2)
            
            # Get correlation insights from Claude if available
            if config.ANTHROPIC_API_KEY:
                game_data = {'week': self.current_week, 'season': self.current_season}
                correlations = self.correlation_analyzer.get_enhanced_correlations(game_data)
                
                correlation_path = config.DATA_DIR / f"correlations_week{self.current_week}.json"
                with open(correlation_path, 'w') as f:
                    json.dump(correlations, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
    
    def generate_lineups(self, lineup_type: str = 'tournament', 
                        num_lineups: int = 20) -> List[Dict]:
        """
        Generate optimized lineups with Monte Carlo simulation
        
        Args:
            lineup_type: 'cash', 'tournament', or 'balanced'
            num_lineups: Number of lineups to generate
            
        Returns:
            List of generated lineups
        """
        try:
            logger.info(f"Generating {num_lineups} {lineup_type} lineups")
            
            # Load processed data
            data = self.load_processed_data()
            if data is None:
                logger.error("No processed data available")
                return []
            
            lineups = []
            
            if lineup_type == 'cash':
                # Single optimal lineup for cash games
                lineup = self.optimizer.optimize_single_lineup(data)
                if lineup:
                    lineups = [lineup]
            
            elif lineup_type == 'tournament':
                # Multiple lineups with correlation
                correlation_matrix = self.correlation_analyzer.correlation_matrix
                
                # Generate diverse lineups
                lineups = self.optimizer.optimize_multiple_lineups(
                    data,
                    num_lineups,
                    max_overlap=6
                )
                
                # Add correlation scores
                for lineup in lineups:
                    lineup['correlation_score'] = self.correlation_analyzer.calculate_lineup_correlation(
                        lineup['players']
                    )
            
            else:  # balanced
                # Mix of cash and tournament lineups
                cash_lineup = self.optimizer.optimize_single_lineup(data)
                if cash_lineup:
                    lineups.append(cash_lineup)
                
                tournament_lineups = self.optimizer.optimize_multiple_lineups(
                    data,
                    num_lineups - 1,
                    max_overlap=7
                )
                lineups.extend(tournament_lineups)
            
            # Run Monte Carlo simulations (already in optimizer.py)
            if lineups:
                logger.info("Running Monte Carlo simulations...")
                sim_results = self.simulator.simulate_tournament(lineups)
                logger.info(f"Simulation results: ROI={sim_results['avg_roi']:.2%}, "
                          f"Cash Rate={sim_results['min_cash_rate']:.2%}, "
                          f"Sharpe Ratio={sim_results['sharpe_ratio']:.2f}")
                
                # Add simulation results to lineups
                for lineup in lineups:
                    lineup['simulation'] = sim_results
                
                # Get AI analysis for top lineups
                if lineups[:3]:  # Analyze top 3 lineups
                    for lineup in lineups[:3]:
                        context = {'week': self.current_week}
                        lineup_analysis = self.ai_analyzer.analyze_lineup(
                            lineup['players'], 
                            context
                        )
                        lineup['ai_analysis'] = lineup_analysis
            
            # Save lineups
            self.save_generated_lineups(lineups)
            
            logger.info(f"Generated {len(lineups)} lineups successfully")
            return lineups
            
        except Exception as e:
            logger.error(f"Error generating lineups: {e}")
            return []
    
    def _merge_salary_updates(self, data: pl.DataFrame, new_salaries: pd.DataFrame) -> pl.DataFrame:
        """Merge updated salaries with existing data"""
        try:
            # Convert new salaries to Polars
            salaries_pl = pl.from_pandas(new_salaries)
            
            # Update salary column in existing data
            # This is a simplified merge - you may need to handle name matching
            data = data.drop('Salary')
            data = data.join(
                salaries_pl.select(['Name', 'Salary']),
                on='Name',
                how='left'
            )
            
            # Recalculate value with new salaries
            data = data.with_columns([
                (pl.col('adjusted_projection') / pl.col('Salary') * 1000).alias('value')
            ])
            
            return data
            
        except Exception as e:
            logger.error(f"Error merging salary updates: {e}")
            return data
    
    # ... rest of the methods remain the same ...
    
    async def monitor_late_swaps(self):
        """Monitor for late swap opportunities"""
        try:
            logger.info("Checking for late swap opportunities")
            
            # Load current lineups
            lineups = self.load_generated_lineups()
            if not lineups:
                return
            
            # Get latest news
            async with self.data_collector as collector:
                news = await collector.get_espn_data(self.current_week, self.current_season)
            
            # Check each lineup for swap opportunities
            for lineup in lineups:
                for player in lineup['players']:
                    # Check if player has news updates
                    player_news = self._check_player_news(player['name'], news)
                    
                    if player_news and 'injury' in player_news.lower():
                        # Find alternatives
                        alternatives = self._find_swap_alternatives(
                            player, 
                            lineup['salary_remaining']
                        )
                        
                        # AI analysis for swap decision (uses best model)
                        swap_rec = self.ai_analyzer.analyze_late_swap(
                            player,
                            alternatives,
                            {'latest_news': player_news}
                        )
                        
                        if swap_rec['swap']:
                            logger.warning(
                                f"SWAP ALERT: Replace {player['name']} with "
                                f"{swap_rec['target']} - {swap_rec['reasoning']}"
                            )
                            
                            # Send notification
                            self.send_notification(swap_rec)
            
        except Exception as e:
            logger.error(f"Error monitoring late swaps: {e}")
    
    def save_processed_data(self, data: pl.DataFrame):
        """Save processed data to disk"""
        try:
            path = config.DATA_DIR / f"processed_data_week{self.current_week}.parquet"
            data.write_parquet(path)
            logger.info(f"Saved processed data to {path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def load_processed_data(self) -> Optional[pl.DataFrame]:
        """Load processed data from disk"""
        try:
            path = config.DATA_DIR / f"processed_data_week{self.current_week}.parquet"
            if path.exists():
                return pl.read_parquet(path)
            return None
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return None
    
    def save_generated_lineups(self, lineups: List[Dict]):
        """Save generated lineups"""
        try:
            path = config.DATA_DIR / f"lineups_week{self.current_week}_{datetime.now():%Y%m%d_%H%M}.json"
            with open(path, 'w') as f:
                json.dump(lineups, f, indent=2, default=str)
            logger.info(f"Saved {len(lineups)} lineups to {path}")
        except Exception as e:
            logger.error(f"Error saving lineups: {e}")
    
    def load_generated_lineups(self) -> List[Dict]:
        """Load most recent generated lineups"""
        try:
            pattern = f"lineups_week{self.current_week}_*.json"
            lineup_files = sorted(config.DATA_DIR.glob(pattern))
            
            if lineup_files:
                with open(lineup_files[-1], 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading lineups: {e}")
            return []
    
    def _check_player_news(self, player_name: str, news_data: Dict) -> Optional[str]:
        """Check for player news updates"""
        # Implementation would parse news data for player mentions
        return None
    
    def _find_swap_alternatives(self, player: Dict, salary_available: int) -> List[Dict]:
        """Find swap alternatives for a player"""
        data = self.load_processed_data()
        if data is None:
            return []
        
        # Find players at same position with similar or lower salary
        alternatives = data.filter(
            (pl.col('Position') == player['position']) &
            (pl.col('Salary') <= player['salary'] + salary_available) &
            (pl.col('Name') != player['name'])
        ).sort('adjusted_projection', descending=True).head(5)
        
        return alternatives.to_pandas().to_dict('records')
    
    def send_notification(self, message: Dict):
        """Send notification (implement your preferred method)"""
        logger.warning(f"NOTIFICATION: {message}")
    
    def print_lineup(self, lineup: Dict):
        """Pretty print a lineup with Monte Carlo results"""
        print("\n" + "="*50)
        print("OPTIMIZED LINEUP")
        print("="*50)
        
        for player in lineup['players']:
            print(f"{player['position']:4} {player['name']:20} "
                  f"${player['salary']:5} {player['projection']:5.1f}pts")
        
        print("-"*50)
        print(f"Total Salary: ${lineup['total_salary']:,}")
        print(f"Remaining: ${lineup['salary_remaining']:,}")
        print(f"Projected: {lineup['total_projection']:.1f} points")
        
        if 'correlation_score' in lineup:
            print(f"Correlation: {lineup['correlation_score']:.2f}")
        
        if 'simulation' in lineup:
            sim = lineup['simulation']
            print(f"\nMonte Carlo Simulation Results:")
            print(f"  Expected ROI: {sim['avg_roi']:.1%}")
            print(f"  Cash Rate: {sim['min_cash_rate']:.1%}")
            print(f"  Top 10 Rate: {sim['top_10_rate']:.2%}")
            print(f"  Sharpe Ratio: {sim['sharpe_ratio']:.2f}")
        
        if 'ai_analysis' in lineup:
            print(f"\nAI Analysis:")
            analysis = lineup['ai_analysis']
            if 'correlation' in analysis:
                print(f"  Correlation Score: {analysis['correlation']}/10")
            if 'gpp_viability' in analysis:
                print(f"  GPP Viability: {analysis['gpp_viability']}/10")


async def main():
    """Main entry point"""
    app = DFSOptimizerApp()
    
    try:
        # Initialize application
        await app.initialize()
        
        print("\n" + "="*60)
        print("DFS OPTIMIZER INITIALIZED")
        print("="*60)
        print(f"✓ Automated FanDuel salary updates every {config.SALARY_UPDATE_HOURS} hours")
        print(f"✓ AI Analysis using: {config.AI_MODEL_PREFERENCE}")
        print(f"✓ Weekly AI Budget: ${config.AI_WEEKLY_BUDGET:.2f}")
        print(f"✓ Monte Carlo Simulations: Enabled (10,000 iterations)")
        print("="*60 + "\n")
        
        # Generate initial lineups
        print("\nGenerating Cash Game Lineup...")
        cash_lineups = app.generate_lineups('cash', 1)
        if cash_lineups:
            app.print_lineup(cash_lineups[0])
        
        print("\nGenerating Tournament Lineups with Monte Carlo...")
        tournament_lineups = app.generate_lineups('tournament', 20)
        if tournament_lineups:
            print(f"\nGenerated {len(tournament_lineups)} tournament lineups")
            print("\nTop Tournament Lineup:")
            app.print_lineup(tournament_lineups[0])
        
        # Keep running for continuous monitoring
        print("\n" + "="*60)
        print("OPTIMIZER RUNNING CONTINUOUSLY")
        print("="*60)
        print("✓ FanDuel salaries update automatically")
        print("✓ Data updates every 30 minutes")
        print("✓ Injury reports update every 15 minutes")
        print("✓ Late swap monitoring active on Sundays")
        print("✓ AI spending tracked against weekly budget")
        print("\nPress Ctrl+C to stop.")
        print("="*60 + "\n")
        
        while True:
            await asyncio.sleep(60)  # Keep alive
            
    except KeyboardInterrupt:
        logger.info("Shutting down optimizer")
        
        # Print final AI spending report
        spend_report = app.ai_analyzer.get_weekly_spend_report()
        print(f"\nFinal AI Spending Report:")
        print(f"  OpenAI: ${spend_report['openai_spend']:.2f}")
        print(f"  Claude: ${spend_report['claude_spend']:.2f}")
        print(f"  Total: ${spend_report['total_spend']:.2f} / ${spend_report['budget']:.2f}")
        print(f"  Remaining: ${spend_report['remaining']:.2f}")
        
        app.scheduler.shutdown()
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run the application
    asyncio.run(main())
