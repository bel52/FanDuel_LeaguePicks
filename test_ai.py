import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_ai_analysis():
    """Test the dual AI analyzer"""
    try:
        from data_collector import get_fresh_data
        from ai_analyzer import DualAIDFSAnalyzer
        
        logger.info("Getting fresh data...")
        data = await get_fresh_data()
        
        if not data or not data.get('players'):
            logger.error("No player data available")
            return False
        
        logger.info(f"Testing AI analysis with {len(data['players'])} players...")
        
        analyzer = DualAIDFSAnalyzer()
        analysis = analyzer.analyze_slate_for_optimization(
            data['players'], 
            data.get('weather', {}), 
            data.get('vegas_odds', {}), 
            'gpp'
        )
        
        print("\n" + "="*60)
        print("AI ANALYSIS RESULTS")
        print("="*60)
        print(f"Strategy: {analysis.get('ai_strategy', 'None')[:200]}...")
        print(f"Leverage Players: {analysis.get('leverage_players', [])}")
        print(f"Avoid Players: {analysis.get('avoid_players', [])}")
        print(f"Stack Teams: {analysis.get('stack_teams', [])}")
        print(f"AI Source: {analysis.get('analysis_source', 'unknown')}")
        print(f"Confidence: {analysis.get('ai_confidence', 0):.1f}")
        
        cost_summary = analyzer.get_cost_summary()
        print(f"AI Cost: ${cost_summary['weekly_spend']:.3f}")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ai_analysis())
    if success:
        print("✅ AI test PASSED")
    else:
        print("❌ AI test FAILED")
