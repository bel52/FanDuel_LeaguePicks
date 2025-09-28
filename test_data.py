#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from data_collector import get_fresh_data

async def test_data_collection():
    """Test data collection directly"""
    logger.info("Testing enhanced data collection...")
    
    try:
        data = await get_fresh_data()
        
        if data and 'players' in data:
            logger.info(f"✅ SUCCESS: {len(data['players'])} players collected")
            
            # Show enhanced data quality
            quality = data.get('data_quality', {})
            logger.info(f"📊 Data Quality Summary:")
            logger.info(f"   • Average ownership: {quality.get('avg_ownership', 0):.1f}%")
            logger.info(f"   • Vegas games: {quality.get('vegas_games', 0)}")
            logger.info(f"   • Real projections: {quality.get('real_projections', 0)}")
            logger.info(f"   • Salary range: ${quality.get('salary_range', {}).get('min', 0):,} - ${quality.get('salary_range', {}).get('max', 0):,}")
            
            # Show sample players
            players = data['players'][:5]
            logger.info(f"📋 Sample Players:")
            for player in players:
                logger.info(f"   • {player['name']} ({player['position']}) - ${player['salary']:,} - {player['ownership']:.1f}% owned")
            
            return True
        else:
            logger.error("❌ No data returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_data_collection())
    if success:
        print("✅ Data collection test PASSED")
    else:
        print("❌ Data collection test FAILED")
