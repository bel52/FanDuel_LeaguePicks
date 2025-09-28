"""
Real-time NFL Breaking News Monitor for DFS Lineup Adjustments
Uses free sources to avoid API costs
"""
import asyncio
import aiohttp
import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path


class NFLBreakingNewsMonitor:
    """Monitors free NFL news sources for lineup-impacting events"""

    def __init__(self):
        self.news_cache = []
        self.last_check = None
        self.critical_keywords = [
            # Injury-related
            'ruled out', 'inactive', 'injury report', 'doubtful', 'questionable',
            'emergency start', 'elevated', 'promoted', 'scratched', 'limited',

            # Weather-related
            'heavy rain', 'wind gusts', 'snow', 'severe weather', 'game conditions',

            # Lineup changes
            'starting lineup', 'depth chart', 'role change', 'snap count',
            'backup quarterback', 'goal line back'
        ]

        # Free news sources (no API keys required)
        self.news_sources = {
            'nfl_news_rss': 'http://www.nfl.com/rss/rsslanding?searchString=news',
            'espn_nfl_rss': 'https://www.espn.com/espn/rss/nfl/news',
            'weather_alerts': 'https://api.weather.gov/alerts/active'
        }

    async def check_breaking_news(self) -> List[Dict[str, Any]]:
        """Check all sources for breaking news in the last hour"""
        try:
            current_time = datetime.now()

            # Only check if it's been 15+ minutes since last check
            if (self.last_check and
                (current_time - self.last_check).total_seconds() < 900):
                return self.news_cache

            logger.info("🔍 Checking for breaking NFL news...")

            all_news = []

            # Check NFL official news (limited parsing to avoid blocking)
            nfl_news = await self._check_nfl_headlines()
            all_news.extend(nfl_news)

            # Check weather alerts for game cities
            weather_alerts = await self._check_weather_alerts()
            all_news.extend(weather_alerts)

            # Filter for critical news only
            critical_news = self._filter_critical_news(all_news)

            self.news_cache = critical_news
            self.last_check = current_time

            if critical_news:
                logger.info(f"🚨 Found {len(critical_news)} critical news items")
                for news in critical_news:
                    logger.info(f"📰 {news['headline'][:100]}...")

            return critical_news

        except Exception as e:
            logger.error(f"Error checking breaking news: {e}")
            return []

    async def _check_nfl_headlines(self) -> List[Dict[str, Any]]:
        """Check NFL headlines for critical updates"""
        headlines = []

        try:
            # Simple headline scraping (very basic to avoid blocking)
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        'https://www.nfl.com/news',
                        timeout=aiohttp.ClientTimeout(total=10),
                        headers={'User-Agent': 'Mozilla/5.0 (compatible; DFS-News/1.0)'}
                    ) as response:
                        if response.status == 200:
                            text = await response.text()

                            # Very basic headline extraction
                            headline_patterns = [
                                r'<h[1-6][^>]*>([^<]*(?:injury|inactive|ruled out|doubtful|questionable)[^<]*)</h[1-6]>',
                                r'<title>([^<]*(?:injury|inactive|ruled out|doubtful|questionable)[^<]*)</title>',
                                r'<span[^>]*>([^<]*(?:injury|inactive|ruled out|doubtful|questionable)[^<]*)</span>'
                            ]

                            for pattern in headline_patterns:
                                matches = re.findall(pattern, text, re.IGNORECASE)
                                for match in matches[:5]:  # Limit to 5 matches
                                    if len(match.strip()) > 10:  # Must be substantial
                                        headlines.append({
                                            'source': 'NFL.com',
                                            'headline': match.strip(),
                                            'timestamp': datetime.now(),
                                            'url': 'https://www.nfl.com/news',
                                            'impact_type': 'injury_report'
                                        })

                except Exception as e:
                    logger.debug(f"NFL headline check failed: {e}")

        except Exception as e:
            logger.error(f"Error checking NFL headlines: {e}")

        return headlines[:3]  # Return max 3 headlines

    async def _check_weather_alerts(self) -> List[Dict[str, Any]]:
        """Check weather.gov for alerts affecting NFL games"""
        alerts = []

        try:
            # Get weather alerts from weather.gov (free government API)
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(
                        'https://api.weather.gov/alerts/active',
                        timeout=aiohttp.ClientTimeout(total=15),
                        headers={'User-Agent': 'DFS-Optimizer/1.0'}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()

                            # Look for weather alerts in NFL cities
                            nfl_cities = [
                                'Buffalo', 'Miami', 'New England', 'New York',
                                'Baltimore', 'Cincinnati', 'Cleveland', 'Pittsburgh',
                                'Houston', 'Indianapolis', 'Jacksonville', 'Tennessee',
                                'Denver', 'Kansas City', 'Las Vegas', 'Los Angeles',
                                'Dallas', 'New York', 'Philadelphia', 'Washington',
                                'Chicago', 'Detroit', 'Green Bay', 'Minnesota',
                                'Atlanta', 'Carolina', 'New Orleans', 'Tampa Bay',
                                'Arizona', 'Los Angeles', 'San Francisco', 'Seattle'
                            ]

                            features = data.get('features', [])
                            for feature in features[:10]:  # Limit to 10 alerts
                                properties = feature.get('properties', {})
                                headline = properties.get('headline', '')
                                areas = properties.get('areaDesc', '')

                                # Check if alert affects NFL cities
                                for city in nfl_cities:
                                    if city.lower() in areas.lower() or city.lower() in headline.lower():
                                        # Check for severe weather keywords
                                        severe_keywords = ['wind', 'snow', 'rain', 'storm', 'freeze']
                                        if any(keyword in headline.lower() for keyword in severe_keywords):
                                            alerts.append({
                                                'source': 'Weather.gov',
                                                'headline': f"Weather Alert: {headline}",
                                                'timestamp': datetime.now(),
                                                'url': 'https://weather.gov',
                                                'impact_type': 'weather',
                                                'affected_area': areas
                                            })
                                            break

                except Exception as e:
                    logger.debug(f"Weather alert check failed: {e}")

        except Exception as e:
            logger.error(f"Error checking weather alerts: {e}")

        return alerts[:2]  # Return max 2 weather alerts

    def _filter_critical_news(self, all_news: List[Dict]) -> List[Dict]:
        """Filter news for only critical DFS-impacting events"""
        critical_news = []

        for news_item in all_news:
            headline = news_item.get('headline', '').lower()

            # Check for critical keywords
            critical_found = False
            for keyword in self.critical_keywords:
                if keyword in headline:
                    critical_found = True
                    break

            if critical_found:
                # Add impact assessment
                news_item['dfs_impact'] = self._assess_dfs_impact(headline)
                critical_news.append(news_item)

        # Sort by DFS impact (highest first)
        critical_news.sort(key=lambda x: x.get('dfs_impact', 0), reverse=True)

        return critical_news

    def _assess_dfs_impact(self, headline: str) -> int:
        """Assess DFS impact on scale of 1-10"""
        headline_lower = headline.lower()
        impact_score = 0

        # High impact keywords
        if any(word in headline_lower for word in ['ruled out', 'inactive', 'emergency start']):
            impact_score += 8

        # Medium impact keywords
        if any(word in headline_lower for word in ['doubtful', 'questionable', 'limited']):
            impact_score += 5

        # Weather impact
        if any(word in headline_lower for word in ['heavy rain', 'wind gusts', 'snow']):
            impact_score += 6

        # Lineup changes
        if any(word in headline_lower for word in ['starting lineup', 'depth chart', 'promoted']):
            impact_score += 7

        return min(impact_score, 10)

    def get_player_specific_news(self, player_names: List[str]) -> List[Dict]:
        """Get news specific to given players"""
        player_news = []

        for news_item in self.news_cache:
            headline = news_item.get('headline', '').lower()

            for player_name in player_names:
                # Simple name matching (first/last name)
                name_parts = player_name.lower().split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]

                    if (first_name in headline and last_name in headline) or player_name.lower() in headline:
                        news_item['affected_player'] = player_name
                        player_news.append(news_item)
                        break

        return player_news

    async def save_news_log(self):
        """Save news to file for debugging"""
        try:
            if self.news_cache:
                log_file = Path('logs') / f'news_log_{datetime.now().strftime("%Y%m%d")}.json'
                log_file.parent.mkdir(exist_ok=True)

                with open(log_file, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'news_items': self.news_cache
                    }, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving news log: {e}")


# Integration functions for main app
async def get_breaking_news() -> List[Dict[str, Any]]:
    """Get current breaking news"""
    monitor = NFLBreakingNewsMonitor()
    return await monitor.check_breaking_news()


async def get_player_news(player_names: List[str]) -> List[Dict[str, Any]]:
    """Get news affecting specific players"""
    monitor = NFLBreakingNewsMonitor()
    await monitor.check_breaking_news()  # Refresh first
    return monitor.get_player_specific_news(player_names)