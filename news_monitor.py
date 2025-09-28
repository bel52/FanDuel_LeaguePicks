"""
Reliable NFL Breaking News Monitor using RSS feeds
Focuses on injury reports and lineup changes that impact DFS
"""
import asyncio
import aiohttp
import feedparser
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import xml.etree.ElementTree as ET


class NFLBreakingNewsMonitor:
    """Monitor reliable RSS feeds for DFS-impacting NFL news"""

    def __init__(self):
        self.news_cache = []
        self.last_check = None

        # Working RSS sources - tested and reliable
        self.rss_sources = {
            'espn_nfl': 'https://www.espn.com/espn/rss/nfl/news',
            'cbs_sports': 'https://www.cbssports.com/rss/headlines/nfl/',
            'yahoo_sports': 'https://sports.yahoo.com/nfl/rss.xml',
        }

        # Keywords that matter for DFS decisions
        self.critical_keywords = [
            # Injury status changes
            'ruled out', 'inactive', 'doubtful', 'questionable', 'limited',
            'injury report', 'practices', 'dnp', 'did not practice',

            # Lineup changes
            'starting', 'backup', 'elevated', 'promoted', 'signed',
            'scratched', 'emergency', 'activated', 'waived',

            # Role changes
            'snap count', 'touches', 'targets', 'goal line', 'red zone',
            'workload', 'carries', 'usage'
        ]

    async def get_breaking_news(self) -> List[Dict[str, Any]]:
        """Get breaking news from RSS feeds - much more reliable"""
        try:
            current_time = datetime.now()

            # Only check every 10 minutes to avoid spam
            if (self.last_check and
                (current_time - self.last_check).total_seconds() < 600):
                return self.news_cache

            logger.info("🔍 Checking RSS feeds for NFL news...")

            all_news = []

            # Check each RSS source
            for source_name, rss_url in self.rss_sources.items():
                try:
                    news_items = await self._parse_rss_feed(source_name, rss_url)
                    all_news.extend(news_items)
                    logger.info(f"📰 {source_name}: {len(news_items)} items")
                except Exception as e:
                    logger.warning(f"RSS source {source_name} failed: {e}")
                    continue

            # Filter for recent and critical news only
            recent_news = self._filter_recent_and_critical(all_news)

            self.news_cache = recent_news
            self.last_check = current_time

            if recent_news:
                logger.info(f"🚨 Found {len(recent_news)} critical news items")
                for news in recent_news[:3]:  # Show top 3
                    logger.info(f"📰 {news['headline'][:80]}...")
            else:
                logger.info("📰 No critical NFL news at this time")

            return recent_news

        except Exception as e:
            logger.error(f"Error checking breaking news: {e}")
            return []

    async def _parse_rss_feed(self, source_name: str, rss_url: str) -> List[Dict[str, Any]]:
        """Parse RSS feed with proper async handling"""
        news_items = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    rss_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={'User-Agent': 'Mozilla/5.0 (compatible; DFS-News/1.0)'}
                ) as response:

                    if response.status != 200:
                        logger.warning(f"RSS {source_name} returned {response.status}")
                        return []

                    xml_content = await response.text()

                    # Parse with feedparser (handles RSS/Atom differences)
                    feed = feedparser.parse(xml_content)

                    if not feed.entries:
                        logger.warning(f"No entries in RSS feed: {source_name}")
                        return []

                    # Process entries
                    for entry in feed.entries[:10]:  # Limit to 10 most recent
                        try:
                            title = entry.get('title', '').strip()
                            summary = entry.get('summary', '').strip()
                            published = entry.get('published', '')
                            link = entry.get('link', '')

                            if not title or len(title) < 10:
                                continue

                            # Parse publish date
                            pub_date = self._parse_publish_date(published)

                            news_item = {
                                'source': source_name,
                                'headline': title,
                                'summary': summary[:200] if summary else '',
                                'timestamp': pub_date,
                                'url': link,
                                'impact_type': self._classify_news_type(title + ' ' + summary)
                            }

                            news_items.append(news_item)

                        except Exception as e:
                            logger.debug(f"Error parsing RSS entry: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error fetching RSS {source_name}: {e}")

        return news_items

    def _parse_publish_date(self, date_str: str) -> datetime:
        """Parse various RSS date formats"""
        if not date_str:
            return datetime.now()

        try:
            # Try common RSS date formats
            import email.utils
            parsed = email.utils.parsedate_tz(date_str)
            if parsed:
                return datetime(*parsed[:6])
        except:
            pass

        # Fallback to current time
        return datetime.now()

    def _classify_news_type(self, text: str) -> str:
        """Classify news type for DFS impact"""
        text_lower = text.lower()

        # Injury-related
        injury_keywords = ['injury', 'hurt', 'pain', 'strain', 'tear', 'sprain', 'concussion']
        if any(keyword in text_lower for keyword in injury_keywords):
            return 'injury_report'

        # Lineup changes
        lineup_keywords = ['starting', 'lineup', 'depth chart', 'promoted', 'elevated']
        if any(keyword in text_lower for keyword in lineup_keywords):
            return 'lineup_change'

        # Status updates
        status_keywords = ['ruled out', 'questionable', 'doubtful', 'limited', 'dnp']
        if any(keyword in text_lower for keyword in status_keywords):
            return 'status_update'

        return 'general_news'

    def _filter_recent_and_critical(self, all_news: List[Dict]) -> List[Dict]:
        """Filter for recent news with DFS impact - RELAXED for more coverage"""
        critical_news = []
        cutoff_time = datetime.now() - timedelta(hours=24)  # Last 24 hours (more coverage)

        for news_item in all_news:
            # Check if recent enough
            news_time = news_item.get('timestamp', datetime.now())
            if news_time < cutoff_time:
                continue

            headline = news_item.get('headline', '').lower()
            summary = news_item.get('summary', '').lower()
            full_text = headline + ' ' + summary
            impact_type = news_item.get('impact_type', '')

            # Check for critical DFS keywords OR injury/status news types
            critical_score = 0
            for keyword in self.critical_keywords:
                if keyword in full_text:
                    critical_score += 1

            # RELAXED: Include injury reports and status updates even without keywords
            if impact_type in ['injury_report', 'status_update', 'lineup_change']:
                critical_score += 2

            # RELAXED: Include any news with player names + "expected to play"
            if 'expected to play' in full_text or 'will play' in full_text:
                critical_score += 3

            # RELAXED: Include concussion protocol news
            if 'concussion' in full_text or 'protocol' in full_text:
                critical_score += 2

            # Accept news with any critical score
            if critical_score > 0:
                news_item['dfs_impact'] = min(10, critical_score * 2)
                critical_news.append(news_item)

        # Sort by DFS impact and recency
        critical_news.sort(
            key=lambda x: (x.get('dfs_impact', 0), x.get('timestamp', datetime.min)),
            reverse=True
        )

        return critical_news[:20]  # Return top 20

    def get_player_specific_news(self, player_names: List[str]) -> List[Dict]:
        """Get news affecting specific players"""
        player_news = []

        for news_item in self.news_cache:
            headline = news_item.get('headline', '').lower()
            summary = news_item.get('summary', '').lower()
            full_text = headline + ' ' + summary

            for player_name in player_names:
                # Simple name matching
                name_parts = player_name.lower().split()
                if len(name_parts) >= 2:
                    first_name = name_parts[0]
                    last_name = name_parts[-1]

                    # Look for first + last name or full name
                    if ((first_name in full_text and last_name in full_text) or
                        player_name.lower() in full_text):

                        news_item_copy = news_item.copy()
                        news_item_copy['affected_player'] = player_name
                        player_news.append(news_item_copy)
                        break

        return player_news

    def assess_dfs_impact(self, headline: str, summary: str = '') -> int:
        """Assess DFS impact on scale of 1-10"""
        text = (headline + ' ' + summary).lower()
        impact = 0

        # High impact: definitive status changes
        if any(word in text for word in ['ruled out', 'inactive', 'emergency start']):
            impact += 8

        # Medium impact: uncertain status
        if any(word in text for word in ['doubtful', 'questionable', 'limited']):
            impact += 5

        # Role changes
        if any(word in text for word in ['starting', 'promoted', 'elevated', 'backup']):
            impact += 6

        # Practice participation
        if any(word in text for word in ['dnp', 'did not practice', 'full practice']):
            impact += 3

        return min(10, impact)


# Integration functions
async def get_breaking_news() -> List[Dict[str, Any]]:
    """Get current breaking news - main integration point"""
    monitor = NFLBreakingNewsMonitor()
    return await monitor.get_breaking_news()


async def get_player_news(player_names: List[str]) -> List[Dict[str, Any]]:
    """Get news affecting specific players"""
    monitor = NFLBreakingNewsMonitor()
    await monitor.get_breaking_news()  # Refresh cache first
    return monitor.get_player_specific_news(player_names)