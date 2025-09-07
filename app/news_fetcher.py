# Create this new file at app/news_fetcher.py
import feedparser
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ESPN's public RSS feed for NFL news
ESPN_NFL_NEWS_URL = "https://www.espn.com/espn/rss/nfl/news"

def fetch_latest_news() -> List[Dict[str, str]]:
    """
    Fetches the latest NFL news headlines from ESPN's RSS feed.
    """
    try:
        feed = feedparser.parse(ESPN_NFL_NEWS_URL)
        if feed.bozo:
            # Bozo bit is set if the feed is malformed
            raise Exception(f"Feed is malformed: {feed.bozo_exception}")

        news_items = []
        for entry in feed.entries:
            news_items.append({
                "headline": entry.title,
                "summary": entry.summary,
                "link": entry.link
            })
        
        logger.info(f"Successfully fetched {len(news_items)} news items from ESPN.")
        return news_items

    except Exception as e:
        logger.error(f"Failed to fetch or parse news feed: {e}")
        return []
