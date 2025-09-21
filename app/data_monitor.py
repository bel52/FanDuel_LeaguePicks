import asyncio
import logging
import aiohttp
import json
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import os

from app.cache_manager import CacheManager
from app.ai_integration import AIAnalyzer
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

@dataclass
class PlayerUpdate:
    player_name: str
    team: str
    update_type: str  # "injury", "weather", "news", "inactive"
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: datetime
    source: str

class RealTimeDataMonitor:
    """Monitors multiple data sources for real-time NFL updates"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self.ai_analyzer = AIAnalyzer()
        
        # Rate limiters for different APIs
        self.espn_limiter = AsyncLimiter(max_rate=60, time_period=60)  # 60/min
        self.news_limiter = AsyncLimiter(max_rate=100, time_period=3600)  # 100/hour
        self.reddit_limiter = AsyncLimiter(max_rate=60, time_period=60)  # 60/min
        
        # Monitoring intervals (seconds)
        self.news_interval = int(os.getenv("NEWS_CHECK_INTERVAL", "300"))  # 5 min
        self.weather_interval = int(os.getenv("WEATHER_CHECK_INTERVAL", "3600"))  # 1 hour
        self.injury_interval = int(os.getenv("INJURY_CHECK_INTERVAL", "600"))  # 10 min
        
        # Player name normalization cache
        self.player_mapping = {}
        
        # Initialize NFL stadiums coordinates for weather
        self.stadium_coords = self._load_stadium_coordinates()
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        logger.info("Starting real-time data monitoring...")
        
        tasks = [
            self._monitor_injury_reports(),
            self._monitor_weather_updates(),
            self._monitor_breaking_news(),
            self._monitor_reddit_updates(),
            self._monitor_espn_updates()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _monitor_injury_reports(self):
        """Monitor injury reports from multiple sources"""
        while True:
            try:
                # ESPN injury reports
                async with self.espn_limiter:
                    espn_updates = await self._fetch_espn_injuries()
                
                # NFL Data Py updates
                nfl_updates = await self._fetch_nfl_data_injuries()
                
                # Process all updates
                all_updates = espn_updates + nfl_updates
                await self._process_player_updates(all_updates)
                
            except Exception as e:
                logger.error(f"Error in injury monitoring: {e}")
            
            await asyncio.sleep(self.injury_interval)
    
    async def _monitor_weather_updates(self):
        """Monitor weather conditions for all NFL stadiums"""
        while True:
            try:
                weather_updates = []
                
                for team, coords in self.stadium_coords.items():
                    weather_data = await self._fetch_weather_gov_data(coords)
                    if weather_data:
                        update = await self._analyze_weather_impact(team, weather_data)
                        if update:
                            weather_updates.append(update)
                
                await self._process_player_updates(weather_updates)
                
            except Exception as e:
                logger.error(f"Error in weather monitoring: {e}")
            
            await asyncio.sleep(self.weather_interval)
    
    async def _monitor_breaking_news(self):
        """Monitor RSS feeds and news APIs for breaking NFL news"""
        while True:
            try:
                news_updates = []
                
                # ESPN RSS feeds
                espn_news = await self._fetch_espn_rss()
                news_updates.extend(espn_news)
                
                # NewsAPI if available
                if os.getenv("NEWS_API_KEY"):
                    async with self.news_limiter:
                        api_news = await self._fetch_news_api()
                        news_updates.extend(api_news)
                
                # Process news for player impacts
                await self._process_news_updates(news_updates)
                
            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
            
            await asyncio.sleep(self.news_interval)
    
    async def _monitor_reddit_updates(self):
        """Monitor Reddit for breaking fantasy football news"""
        while True:
            try:
                async with self.reddit_limiter:
                    reddit_updates = await self._fetch_reddit_updates()
                    await self._process_news_updates(reddit_updates)
                
            except Exception as e:
                logger.error(f"Error in Reddit monitoring: {e}")
            
            await asyncio.sleep(self.news_interval)
    
    async def _monitor_espn_updates(self):
        """Monitor ESPN hidden APIs for real-time updates"""
        while True:
            try:
                async with self.espn_limiter:
                    # Fetch scoreboard for live game data
                    scoreboard = await self._fetch_espn_scoreboard()
                    
                    # Fetch player news
                    player_news = await self._fetch_espn_player_news()
                    
                    # Process updates
                    all_updates = scoreboard + player_news
                    await self._process_player_updates(all_updates)
                
            except Exception as e:
                logger.error(f"Error in ESPN monitoring: {e}")
            
            await asyncio.sleep(300)  # 5 minutes
    
    async def _fetch_espn_injuries(self) -> List[PlayerUpdate]:
        """Fetch injury reports from ESPN API"""
        updates = []
        
        try:
            # NFL teams API endpoint
            url = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for team in data.get('items', []):
                            team_id = team['id']
                            
                            # Fetch team injuries
                            injury_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/teams/{team_id}/injuries"
                            
                            async with session.get(injury_url) as injury_response:
                                if injury_response.status == 200:
                                    injury_data = await injury_response.json()
                                    
                                    for injury in injury_data.get('items', []):
                                        player_name = injury.get('athlete', {}).get('displayName', '')
                                        status = injury.get('status', {}).get('type', {}).get('name', '')
                                        description = injury.get('status', {}).get('type', {}).get('description', '')
                                        
                                        if player_name and status:
                                            severity = self._calculate_injury_severity(status)
                                            
                                            updates.append(PlayerUpdate(
                                                player_name=player_name,
                                                team=team.get('abbreviation', ''),
                                                update_type="injury",
                                                severity=severity,
                                                description=f"{status}: {description}",
                                                timestamp=datetime.now(),
                                                source="ESPN"
                                            ))
        
        except Exception as e:
            logger.error(f"Error fetching ESPN injuries: {e}")
        
        return updates
    
    async def _fetch_nfl_data_injuries(self) -> List[PlayerUpdate]:
        """Fetch injury data using nfl-data-py"""
        updates = []
        
        try:
            import nfl_data_py as nfl
            
            # Get current week injury reports
            current_year = datetime.now().year
            injuries = nfl.import_injuries([current_year])
            
            # Filter to current week
            current_week = self._get_current_nfl_week()
            current_injuries = injuries[injuries['week'] == current_week]
            
            for _, injury in current_injuries.iterrows():
                player_name = injury.get('full_name', '')
                team = injury.get('team', '')
                status = injury.get('report_status', '')
                
                if player_name and status:
                    severity = self._calculate_injury_severity(status)
                    
                    updates.append(PlayerUpdate(
                        player_name=player_name,
                        team=team,
                        update_type="injury",
                        severity=severity,
                        description=f"Status: {status}",
                        timestamp=datetime.now(),
                        source="NFL-Data-Py"
                    ))
        
        except Exception as e:
            logger.error(f"Error fetching NFL data injuries: {e}")
        
        return updates
    
    async def _fetch_weather_gov_data(self, coords: Dict[str, float]) -> Optional[Dict]:
        """Fetch weather data from weather.gov API"""
        try:
            lat, lon = coords['lat'], coords['lon']
            
            # First get the grid info
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            
            async with aiohttp.ClientSession() as session:
                headers = {"User-Agent": "DFS-Optimizer/1.0"}
                
                async with session.get(points_url, headers=headers) as response:
                    if response.status == 200:
                        points_data = await response.json()
                        forecast_url = points_data['properties']['forecast']
                        
                        # Get the forecast
                        async with session.get(forecast_url, headers=headers) as forecast_response:
                            if forecast_response.status == 200:
                                forecast_data = await forecast_response.json()
                                return forecast_data['properties']['periods'][0]  # Current period
        
        except Exception as e:
            logger.error(f"Error fetching weather.gov data for {coords}: {e}")
        
        return None
    
    async def _fetch_espn_rss(self) -> List[Dict]:
        """Fetch breaking news from ESPN RSS feeds"""
        news_updates = []
        
        rss_feeds = [
            "https://www.espn.com/espn/rss/nfl/news",
            "https://www.espn.com/espn/rss/news"
        ]
        
        try:
            for feed_url in rss_feeds:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:10]:  # Latest 10 items
                    # Extract player names from title/description
                    player_names = self._extract_player_names(entry.title + " " + entry.get('summary', ''))
                    
                    if player_names:
                        news_updates.append({
                            'title': entry.title,
                            'summary': entry.get('summary', ''),
                            'players': player_names,
                            'timestamp': datetime.now(),
                            'source': 'ESPN RSS',
                            'url': entry.link
                        })
        
        except Exception as e:
            logger.error(f"Error fetching ESPN RSS: {e}")
        
        return news_updates
    
    async def _fetch_news_api(self) -> List[Dict]:
        """Fetch NFL news from NewsAPI"""
        news_updates = []
        
        try:
            api_key = os.getenv("NEWS_API_KEY")
            if not api_key:
                return news_updates
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'NFL injury OR inactive OR questionable OR doubtful',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'apiKey': api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            player_names = self._extract_player_names(
                                article.get('title', '') + " " + article.get('description', '')
                            )
                            
                            if player_names:
                                news_updates.append({
                                    'title': article['title'],
                                    'summary': article.get('description', ''),
                                    'players': player_names,
                                    'timestamp': datetime.now(),
                                    'source': 'NewsAPI',
                                    'url': article['url']
                                })
        
        except Exception as e:
            logger.error(f"Error fetching NewsAPI: {e}")
        
        return news_updates
    
    async def _fetch_reddit_updates(self) -> List[Dict]:
        """Fetch updates from Reddit NFL and fantasy football subreddits"""
        news_updates = []
        
        try:
            # Reddit API endpoints (no auth needed for public posts)
            subreddits = ['nfl', 'fantasyfootball']
            
            async with aiohttp.ClientSession() as session:
                for subreddit in subreddits:
                    url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=25"
                    headers = {'User-Agent': 'DFS-Optimizer/1.0'}
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for post in data.get('data', {}).get('children', []):
                                post_data = post['data']
                                title = post_data.get('title', '')
                                text = post_data.get('selftext', '')
                                
                                # Look for injury/news keywords
                                if any(keyword in title.lower() for keyword in 
                                      ['injury', 'inactive', 'questionable', 'doubtful', 'out', 'ruled out']):
                                    
                                    player_names = self._extract_player_names(title + " " + text)
                                    
                                    if player_names:
                                        news_updates.append({
                                            'title': title,
                                            'summary': text[:200] + "..." if len(text) > 200 else text,
                                            'players': player_names,
                                            'timestamp': datetime.now(),
                                            'source': f'Reddit r/{subreddit}',
                                            'url': f"https://reddit.com{post_data.get('permalink', '')}"
                                        })
        
        except Exception as e:
            logger.error(f"Error fetching Reddit updates: {e}")
        
        return news_updates
    
    async def _fetch_espn_scoreboard(self) -> List[PlayerUpdate]:
        """Fetch live game data from ESPN scoreboard"""
        updates = []
        
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for event in data.get('events', []):
                            # Check for in-progress games
                            status = event.get('status', {}).get('type', {}).get('name', '')
                            
                            if status in ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME']:
                                # Extract team performance data for potential late swaps
                                for competition in event.get('competitions', []):
                                    for competitor in competition.get('competitors', []):
                                        team = competitor.get('team', {}).get('abbreviation', '')
                                        score = int(competitor.get('score', 0))
                                        
                                        # Create game status update
                                        updates.append(PlayerUpdate(
                                            player_name=f"{team}_TEAM",
                                            team=team,
                                            update_type="game_status",
                                            severity=0.3,  # Medium impact
                                            description=f"Live game: {score} points",
                                            timestamp=datetime.now(),
                                            source="ESPN Scoreboard"
                                        ))
        
        except Exception as e:
            logger.error(f"Error fetching ESPN scoreboard: {e}")
        
        return updates
    
    async def _fetch_espn_player_news(self) -> List[PlayerUpdate]:
        """Fetch player-specific news from ESPN"""
        updates = []
        
        try:
            # This would require player IDs - simplified version
            url = "https://site.api.espn.com/apis/fantasy/v2/games/ffl/news"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for news_item in data.get('items', [])[:20]:  # Latest 20
                            player_name = news_item.get('athlete', {}).get('displayName', '')
                            headline = news_item.get('headline', '')
                            description = news_item.get('description', '')
                            
                            if player_name and any(keyword in headline.lower() for keyword in 
                                                 ['injury', 'inactive', 'questionable', 'out']):
                                
                                severity = self._calculate_news_severity(headline + " " + description)
                                
                                updates.append(PlayerUpdate(
                                    player_name=player_name,
                                    team="",  # Would need to extract
                                    update_type="news",
                                    severity=severity,
                                    description=headline,
                                    timestamp=datetime.now(),
                                    source="ESPN Player News"
                                ))
        
        except Exception as e:
            logger.error(f"Error fetching ESPN player news: {e}")
        
        return updates
    
    async def _analyze_weather_impact(self, team: str, weather_data: Dict) -> Optional[PlayerUpdate]:
        """Analyze weather impact on team/players"""
        try:
            conditions = weather_data.get('shortForecast', '').lower()
            wind_speed = weather_data.get('windSpeed', '')
            temperature = weather_data.get('temperature', 0)
            
            # Extract wind speed number
            wind_mph = 0
            if wind_speed:
                wind_match = re.search(r'(\d+)', wind_speed)
                if wind_match:
                    wind_mph = int(wind_match.group(1))
            
            # Calculate weather severity
            severity = 0.0
            description_parts = []
            
            if wind_mph >= 20:
                severity += 0.4
                description_parts.append(f"High winds ({wind_mph} mph)")
            elif wind_mph >= 15:
                severity += 0.2
                description_parts.append(f"Moderate winds ({wind_mph} mph)")
            
            if any(term in conditions for term in ['rain', 'storm', 'snow']):
                severity += 0.3
                description_parts.append(f"Precipitation: {conditions}")
            
            if temperature <= 20:
                severity += 0.2
                description_parts.append(f"Cold weather ({temperature}°F)")
            
            if severity > 0.1:  # Only report significant weather
                return PlayerUpdate(
                    player_name=f"{team}_WEATHER",
                    team=team,
                    update_type="weather",
                    severity=min(severity, 1.0),
                    description="; ".join(description_parts),
                    timestamp=datetime.now(),
                    source="Weather.gov"
                )
        
        except Exception as e:
            logger.error(f"Error analyzing weather for {team}: {e}")
        
        return None
    
    async def _process_player_updates(self, updates: List[PlayerUpdate]):
        """Process and store player updates"""
        if not updates:
            return
        
        high_priority_updates = [u for u in updates if u.severity >= 0.5]
        
        for update in high_priority_updates:
            # Cache the update
            cache_key = f"player_update:{update.player_name}:{update.update_type}"
            await self.cache_manager.set(cache_key, {
                'player_name': update.player_name,
                'team': update.team,
                'update_type': update.update_type,
                'severity': update.severity,
                'description': update.description,
                'timestamp': update.timestamp.isoformat(),
                'source': update.source
            }, ttl=3600)  # 1 hour
            
            logger.info(f"High priority update: {update.player_name} - {update.description} (severity: {update.severity})")
        
        # Store all updates for batch processing
        all_updates_key = f"all_updates:{datetime.now().strftime('%Y-%m-%d-%H')}"
        existing_updates = await self.cache_manager.get(all_updates_key) or []
        
        new_updates = [
            {
                'player_name': u.player_name,
                'team': u.team,
                'update_type': u.update_type,
                'severity': u.severity,
                'description': u.description,
                'timestamp': u.timestamp.isoformat(),
                'source': u.source
            }
            for u in updates
        ]
        
        existing_updates.extend(new_updates)
        await self.cache_manager.set(all_updates_key, existing_updates, ttl=7200)  # 2 hours
    
    async def _process_news_updates(self, news_updates: List[Dict]):
        """Process news updates for player impacts"""
        for news in news_updates:
            for player_name in news.get('players', []):
                # Use AI to analyze news impact
                try:
                    current_proj = await self._get_player_projection(player_name)
                    if current_proj:
                        new_proj, reasoning = await self.ai_analyzer.analyze_player_news_impact(
                            player_name, [news], current_proj
                        )
                        
                        if abs(new_proj - current_proj) > 1.0:  # Significant change
                            severity = min(abs(new_proj - current_proj) / 10.0, 1.0)
                            
                            update = PlayerUpdate(
                                player_name=player_name,
                                team="",  # Would need team mapping
                                update_type="news",
                                severity=severity,
                                description=f"Projection: {current_proj:.1f} → {new_proj:.1f} - {reasoning}",
                                timestamp=datetime.now(),
                                source=news['source']
                            )
                            
                            await self._process_player_updates([update])
                
                except Exception as e:
                    logger.error(f"Error processing news for {player_name}: {e}")
    
    def _extract_player_names(self, text: str) -> List[str]:
        """Extract potential NFL player names from text"""
        # This is a simplified version - you'd want a more comprehensive player database
        common_names = []
        
        # Pattern for "FirstName LastName" in title case
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        matches = re.findall(name_pattern, text)
        
        # Filter out common non-player names
        exclude_terms = {
            'New York', 'Las Vegas', 'New England', 'San Francisco', 'Green Bay',
            'Kansas City', 'Los Angeles', 'Tampa Bay', 'Monday Night', 'Sunday Night'
        }
        
        for match in matches:
            if match not in exclude_terms and len(match.split()) == 2:
                common_names.append(match)
        
        return list(set(common_names))  # Remove duplicates
    
    def _calculate_injury_severity(self, status: str) -> float:
        """Calculate injury severity score"""
        status_lower = status.lower()
        
        if 'out' in status_lower or 'inactive' in status_lower:
            return 1.0
        elif 'doubtful' in status_lower:
            return 0.8
        elif 'questionable' in status_lower:
            return 0.6
        elif 'probable' in status_lower:
            return 0.3
        elif 'limited' in status_lower:
            return 0.4
        else:
            return 0.2
    
    def _calculate_news_severity(self, text: str) -> float:
        """Calculate news impact severity"""
        text_lower = text.lower()
        severity = 0.0
        
        # High impact keywords
        if any(word in text_lower for word in ['suspended', 'arrested', 'surgery', 'season-ending']):
            severity += 0.8
        
        # Medium impact keywords
        if any(word in text_lower for word in ['injury', 'hurt', 'limited', 'questionable']):
            severity += 0.5
        
        # Low impact keywords
        if any(word in text_lower for word in ['probable', 'expected to play', 'cleared']):
            severity += 0.2
        
        return min(severity, 1.0)
    
    async def _get_player_projection(self, player_name: str) -> Optional[float]:
        """Get current player projection from cache or data"""
        # This would integrate with your existing player data
        # For now, return a placeholder
        return 10.0  # Would lookup actual projection
    
    def _get_current_nfl_week(self) -> int:
        """Calculate current NFL week"""
        # NFL season typically starts first Thursday in September
        # This is a simplified calculation
        now = datetime.now()
        if now.month >= 9:
            # Rough calculation - would need more precise logic
            return min(((now - datetime(now.year, 9, 1)).days // 7) + 1, 18)
        else:
            return 1
    
    def _load_stadium_coordinates(self) -> Dict[str, Dict[str, float]]:
        """Load NFL stadium coordinates for weather monitoring"""
        # Simplified stadium coordinates - you'd want a complete database
        return {
            'BUF': {'lat': 42.7738, 'lon': -78.7870},  # Buffalo
            'MIA': {'lat': 25.9580, 'lon': -80.2389},  # Miami
            'NE': {'lat': 42.0909, 'lon': -71.2643},   # New England
            'NYJ': {'lat': 40.8135, 'lon': -74.0745},  # New York Jets
            'BAL': {'lat': 39.2780, 'lon': -76.6227},  # Baltimore
            'CIN': {'lat': 39.0955, 'lon': -84.5161},  # Cincinnati
            'CLE': {'lat': 41.5061, 'lon': -81.6995},  # Cleveland
            'PIT': {'lat': 40.4469, 'lon': -80.0158},  # Pittsburgh
            'HOU': {'lat': 29.6847, 'lon': -95.4107},  # Houston
            'IND': {'lat': 39.7601, 'lon': -86.1639},  # Indianapolis
            'JAX': {'lat': 30.3240, 'lon': -81.6373},  # Jacksonville
            'TEN': {'lat': 36.1665, 'lon': -86.7713},  # Tennessee
            'DEN': {'lat': 39.7439, 'lon': -105.0201}, # Denver
            'KC': {'lat': 39.0489, 'lon': -94.4839},   # Kansas City
            'LV': {'lat': 36.0909, 'lon': -115.1833},  # Las Vegas
            'LAC': {'lat': 33.8644, 'lon': -118.2610}, # Los Angeles Chargers
            'DAL': {'lat': 32.7473, 'lon': -97.0945},  # Dallas
            'NYG': {'lat': 40.8135, 'lon': -74.0745},  # New York Giants
            'PHI': {'lat': 39.9008, 'lon': -75.1675},  # Philadelphia
            'WAS': {'lat': 38.9077, 'lon': -76.8644},  # Washington
            'CHI': {'lat': 41.8623, 'lon': -87.6167},  # Chicago
            'DET': {'lat': 42.3400, 'lon': -83.0456},  # Detroit
            'GB': {'lat': 44.5013, 'lon': -88.0622},   # Green Bay
            'MIN': {'lat': 44.9737, 'lon': -93.2581},  # Minnesota
            'ATL': {'lat': 33.7553, 'lon': -84.4006},  # Atlanta
            'CAR': {'lat': 35.2271, 'lon': -80.8526},  # Carolina
            'NO': {'lat': 29.9511, 'lon': -90.0812},   # New Orleans
            'TB': {'lat': 27.9759, 'lon': -82.5034},   # Tampa Bay
            'ARI': {'lat': 33.5276, 'lon': -112.2626}, # Arizona
            'LAR': {'lat': 34.0141, 'lon': -118.2879}, # Los Angeles Rams
            'SF': {'lat': 37.4032, 'lon': -121.9698},  # San Francisco
            'SEA': {'lat': 47.5952, 'lon': -122.3316}  # Seattle
        }

    async def get_recent_updates(self, hours: int = 24) -> List[Dict]:
        """Get recent player updates for the dashboard"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get updates from current hour cache
        current_hour_key = f"all_updates:{datetime.now().strftime('%Y-%m-%d-%H')}"
        current_updates = await self.cache_manager.get(current_hour_key) or []
        
        # Filter by time
        recent_updates = [
            update for update in current_updates
            if datetime.fromisoformat(update['timestamp']) > cutoff_time
        ]
        
        # Sort by severity and timestamp
        recent_updates.sort(key=lambda x: (x['severity'], x['timestamp']), reverse=True)
        
        return recent_updates[:50]  # Return top 50
