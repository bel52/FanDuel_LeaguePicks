"""
Automated FanDuel salary data extraction
"""
import asyncio
from playwright.async_api import async_playwright
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
import redis
from config import config
import re

class FanDuelScraper:
    """Automated FanDuel salary extraction"""
    
    def __init__(self):
        self.base_url = "https://www.fanduel.com"
        self.redis_client = redis.from_url(config.REDIS_URL)
        self.cache_ttl = 3600 * 6  # 6 hours cache
        
    async def get_nfl_salaries(self, week: Optional[int] = None) -> pd.DataFrame:
        """
        Get current NFL salaries from FanDuel
        
        Args:
            week: Specific week number (optional)
            
        Returns:
            DataFrame with player salaries
        """
        # Check cache first
        cache_key = f"fanduel_salaries:nfl:{week or 'current'}"
        cached_data = self.redis_client.get(cache_key)
        
        if cached_data:
            logger.info("Using cached FanDuel salary data")
            return pd.read_json(cached_data)
        
        logger.info("Fetching fresh FanDuel salary data")
        
        # Try Playwright first (faster, more reliable)
        try:
            salaries = await self._scrape_with_playwright()
            if salaries is not None and not salaries.empty:
                # Cache the data
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    salaries.to_json()
                )
                return salaries
        except Exception as e:
            logger.warning(f"Playwright scraping failed: {e}")
        
        # Fallback to Selenium with undetected-chromedriver
        try:
            salaries = self._scrape_with_selenium()
            if salaries is not None and not salaries.empty:
                # Cache the data
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    salaries.to_json()
                )
                return salaries
        except Exception as e:
            logger.error(f"Selenium scraping failed: {e}")
        
        # Final fallback: use API endpoint if available
        try:
            salaries = await self._fetch_from_api()
            if salaries is not None and not salaries.empty:
                self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    salaries.to_json()
                )
                return salaries
        except Exception as e:
            logger.error(f"API fetch failed: {e}")
        
        return pd.DataFrame()
    
    async def _scrape_with_playwright(self) -> Optional[pd.DataFrame]:
        """Scrape using Playwright (headless)"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            page = await context.new_page()
            
            try:
                # Navigate to FanDuel NFL contests
                await page.goto(f"{self.base_url}/contests?sport=NFL", wait_until='networkidle')
                
                # Wait for content to load
                await page.wait_for_selector('.contest-card', timeout=10000)
                
                # Find and click on a main slate contest
                contests = await page.query_selector_all('.contest-card')
                
                for contest in contests[:5]:  # Check first 5 contests
                    title = await contest.query_selector('.contest-name')
                    if title:
                        title_text = await title.inner_text()
                        if 'main' in title_text.lower() or 'sun' in title_text.lower():
                            await contest.click()
                            break
                
                # Wait for player list to load
                await page.wait_for_selector('.player-list-item', timeout=10000)
                
                # Extract player data
                players_data = await page.evaluate("""
                    () => {
                        const players = [];
                        document.querySelectorAll('.player-list-item').forEach(item => {
                            const name = item.querySelector('.player-name')?.innerText || '';
                            const position = item.querySelector('.player-position')?.innerText || '';
                            const team = item.querySelector('.player-team')?.innerText || '';
                            const salary = item.querySelector('.player-salary')?.innerText || '';
                            const fppg = item.querySelector('.player-fppg')?.innerText || '';
                            
                            if (name && salary) {
                                players.push({
                                    Name: name,
                                    Position: position,
                                    Team: team,
                                    Salary: parseInt(salary.replace(/[$,]/g, '')) || 0,
                                    FPPG: parseFloat(fppg) || 0
                                });
                            }
                        });
                        return players;
                    }
                """)
                
                await browser.close()
                
                if players_data:
                    df = pd.DataFrame(players_data)
                    logger.info(f"Scraped {len(df)} players from FanDuel")
                    return df
                
            except Exception as e:
                logger.error(f"Playwright scraping error: {e}")
                await browser.close()
                return None
    
    def _scrape_with_selenium(self) -> Optional[pd.DataFrame]:
        """Scrape using Selenium with undetected-chromedriver"""
        try:
            # Setup undetected Chrome
            options = uc.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            
            driver = uc.Chrome(options=options)
            
            try:
                # Navigate to FanDuel
                driver.get(f"{self.base_url}/contests?sport=NFL")
                
                # Wait for page load
                wait = WebDriverWait(driver, 15)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "contest-card")))
                
                # Find main slate contest
                contests = driver.find_elements(By.CLASS_NAME, "contest-card")
                
                for contest in contests[:5]:
                    try:
                        title = contest.find_element(By.CLASS_NAME, "contest-name").text
                        if 'main' in title.lower() or 'sun' in title.lower():
                            contest.click()
                            break
                    except:
                        continue
                
                # Wait for player data
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "player-list-item")))
                time.sleep(2)  # Additional wait for dynamic content
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                players_data = []
                for player_item in soup.find_all(class_='player-list-item'):
                    try:
                        name = player_item.find(class_='player-name').text.strip()
                        position = player_item.find(class_='player-position').text.strip()
                        team = player_item.find(class_='player-team').text.strip()
                        salary_text = player_item.find(class_='player-salary').text.strip()
                        salary = int(re.sub(r'[^\d]', '', salary_text))
                        
                        fppg_elem = player_item.find(class_='player-fppg')
                        fppg = float(fppg_elem.text.strip()) if fppg_elem else 0
                        
                        players_data.append({
                            'Name': name,
                            'Position': position,
                            'Team': team,
                            'Salary': salary,
                            'FPPG': fppg
                        })
                    except Exception as e:
                        continue
                
                driver.quit()
                
                if players_data:
                    df = pd.DataFrame(players_data)
                    logger.info(f"Scraped {len(df)} players via Selenium")
                    return df
                
            except Exception as e:
                logger.error(f"Selenium error: {e}")
                driver.quit()
                return None
                
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            return None
    
    async def _fetch_from_api(self) -> Optional[pd.DataFrame]:
        """Try to fetch from unofficial API endpoints"""
        import aiohttp
        
        try:
            # Known FanDuel API endpoints (may change)
            api_urls = [
                "https://api.fanduel.com/contests/nfl/main/players",
                "https://api.fanduel.com/fixture-lists/nfl/main",
                "https://fanduel.com/api/nfl/contests/main/players"
            ]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Referer': 'https://www.fanduel.com/'
            }
            
            async with aiohttp.ClientSession() as session:
                for url in api_urls:
                    try:
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                # Parse the API response (structure varies)
                                players_data = self._parse_api_response(data)
                                
                                if players_data:
                                    df = pd.DataFrame(players_data)
                                    logger.info(f"Fetched {len(df)} players from API")
                                    return df
                                    
                    except Exception as e:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"API fetch error: {e}")
            return None
    
    def _parse_api_response(self, data: Dict) -> List[Dict]:
        """Parse various API response formats"""
        players_data = []
        
        # Try different response structures
        if 'players' in data:
            for player in data['players']:
                players_data.append({
                    'Name': player.get('name', ''),
                    'Position': player.get('position', ''),
                    'Team': player.get('team', ''),
                    'Salary': player.get('salary', 0),
                    'FPPG': player.get('fppg', 0)
                })
        elif 'fixtures' in data:
            for fixture in data['fixtures']:
                if 'players' in fixture:
                    for player in fixture['players']:
                        players_data.append({
                            'Name': player.get('display_name', ''),
                            'Position': player.get('position', ''),
                            'Team': player.get('team_code', ''),
                            'Salary': player.get('salary', 0),
                            'FPPG': player.get('fppg', 0)
                        })
        
        return players_data
    
    def merge_with_projections(self, salary_df: pd.DataFrame, 
                              projections_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge FanDuel salaries with projections
        
        Args:
            salary_df: FanDuel salary data
            projections_df: Player projections
            
        Returns:
            Merged DataFrame
        """
        # Standardize player names for matching
        salary_df['Name_Clean'] = salary_df['Name'].str.lower().str.strip()
        projections_df['Name_Clean'] = projections_df['Name'].str.lower().str.strip()
        
        # Merge on cleaned names
        merged = pd.merge(
            salary_df,
            projections_df,
            on='Name_Clean',
            how='left',
            suffixes=('', '_proj')
        )
        
        # Clean up
        merged = merged.drop(columns=['Name_Clean', 'Name_proj'], errors='ignore')
        
        return merged


class AlternativeSalarySource:
    """Backup methods for getting salary data"""
    
    @staticmethod
    async def get_from_dfs_sites() -> pd.DataFrame:
        """Get salary data from DFS aggregator sites"""
        sources = [
            "https://www.dailyfantasynerd.com/optimizer/fanduel/nfl",
            "https://rotogrinders.com/projected-stats/nfl",
            "https://www.linestarapp.com/DesktopSlate/All/Sport/2/Site/4"
        ]
        
        # Implementation would scrape these sites
        # This is a fallback if FanDuel direct access fails
        return pd.DataFrame()
    
    @staticmethod
    async def estimate_salaries(player_stats: pd.DataFrame) -> pd.DataFrame:
        """Estimate salaries based on player performance"""
        # Use historical salary-to-performance ratios
        # This is a last resort fallback
        
        if player_stats.empty:
            return pd.DataFrame()
        
        # Basic estimation formula
        player_stats['Estimated_Salary'] = (
            player_stats['avg_points'] * 500 + 3000
        ).clip(3000, 12000).astype(int)
        
        # Adjust by position
        position_multipliers = {
            'QB': 1.1,
            'RB': 1.0,
            'WR': 0.95,
            'TE': 0.85,
            'DST': 0.7
        }
        
        for pos, mult in position_multipliers.items():
            mask = player_stats['Position'] == pos
            player_stats.loc[mask, 'Estimated_Salary'] *= mult
        
        player_stats['Salary'] = player_stats['Estimated_Salary'].astype(int)
        
        return player_stats
