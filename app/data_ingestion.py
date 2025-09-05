import os
import logging
import pandas as pd

from .normalizer import normalize

# Optional Playwright usage
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

class FantasyProsAutomation:
    def __init__(self):
        self.auth_state_file = "fantasypros_auth.json"

    def authenticate_and_save_state(self, username, password):
        if sync_playwright is None:
            logging.error("Playwright not installed/available.")
            return False
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto('https://fantasypros.com/login')
            page.fill('#username', username)
            page.fill('#password', password)
            page.click('button[type=submit]')
            page.wait_for_url('**/dashboard**')
            context.storage_state(path=self.auth_state_file)
            browser.close()
            logging.info("FantasyPros login successful; state saved.")
            return True

    def collect_data_with_rate_limiting(self):
        if sync_playwright is None:
            logging.error("Playwright not available for collection.")
            return None
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(storage_state=self.auth_state_file)
            page = context.new_page()
            import time
            time.sleep(2)  # respectful delay
            page.goto('https://fantasypros.com/premium-data')
            data = None
            try:
                data = page.evaluate("() => window.playerData")
            except Exception as e:
                logging.error(f"Data extraction failed: {e}")
            browser.close()
            return data

def _load_csv(path: str, pos_hint: str):
    try:
        df = pd.read_csv(path)
        return normalize(df, pos_hint=pos_hint)
    except Exception as e:
        logging.error(f"Failed to read/normalize {path}: {e}")
        return None

def load_weekly_data():
    """Load and merge weekly projections from data/input/{qb,rb,wr,te,dst}.csv.
       If none present, try FantasyPros scraping (requires creds)."""
    data_dir = "data/input"
    plan = [
        ("qb.csv","QB"),
        ("rb.csv","RB"),
        ("wr.csv","WR"),
        ("te.csv","TE"),
        ("dst.csv","DST"),
    ]
    frames = []
    existing = []
    for fname, phint in plan:
        fpath = os.path.join(data_dir, fname)
        if os.path.isfile(fpath):
            existing.append(fpath)
            frames.append(_load_csv(fpath, phint))
    frames = [f for f in frames if f is not None and not f.empty]
    if frames:
        df = pd.concat(frames, ignore_index=True)
        # Deduplicate by name/team/pos, keep max proj
        df = (df.sort_values('PROJ PTS', ascending=False)
                .drop_duplicates(subset=['PLAYER NAME','TEAM','POS'], keep='first')
                .reset_index(drop=True))
        logging.info(f"Loaded {len(df)} players from: {', '.join(os.path.basename(x) for x in existing)}")
        return df

    # Else: attempt scrape if creds provided
    user = os.getenv('FANTASYPROS_USER')
    pwd = os.getenv('FANTASYPROS_PASS')
    if not user or not pwd:
        logging.error("No local CSVs and FantasyPros credentials not provided.")
        return None
    logging.info("No local CSVs found. Starting FantasyPros scrape...")
    fp = FantasyProsAutomation()
    if not os.path.exists("fantasypros_auth.json"):
        ok = fp.authenticate_and_save_state(user, pwd)
        if not ok:
            return None
    data = fp.collect_data_with_rate_limiting()
    if data is None:
        return None
    try:
        df = pd.DataFrame(data)
        df = normalize(df, pos_hint=None)
        logging.info(f"Scraped data normalized: {len(df)} players.")
        return df
    except Exception as e:
        logging.error(f"Error processing scraped JSON: {e}")
        return None
