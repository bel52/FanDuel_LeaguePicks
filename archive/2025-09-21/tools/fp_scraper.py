#!/usr/bin/env python3
"""
FantasyPros DFS Cheat Sheet scraper (FanDuel / Main).
- Logs in once (storage state persisted), then runs headless thereafter.
- Iterates positions: QB, RB, WR, TE, DST
- Extracts table columns and writes app-ready CSVs:
    data/fantasypros/{qb,rb,wr,te,dst}.csv
- Auto-detects headers like "PROJ PTS" / "Projection" and "SALARY" / "FD Salary".
- Defaults to current (visible) week; optional --week to switch.

Usage:
  # First run (interactive login, headful):
  python tools/fp_scraper.py --headful --save-state .auth/fp_state.json

  # Subsequent runs (headless, reusing login):
  python tools/fp_scraper.py --state .auth/fp_state.json

  # Force week:
  python tools/fp_scraper.py --state .auth/fp_state.json --week 2
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from playwright.sync_api import sync_playwright, Page, expect

SITE_URL = "https://www.fantasypros.com/nfl/dfs/cheatsheets/"
POS_TABS = ["QB", "RB", "WR", "TE", "DST"]
OUT_MAP = {"QB":"qb.csv","RB":"rb.csv","WR":"wr.csv","TE":"te.csv","DST":"dst.csv"}
OUT_DIR = Path("data/fantasypros")

EXPECTED_HEADERS = ["PLAYER NAME","OPP","PROJ PTS","SALARY"]

def norm_header(h: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (h or '').strip().lower())

def clean_money(x: str) -> int:
    s = (x or '').replace('$','').replace(',','').strip()
    try:
        return int(float(s))
    except Exception:
        return 0

def to_float(x: str) -> float:
    s = (x or '').replace(',','').strip()
    try:
        return float(s)
    except Exception:
        return 0.0

def build_player_name(name: str, team: str, pos: str) -> str:
    name = (name or '').strip()
    team = (team or '').strip().upper()
    pos  = (pos or '').strip().upper()
    return f"{name} ({team} - {pos})"

def find_dropdown_and_select(page: Page, label_text: str, item_text: str) -> None:
    """
    Clicks a labeled dropdown (e.g., 'FanDuel' / 'Main') and selects the item.
    Works by locating the button near the visible text, then clicking menu item.
    """
    # Try button with exact text first
    button = page.get_by_role("button", name=re.compile(rf"^{re.escape(item_text)}$", re.I))
    alt = page.get_by_text(label_text, exact=False)
    # Open menu by clicking the current selection button near label, or any known button
    try:
        button.first.click(timeout=3000)
    except Exception:
        # Try the generic dropdown caret near label text
        try:
            alt.locator("xpath=..").locator("button").first.click(timeout=3000)
        except Exception:
            # Fallback: click any dropdown button
            page.locator("button").filter(has_text=re.compile(label_text, re.I)).first.click(timeout=3000)
    # Now select menu item
    page.get_by_role("menuitem", name=re.compile(rf"^{re.escape(item_text)}$", re.I)).first.click(timeout=5000)

def maybe_switch_week(page: Page, week: Optional[int]) -> None:
    if week is None:
        return
    # Look for a "Week" control then pick target week
    # Often the page shows "Week N" near the title.
    try:
        # Click the "Week" menu button (contains 'Week')
        page.get_by_role("button", name=re.compile(r"Week\s*\d+", re.I)).first.click(timeout=3000)
        page.get_by_role("menuitem", name=re.compile(rf"^\s*Week\s*{week}\s*$", re.I)).first.click(timeout=5000)
    except Exception:
        # Some layouts use a dropdown near the page title
        try:
            page.locator("text=Week").locator("xpath=..").locator("button").first.click(timeout=3000)
            page.get_by_role("menuitem", name=re.compile(rf"^\s*Week\s*{week}\s*$", re.I)).first.click(timeout=5000)
        except Exception:
            print(f"[warn] Could not switch week to {week}. Continuing with current page week.")

def goto_tab(page: Page, pos: str) -> None:
    # Tabs are usually role="tab" with text QB/RB/WR/TE/DST
    tab = page.get_by_role("tab", name=re.compile(rf"^{pos}$", re.I))
    tab.first.click(timeout=5000)

def wait_for_table(page: Page) -> None:
    # Wait for at least one table row to appear
    # Using table role -> grid / rowgroup may vary; we fallback to 'table' css.
    page.wait_for_timeout(250)  # tiny debounce
    for _ in range(40):
        rows = page.locator("table tbody tr")
        if rows.count() > 0:
            return
        page.wait_for_timeout(250)
    raise RuntimeError("Timed out waiting for table rows.")

def header_index_map(page: Page) -> Dict[str, int]:
    headers = []
    ths = page.locator("table thead tr th")
    cnt = ths.count()
    for i in range(cnt):
        txt = ths.nth(i).inner_text().strip()
        headers.append(txt)
    # Build a map for important columns
    hmap: Dict[str, int] = {}
    for i, h in enumerate(headers):
        hn = norm_header(h)
        if hn in ("player","playername","name"):
            hmap.setdefault("PLAYER", i)
        if hn in ("opp","opponent"):
            hmap.setdefault("OPP", i)
        if hn in ("projpts","projections","proj","projection","fpts","points","predscore","predictedscore"):
            hmap.setdefault("PROJ", i)
        if hn in ("salary","fdsalary","fd","cost","sal"):
            hmap.setdefault("SAL", i)
        if hn in ("team","tm"):
            hmap.setdefault("TEAM", i)
        if hn in ("position","pos"):
            hmap.setdefault("POS", i)
    return hmap

def scrape_position(page: Page, pos: str) -> List[Dict[str, str]]:
    goto_tab(page, pos)
    wait_for_table(page)
    hmap = header_index_map(page)
    # basic row extraction
    rows = page.locator("table tbody tr")
    n = rows.count()
    out: List[Dict[str,str]] = []
    for i in range(n):
        tds = rows.nth(i).locator("td")
        td_cnt = tds.count()
        if td_cnt == 0:
            continue
        # Helper to read a cell safely
        def cell(idx: Optional[int]) -> str:
            if idx is None or idx >= td_cnt:
                return ""
            return (tds.nth(idx).inner_text() or "").strip()

        # Player text + team/pos may be in same cell via <a> or sub-spans
        player_txt = cell(hmap.get("PLAYER"))
        team_txt   = cell(hmap.get("TEAM"))
        pos_txt    = pos  # default to tab if no explicit pos col
        if not team_txt:
            # Try to parse "(TEAM - POS)" that may be rendered with the name
            m = re.search(r'\(([A-Z]{2,3})\s*-\s*(QB|RB|WR|TE|DST)\)', player_txt)
            if m:
                team_txt = m.group(1)
                pos_txt  = m.group(2)
        opp_txt    = cell(hmap.get("OPP"))
        proj_txt   = cell(hmap.get("PROJ"))
        sal_txt    = cell(hmap.get("SAL"))

        # If the player cell has a link with just the name, try to pull just the name text
        # Remove any "(TEAM - POS)" suffix if present
        name = player_txt
        name = re.sub(r'\s*\(([A-Z]{2,3})\s*-\s*(QB|RB|WR|TE|DST)\)\s*', '', name).strip()

        row = {
            "PLAYER NAME": build_player_name(name, team_txt, pos_txt),
            "OPP": (opp_txt or "").upper(),
            "PROJ PTS": f"{to_float(proj_txt):.2f}",
            "SALARY": str(clean_money(sal_txt)),
        }
        # Simple validation
        if name and row["SALARY"].isdigit():
            out.append(row)
    return out

def write_csv(path: Path, rows: List[Dict[str,str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=EXPECTED_HEADERS)
        wr.writeheader()
        wr.writerows(rows)

def run(state_in: Optional[str], state_out: Optional[str], week: Optional[int], headful: bool) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        context_kwargs = {}
        if state_in:
            context_kwargs["storage_state"] = state_in
        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        page.goto(SITE_URL, wait_until="domcontentloaded")
        if headful:
            print("\nIf not logged in, please log in to FantasyPros now, then return here.\n"
                  "Once the Cheat Sheet loads fully, the script will continue.\n")
        # Wait for the main cheat sheet title area
        try:
            page.wait_for_selector("text=Cheat Sheet", timeout=15000)
        except Exception:
            pass

        # Ensure Site: FanDuel and Slate: Main
        try:
            # The two dropdowns are usually visible near the right/top of the table header
            find_dropdown_and_select(page, "Site", "FanDuel")
        except Exception:
            pass
        try:
            find_dropdown_and_select(page, "Slate", "Main")
        except Exception:
            pass

        # Switch week if requested
        maybe_switch_week(page, week)

        # Scrape each position
        all_counts: Dict[str,int] = {}
        for pos in POS_TABS:
            try:
                rows = scrape_position(page, pos)
            except Exception as e:
                print(f"[warn] {pos} scrape failed: {e}")
                rows = []
            out_path = OUT_DIR / OUT_MAP[pos]
            write_csv(out_path, rows)
            all_counts[pos] = len(rows)
            print(f"[{pos}] → {out_path} ({len(rows)} rows)")

        # Save state (persist login) if requested
        if state_out:
            context.storage_state(path=state_out)
            print(f"Saved login/session state to {state_out}")

        context.close()
        browser.close()

        # Basic sanity check
        if sum(all_counts.values()) == 0:
            print("\n[ERROR] No rows scraped. Are you logged in and on the DFS Cheat Sheet?")
            sys.exit(2)
        else:
            print("\nDone.")

def main():
    ap = argparse.ArgumentParser(description="Scrape FantasyPros DFS Cheat Sheet → app-ready CSVs.")
    ap.add_argument("--state", help="Path to existing Playwright storage state (reuses login).", default=None)
    ap.add_argument("--save-state", help="Path to save storage state after run (for future headless runs).", default=None)
    ap.add_argument("--week", type=int, help="Week number to select (optional).", default=None)
    ap.add_argument("--headful", action="store_true", help="Run with a visible browser (first login).")
    args = ap.parse_args()
    run(args.state, args.save_state, args.week, args.headful)

if __name__ == "__main__":
    main()
