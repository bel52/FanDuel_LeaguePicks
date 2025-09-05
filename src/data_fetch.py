import os, requests, csv, re
from src import util

ODDS_API_KEY = os.getenv('ODDS_API_KEY')
ODDS_API_URL = ("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/"
                "odds?regions=us&oddsFormat=american&markets=spreads,totals&apiKey={key}")

def _parse_name_team_pos(s: str):
    # "Ja'Marr Chase (CIN - WR)" -> ("Ja'Marr Chase","CIN","WR")
    name = s.split('(')[0].strip()
    m = re.search(r'\(([^)]+)\)', s)
    team, pos = '', ''
    if m:
        inside = m.group(1)  # "CIN - WR"
        parts = [p.strip() for p in inside.split('-')]
        if len(parts) >= 1: team = parts[0]
        if len(parts) >= 2: pos = parts[1]
    return name, team, pos

def _parse_salary(s: str) -> int:
    s = (s or '').replace('$','').replace(',','').strip()
    try: return int(float(s))
    except: return 0

def _parse_proj(s: str) -> float:
    try: return float((s or '0').strip())
    except: return 0.0

def _kick_slot(kick: str) -> tuple[str,str]:
    """Return ('EARLY'|'LATE'|'' , original) from FantasyPros 'KICKOFF' like 'Sun 1:00PM'."""
    raw = (kick or '').strip()
    k = raw.lower()
    if 'sun' in k and ('1:00pm' in k or '1:00 pm' in k): return 'EARLY', raw
    if 'sun' in k and ('4:05pm' in k or '4:25pm' in k or '4:05 pm' in k or '4:25 pm' in k): return 'LATE', raw
    # all other slates (TNF/SNF/MNF/International/etc.) -> ''
    return '', raw

def _opp_from_fp(opp: str) -> str:
    """'@BUF' -> BUF ; 'BUF' -> BUF"""
    o = (opp or '').strip().upper()
    return o[1:] if o.startswith('@') else o

def _load_fp_file(path: str, sunday_main_only=True):
    out = []
    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        for r in rd:
            name_raw = (r.get('PLAYER NAME') or '').strip()
            if not name_raw: continue
            name, team, pos = _parse_name_team_pos(name_raw)
            if not team or not pos:  # skip defenses in wrong file, kickers, etc.
                continue

            # Sunday Main filter
            kick = (r.get('KICKOFF') or '')
            slot, raw = _kick_slot(kick)
            if sunday_main_only and slot not in ('EARLY','LATE'):
                continue

            salary = _parse_salary(r.get('SALARY'))
            proj   = _parse_proj(r.get('PROJ PTS'))
            opp    = _opp_from_fp(r.get('OPP') or '')

            out.append({
                'Player': name, 'Team': team, 'Opp': opp, 'Pos': pos,
                'Salary': salary, 'ProjFP': proj,
                'KickSlot': slot, 'KickRaw': raw,
            })
    return out

def load_projections():
    """Cheat Sheet CSVs → unified list, deduped by (Player,Team,Pos).
       Expected files in data/fantasypros/: qb.csv rb.csv wr.csv te.csv dst.csv"""
    base = os.path.join('data','fantasypros')
    files = ['qb.csv','rb.csv','wr.csv','te.csv','dst.csv']
    total_counts = {}
    raw = []
    for fn in files:
        p = os.path.join(base, fn)
        rows = _load_fp_file(p)
        total_counts[fn] = sum(1 for _ in open(p, 'r', encoding='utf-8', newline='')) - 1  # rough
        raw.extend(rows)

    # Deduplicate Sunday Main players
    seen = set()
    dedup = []
    for p in raw:
        key = (p['Player'], p['Team'], p['Pos'])
        if key in seen: continue
        seen.add(key)
        dedup.append(p)

    # Per-position counts
    perpos = {'QB':0,'RB':0,'WR':0,'TE':0,'DST':0}
    for p in dedup:
        if p['Pos'] in perpos: perpos[p['Pos']] += 1

    util.logger.info(f"CheatSheets rows per file: {total_counts} | kept Sunday-Main unique: {len(dedup)}")
    util.logger.info(f"Loaded per-pos: {perpos}")
    return dedup

# ---------------- Odds (implied totals) ----------------
def fetch_odds():
    if not ODDS_API_KEY:
        util.logger.warning("ODDS_API_KEY not set; skipping odds fetch.")
        return []
    url = ODDS_API_URL.format(key=ODDS_API_KEY)
    try:
        r = requests.get(url, timeout=12); r.raise_for_status()
        payload = r.json()
    except Exception as e:
        util.logger.error(f"Odds fetch failed: {e}")
        return []
    games = []
    for g in payload:
        home = g.get('home_team'); away = g.get('away_team')
        bks = g.get('bookmakers') or []
        if not bks: continue
        mkts = bks[0].get('markets', [])
        spread = next((m for m in mkts if m.get('key')=='spreads'), None)
        total  = next((m for m in mkts if m.get('key')=='totals'), None)
        over_under = None
        spread_pts = None
        fav = None
        if spread and spread.get('outcomes'):
            for o in spread['outcomes']:
                pts = o.get('point')
                if pts is None: continue
                if float(pts) < 0:
                    fav = o.get('name')
                    spread_pts = float(pts)
        if total and total.get('outcomes'):
            over = next((o for o in total['outcomes'] if o.get('name')=='Over'), None)
            if over and over.get('point') is not None:
                over_under = float(over['point'])
        home_it = away_it = None
        if over_under is not None:
            if spread_pts is not None and fav:
                fav_total = (over_under/2) - (spread_pts/2)
                dog_total = over_under - fav_total
                if fav == home:
                    home_it, away_it = fav_total, dog_total
                elif fav == away:
                    away_it, home_it = fav_total, dog_total
            else:
                home_it = away_it = over_under/2
        games.append({
            'Home': home or '', 'Away': away or '',
            'Total': f"{over_under:.1f}" if over_under is not None else '',
            'SpreadFav': fav or '', 'SpreadPts': f"{spread_pts:.1f}" if spread_pts is not None else '',
            'HomeImplied': f"{home_it:.1f}" if home_it is not None else '',
            'AwayImplied': f"{away_it:.1f}" if away_it is not None else '',
        })
    util.logger.info(f"Fetched odds for {len(games)} games.")
    return games
