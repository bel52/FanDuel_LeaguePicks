import os, csv, logging

logger = logging.getLogger(__name__)

def setup_logging(log_name:str):
    os.makedirs(os.path.dirname(log_name), exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers if script reruns
    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Logging started for {log_name}")

def read_csv(path:str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({k:(v.strip() if isinstance(v,str) else v) for k,v in row.items()})
    return out

def write_csv(path:str, fieldnames:list, rows:list, mode:str='w'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"Wrote {len(rows)} rows to {path}")

def current_week():
    """Infer week as (last Week in data/season/standings.csv) + 1; fallback to 1."""
    try:
        rows = read_csv(os.path.join('data','season','standings.csv'))
        weeks = [int(r['Week']) for r in rows if r.get('Week')]
        return (max(weeks) + 1) if weeks else 1
    except Exception as e:
        logger.warning(f"current_week(): fallback to 1 ({e})")
        return 1

def value_per_1k(proj:float, salary:int)->float:
    denom = (salary or 1)/1000
    return round((proj or 0)/denom, 3)
