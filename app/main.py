import logging
import os
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import PlainTextResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="FanDuel NFL DFS Optimizer", version="3.0.0")

def load_csv_data():
    """Simple CSV data loader"""
    input_dir = "/app/data/input"
    all_players = []
    
    files = ["qb.csv", "rb.csv", "wr.csv", "te.csv", "dst.csv"]
    
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                # Extract player info from different possible column formats
                name = ""
                for name_col in ['PLAYER NAME', 'Player', 'NAME']:
                    if name_col in row and pd.notna(row[name_col]):
                        name = str(row[name_col])
                        break
                
                if not name or name == 'nan':
                    continue
                
                # Extract team and position from name like "Josh Allen (BUF - QB)"
                team = pos = ""
                if '(' in name and ')' in name:
                    parts = name.split('(')[1].split(')')[0]
                    if ' - ' in parts:
                        team, pos = parts.split(' - ')
                        name = name.split('(')[0].strip()
                
                # Get salary and projection
                salary = 0
                for sal_col in ['SALARY', 'Salary', 'SAL', 'PRICE']:
                    if sal_col in row:
                        sal_val = str(row[sal_col]).replace('$', '').replace(',', '')
                        try:
                            salary = int(float(sal_val))
                            break
                        except:
                            continue
                
                proj_pts = 0.0
                for proj_col in ['PROJ PTS', 'Proj Pts', 'PROJ', 'FPTS']:
                    if proj_col in row:
                        try:
                            proj_pts = float(row[proj_col])
                            break
                        except:
                            continue
                
                # Get opponent
                opp = str(row.get('OPP', row.get('Opponent', ''))).replace('@', '').strip()
                
                if name and salary > 0 and proj_pts > 0:
                    all_players.append({
                        'PLAYER NAME': name,
                        'POS': pos,
                        'TEAM': team,
                        'OPP': opp,
                        'SALARY': salary,
                        'PROJ PTS': proj_pts
                    })
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            continue
    
    return pd.DataFrame(all_players)

def simple_optimize(df, game_type="league"):
    """Simple greedy optimization"""
    if df.empty:
        return []
    
    # Calculate value (points per $1000)
    df['value'] = df['PROJ PTS'] / (df['SALARY'] / 1000)
    
    # For H2H, prioritize ceiling (higher projections)
    if game_type == "h2h":
        df = df.sort_values(['PROJ PTS'], ascending=False)
    else:
        # For league, prioritize value
        df = df.sort_values(['value'], ascending=False)
    
    lineup = []
    total_salary = 0
    salary_cap = 60000
    
    # Position requirements
    positions_needed = {
        'QB': 1,
        'RB': 2, 
        'WR': 3,
        'TE': 1,
        'DST': 1
    }
    
    # Add one more flex (RB/WR/TE)
    flex_needed = 1
    
    # First pass - fill required positions
    for _, player in df.iterrows():
        pos = player['POS']
        salary = player['SALARY']
        
        if pos in positions_needed and positions_needed[pos] > 0:
            if total_salary + salary <= salary_cap:
                lineup.append(player)
                total_salary += salary
                positions_needed[pos] -= 1
    
    # Second pass - add flex players
    for _, player in df.iterrows():
        if flex_needed <= 0:
            break
            
        pos = player['POS']
        salary = player['SALARY']
        
        # Check if already in lineup
        if any(p['PLAYER NAME'] == player['PLAYER NAME'] for p in lineup):
            continue
            
        if pos in ['RB', 'WR', 'TE'] and total_salary + salary <= salary_cap:
            lineup.append(player)
            total_salary += salary
            flex_needed -= 1
    
    return lineup

def format_lineup_text(lineup, game_type="league"):
    """Format lineup as readable text"""
    if not lineup:
        return "No lineup generated - check data files"
    
    result = f"FANDUEL NFL DFS LINEUP - {game_type.upper()} STRATEGY\n"
    result += "=" * 50 + "\n\n"
    
    # Header
    result += f"{'POS':<4} {'PLAYER':<20} {'TEAM':<4} {'SALARY':>7} {'PROJ':>6}\n"
    result += "-" * 50 + "\n"
    
    total_salary = 0
    total_proj = 0.0
    
    # Sort lineup by position for display
    pos_order = ['QB', 'RB', 'WR', 'TE', 'DST']
    lineup_sorted = []
    
    for pos in pos_order:
        pos_players = [p for p in lineup if p['POS'] == pos]
        lineup_sorted.extend(pos_players)
    
    # Add any remaining players (flex)
    for player in lineup:
        if player not in lineup_sorted:
            lineup_sorted.append(player)
    
    for i, player in enumerate(lineup_sorted):
        pos_label = player['POS']
        if player['POS'] in ['RB', 'WR', 'TE'] and i >= 7:  # FLEX position
            pos_label = "FLEX"
            
        result += f"{pos_label:<4} {player['PLAYER NAME']:<20} {player['TEAM']:<4} ${player['SALARY']:>6} {player['PROJ PTS']:>6.1f}\n"
        total_salary += player['SALARY']
        total_proj += player['PROJ PTS']
    
    result += "-" * 50 + "\n"
    result += f"TOTAL:{'':<20} {'':<4} ${total_salary:>6} {total_proj:>6.1f}\n"
    result += f"SALARY REMAINING: ${60000 - total_salary}\n"
    
    return result

@app.get("/")
def root():
    return {
        "app": "FanDuel NFL DFS Optimizer",
        "version": "3.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    input_dir = "/app/data/input"
    files_found = []
    
    for pos in ["qb", "rb", "wr", "te", "dst"]:
        if os.path.exists(os.path.join(input_dir, f"{pos}.csv")):
            files_found.append(f"{pos}.csv")
    
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            "data": f"files found: {', '.join(files_found)}" if files_found else "no files found"
        }
    }

@app.get("/data/status")
def data_status():
    input_dir = "/app/data/input"
    status = {"files_present": {}}
    
    for pos in ["qb", "rb", "wr", "te", "dst"]:
        file_path = os.path.join(input_dir, f"{pos}.csv")
        status["files_present"][f"{pos}.csv"] = os.path.exists(file_path)
    
    return status

@app.get("/optimize_text", response_class=PlainTextResponse)
def optimize_text(
    game_type: str = Query("league", regex="^(league|h2h)$"),
    salary_cap: int = Query(60000),
    width: int = Query(100)
):
    """Generate optimized lineup in text format"""
    try:
        # Load data
        df = load_csv_data()
        
        if df.empty:
            return "No player data found. Please ensure CSV files are in /app/data/input/"
        
        # Optimize
        lineup = simple_optimize(df, game_type)
        
        if not lineup:
            return f"Could not generate a valid {game_type} lineup with available players."
        
        # Format and return
        return format_lineup_text(lineup, game_type)
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return f"Error generating lineup: {str(e)}"

@app.get("/optimize")
def optimize_json(
    game_type: str = Query("league", regex="^(league|h2h)$")
):
    """Generate optimized lineup in JSON format"""
    try:
        df = load_csv_data()
        
        if df.empty:
            raise HTTPException(status_code=422, detail="No player data available")
        
        lineup = simple_optimize(df, game_type)
        
        if not lineup:
            raise HTTPException(status_code=422, detail="Could not generate valid lineup")
        
        return {
            "lineup_players": [
                {
                    "name": p['PLAYER NAME'],
                    "position": p['POS'],
                    "team": p['TEAM'],
                    "opponent": p['OPP'],
                    "proj_points": p['PROJ PTS'],
                    "salary": p['SALARY']
                }
                for p in lineup
            ],
            "total_projection": sum(p['PROJ PTS'] for p in lineup),
            "total_salary": sum(p['SALARY'] for p in lineup),
            "game_type": game_type
        }
        
    except Exception as e:
        logger.error(f"JSON optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
