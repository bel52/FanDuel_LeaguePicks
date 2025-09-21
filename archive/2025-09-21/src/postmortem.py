from datetime import datetime
from src import util

def log_week_result(week:int, bot_points:float, opp_points:float|None, standings_path="data/season/standings.csv"):
    row = {
        'Week': week,
        'BotPoints': f"{bot_points:.1f}",
        'OppPoints': f"{opp_points:.1f}" if opp_points is not None else '',
        'Result': ('Win' if opp_points is not None and bot_points>opp_points else
                   'Loss' if opp_points is not None and bot_points<opp_points else
                   'Tie' if opp_points is not None else ''),
        'BotW-L': '', 'SeasonPoints': '', 'Rank': ''
    }
    util.write_csv(standings_path, fieldnames=list(row.keys()), rows=[row], mode='a')
    return row

def log_bankroll(entry_fee:float, winnings:float, bankroll_path="data/bankroll/bankroll.csv"):
    profit = winnings - entry_fee
    roi = (profit/entry_fee*100) if entry_fee>0 else 0.0
    row = {
        'Date': datetime.now().strftime("%Y-%m-%d"),
        'Week': '', 'EntryFee': f"{entry_fee:.2f}", 'Winnings': f"{winnings:.2f}",
        'Profit': f"{profit:.2f}", 'ROI': f"{roi:.1f}%"
    }
    util.write_csv(bankroll_path, fieldnames=list(row.keys()), rows=[row], mode='a')
    return row
