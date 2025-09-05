from src.util import value_per_1k

VALUE_THRESHOLD = 1.6
VALUE_THRESHOLD_TE_DST = 1.5

def meets_value(player):
    v = value_per_1k(player.get('ProjFP',0), player.get('Salary',0))
    if player['Pos'] in ('TE','DST'):
        return v >= VALUE_THRESHOLD_TE_DST
    return v >= VALUE_THRESHOLD

def adjusted_score(player, game_info:dict|None):
    base = float(player.get('ProjFP',0.0))
    adj = base
    if game_info:
        it = game_info.get('implied_total')
        if it is not None:
            if it >= 28: adj += base*0.05
            elif it <= 18: adj -= base*0.05
        wx = game_info.get('weather') or {}
        wind = wx.get('wind')
        precip = wx.get('precipProb') or 0
        desc = str(wx.get('shortForecast','')).lower()
        try:
            if isinstance(wind,str) and wind:
                wind_val = int(wind.split()[0])
            else:
                wind_val = int(wind) if wind is not None else None
        except:
            wind_val = None
        if player['Pos'] in ('QB','WR'):
            if wind_val and wind_val >= 20: adj -= base*0.10
            elif wind_val and wind_val >= 15: adj -= base*0.05
            if precip and precip >= 60: adj -= base*0.05
            if 'heavy' in desc or 'snow' in desc: adj -= base*0.07
        if player['Pos']=='RB':
            if wind_val and wind_val >= 20: adj += base*0.05
            if precip and precip >= 60: adj += base*0.05
        if player['Pos']=='DST':
            if precip and precip >= 60: adj += 0.5
    return round(adj,2)
