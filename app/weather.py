from __future__ import annotations
import requests
from typing import Optional, Dict
from app.config import OPENWEATHER_API_KEY, WEATHER_WIND_PENALTY, WEATHER_RAIN_PENALTY

def fetch_weather(lat: float, lon: float) -> Optional[Dict]:
    if not OPENWEATHER_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "imperial"},
            timeout=20,
        )
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None

def weather_adjustment(weather_json: Optional[Dict]) -> float:
    if not weather_json:
        return 1.0
    try:
        wind = float(weather_json.get("wind", {}).get("speed", 0.0))
        rainy = any(
            k in weather_json.get("weather", [{}])[0].get("main", "").lower()
            for k in ["rain","drizzle","thunderstorm"]
        )
        mult = 1.0
        if wind > 15:
            buckets = int((wind - 15) // 5) + 1
            mult -= buckets * WEATHER_WIND_PENALTY
        if rainy:
            mult -= WEATHER_RAIN_PENALTY
        return max(0.85, mult)
    except Exception:
        return 1.0
