import asyncio
from fanduel_salary_scraper import get_fanduel_salaries

async def main():
    players = await get_fanduel_salaries('data/fanduel_salaries_manual.csv')
    print(f"Loaded {len(players)} players")
    print("First player:", players[0])

asyncio.run(main())
