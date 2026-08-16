"""Kickoff times and lock state.

FanDuel locks each player at his game's scheduled kickoff (in late-swap-eligible
contests); the whole slate never locks at once. Late swap is therefore a function of
one thing: which games have started. This module answers that.

Primary source: nflverse schedules (free, no quota, includes exact kickoff datetimes).
Fallback: The Odds API `commence_time` already captured on TeamLine.

All times are handled in UTC internally and rendered in US/Eastern for display,
because that is how NFL schedules are published and how Brett thinks about them.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from .matching import norm_team

ET = ZoneInfo("America/New_York")


@dataclass
class GameTime:
    team: str
    opponent: str
    kickoff_utc: datetime
    game_id: str = ""

    def locked(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return now >= self.kickoff_utc

    @property
    def kickoff_et(self) -> str:
        return self.kickoff_utc.astimezone(ET).strftime("%a %I:%M %p ET")


class KickoffSchedule:
    def __init__(self, games: dict[str, GameTime]):
        self.by_team = games                       # team abbr -> GameTime

    @classmethod
    def from_nflverse(cls, season: int, week: int) -> "KickoffSchedule":
        import nflreadpy as nfl
        df = nfl.load_schedules([season]).to_pandas()
        df = df[df["week"] == week]
        games: dict[str, GameTime] = {}
        for _, r in df.iterrows():
            # gameday: YYYY-MM-DD, gametime: HH:MM (ET)
            try:
                naive = datetime.strptime(f"{r['gameday']} {r['gametime']}", "%Y-%m-%d %H:%M")
                ko = naive.replace(tzinfo=ET).astimezone(timezone.utc)
            except (ValueError, TypeError):
                continue
            home, away = norm_team(str(r["home_team"])), norm_team(str(r["away_team"]))
            gid = str(r.get("game_id", f"{away}@{home}"))
            games[home] = GameTime(home, away, ko, gid)
            games[away] = GameTime(away, home, ko, gid)
        if not games:
            raise ValueError(f"no schedule rows for {season} week {week}")
        return cls(games)

    @classmethod
    def from_team_lines(cls, team_lines: dict) -> "KickoffSchedule":
        """Fallback: build from Odds API TeamLine.kickoff_iso."""
        games: dict[str, GameTime] = {}
        for t, line in team_lines.items():
            if not line.kickoff_iso:
                continue
            ko = datetime.fromisoformat(line.kickoff_iso.replace("Z", "+00:00"))
            games[norm_team(t)] = GameTime(norm_team(t), norm_team(line.opponent), ko)
        if not games:
            raise ValueError("no kickoff times in team lines")
        return cls(games)

    def locked_teams(self, now: datetime | None = None) -> set[str]:
        return {t for t, g in self.by_team.items() if g.locked(now)}

    def next_lock(self, now: datetime | None = None) -> GameTime | None:
        now = now or datetime.now(timezone.utc)
        future = [g for g in self.by_team.values() if g.kickoff_utc > now]
        return min(future, key=lambda g: g.kickoff_utc) if future else None

    def slate_windows(self) -> list[tuple[str, list[str]]]:
        """Distinct kickoff windows and the teams in each — the late-swap checkpoints."""
        by_ko: dict[datetime, set[str]] = {}
        for g in self.by_team.values():
            by_ko.setdefault(g.kickoff_utc, set()).add(g.team)
        return [(ko.astimezone(ET).strftime("%a %I:%M %p ET"), sorted(teams))
                for ko, teams in sorted(by_ko.items())]

    def summary(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        lines = ["Kickoff windows:"]
        for label, teams in self.slate_windows():
            locked = all(t in self.locked_teams(now) for t in teams)
            mark = "LOCKED" if locked else "open"
            lines.append(f"  {label:20s} [{mark:6s}] {', '.join(teams)}")
        nl = self.next_lock(now)
        if nl:
            mins = int((nl.kickoff_utc - now).total_seconds() // 60)
            lines.append(f"  next lock: {nl.kickoff_et} ({mins} min) — {nl.game_id or nl.team}")
        return "\n".join(lines)
