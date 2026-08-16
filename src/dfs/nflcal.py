"""NFL calendar — what week is it, really.

The season/week should never be something Brett has to work out. Derived from the
nflverse schedule, which carries exact kickoff datetimes, so the answer is right
across the preseason gap, bye-adjacent Tuesdays, the Thursday/Monday tails, and the
January weeks where the calendar year has rolled but the NFL season has not.

Rules, in order:
  1. Season = the season whose schedule window contains today. NFL seasons are keyed
     to their starting year, so January 2027 games belong to season 2026.
  2. Current week = the week whose game window contains now, where a week runs from
     the Tuesday before its first kickoff to the Tuesday after its last. Tuesday is
     the natural boundary: FanDuel posts new slates Tue/Wed, and Monday night still
     belongs to the week that just played.
  3. Before the season opens (August, preseason) the answer is Week 1 — that is the
     week being prepared for, which is what a build should default to.
  4. After the regular season ends, it pins to the final week.

Everything here is a DEFAULT. Every entry point takes an explicit override so Brett
can build any week he wants for testing or review.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


@dataclass
class WeekInfo:
    season: int
    week: int
    reason: str                      # why this week, in plain words
    first_kickoff: datetime | None = None
    last_kickoff: datetime | None = None
    is_preseason: bool = False
    is_postseason: bool = False

    @property
    def label(self) -> str:
        return f"{self.season} Week {self.week}"

    def days_to_kickoff(self, now: datetime | None = None) -> float | None:
        if not self.first_kickoff:
            return None
        now = now or datetime.now(timezone.utc)
        return round((self.first_kickoff - now).total_seconds() / 86400, 1)

    def summary(self, now: datetime | None = None) -> str:
        d = self.days_to_kickoff(now)
        when = ""
        if d is not None:
            when = (f", first kickoff in {d:.1f} days" if d > 0
                    else f", games started {abs(d):.1f} days ago")
        return f"{self.label} — {self.reason}{when}"


@lru_cache(maxsize=8)
def _schedule(season: int):
    """(week -> (first_kickoff_utc, last_kickoff_utc)) for the regular season."""
    import nflreadpy as nfl
    df = nfl.load_schedules([season]).to_pandas()
    if "game_type" in df.columns:
        df = df[df["game_type"] == "REG"]
    out: dict[int, tuple[datetime, datetime]] = {}
    for wk, grp in df.groupby("week"):
        times = []
        for _, r in grp.iterrows():
            try:
                naive = datetime.strptime(f"{r['gameday']} {r['gametime']}", "%Y-%m-%d %H:%M")
                times.append(naive.replace(tzinfo=ET).astimezone(timezone.utc))
            except (ValueError, TypeError):
                continue
        if times:
            out[int(wk)] = (min(times), max(times))
    return out


def current_week(now: datetime | None = None,
                 season_hint: int | None = None) -> WeekInfo:
    """Best guess at the season and week to operate on right now."""
    now = now or datetime.now(timezone.utc)
    et = now.astimezone(ET)
    # A season starting in year Y runs Sep(Y) -> Jan(Y+1); before August, we are still
    # in the tail of the previous season.
    guess = season_hint if season_hint is not None else (
        et.year if et.month >= 8 else et.year - 1)

    for season in ([guess] if season_hint else [guess, guess - 1]):
        try:
            sched = _schedule(season)
        except Exception:
            continue
        if not sched:
            continue
        weeks = sorted(sched)
        first_wk, last_wk = weeks[0], weeks[-1]
        season_start = sched[first_wk][0]
        season_end = sched[last_wk][1]

        if now < season_start:
            gap = (season_start - now).days
            if season_hint or gap <= 60:
                return WeekInfo(season, first_wk,
                                f"season hasn't started — preparing Week {first_wk}",
                                *sched[first_wk], is_preseason=True)
            continue

        if now > season_end + timedelta(days=2):
            if season_hint:
                return WeekInfo(season, last_wk, "regular season complete",
                                *sched[last_wk], is_postseason=True)
            continue

        # Inside the season: a week owns [Tuesday 6am before its first kickoff,
        # Tuesday 6am after its last). New FanDuel slates post Tue/Wed, so Tuesday
        # morning is the natural handoff — Monday night still belongs to the week
        # that just played, and Tuesday you start preparing the next one.
        for wk in weeks:
            start, end = sched[wk]
            lo = _tuesday_6am_on_or_before(start)
            hi = _tuesday_6am_after(end)
            if lo <= now < hi:
                started = now >= start
                why = ("games in progress or just played" if started
                       else "next slate up")
                return WeekInfo(season, wk, why, start, end)
        return WeekInfo(season, last_wk, "between weeks — defaulting to the last slate",
                        *sched[last_wk])

    # No schedule available (offline, or nflverse hasn't published yet).
    return WeekInfo(guess, 1, "schedule unavailable — defaulting to Week 1",
                    is_preseason=True)


def _tuesday_6am_on_or_before(dt: datetime) -> datetime:
    d = dt.astimezone(ET)
    back = (d.weekday() - 1) % 7            # Monday=0, Tuesday=1
    tue = (d - timedelta(days=back)).replace(hour=6, minute=0, second=0, microsecond=0)
    if tue > d:
        tue -= timedelta(days=7)
    return tue.astimezone(timezone.utc)


def _tuesday_6am_after(dt: datetime) -> datetime:
    d = dt.astimezone(ET)
    fwd = ((1 - d.weekday()) % 7) or 7
    tue = (d + timedelta(days=fwd)).replace(hour=6, minute=0, second=0, microsecond=0)
    if tue <= d:
        tue += timedelta(days=7)
    return tue.astimezone(timezone.utc)
