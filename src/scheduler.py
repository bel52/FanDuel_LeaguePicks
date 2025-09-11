"""
Simple scheduler for DFS optimizer using built‑in threading.

This module provides a lightweight alternative to external scheduling
libraries (e.g. APScheduler) by using a background thread that wakes up
periodically and checks whether any tasks are due to run.  Tasks are
configured statically based on the day of week and time and will run
once per occurrence.  Polling for injuries and live scores happens
every N minutes on Sundays.

Environment variables:

  SCHEDULE_ENABLED    – set to '1' to enable scheduled jobs.
  TIME_ZONE           – IANA timezone string (default: America/New_York).
  INJURY_POLL_INTERVAL – minutes between injury/live score checks (default 30).

This scheduler starts automatically via ``app/main.py`` when the
application boots.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    ZoneInfo = None  # type: ignore

from typing import Callable, Dict, Any, Optional

from .data_fetch import (
    load_all_players,
    fetch_odds,
    fetch_weather,
    fetch_injuries,
    fetch_live_scores,
)
from .lineup_builder import build_lineup
from .analysis import monte_carlo_simulation
from .emailer import send_email
from .util import logger


def _format_lineup_text(lineup: Dict[str, Any], score: float) -> str:
    if not lineup:
        return "No valid lineup could be built."
    def row(slot: str, p: Dict[str, Any]) -> str:
        return f"{slot:>4}  {p['Name']:<24} {p['Team']:<3} {p['Pos']:<3}  ${p['Salary']:>5}  Proj:{p['ProjFP']:>5.1f}"
    ordered = ['QB','RB1','RB2','WR1','WR2','WR3','TE','FLEX','DST']
    lines: list[str] = []
    total_salary = 0
    for slot in ordered:
        player = lineup.get(slot)
        if not player:
            continue
        lines.append(row(slot, player))
        total_salary += int(player.get('Salary', 0) or 0)
    lines.append("-" * 60)
    lines.append(f"Total Salary: ${total_salary}   Score: {score:.2f}")
    return "\n".join(lines)


def _compose_email_subject(name: str) -> str:
    return f"DFS Optimizer: {name}"


def _compose_email_body(name: str, lineup: Dict[str, Any], score: float,
                        mc_summary: Optional[Dict[str, float]] = None,
                        scores: Optional[Dict[str, float]] = None) -> str:
    """
    Assemble the email body for a scheduled report.  Includes the
    lineup details, optional Monte‑Carlo summary and live score snippet.

    Parameters
    ----------
    name : str
        Descriptor of the analysis run (e.g. "Sunday Mid‑Game Update").
    lineup : Dict[str, Any]
        Selected lineup keyed by roster slot.
    score : float
        Projected total score for the lineup.
    mc_summary : Optional[Dict[str, float]]
        Summary statistics from the Monte‑Carlo simulation (may be None).
    scores : Optional[Dict[str, float]]
        Mapping of team abbreviations to current scores for live games.  If
        provided, the top scores will be included in the email body.

    Returns
    -------
    str
        Full email body content.
    """
    body_lines: list[str] = [f"{name} results:", "", _format_lineup_text(lineup, score)]
    if mc_summary:
        body_lines.extend(["", "Monte‑Carlo summary:",
            f"Mean: {mc_summary.get('mean', 0):.2f}, "
            f"Median: {mc_summary.get('median', 0):.2f}, "
            f"75th pct: {mc_summary.get('p75', 0):.2f}, "
            f"90th pct: {mc_summary.get('p90', 0):.2f}, "
            f"Max: {mc_summary.get('max', 0):.2f}" ])
    if scores:
        body_lines.append("")
        body_lines.append("Live scores:")
        # Show up to 10 teams sorted by score descending
        for team, pts in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:10]:
            body_lines.append(f"{team}: {pts:.1f}")
    return "\n".join(str(l) for l in body_lines)


def run_analysis(name: str, do_monte_carlo: bool = True,
                 scores: Optional[Dict[str, float]] = None) -> None:
    """
    Perform a generic analysis run: load data, build lineup, optionally
    run Monte‑Carlo simulation, and send an email notification.  If
    ``scores`` is provided, players on teams with active scores are
    considered locked and will not be swapped during lineup construction.

    Parameters
    ----------
    name : str
        A label for the analysis (used in subject/body).
    do_monte_carlo : bool, optional
        Whether to run the Monte‑Carlo simulation.  Also depends on
        ``USE_MONTE_CARLO`` environment variable.
    scores : Optional[Dict[str, float]], optional
        Mapping of team abbreviations to their current scores in
        ongoing games.  If provided, any player whose ``Team`` is in
        ``scores`` will be locked and excluded from swaps.
    """
    logger.info(f"Starting scheduled job: {name}")
    players = load_all_players()
    odds = fetch_odds() if os.getenv('USE_ODDS', '0') == '1' else []
    weather = fetch_weather() if os.getenv('USE_WEATHER', '0') == '1' else []
    injuries = fetch_injuries()
    # Determine locked player names based on active scores (teams)
    locked_names: Optional[set[str]] = None
    if scores:
        teams_in_play = set(scores.keys())
        # Players whose team is currently playing are locked
        locked_names = {p['Name'] for p in players if p.get('Team') in teams_in_play}
    lineup, score = build_lineup(players, odds_data=odds, weather_data=weather,
                                 injuries=injuries, locked=locked_names)
    if lineup is None:
        msg = f"{name}: no valid lineup constructed."
        send_email(_compose_email_subject(name), msg)
        logger.warning(msg)
        return
    mc_summary: Optional[Dict[str, float]] = None
    if do_monte_carlo and os.getenv('USE_MONTE_CARLO', '0') == '1':
        try:
            mc_iters = int(os.getenv('MC_ITERS', '1000'))
        except ValueError:
            mc_iters = 1000
        mc_summary = monte_carlo_simulation(players, iters=mc_iters, odds_data=odds, weather_data=weather)
    body = _compose_email_body(name, lineup, score, mc_summary, scores)
    send_email(_compose_email_subject(name), body)
    logger.info(f"Finished scheduled job: {name} with score {score:.2f}")


def run_full_analysis() -> None:
    run_analysis("Weekly Full Analysis", do_monte_carlo=True)


def run_daily_update() -> None:
    run_analysis("Daily Update", do_monte_carlo=False)


def run_pre_slate_update() -> None:
    run_analysis("Sunday Pre‑Slate Update", do_monte_carlo=False)


def run_mid_game_update() -> None:
    """
    Run the mid‑game update.  This pulls the latest live scores to
    identify games already in progress and locks players accordingly
    before constructing a new lineup.  It omits Monte‑Carlo simulation
    for speed.  Live scores are included in the email report.
    """
    try:
        scores = fetch_live_scores()
    except Exception as exc:
        logger.warning(f"Could not fetch live scores for mid‑game update: {exc}")
        scores = {}
    run_analysis("Sunday Mid‑Game Update", do_monte_carlo=False, scores=scores)


def run_injury_score_update() -> None:
    logger.info("Periodic injury/score update triggered")
    try:
        fetch_injuries()
        fetch_live_scores()
    except Exception as exc:
        logger.warning(f"Injury/score update failed: {exc}")


def _scheduler_loop() -> None:
    """Background loop that checks the current time and runs tasks."""
    # Determine timezone for scheduling
    tz_name = os.getenv('TIME_ZONE', 'America/New_York')
    zone = None
    if ZoneInfo is not None:
        try:
            zone = ZoneInfo(tz_name)
        except Exception:
            zone = None
    # Poll interval for injury/score updates
    # Determine polling interval for injury and live score checks.  Use the
    # lesser of INJURY_POLL_INTERVAL and SCORE_POLL_INTERVAL (defaults to 30).
    try:
        inj_int = int(os.getenv('INJURY_POLL_INTERVAL', '30'))
    except ValueError:
        inj_int = 30
    try:
        score_int = int(os.getenv('SCORE_POLL_INTERVAL', str(inj_int)))
    except ValueError:
        score_int = inj_int
    poll_interval = min(inj_int, score_int) if inj_int > 0 and score_int > 0 else 30
    # Last run tracking to prevent multiple executions within the same minute
    last_runs: Dict[str, tuple[int, int]] = {}
    while True:
        now = datetime.now(zone) if zone else datetime.now()
        wd = now.weekday()  # Monday=0, Sunday=6
        hm = (now.hour, now.minute)
        # Wednesday full analysis at 17:00
        if wd == 2 and hm == (17, 0):
            if last_runs.get('full') != hm:
                run_full_analysis()
                last_runs['full'] = hm
        # Daily update Thu–Sat at 17:00
        if wd in (3, 4, 5) and hm == (17, 0):
            if last_runs.get('daily') != hm:
                run_daily_update()
                last_runs['daily'] = hm
        # Pre‑slate update Sunday 10:00
        if wd == 6 and hm == (10, 0):
            if last_runs.get('pre') != hm:
                run_pre_slate_update()
                last_runs['pre'] = hm
        # Mid‑game update Sunday 14:45
        if wd == 6 and hm == (14, 45):
            if last_runs.get('mid') != hm:
                run_mid_game_update()
                last_runs['mid'] = hm
        # Injury/score polling every poll_interval minutes on Sunday
        if wd == 6 and now.minute % poll_interval == 0:
            key = f"poll:{now.hour}:{now.minute}"
            if last_runs.get(key) != hm:
                run_injury_score_update()
                last_runs[key] = hm
        # Sleep for 30 seconds before rechecking
        time.sleep(30)


def start_scheduler() -> Optional[threading.Thread]:
    """Start the scheduling thread if enabled.  Returns the thread instance."""
    if os.getenv('SCHEDULE_ENABLED', '0') != '1':
        logger.info("Scheduler not started: SCHEDULE_ENABLED is not '1'")
        return None
    thread = threading.Thread(target=_scheduler_loop, daemon=True)
    thread.start()
    logger.info("Background scheduler thread started")
    return thread
