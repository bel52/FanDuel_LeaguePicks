"""nflverse whole-pool actuals (2026-09-26). Offline: frames are injected.

Ground truth used below is FanDuel's own Week 2 2026 contest page: CeeDee Lamb 34.3,
Ladd McConkey 5.0, Jeremiyah Love 6.0, Carolina DEF 26.0."""
import json
import pandas as pd
import pytest

from dfs.actuals import skill_points, dst_points, load_week, resolve, ingest
from dfs.results import ResultLog


def test_skill_points_match_fanduel_page():
    cee = {"receptions": 8, "receiving_yards": 153, "receiving_tds": 2}
    assert skill_points(cee) == 34.3                       # includes the 100-yd bonus
    assert skill_points({"receptions": 3, "receiving_yards": 35}) == 5.0
    love = {"carries": 9, "rushing_yards": 29, "receptions": 3, "receiving_yards": 16}
    assert skill_points(love) == 6.0
    dak = {"passing_yards": 279, "passing_tds": 4, "rushing_yards": 26}
    assert skill_points(dak) == 29.76
    assert skill_points({"receiving_yards": float("nan"), "special_teams_tds": 1}) == 6.0


def test_dst_uses_opponent_sacks_not_defender_sum():
    """Carolina W2 2026: defenders were credited 2.0 sacks, FanDuel scored 3 —
    the opponent's sacks_suffered is the team number."""
    car = {"def_sacks": 2.0, "fumble_recovery_opp": 2, "def_tds": 1,
           "special_teams_tds": 0, "def_safeties": 0}
    atl = {"sacks_suffered": 3, "passing_interceptions": 3}
    assert dst_points(car, atl, points_allowed=3) == 26.0
    assert dst_points({}, {}, points_allowed=0) == 10.0          # shutout
    assert dst_points({}, {}, points_allowed=35) == -4.0


class FakeLoader:
    """Week 2: CAR 34 @ ATL 3 is final. DAL/WAS is still in progress."""
    def schedules(self, s):
        return pd.DataFrame([
            {"week": 2, "game_type": "REG", "home_team": "ATL", "away_team": "CAR",
             "home_score": 3.0, "away_score": 34.0},
            {"week": 2, "game_type": "REG", "home_team": "DAL", "away_team": "WAS",
             "home_score": float("nan"), "away_score": float("nan")}])

    def team_stats(self, s):
        return pd.DataFrame([
            {"week": 2, "season_type": "REG", "team": "CAR", "opponent_team": "ATL",
             "sacks_suffered": 1, "passing_interceptions": 0, "fumble_recovery_opp": 2,
             "def_tds": 1, "special_teams_tds": 0, "def_safeties": 0},
            {"week": 2, "season_type": "REG", "team": "ATL", "opponent_team": "CAR",
             "sacks_suffered": 3, "passing_interceptions": 3, "fumble_recovery_opp": 0,
             "def_tds": 0, "special_teams_tds": 0, "def_safeties": 0}])

    def rosters(self, s):
        rows = [("G1", "CAR", "WR", "Tetairoa McMillan", "Tetairoa", "McMillan", "P1"),
                ("G2", "ATL", "RB", "Bijan Robinson", "Bijan", "Robinson", "P2"),
                ("G3", "ATL", "WR", "Backup Guy", "Backup", "Guy", "P3"),
                ("G4", "CAR", "RB", "Chuba Hubbard", "Chuba", "Hubbard", "P4"),
                ("G5", "WAS", "WR", "Terry McLaurin", "Terry", "McLaurin", "P5")]
        return pd.DataFrame([{"week": 2, "game_type": "REG", "gsis_id": g, "team": t,
                              "position": p, "full_name": fn, "football_name": f,
                              "first_name": f, "last_name": l, "pfr_id": pfr}
                             for g, t, p, fn, f, l, pfr in rows])

    def player_stats(self, s):
        return pd.DataFrame([
            {"week": 2, "season_type": "REG", "player_id": "G1", "team": "CAR",
             "position": "WR", "player_display_name": "Tetairoa McMillan",
             "receptions": 5, "receiving_yards": 101},
            {"week": 2, "season_type": "REG", "player_id": "G2", "team": "ATL",
             "position": "RB", "player_display_name": "Bijan Robinson",
             "carries": 16, "rushing_yards": 72, "receptions": 3, "receiving_yards": 9},
            {"week": 2, "season_type": "REG", "player_id": "G5", "team": "WAS",
             "position": "WR", "player_display_name": "Terry McLaurin",
             "receptions": 2, "receiving_yards": 50}])

    def snaps(self, s):
        return pd.DataFrame([
            {"week": 2, "game_type": "REG", "team": "CAR", "pfr_player_id": "P1",
             "offense_snaps": 55, "defense_snaps": 0, "st_snaps": 2},
            {"week": 2, "game_type": "REG", "team": "ATL", "pfr_player_id": "P2",
             "offense_snaps": 48, "defense_snaps": 0, "st_snaps": 0}])


def _pool():
    return [{"fd_id": "a", "name": "Tetairoa McMillan", "position": "WR", "team": "CAR", "salary": 6800},
            {"fd_id": "b", "name": "Bijan Robinson", "position": "RB", "team": "ATL", "salary": 8900},
            {"fd_id": "c", "name": "Backup Guy", "position": "WR", "team": "ATL", "salary": 4500},
            {"fd_id": "d", "name": "Totally Unknown", "position": "WR", "team": "CAR", "salary": 7000},
            {"fd_id": "e", "name": "Carolina Panthers", "position": "D", "team": "CAR", "salary": 3300},
            {"fd_id": "f", "name": "Terry McLaurin", "position": "WR", "team": "WAS", "salary": 6000},
            {"fd_id": "g", "name": "Chuba Hubbard", "position": "RB", "team": "CAR", "salary": 6700}]


def test_resolve_grades_final_games_only_and_never_invents_zeros():
    wo = load_week(2026, 2, FakeLoader())
    assert wo.final_teams == {"CAR", "ATL"} and "WAS" in wo.pending_teams
    res, unmatched = resolve(_pool(), wo)
    by = {r.fd_id: r for r in res}
    assert by["a"].points == 15.6 and by["a"].played == 1          # McMillan
    assert by["b"].points == 9.6 and by["b"].played == 1            # Bijan
    assert by["c"].points == 0.0 and by["c"].how == "roster-zero"   # rostered, no stats
    assert by["c"].played == 0                                      # no snap row
    assert by["e"].points == 26.0                                   # CAR DEF
    assert by["g"].points == 0.0 and by["g"].played == 0            # Hubbard: no snaps
    assert "f" not in by                                            # game not final
    assert [u["name"] for u in unmatched] == ["Totally Unknown"]    # left ungraded


def _seed(tmp_path):
    rl = ResultLog(tmp_path / "r.db")
    with rl._c() as c:
        for p in _pool():
            c.execute("""INSERT INTO player_results (season,week,fd_id,name,position,team,
                         salary,projection,proj_fp,proj_props,proj_blend)
                         VALUES (2026,2,?,?,?,?,?,10,9,11,10.2)""",
                      (p["fd_id"], p["name"], p["position"], p["team"], p["salary"]))
    return rl


def test_ingest_writes_actuals_and_grades_the_shadow(tmp_path):
    rl = _seed(tmp_path)
    shadow = [{"fd_id": x, "name": x} for x in ("a", "b", "c", "e")]
    with rl._c() as c:
        c.execute("""INSERT INTO entries (season,week,contest,submitted_ts,lineup_json,status)
                     VALUES (2026,2,'Leather League [shadow:model]','t',?,'pending')""",
                  (json.dumps(shadow),))
        c.execute("""INSERT INTO entries (season,week,contest,submitted_ts,lineup_json,status,
                     actual_score) VALUES (2026,2,'Leather League','t','[]','confirmed',40)""")
    (rep,) = ingest(rl, 2026, [2], FakeLoader())
    assert rep.written == 5 and rep.zeros == 2 and not rep.error
    assert [u["name"] for u in rep.unmatched] == ["Totally Unknown"]
    assert rep.shadows_graded == [{"contest": "Leather League [shadow:model]",
                                   "score": 51.2}]       # 15.6 + 9.6 + 0 + 26.0
    with rl._c() as c:
        r = c.execute("SELECT actual, actual_src, played, played_src FROM player_results "
                      "WHERE fd_id='a'").fetchone()
    assert (r["actual"], r["actual_src"], r["played"], r["played_src"]) == (15.6, "nflverse", 1, "snaps")
    assert rl.arm_scoreboard(2026)[0]["shadow"] == 51.2
    # re-running is harmless
    (again,) = ingest(rl, 2026, [2], FakeLoader())
    assert again.written == 5


def test_page_actual_never_overwrites_nflverse_and_conflicts_surface(tmp_path):
    rl = _seed(tmp_path)
    ingest(rl, 2026, [2], FakeLoader())
    # a hand-transcribed capture says McMillan scored 14.6 (a typo)
    rl.log_actuals_by_name(2026, 2, {"tetairoa mcmillan": 14.6})
    with rl._c() as c:
        r = c.execute("SELECT actual, actual_fd, actual_src FROM player_results "
                      "WHERE fd_id='a'").fetchone()
    assert (r["actual"], r["actual_fd"], r["actual_src"]) == (15.6, 14.6, "nflverse")
    assert rl.actual_conflicts(2026, 2) == [{"name": "Tetairoa McMillan",
                                              "fd": 14.6, "nfl": 15.6}]
    # before nflverse grades a player, the page value is used
    rl.log_actuals_by_name(2026, 2, {"terry mclaurin": 6.0})
    with rl._c() as c:
        r = c.execute("SELECT actual, actual_src FROM player_results WHERE fd_id='f'").fetchone()
    assert (r["actual"], r["actual_src"]) == (6.0, "fd_page")


def test_ingest_survives_a_feed_outage(tmp_path):
    rl = _seed(tmp_path)
    class Down:
        def schedules(self, s): raise ConnectionError("nflverse down")
    (rep,) = ingest(rl, 2026, [2], Down())
    assert rep.error and "unavailable" in rep.line() and "retry" in rep.line()


def test_log_entry_no_longer_erases_projection_components(tmp_path):
    """LIVE BUG: log_entry used INSERT OR REPLACE, which deletes the row and re-inserts
    only its listed columns — wiping proj_fp/props/blend and p_active for exactly the
    players we rostered, right after log_projection_components wrote them."""
    from types import SimpleNamespace as NS
    rl = _seed(tmp_path)
    p = NS(fd_id="a", name="Tetairoa McMillan", position="WR", team="CAR", salary=6800,
           projection=11.85, proj_source="blend", implied_team_total=23.0)
    rl.log_entry(2026, 2, "Leather League", [p])
    with rl._c() as c:
        r = c.execute("SELECT proj_fp, proj_props, proj_blend, projection, in_lineup "
                      "FROM player_results WHERE fd_id='a'").fetchone()
    assert (r["proj_fp"], r["proj_props"], r["proj_blend"]) == (9, 11, 10.2)
    assert r["projection"] == 11.85 and r["in_lineup"] == 1


def test_migration_marks_existing_actuals_as_page_sourced(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    ResultLog(db)                                         # current schema...
    con = sqlite3.connect(db)                              # ...then simulate a pre-09-26 db
    con.executescript("""
      CREATE TABLE pr_old AS SELECT season,week,fd_id,name,position,team,salary,projection,
        actual,in_lineup,proj_fp,proj_props,proj_blend,p_active,opp_implied_total,played,
        status_at_lock FROM player_results;
      DROP TABLE player_results; ALTER TABLE pr_old RENAME TO player_results;
      INSERT INTO player_results (season,week,fd_id,name,actual) VALUES (2026,1,'x','X',12.5);
    """)
    con.commit(); con.close()
    rl = ResultLog(db)
    with rl._c() as c:
        r = c.execute("SELECT actual, actual_fd, actual_src FROM player_results").fetchone()
    assert (r["actual"], r["actual_fd"], r["actual_src"]) == (12.5, 12.5, "fd_page")
