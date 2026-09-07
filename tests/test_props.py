"""Tests for the market-implied projection layer.

Fixtures are REAL captures, not hand-written: `odds_props_tbcin.json` is the live
2026 Week 1 TB@CIN board (FanDuel + DraftKings, six markets, pulled 2026-09-07) and
`odds_events_w1.json` is the real Week 1 event listing. Synthetic fixtures would have
hidden the two things that actually matter here — that the board carries no team or
position on a props outcome, and that different books price different thresholds for
the same player.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dfs.blend import apply_availability, apply_props
from dfs.injuries import (InjuryRecord, Status, annotate_availability,
                          play_probability)
from dfs.matching import load_aliases
from dfs.props import (ANYTIME_TD_OVERROUND, MULTI_TD_FACTOR, PRIMARY_MARKET,
                       SCALE_REJECT, american_to_prob, devig_two_way,
                       match_props_to_slate, mean_from_line, parse_event_board,
                       poisson_lambda_from_tail, props_points)
from dfs.slate import PlayerSlate, SlatePlayer, SlateType

FIX = Path(__file__).parent / "fixtures"


def board():
    return json.loads((FIX / "odds_props_tbcin.json").read_text())


def parsed():
    return parse_event_board(board())


# --------------------------------------------------------------------------
# odds math
# --------------------------------------------------------------------------

def test_american_to_prob_both_signs():
    assert american_to_prob(-140) == pytest.approx(140 / 240, abs=1e-9)
    assert american_to_prob(125) == pytest.approx(100 / 225, abs=1e-9)


def test_devig_two_way_removes_overround():
    # -114 / -114 is the standard FanDuel yardage price: a true coin flip.
    assert devig_two_way(-114, -114) == pytest.approx(0.5, abs=1e-9)
    # A favoured Over must come back above 0.5 but below its vigged probability.
    raw = american_to_prob(-145)
    p = devig_two_way(-145, 114)
    assert 0.5 < p < raw


def test_mean_equals_line_at_even_money():
    """The whole scale-anchoring argument rests on this: at 50/50 the market's
    central estimate IS the line, with no spread assumption doing hidden work."""
    assert mean_from_line(87.5, 0.5, 36.0) == pytest.approx(87.5, abs=1e-9)
    assert mean_from_line(87.5, 0.6, 36.0) > 87.5
    assert mean_from_line(87.5, 0.4, 36.0) < 87.5


def test_mean_from_line_never_negative():
    assert mean_from_line(1.5, 0.01, 36.0) >= 0.0


def test_poisson_lambda_round_trips():
    for lam in (0.8, 1.6, 2.2, 3.1):
        # P(X >= 2) for a Poisson(lam)
        import math
        tail = 1.0 - math.exp(-lam) * (1 + lam)
        assert poisson_lambda_from_tail(1.5, tail) == pytest.approx(lam, abs=0.02)


def test_pass_td_lambda_agrees_across_books_and_thresholds():
    """The load-bearing validation for pass TDs.

    DraftKings priced Burrow's pass TDs at 1.5 and FanDuel at 2.5 on the same board.
    A Poisson solve from either threshold has to land in the same place, or the
    single-threshold approach is not identifying anything real.
    """
    b = board()
    lams = {}
    for bk in b["bookmakers"]:
        mkt = next(m for m in bk["markets"] if m["key"] == "player_pass_tds")
        outs = [o for o in mkt["outcomes"] if o["description"] == "Joe Burrow"]
        over = next(o for o in outs if o["name"] == "Over")
        under = next(o for o in outs if o["name"] == "Under")
        p = devig_two_way(over["price"], under["price"])
        lams[bk["key"]] = poisson_lambda_from_tail(over["point"], p)
    assert set(lams) == {"fanduel", "draftkings"}
    assert lams["fanduel"] == pytest.approx(lams["draftkings"], abs=0.25)
    assert 1.5 < lams["fanduel"] < 3.0


# --------------------------------------------------------------------------
# board parsing
# --------------------------------------------------------------------------

def test_parse_extracts_expected_players_and_markets():
    pp = parsed()
    burrow = pp["joe burrow"]
    assert burrow.event_teams == ("TB", "CIN")
    assert {"pass_yds", "pass_tds", "rush_yds", "anytime_td"} <= burrow.markets
    assert 240 < burrow.stats["pass_yds"] < 290
    chase = pp["jamarr chase"]
    assert {"rec_yds", "rec_rec", "anytime_td"} <= chase.markets
    assert 80 < chase.stats["rec_yds"] < 95
    assert 6.0 < chase.stats["rec_rec"] < 8.0


def test_parse_uses_both_books():
    pp = parsed()
    assert pp["jamarr chase"].books == {"fanduel", "draftkings"}


def test_anytime_td_probability_is_devigged_down():
    """TD boards are one-sided, so the overround haircut is the only vig removal
    available. The result must be strictly below the raw implied probability."""
    pp = parsed()
    raw_fd = american_to_prob(-130)          # Chase, FanDuel
    assert pp["jamarr chase"].stats["anytime_td_prob"] < raw_fd
    assert pp["jamarr chase"].stats["anytime_td_prob"] > raw_fd / (ANYTIME_TD_OVERROUND + 0.2)


def test_defense_entries_do_not_become_players():
    """'Cincinnati Bengals Defense' and 'Cincinnati Bengals D/ST' appear on the
    anytime-TD board. They must never be matched to the D slot, which has no props."""
    pp = parsed()
    slate = _slate([("D", "CIN", "Cincinnati Bengals", 3200)])
    assert match_props_to_slate(slate.players, pp) == {}


def test_unknown_event_teams_raise():
    b = board()
    b["home_team"] = "Cincinnati Bengalz"
    with pytest.raises(Exception):
        parse_event_board(b)


def test_missing_side_of_two_way_is_ignored():
    """A market with an Over but no Under cannot be devigged; it must be dropped
    rather than assumed to be even money."""
    b = board()
    for bk in b["bookmakers"]:
        for m in bk["markets"]:
            if m["key"] == "player_reception_yds":
                m["outcomes"] = [o for o in m["outcomes"]
                                 if not (o["description"] == "Tee Higgins"
                                         and o["name"] == "Under")]
    pp = parse_event_board(b)
    assert "rec_yds" not in pp["tee higgins"].markets


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------

def _slate(rows) -> PlayerSlate:
    players = []
    for i, (pos, team, name, sal) in enumerate(rows):
        opp = "TB" if team == "CIN" else "CIN"
        players.append(SlatePlayer(fd_id=f"x-{i}", name=name, position=pos, team=team,
                                   opponent=opp, salary=sal, game="TB@CIN"))
    return PlayerSlate(slate_id="t", slate_type=SlateType.FULL, season=2026, week=1,
                       players=players)


class _FP:
    def __init__(self, points, stats=None):
        self.points = points
        self.stats = stats or {}


def test_match_scoped_by_event_teams():
    pp = parsed()
    slate = _slate([("WR", "CIN", "Ja'Marr Chase", 8900),
                    ("QB", "CIN", "Joe Burrow", 8200),
                    ("RB", "DET", "Jahmyr Gibbs", 9100)])
    m = match_props_to_slate(slate.players, pp)
    assert len(m) == 2                      # Gibbs is not in this game
    assert m["x-0"].name == "jamarr chase"


def test_match_rejects_same_surname_different_first_name():
    """A props outcome carries no team and no position, so surname collisions are the
    real risk. Only the event's own teams are candidates and first names must be
    compatible — the guard matching.py needed after Bijan/Brian Robinson."""
    pp = parsed()
    slate = _slate([("RB", "CIN", "Devon Brown", 5000)])
    assert match_props_to_slate(slate.players, pp) == {}


# --------------------------------------------------------------------------
# scoring + scale anchoring
# --------------------------------------------------------------------------

def test_props_points_are_anchored_to_fp_level():
    """Anchoring is the design's core claim: markets set the DISTRIBUTION of points
    across players, FP sets the LEVEL. So the median props/FP ratio must come out at
    1.0 by construction, whatever the raw skew was."""
    pp = parsed()
    rows = [("WR", "CIN", "Ja'Marr Chase", 8900), ("WR", "CIN", "Tee Higgins", 6400),
            ("WR", "TB", "Emeka Egbuka", 6400), ("WR", "TB", "Chris Godwin", 5500),
            ("WR", "TB", "Jalen McMillan", 5000)]
    slate = _slate(rows)
    fp = {p.fd_id: _FP(points=v) for p, v in zip(slate.players,
                                                 [17.0, 11.5, 10.2, 8.0, 6.5])}
    pts, rep = props_points(slate.players, pp, fp)
    assert len(pts) == 5
    ratios = sorted(pts[p.fd_id] / fp[p.fd_id].points for p in slate.players)
    assert ratios[len(ratios) // 2] == pytest.approx(1.0, abs=0.02)
    assert rep.scale["WR"] > 0


def test_scale_gate_rejects_a_broken_board():
    """If the parser or the board schema breaks, the position's scale factor blows
    out. That must discard props for the position, not silently rescale garbage."""
    pp = parsed()
    slate = _slate([("WR", "CIN", "Ja'Marr Chase", 8900),
                    ("WR", "CIN", "Tee Higgins", 6400),
                    ("WR", "TB", "Emeka Egbuka", 6400),
                    ("WR", "TB", "Chris Godwin", 5500)])
    # FP an order of magnitude off => scale far outside the reject band
    fp = {p.fd_id: _FP(points=200.0) for p in slate.players}
    pts, rep = props_points(slate.players, pp, fp)
    assert pts == {}
    assert rep.rejected and "WR" in rep.rejected[0]
    assert SCALE_REJECT[0] < 1.0 < SCALE_REJECT[1]


def test_thin_scale_sample_skips_the_position():
    """One QB on the board cannot support a scale factor, and neither can a
    two-player global sample. Props must be dropped, not scaled on noise."""
    pp = parsed()
    slate = _slate([("QB", "CIN", "Joe Burrow", 8200)])
    pts, rep = props_points(slate.players, pp,
                            {"x-0": _FP(points=21.6, stats={"pass_ints": 0.65})})
    assert pts == {}
    assert rep.rejected and "sample too small" in rep.rejected[0]


def test_all_positions_factor_used_when_one_position_is_thin(capsys):
    """A single TE among a well-sampled slate is scaled on the all-positions factor,
    with a warning — not silently, and not dropped."""
    pp = parsed()
    rows = [("WR", "CIN", "Ja'Marr Chase", 8900), ("WR", "CIN", "Tee Higgins", 6400),
            ("WR", "TB", "Emeka Egbuka", 6400), ("WR", "TB", "Chris Godwin", 5500),
            ("WR", "TB", "Jalen McMillan", 5000), ("RB", "CIN", "Chase Brown", 7500),
            ("RB", "TB", "Bucky Irving", 7000), ("RB", "TB", "Sean Tucker", 4600),
            ("TE", "TB", "Cade Otton", 5000)]
    slate = _slate(rows)
    fpv = [17.0, 11.5, 10.2, 8.0, 6.5, 15.1, 13.0, 5.0, 8.4]
    fp = {p.fd_id: _FP(points=v) for p, v in zip(slate.players, fpv)}
    pts, rep = props_points(slate.players, pp, fp)
    assert rep.sample["WR"].startswith("n=")
    assert "all-positions" in rep.sample["TE"]
    assert any("only 1 priced" in w for w in rep.warnings)
    # Eight of nine, not nine: Sean Tucker carries only a TD price on this board and
    # no rushing-yards line, so he has no primary market and stays FP-only.
    assert len(pts) == 8
    assert slate.players[7].fd_id not in pts


def test_interceptions_lower_the_raw_props_score():
    from dfs.props import _merged_stats
    from dfs.scoring import score
    pp = parsed()["joe burrow"]
    a, _ = score(_merged_stats(pp, {"pass_ints": 0.0}, "QB"), "QB")
    b, _ = score(_merged_stats(pp, {"pass_ints": 0.8}, "QB"), "QB")
    assert a - b == pytest.approx(0.8, abs=0.01)


def test_anytime_td_becomes_position_appropriate_tds():
    from dfs.props import _merged_stats
    pp = parsed()
    wr = _merged_stats(pp["jamarr chase"], {}, "WR")
    rb = _merged_stats(pp["chase brown"], {}, "RB")
    assert wr["rec_tds"] > 0 and "rush_tds" not in wr
    assert rb["rush_tds"] > 0 and "rec_tds" not in rb
    assert wr["rec_tds"] == pytest.approx(
        MULTI_TD_FACTOR * pp["jamarr chase"].stats["anytime_td_prob"], abs=1e-3)


def test_position_without_its_primary_market_is_skipped():
    """A WR with a TD price but no receiving-yards line is not priced: one market is
    not enough to build a stat line from, and a partial line under-projects."""
    b = board()
    for bk in b["bookmakers"]:
        bk["markets"] = [m for m in bk["markets"] if m["key"] != "player_reception_yds"]
    pp = parse_event_board(b)
    slate = _slate([("WR", "CIN", "Ja'Marr Chase", 8900)])
    pts, _ = props_points(slate.players, pp, {"x-0": _FP(points=17.0)})
    assert pts == {}
    assert PRIMARY_MARKET["WR"] == "rec_yds"


# --------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------

def _dist():
    return json.loads((Path(__file__).parents[1] / "data" / "distributions.json").read_text())


def test_blend_is_a_weighted_average_and_keeps_components():
    slate = _slate([("WR", "CIN", "Ja'Marr Chase", 8900)])
    p = slate.players[0]
    p.proj_fp = 10.0
    p.projection = 10.0
    changed = apply_props(slate, {"x-0": 20.0}, _dist(), weights={"WR": 0.6})
    assert changed == [p]
    assert p.proj_props == 20.0
    assert p.proj_blend == pytest.approx(16.0)
    assert p.projection == pytest.approx(16.0)
    assert p.proj_fp == 10.0                      # component preserved
    assert "props 60%" in p.proj_source


def test_blend_skips_players_with_no_props():
    slate = _slate([("WR", "CIN", "Tee Higgins", 6400)])
    p = slate.players[0]
    p.proj_fp = p.projection = 11.0
    assert apply_props(slate, {}, _dist()) == []
    assert p.projection == 11.0
    assert p.proj_props is None


def test_blend_weight_zero_is_a_no_op_on_value():
    slate = _slate([("WR", "CIN", "Tee Higgins", 6400)])
    p = slate.players[0]
    p.proj_fp = p.projection = 11.0
    apply_props(slate, {"x-0": 99.0}, _dist(), weights={"WR": 0.0})
    assert p.projection == 11.0


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------

def test_play_probability_prefers_an_explicit_percentage():
    rec = InjuryRecord(name="X", team="CIN", status=Status.QUESTIONABLE,
                       detail="Knee · practice LP/LP · 40% to play")
    assert play_probability(rec) == pytest.approx(0.40)


def test_play_probability_uses_practice_trend_when_no_percentage():
    full = InjuryRecord(name="X", team="CIN", status=Status.QUESTIONABLE,
                        detail="Knee · practice DNP/LP/FP")
    dnp = InjuryRecord(name="Y", team="CIN", status=Status.QUESTIONABLE,
                       detail="Knee · practice LP/DNP")
    assert play_probability(full) > play_probability(dnp)
    assert play_probability(full) > 0.85
    assert play_probability(dnp) < 0.55


def test_play_probability_defaults_to_certain_when_healthy():
    assert play_probability(None) == 1.0
    assert play_probability(InjuryRecord(name="X", team="CIN",
                                         status=Status.ACTIVE)) == 1.0


def test_availability_discount_reports_and_preserves_the_undiscounted_number():
    """Design rule 5 refined: a haircut is allowed only if it stays visible."""
    slate = _slate([("WR", "CIN", "Ja'Marr Chase", 8900),
                    ("QB", "CIN", "Joe Burrow", 8200)])
    chase, burrow = slate.players
    for p, v in ((chase, 18.0), (burrow, 21.0)):
        p.proj_fp = p.proj_blend = p.projection = v
    annotate_availability(slate, {
        "jamarr chase": InjuryRecord(name="Ja'Marr Chase", team="CIN",
                                     status=Status.QUESTIONABLE,
                                     detail="Knee · 75% to play"),
    })
    disc = apply_availability(slate, _dist())
    assert [p.name for p in disc] == ["Ja'Marr Chase"]
    assert chase.projection == pytest.approx(13.5)
    assert chase.proj_blend == 18.0            # undiscounted number retained
    assert chase.p_active == pytest.approx(0.75)
    assert "p_active=0.75" in chase.proj_source
    assert burrow.projection == 21.0           # healthy player untouched


def test_availability_is_idempotent():
    """The swap path re-runs the projection layers on an already-built slate; a
    double discount would compound into a phantom benching."""
    slate = _slate([("WR", "CIN", "Ja'Marr Chase", 8900)])
    p = slate.players[0]
    p.proj_fp = p.proj_blend = p.projection = 18.0
    p.p_active = 0.5
    apply_availability(slate, _dist())
    apply_availability(slate, _dist())
    assert p.projection == pytest.approx(9.0)


# --------------------------------------------------------------------------
# aliases
# --------------------------------------------------------------------------

def test_alias_file_normalizes_both_sides(tmp_path):
    f = tmp_path / "a.json"
    f.write_text(json.dumps({"players": {"Marquise Brown": "Hollywood Brown"}}))
    al = load_aliases(f)
    assert al == {"marquise brown": "hollywood brown"}


def test_missing_alias_file_is_not_fatal(tmp_path):
    assert load_aliases(tmp_path / "nope.json") == {}


def test_shipped_alias_file_parses():
    """The shipped file is a documented empty stub; a syntax error in it would
    silently disable aliasing for every build."""
    p = Path(__file__).parents[1] / "data" / "aliases.json"
    assert isinstance(load_aliases(p), dict)


def test_alias_resolves_an_otherwise_unmatched_player():
    from dfs.matching import match_slate

    class P:
        def __init__(self, name, team, pos, pts):
            self.name, self.team, self.position, self.points = name, team, pos, pts

    slate = _slate([("WR", "CIN", "Marquise Brown", 6000)])
    projections = [P("Hollywood Brown", "CIN", "WR", 12.0)]
    m, rep = match_slate(slate.players, projections)
    assert m == {} and rep.unmatched
    m, rep = match_slate(slate.players, projections,
                         aliases={"marquise brown": "hollywood brown"})
    assert m["x-0"].points == 12.0
    assert rep.by_method.get("alias") == 1
    assert rep.alias_hits == [("Marquise Brown", "Hollywood Brown")]


# --------------------------------------------------------------------------
# vegas: the board carries the whole season, not the slate
# --------------------------------------------------------------------------

def _odds_payload():
    """Two events involving BUF, in board order: the real Week 1 game first, then a
    later week. Lines are the real 2026 Week 1 numbers (BUF@HOU 44.5, BUF -1.5 =>
    23.0 implied). The Week 2 game is given a fat total so a wrong-week pick is
    unmistakable in the assertion.
    """
    def book(total, spreads, upd="2026-09-07T20:00:00Z"):
        return {"key": "fanduel", "title": "FanDuel", "last_update": upd, "markets": [
            {"key": "totals", "last_update": upd,
             "outcomes": [{"name": "Over", "point": total},
                          {"name": "Under", "point": total}]},
            {"key": "spreads", "last_update": upd,
             "outcomes": [{"name": n, "point": p} for n, p in spreads.items()]},
        ]}
    return [
        {"id": "w1", "commence_time": "2026-09-13T17:00:00Z",
         "home_team": "Houston Texans", "away_team": "Buffalo Bills",
         "bookmakers": [book(44.5, {"Buffalo Bills": -1.5, "Houston Texans": 1.5})]},
        {"id": "w2", "commence_time": "2026-09-18T00:15:00Z",
         "home_team": "Buffalo Bills", "away_team": "Detroit Lions",
         "bookmakers": [book(53.5, {"Buffalo Bills": -9.0, "Detroit Lions": 9.0})]},
    ]


def _client(payload):
    from dfs.vegas import OddsClient
    oc = OddsClient(api_key="test")
    oc._get = lambda path, params: payload          # no network
    return oc


def test_slate_games_pins_vegas_to_the_right_week():
    """The regression. The /odds endpoint returns every event the book has priced --
    272 of them on 2026-09-07, 17 involving Buffalo -- and team-keyed output meant the
    LAST event won. Week 1 builds were tilted on future weeks' lines.
    """
    oc = _client(_odds_payload())
    out = oc.team_lines(slate_teams={"BUF", "HOU"}, slate_games={"BUF@HOU"})
    assert oc.games_matched == 1
    assert oc.events_seen == 2
    assert out["BUF"].game_total == 44.5
    assert out["BUF"].implied_total == pytest.approx(23.0)
    assert out["BUF"].kickoff_iso.startswith("2026-09-13")
    assert "DET" not in out                       # the later game is not on this slate


def test_earliest_kickoff_wins_without_a_game_filter():
    """Belt and braces: callers that can only supply team names still must not pick up
    a future week's line."""
    oc = _client(_odds_payload())
    out = oc.team_lines(slate_teams={"BUF", "HOU", "DET"})
    assert out["BUF"].implied_total == pytest.approx(23.0)
    assert out["BUF"].kickoff_iso.startswith("2026-09-13")


def test_kickoff_times_follow_the_selected_game():
    """KickoffSchedule.from_team_lines is the nflverse fallback that decides which
    slots are still unlocked in a late swap, so a wrong-week kickoff is a wrong lock
    state, not just a cosmetic error."""
    from dfs.kickoffs import KickoffSchedule
    oc = _client(_odds_payload())
    out = oc.team_lines(slate_teams={"BUF", "HOU"}, slate_games={"BUF@HOU"})
    sched = KickoffSchedule.from_team_lines(out)
    assert all(k.startswith("2026-09-13") for k in
               [out["BUF"].kickoff_iso, out["HOU"].kickoff_iso])
    assert sched is not None


def test_book_preference_and_provenance_recorded():
    oc = _client(_odds_payload())
    out = oc.team_lines(slate_games={"BUF@HOU"})
    assert out["BUF"].book == "fanduel"
    assert oc.books_used == {"fanduel": 1}
    assert oc.newest_update == "2026-09-07T20:00:00Z"


def test_total_and_spread_never_cross_books():
    """A book offering a total but no spread must not lend its total to another book's
    spread -- that pairing exists on no real board."""
    payload = _odds_payload()[:1]
    payload[0]["bookmakers"] = [
        {"key": "fanduel", "last_update": "z", "markets": [
            {"key": "totals", "last_update": "z",
             "outcomes": [{"name": "Over", "point": 60.0}]}]},
        {"key": "draftkings", "last_update": "z", "markets": [
            {"key": "totals", "last_update": "z",
             "outcomes": [{"name": "Over", "point": 44.5}]},
            {"key": "spreads", "last_update": "z",
             "outcomes": [{"name": "Buffalo Bills", "point": -1.5},
                          {"name": "Houston Texans", "point": 1.5}]}]},
    ]
    oc = _client(payload)
    out = oc.team_lines(slate_games={"BUF@HOU"})
    # FanDuel is preferred but incomplete here, so DraftKings' own pair is used whole.
    assert out["BUF"].book == "draftkings"
    assert out["BUF"].game_total == 44.5


# --------------------------------------------------------------------------
# props cache
# --------------------------------------------------------------------------

def _stub_client(tmp_path, payload, max_age_hours=6.0):
    from dfs.props import PropsClient
    pc = PropsClient(api_key="test", cache_dir=tmp_path, max_age_hours=max_age_hours)
    calls = []

    def _get(path, params):
        calls.append(path)
        return dict(payload)

    pc._get = _get
    pc.calls = calls
    return pc


def test_cache_hit_avoids_a_second_fetch(tmp_path):
    pc = _stub_client(tmp_path, board())
    pc.event_board("ed6d24ff")
    pc.event_board("ed6d24ff")
    assert len(pc.calls) == 1
    assert pc.cache_hits == 1


def test_cache_freshness_uses_the_stamp_not_the_mtime(tmp_path):
    """A cached board copied by git (clone, pull, checkout) gets a current mtime while
    its contents are days old. Freshness must come from the stamp written inside the
    file, or a stale board is served as live market data."""
    import os
    import time as _t
    pc = _stub_client(tmp_path, board(), max_age_hours=6.0)
    pc.event_board("ed6d24ff")
    f = tmp_path / "props-ed6d24ff.json"
    stale = json.loads(f.read_text())
    stale["_fetched_at"] = "2026-09-01T12:00:00+00:00"          # six days old
    f.write_text(json.dumps(stale))
    os.utime(f, (_t.time(), _t.time()))                          # fresh mtime, as git leaves it
    pc2 = _stub_client(tmp_path, board(), max_age_hours=6.0)
    pc2.event_board("ed6d24ff")
    assert pc2.calls == [f"sports/americanfootball_nfl/events/ed6d24ff/odds"]
    assert pc2.cache_hits == 0
    assert pc2.stale_cache == 1


def test_unstamped_cache_file_is_refetched(tmp_path):
    f = tmp_path / "props-ed6d24ff.json"
    f.write_text(json.dumps(board()))            # no _fetched_at
    pc = _stub_client(tmp_path, board())
    pc.event_board("ed6d24ff")
    assert len(pc.calls) == 1


def test_corrupt_cache_file_is_refetched(tmp_path):
    (tmp_path / "props-ed6d24ff.json").write_text("{not json")
    pc = _stub_client(tmp_path, board())
    pc.event_board("ed6d24ff")
    assert len(pc.calls) == 1


def test_props_cache_directory_is_gitignored():
    """A tracked cache is rewritten by every checkout. Guard the ignore rule itself."""
    ig = (Path(__file__).parents[1] / ".gitignore").read_text()
    assert "data/props/" in ig
