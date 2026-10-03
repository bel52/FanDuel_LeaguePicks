"""2026-10-03: FantasyPros DOUBTFUL de-escalation and Odds API credit budget."""
import json
from datetime import datetime, timezone, timedelta

from dfs.injuries import (records_from_fantasypros, official_deescalate, InjuryRecord,
                          Status, Action, play_probability, DEESCALATE_MIN_PROB)
from dfs.matching import norm_name


def _fp(**kw):
    base = {"name": "Mike Evans", "team_id": "TB", "injury_type": "Hamstring"}
    base.update(kw)
    return records_from_fantasypros([base])[norm_name(base["name"])]


def test_fp_doubtful_with_high_probability_stays_in_pool():
    """Week 3 2026: Evans removed as DOUBTFUL at 87% to play."""
    r = _fp(status="Doubtful", probability_of_playing="0.87")
    assert r.status is Status.QUESTIONABLE and r.action is Action.FLAG
    assert abs(play_probability(r) - 0.87) < 1e-9
    assert r.detail.startswith("FP doubtful")      # visible in the 48-char sweep line


def test_practice_escalation_respects_probability():
    r = _fp(status="Q", practice_1="DNP", practice_2="DNP", probability_of_playing="0.62")
    assert r.status is Status.QUESTIONABLE
    assert abs(play_probability(r) - 0.62) < 1e-9


def test_low_probability_doubtful_still_removed():
    r = _fp(status="Doubtful", probability_of_playing="0.20")
    assert r.status is Status.DOUBTFUL and r.action is Action.REMOVE
    assert _fp(status="Doubtful").action is Action.REMOVE          # no probability
    assert _fp(status="Doubtful", probability_of_playing=f"{(DEESCALATE_MIN_PROB-1)/100}").action is Action.REMOVE


def test_probability_never_creates_a_removal():
    r = _fp(status="Q", practice_1="FP", probability_of_playing="0.05")
    assert r.status is Status.QUESTIONABLE


def test_official_questionable_overrules_fp_doubtful():
    fp = _fp(status="Doubtful", practice_1="DNP", practice_2="LP")
    off = InjuryRecord(name="Mike Evans", team="TB", status=Status.QUESTIONABLE,
                       source="sleeper")
    rep = official_deescalate(fp, off)
    assert rep is not None and rep.status is Status.QUESTIONABLE
    assert "practice DNP/LP" in rep.detail and "official QUESTIONABLE" in rep.detail
    assert abs(play_probability(rep) - 0.75) < 1e-9                  # last session LP


def test_official_deescalation_is_narrow():
    off_q = InjuryRecord(name="X", team="TB", status=Status.QUESTIONABLE, source="sleeper")
    csv_d = InjuryRecord(name="X", team="TB", status=Status.DOUBTFUL, source="fanduel_csv")
    fp_out = InjuryRecord(name="X", team="TB", status=Status.OUT, source="fantasypros")
    fp_d = InjuryRecord(name="X", team="TB", status=Status.DOUBTFUL, source="fantasypros")
    off_d = InjuryRecord(name="X", team="TB", status=Status.DOUBTFUL, source="sleeper")
    assert official_deescalate(csv_d, off_q) is None      # official-ish source kept
    assert official_deescalate(fp_out, off_q) is None     # OUT never overruled
    assert official_deescalate(fp_d, off_d) is None       # official agrees
    assert official_deescalate(fp_d, None) is None        # absence is not evidence


# ---- credit budget ----
from dfs.props import PropsClient, CREDITS_PER_EVENT


def _client(tmp_path, remaining, **kw):
    pc = PropsClient(api_key="k", cache_dir=tmp_path, **kw)
    calls = []
    events = [{"id": f"e{i}", "home_team": h, "away_team": a}
              for i, (a, h) in enumerate([("Buffalo Bills", "Houston Texans"),
                                          ("Tampa Bay Buccaneers", "Cincinnati Bengals")])]

    def fake_get(path, params):
        calls.append(path)
        pc.last_quota = {"remaining": str(remaining), "used": "0", "last": "0"}
        if path.endswith("/events"):
            return events
        return {"home_team": "x", "away_team": "y", "bookmakers": []}
    pc._get = fake_get
    return pc, calls


def test_budget_blocks_fetch_that_would_breach_reserve(tmp_path):
    pc, calls = _client(tmp_path, remaining=2 * CREDITS_PER_EVENT + 5, reserve_credits=20)
    pc.slate_props({"BUF@HOU", "TB@CIN"})
    assert [c for c in calls if c.endswith("/odds")] == []
    assert pc.cache_only and "BUDGET" in pc.budget_note


def test_budget_allows_fetch_with_headroom(tmp_path):
    pc, calls = _client(tmp_path, remaining=500, reserve_credits=20)
    pc.slate_props({"BUF@HOU", "TB@CIN"})
    assert len([c for c in calls if c.endswith("/odds")]) == 2
    assert "budget ok" in pc.budget_note


def test_cache_only_spends_nothing_but_uses_fresh_cache(tmp_path):
    fresh = {"_fetched_at": datetime.now(timezone.utc).isoformat(),
             "home_team": "Houston Texans", "away_team": "Buffalo Bills", "bookmakers": []}
    (tmp_path / "props-e0.json").write_text(json.dumps(fresh))
    pc, calls = _client(tmp_path, remaining=500, cache_only=True)
    pc.slate_props({"BUF@HOU", "TB@CIN"})
    assert [c for c in calls if c.endswith("/odds")] == []
    assert pc.cache_hits == 1
    assert any("TB@CIN" in e and "cache-only" in e for e in pc.errors)


def test_web_test_run_passes_cache_only():
    import inspect
    from dfs import web
    assert '"--props-cache-only"' in inspect.getsource(web.api_build)


def test_official_layer_restores_only_in_gate_mode(monkeypatch, capsys):
    from types import SimpleNamespace
    from dfs import cli
    fp_d = InjuryRecord(name="Mike Evans", team="TB", status=Status.DOUBTFUL,
                        detail="Hamstring · practice DNP/DNP", source="fantasypros")
    off = {norm_name("Mike Evans"): InjuryRecord(name="Mike Evans", team="TB",
                                                 status=Status.QUESTIONABLE,
                                                 source="sleeper")}

    class FakeSleeper:
        def __init__(self, **kw): pass
        def players(self): return ({}, 0.1, True)
    monkeypatch.setattr(cli, "SleeperClient", FakeSleeper)
    monkeypatch.setattr(cli, "records_from_sleeper", lambda payload: off)
    player = SimpleNamespace(name="Mike Evans", position="WR", team="TB",
                             salary=7400, fd_id="1")
    slate = SimpleNamespace(players=[player])
    out = cli._official_inactives_layer(
        slate, {norm_name("Mike Evans"): fp_d}, SimpleNamespace(official_inactives="log"))
    assert out[norm_name("Mike Evans")].status is Status.DOUBTFUL      # LOG applies nothing
    assert "WOULD be overruled" in capsys.readouterr().out
    out = cli._official_inactives_layer(
        slate, {norm_name("Mike Evans"): fp_d}, SimpleNamespace(official_inactives="gate"))
    rec = out[norm_name("Mike Evans")]
    assert rec.status is Status.QUESTIONABLE and rec.action is Action.FLAG
    assert "practice DNP/DNP" in rec.detail
    assert "RESTORED" in capsys.readouterr().out
