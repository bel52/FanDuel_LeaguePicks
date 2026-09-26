"""Teammate effects (2026-09-26). Rates measured on nflverse 2023-25; see teammates.py."""
from types import SimpleNamespace as NS

from dfs.teammates import (adjust_for_absences, ABSORB_REC, SAME_POS_MULT,
                           MAX_BOOST_PTS, rec_points)
from dfs.injuries import InjuryRecord, Status, play_probability
from dfs.matching import norm_name


def _sp(fd, name, pos, team, proj):
    return NS(fd_id=fd, name=name, position=pos, team=team, proj_fp=proj,
              projection=proj, proj_blend=proj, proj_source="fp", teammate_adj=None)


def _fp(name, pos, team, **stats):
    return NS(name=name, position=pos, team=team, stats=stats)


def _world():
    """BAL, Week 3 shape: Zay Flowers doubtful (30% to play) and removed from the
    pool; Bateman, Andrews and Henry remain. A DAL receiver must be untouched."""
    fps = {"bat": _fp("Rashod Bateman", "WR", "BAL", rec_rec=4, rec_yds=50),
           "and": _fp("Mark Andrews", "TE", "BAL", rec_rec=4, rec_yds=40),
           "hen": _fp("Derrick Henry", "RB", "BAL", rec_rec=1, rec_yds=8, rush_att=20,
                      rush_yds=90, rush_tds=0.8),
           "cee": _fp("CeeDee Lamb", "WR", "DAL", rec_rec=7, rec_yds=90)}
    players = [_sp("bat", "Rashod Bateman", "WR", "BAL", 9.7),
               _sp("and", "Mark Andrews", "TE", "BAL", 10.3),
               _sp("hen", "Derrick Henry", "RB", "BAL", 17.0),
               _sp("cee", "CeeDee Lamb", "WR", "DAL", 16.4)]
    slate = NS(players=players, fp_by_id=fps)
    flowers = _fp("Zay Flowers", "WR", "BAL", rec_rec=6, rec_yds=70, rec_tds=0.4)
    inj = {norm_name("Zay Flowers"): InjuryRecord(
        name="Zay Flowers", team="BAL", status=Status.DOUBTFUL,
        detail="Hamstring · practice DNP/DNP/LIMIT · 30% to play")}
    return slate, list(fps.values()) + [flowers], inj, flowers


def test_vacated_receiving_flows_to_teammates_by_volume_and_position():
    slate, fp, inj, flowers = _world()
    adjs = adjust_for_absences(slate, fp, inj, play_probability)
    by = {a.player.fd_id: a.boost for a in adjs}
    vacated = rec_points(flowers.stats) * 0.70                 # 70% he sits
    w = {"bat": 4 * SAME_POS_MULT, "and": 4.0, "hen": 1.0}
    total_w = sum(w.values())
    for k in w:
        assert abs(by[k] - round(ABSORB_REC * vacated * w[k] / total_w, 2)) < 0.02
    assert by["bat"] > by["and"] > by["hen"] > 0                # same position first
    assert "cee" not in by                                      # other team untouched
    bat = next(p for p in slate.players if p.fd_id == "bat")
    assert bat.proj_fp == round(9.7 + by["bat"], 2) and bat.teammate_adj == by["bat"]
    assert "teammates" in bat.proj_source
    # total redistributed never exceeds the conservative absorption of what was vacated
    assert sum(by.values()) <= ABSORB_REC * vacated + 0.03


def test_no_boost_when_consensus_already_zeroed_the_absent_player():
    slate, fp, inj, flowers = _world()
    flowers.stats = {}                                          # FP has him at zero
    assert adjust_for_absences(slate, fp, inj, play_probability) == []


def test_near_certain_players_vacate_nothing():
    slate, fp, inj, _ = _world()
    inj[norm_name("Zay Flowers")].detail = "Hamstring · 95% to play"
    inj[norm_name("Zay Flowers")].status = Status.QUESTIONABLE
    assert adjust_for_absences(slate, fp, inj, play_probability) == []


def test_boost_is_capped():
    slate, fp, inj, flowers = _world()
    flowers.stats = {"rec_rec": 12, "rec_yds": 200, "rec_tds": 3}    # absurd star
    adjs = adjust_for_absences(slate, fp, inj, play_probability)
    bat = next(a for a in adjs if a.player.fd_id == "bat")
    assert bat.capped and bat.boost <= min(MAX_BOOST_PTS, 0.35 * 9.7) + 1e-9


def test_rushing_goes_only_to_backs():
    slate, fp, inj, _ = _world()
    slate.players.append(_sp("rb2", "Justice Hill", "RB", "BAL", 5.0))
    slate.fp_by_id["rb2"] = _fp("Justice Hill", "RB", "BAL", rec_rec=2, rec_yds=15,
                                rush_att=5, rush_yds=20)
    inj[norm_name("Derrick Henry")] = InjuryRecord(name="Derrick Henry", team="BAL",
                                                   status=Status.OUT, detail="Ankle")
    slate.players = [p for p in slate.players if p.fd_id != "hen"]   # swept as OUT
    adjs = adjust_for_absences(slate, fp + [slate.fp_by_id["rb2"]], inj, play_probability)
    by = {a.player.fd_id: a for a in adjs}
    assert all(s[0] != a.player.name for a in adjs for s in a.sources)  # never self-boosts
    rush_to_wr = [s for s in by["bat"].sources if s[0] == "Derrick Henry"]
    assert rush_to_wr and rush_to_wr[0][2] < by["rb2"].sources[-1][2]   # WR: catches only
    hill_src = [s for s in by["rb2"].sources if s[0] == "Derrick Henry"]
    assert hill_src and hill_src[0][1] == 1.0                # OUT = 100% sits


def test_boost_reaches_the_blend_only_at_the_fp_weight():
    """Markets already reprice teammates; the props component must not be boosted."""
    from dfs.blend import apply_props
    slate, fp, inj, _ = _world()
    adjs = adjust_for_absences(slate, fp, inj, play_probability)
    boost = next(a.boost for a in adjs if a.player.fd_id == "bat")
    for p in slate.players:
        p.proj_props = None
    real = NS(players=slate.players)
    apply_props(real, {"bat": 11.0}, distributions={}, weights={"WR": 0.6})
    bat = next(p for p in slate.players if p.fd_id == "bat")
    assert abs(bat.projection - (0.6 * 11.0 + 0.4 * (9.7 + boost))) < 0.011


def test_cli_layer_respects_the_off_switch(capsys):
    from dfs import cli
    slate, fp, inj, _ = _world()
    assert cli._teammate_layer(slate, fp, inj, NS(no_teammates=True)) == []
    assert cli._teammate_layer(slate, fp, {}, NS(no_teammates=False)) == []   # no feed
    adjs = cli._teammate_layer(slate, fp, inj, NS(no_teammates=False))
    out = capsys.readouterr().out
    assert adjs and "teammate effects applied" in out and "Zay Flowers 70% out" in out


def test_cli_layer_never_breaks_the_caller(capsys, monkeypatch):
    """A Sunday swap check must survive any failure in this layer."""
    from dfs import cli
    import dfs.teammates as tm
    slate, fp, inj, _ = _world()
    def boom(*a, **k): raise KeyError("unexpected feed shape")
    monkeypatch.setattr(tm, "adjust_for_absences", boom)
    before = [p.projection for p in slate.players]
    assert cli._teammate_layer(slate, fp, inj, NS(no_teammates=False)) == []
    assert "teammate effects skipped" in capsys.readouterr().out
    assert [p.projection for p in slate.players] == before
