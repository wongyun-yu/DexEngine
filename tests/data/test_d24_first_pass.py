"""D24 in the first pass (object based): the main-object collision hull deeper than
object_desk.OBJ_BELOW_M below the desk surface at any frame drops the tree, by height alone; a clip on the manual hold
table is held instead of dropped."""

from pathlib import Path

import pytest

from dexengine.data import candidate_rules as CR
from dexengine.data import candidates as CA
from dexengine.placement.scene.object_desk import OBJ_BELOW_M


def _ti(depth_m: float) -> dict:
    return {
        "support": {"desk_z": 0.10, "obj_zmin_hull_traj": 0.10 - depth_m},
        "rules": {"D24": {"version": CR.D24_INPUT_VERSION}},
    }


@pytest.fixture
def mains(monkeypatch):
    monkeypatch.setattr(CR, "no_main_object", lambda tree_dir, ti: False)


@pytest.mark.parametrize(
    ("depth", "want"),
    [(OBJ_BELOW_M + 0.005, CR.D24_BELOW), (OBJ_BELOW_M - 0.005, CR.D24_OK), (-0.03, CR.D24_OK)],
)
def test_state_compares_the_hull_with_the_desk(mains, depth, want):
    assert CR.d24_state("t", _ti(depth), True) == want
    assert want in CR.D24_MEASURED


def test_state_off_scene_is_not_measured(mains):
    assert CR.d24_state("t", _ti(0.5), False) == CR.D24_OFF_SCENE


def _verdict(monkeypatch, clip: str, held: dict):
    monkeypatch.setattr(CA, "d24_state", lambda tdir, d, applies: CR.D24_BELOW)
    monkeypatch.setattr(CA, "rule_on", lambda rule, scene: rule == "D24")
    monkeypatch.setattr(CA, "_HELD", dict(held))
    return CA._desk_verdict("hot3d", Path("/x") / clip, {}, {}, {}, "table")


def test_below_is_dropped(monkeypatch):
    verdict, _infos = _verdict(monkeypatch, "c1", {})
    assert verdict == ("drop", "obj_below_desk(D24)")


def test_a_held_clip_is_held_not_dropped(monkeypatch, capsys):
    verdict, infos = _verdict(monkeypatch, "c1", {"c1": "D24_BELOW: waits"})
    assert verdict != ("drop", "obj_below_desk(D24)")
    assert "obj_below_desk(D24):user_hold" in infos
    assert "[USER_HOLD] D24 below c1: D24_BELOW: waits" in capsys.readouterr().out


def test_held_set_is_the_repo_hold_table(monkeypatch):
    """The held set is read from the one table (cells_cond.hold_rows) once."""
    from dexengine.record.augment import cells_cond as ACC

    monkeypatch.setattr(CA, "_HELD", None)
    calls = []
    monkeypatch.setattr(ACC, "hold_rows", lambda path=None: calls.append(path) or {"c9": "D24_BELOW: x"})
    assert CA._held_below("c9") and not CA._held_below("c1")
    assert calls == [None]
