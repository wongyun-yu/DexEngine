"""The f0 object repair's bodies (repair_bodies, repair_obj THE BODIES): per-body thresholds, the slot set, and the
identity of a side without other bodies."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from dexengine.retarget.f0 import repair_bodies as RB
from dexengine.retarget.f0 import repair_obj as RO
from tests.imports import own_module


def _cube(c, h):
    return np.array(
        [[c[0] + sx * h, c[1] + sy * h, c[2] + sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )


def _row(pos, quat=(1.0, 0.0, 0.0, 0.0)):
    return np.r_[np.asarray(pos, float), np.asarray(quat, float)]


def _two_bodies(slot_x=0.1):
    """A main cube at the origin (first seven qpos columns), a slot cube at (slot_x, 0, 0) (next seven); both 4 cm."""
    main = RO.CookedObject([_cube((0, 0, 0), 0.02)], "m")
    slot = RO.CookedObject([_cube((0, 0, 0), 0.02)], "s")
    bs = RB.BodySet(main, 0, [("lid", "slot", slot, 7)], "d")
    return bs, np.r_[_row((0, 0, 0)), _row((slot_x, 0, 0))]


def _tip(c):
    return [("right_tip", [_cube(c, 0.01)])]


def test_each_body_is_measured_with_its_own_class():
    """A finger 1.5 mm inside the slot and apart from the main object: the Sat's main value is 0 and the slot reads
    1.5 mm; the pieces sit in the main object's frame at the row's poses."""
    bs, q = _two_bodies()
    at = bs.at(q)
    assert at.bodies == [("main", "main"), ("lid", "slot")] and at.body_of == [0, 1]
    sat = RO.sat_bodies(_tip((0.1 - 0.02 - 0.01 + 0.0015, 0, 0)), at, np.eye(3), np.zeros(3))
    assert sat[0] == 0.0 and sat.others[0][:3] == ("lid", "slot", pytest.approx(0.0015))


@pytest.mark.parametrize("slot_mm,main_mm,opened", [(1.5, 0.0, True), (0.8, 0.0, False), (0.0, 3.0, False)])
def test_a_slot_opens_above_1mm_and_the_main_above_5mm(capsys, slot_mm, main_mm, opened):
    sat = RO.Sat([("main", "main", main_mm / 1000.0, "tip"), ("lid", "slot", slot_mm / 1000.0, "tip")])
    rep = {"side": "right"}
    assert RO.gate(rep, sat, "", False) is opened
    assert rep["obj_sat_mm"] == main_mm and rep["obj_bodies"][0]["sat_mm"] == slot_mm
    assert "slot:lid=" in capsys.readouterr().out


def test_residual_and_release_are_per_body():
    """A slot keeps ≤ 1 mm (the main keeps ≤ 2 mm): 1.2 mm in the slot fails, 1.5 mm in the main passes; at release the
    slot must be below 1 mm."""
    sat = RO.Sat([("main", "main", 0.0015, "tip"), ("lid", "slot", 0.0012, "tip")])
    assert RO.failed_checks(sat[0], sat[1], 0.0, 0.0, sat) == "slot lid SAT 1.20mm > 1mm (tip)"
    ok = RO.Sat([("main", "main", 0.0015, "tip"), ("lid", "slot", 0.0009, "tip")])
    assert RO.failed_checks(ok[0], ok[1], 0.0, 0.0, ok) == ""
    n = RO.window_frames()
    late = [ok] * n
    late[RO.release_frame()] = RO.Sat([("main", "main", 0.0, ""), ("lid", "slot", 0.001, "tip")])
    assert "slot lid SAT 1.00mm ≥ 1mm" in RO.release_failed([1.5] * n, late)
    assert RO.release_failed([1.5] * n, [ok] * n) == ""


def test_a_side_without_other_bodies_is_its_main_object(monkeypatch, tmp_path, capsys):
    """Identity: no other main object, no hand-on slot and no hand-on still → with_bodies returns the main object itself
    (same pieces and digest), the report lists no slots and the key's policy line is the one before the rule."""
    main = RO.CookedObject([_cube((0, 0, 0), 0.02)], "m")
    o = {"name": "mug", "role": "main", "qpos_start": 0}
    (tmp_path / "task_info.json").write_text(json.dumps({"objects": [o]}))
    (tmp_path / "0").mkdir()
    np.savez(tmp_path / "0/trajectory_kinematic.npz", qpos=np.zeros((2, 7)))
    from dexengine.data.objects import extra_slots as XS

    monkeypatch.setattr(XS, "clip_rows", lambda *a, **k: [])  # no extra row: no slot, no still
    assert RB.with_bodies(main, o, tmp_path, "right", "taco") is main
    assert RO.policy_line(False) == RO.policy_line() and "slot" not in RO.policy_line()
    assert "slot open>1.0" in RO.policy_line(True)
    assert "main mug + -" in capsys.readouterr().out


def test_hand_on_slots_are_moving_rows_whose_hand_collides(monkeypatch):
    """Only SLOT rows (kinematic) with the training preset's hand channel ON are slots: a pinned still
    (STATIC_SPAWN) and a slot whose hand channel is OFF are not."""
    from dexengine.data.objects import extra_slots as XS
    from dexengine.rl.tasks.scene import collision_presets as CP

    rows = [
        {"name": "lid", "cls": XS.SLOT, "kind": "moving"},
        {"name": "rack", "cls": XS.STATIC_SPAWN, "kind": "touched"},
        {"name": "cap", "cls": XS.SLOT, "kind": "carried_off"},
    ]
    monkeypatch.setattr(
        CP, "extra_collide", lambda kind, full=None, ds=None: {"hand": "OFF" if kind == "carried_off" else "ON"}
    )
    ti = {"objects": [{"name": n, "qpos_start": 7 * i} for i, n in enumerate(("lid", "rack", "cap"))]}
    got = RB.hand_on_slots(rows, ti, "oakink2")
    assert [(o["name"], k) for o, k in got] == [("lid", "moving")]


def test_the_projection_targets_each_body_with_its_residual(monkeypatch):
    """A finger 1.5 mm inside a slot (apart from the main object) is projected to the slot's own target (residual
    1 mm × STRICT_MARGIN_SCALE), not to the main object's (min(OBJ_TOUCH_MM, 2 mm × scale) = 1 mm); the touch measures
    read the main object's pieces only."""
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    S = own_module("dexengine.retarget.f0.repair_obj_solve")
    at = _two_bodies()[0].at(_two_bodies()[1])
    nf = S.HC.nd - 6
    strict = types.SimpleNamespace(links=[("right_tip", 0, None, None)], parts={"right_tip": [_cube((0, 0, 0), 0.01)]})
    box = (np.full(nf, -9.0), np.full(nf, 9.0))
    P = S.Projection(strict, at, np.eye(3), np.zeros(3), (np.zeros(3), np.eye(3), np.zeros(nf)), -1.0, box, [])
    c0 = np.array([0.1 - 0.02 - 0.01 + 0.0015, 0.0, 0.0])
    P.link_T = lambda dq: [(np.eye(3), c0 + np.asarray(dq[:3], float))]
    assert P.res_mm.tolist() == [RO.OBJ_RESIDUAL_MM, RO.SLOT_RESIDUAL_MM] and len(P.ostarts) == 1

    def slot_mm(dq):
        return RO.sat_bodies(_tip(c0 + dq[:3]), at, np.eye(3), np.zeros(3)).others[0][2] * 1000.0

    r_t = np.minimum(RO.OBJ_TOUCH_MM, P.res_mm * S.STRICT_MARGIN_SCALE) / 1000.0
    assert slot_mm(P.solve(r_t)) == pytest.approx(RO.SLOT_RESIDUAL_MM * S.STRICT_MARGIN_SCALE, abs=0.02)
    assert slot_mm(P.solve(0.001)) == pytest.approx(1.0, abs=0.02)  # one main-object target leaves the slot at 1 mm
