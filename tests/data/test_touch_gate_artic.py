"""S-touch-gate on articulated bodies (CPU): an articulated main object is measured on every link at each frame's joint
values (the f0 articulated model), an articulated extra on the rigid USD its slot spawns, and a body without cooked
parts is a declared unmeasured pair."""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import touch_gate as TG
from dexengine.retarget.f0 import repair_obj as RO

T = 21
LID = {
    "name": "lid",
    "parent": "box",
    "joint": "box_hinge",
    "type": "hinge",
    "axis": np.array([0.0, 0.0, 1.0]),
    "anchor": np.zeros(3),
    "b_pos": np.array([0.1, 0.0, 0.0]),
    "b_R": np.eye(3),
}


def _cube(c, h):
    return np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float) + np.asarray(c, float)


def _plate():
    return [np.array([[x, y, z] for x in (-0.03, 0.03) for y in (-0.01, 0.01) for z in (-0.03, 0.03)], float)]


def _scene(monkeypatch, cooked=None, plate_artic=False):
    """A box (main, root cube at the origin) whose lid swings on a z hinge from θ = 0 (x 0.13–0.17) to θ = π/2
    (x 0.08–0.12, y 0.03–0.07); a touched plate pinned at y 0.062–0.082 in front of the open lid: 8 mm deep at θ = π/2,
    clear at θ = 0 (a rigid reading at the rest pose sees no contact). Mid-swing the lid's outer corner reaches up to
    y 0.073, so the deepest exit distance lies between 8 and 11 mm."""
    q = np.zeros((T, 15))
    q[:, 3] = q[:, 10] = 1.0
    q[:, 7:9] = (0.1, 0.072)
    q[:, 14] = np.linspace(0.0, np.pi / 2, T)
    art = RO.ArticObject(
        RO.CookedObject([_cube((0, 0, 0), 0.02)], "d"),
        [(LID, 14, RO.CookedObject([_cube((0.05, 0, 0), 0.02)], "d"))],
        {"joint": "box_hinge"},
        "d",
    )
    monkeypatch.setattr(RO, "cooked", lambda o, ds, tree: cooked if cooked is not None else (art, ""))
    monkeypatch.setattr(TG, "parts_of", lambda ds, o: _plate() if o["name"] == "plate" else None)
    monkeypatch.setattr(CS, "parts", lambda ds, oid: CS.Parts(_plate()))  # the still row as convex parts
    TG.MEMO.clear()
    box = {"name": "box", "obj_id": "box", "role": "main", "qpos_start": 0, "mesh_dir": "m", "articulation": {"j": 1}}
    plate = {"name": "plate", "obj_id": "plate", "role": "extra", "qpos_start": 7, "mesh_dir": "m"}
    if plate_artic:
        plate["articulation"] = {"j": 2}
    ti = {"dataset_name": "oakink2", "objects": [box, plate]}
    return ti, q, box, plate


def test_an_articulated_main_object_is_measured_on_every_link_at_each_frame(monkeypatch):
    ti, q, box, plate = _scene(monkeypatch)
    g = TG.gate_row("/t", ti, q, plate, [box])
    assert g["why"] == "true-shape overlap" and not g["on"] and g["drop"]["partner"] == "box"
    assert 0.008 <= g["pairs"]["box"]["depth_m"] <= 0.011
    q[:, 14] = 0.0  # lid closed the whole clip: clear
    TG.MEMO.clear()
    assert TG.gate_row("/t", ti, q, plate, [box])["why"] == "clear"


def test_an_articulated_extra_is_measured_on_the_rigid_usd_its_slot_spawns(monkeypatch):
    ti, q, box, plate = _scene(monkeypatch, plate_artic=True)
    assert 0.008 <= TG.gate_row("/t", ti, q, plate, [box])["pairs"]["box"]["depth_m"] <= 0.011


def test_a_body_without_cooked_parts_is_a_declared_unmeasured_pair(monkeypatch):
    ti, q, box, plate = _scene(monkeypatch, cooked=(None, "no cooked hull (no USD)"))
    g = TG.gate_row("/t", ti, q, plate, [box])
    assert g["why"] == "unmeasured" and not g["on"]
    assert g["pairs"]["box"] == {"depth_m": None, "pass": False, "unmeasured": "no cooked hull (no USD)"}
