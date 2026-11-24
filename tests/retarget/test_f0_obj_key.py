"""The f0 object geometry key (repair_geom.object_cache_key): the process cache and the GPU evaluator's obj_key.

Measured with the key before (mesh, surface points, θ0 rounded to 1e-4 rad), the clip solved second in a process
got the first one's geometry: another joint record at the same angle (close_laptop_lid 6b5c334699d5_s002 left n_pen 483
after 39c30d86600d_s005, 0 alone) and the same record at another angle inside the rounding (pull_out_drawer
2e0d01782266_s000 right n_pen 461 alone, 459 · 460 after other drawer clips).
- a clip's geometry is built from its own record and its own angle in either processing order (synthetic clips through
  repair_side._load_side; the old key fails both — the tests never read the live data root, the real clips are measured
  in the commit);
- another record or another angle is another key; the same record written in another key order is not;
- a rigid key is (mesh_dir, sp_dir) as before, whatever the angle; an articulated record without a geometry field dies.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest

from tests.imports import own_module

REC_A = {
    "type": "hinge",
    "axis": [-1.0, 0.0, 0.0],
    "pos": [0.0, 0.0, 0.0],
    "range": [-0.1, 2.0],
    "qpos_index": 0,
    "parts": {"base": "visual.obj", "moving": "../lid/visual.obj"},
    "child_pose": {"pos": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]},
}
REC_B = dict(REC_A, axis=[1.0, 0.0, 0.0], parts={"base": "../lid/visual.obj", "moving": "visual.obj"})


def _geom(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair_geom")


def test_the_record_is_in_the_key(monkeypatch):
    G = _geom(monkeypatch)
    ka, kb = G.object_cache_key("m", "s", REC_A, -0.0), G.object_cache_key("m", "s", REC_B, 0.0)
    assert ka != kb  # same mesh, same angle, another joint
    assert ka[:4] == kb[:4] == ("m", "s", "artic", 0.0)  # what the old key held: equal
    reordered = {k: REC_A[k] for k in reversed(list(REC_A))}
    assert G.object_cache_key("m", "s", reordered, 0.0) == ka  # the same record written in another key order
    assert G.object_cache_key("m", "s", REC_A, 0.00003) != ka  # another angle, the same to 1e-4 rad


def test_rigid_key_is_unchanged(monkeypatch):
    G = _geom(monkeypatch)
    assert G.object_cache_key("m", "s", None, 0.0) == ("m", "s")
    assert G.object_cache_key("m", "s", None, 1.3) == ("m", "s")  # a rigid body has no angle
    assert G.object_cache_key("m", "s", REC_A, 0.0) != ("m", "s")


@pytest.mark.parametrize(
    "rec",
    [{}, {k: v for k, v in REC_A.items() if k != "axis"}, dict(REC_A, parts={"base": "visual.obj"}), "hinge"],
)
def test_an_articulated_record_without_its_geometry_dies(monkeypatch, rec):
    G = _geom(monkeypatch)
    with pytest.raises(SystemExit, match="F0_OBJ_KEY"):
        G.object_cache_key("m", "s", rec, 0.0)


#: two clips on one mesh that the key before the fix merged: (record, frame-0 angle) of each
CASES = {
    "record": {"cA": (REC_A, 0.0), "cB": (REC_B, 0.0)},  # another joint at the same angle (the laptop pair)
    "angle": {"cA": (REC_A, -0.00133), "cB": (REC_A, -0.00127)},  # the same joint, angles equal to 1e-4 (the drawers)
}


def _old_key(m, s, a, th):
    return (m, s) if a is None else (m, s, "artic", round(float(th), 4))


def _two_clips(monkeypatch, tmp_path, case):
    """Two synthetic clips on one mesh; ObjectGeom returns what it was built from (axis x, angle)."""
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    RS = own_module("dexengine.retarget.f0.repair_side")
    H, T = RS.HC.H, 4
    recs = {c: (dict(rec, qpos_index=H + 7), th) for c, (rec, th) in CASES[case].items()}
    for c, (_rec, th) in recs.items():
        d = tmp_path / "data" / c
        (d / "0").mkdir(parents=True)
        q = np.zeros((T, H + 8))
        q[:, H + 3] = 1.0
        q[:, H + 7] = th
        np.savez(d / "0/trajectory_kinematic.npz", qpos=q)
        (d / "task_info.json").write_text("{}")
        (tmp_path / "w/wrist_poses_right").mkdir(parents=True, exist_ok=True)
        np.savez(
            tmp_path / f"w/wrist_poses_right/{c}.npz",
            positions=np.zeros((T, 3)),
            quaternions=np.tile([1.0, 0, 0, 0], (T, 1)),
        )
    lay = types.SimpleNamespace(hand_qpos_dim=H, hand_slice=lambda s: (0, H), main_slice=lambda s: (H, H + 7))
    monkeypatch.setattr(RS.RSEP, "applied", lambda tree, qq: qq)
    monkeypatch.setattr(RS.QL, "read", lambda *a, **k: lay)
    monkeypatch.setattr(RS, "main_object_dirs", lambda tree, *a: ("m", "s", recs[Path(tree).name][0]))
    monkeypatch.setattr(RS, "desk_height", lambda: -1.0)
    monkeypatch.setattr(RS, "scene_z_offset", lambda tree: 0.0)
    monkeypatch.setitem(RS._HAND_CACHE, (RS.HC.name, "right"), types.SimpleNamespace(pen_margin=0.0))
    monkeypatch.setattr(RS, "ObjectGeom", lambda m, sp_dir, articulation, theta: (articulation["axis"][0], theta))
    monkeypatch.setattr(RS, "_OBJ_CACHE", {})
    want = {c: (rec["axis"][0], th) for c, (rec, th) in recs.items()}
    return RS, recs, want


def _load(RS, tmp_path, order):
    return {c: RS._load_side(c, "right", tmp_path / "data", tmp_path / "w", 0.0, "oakink2")[5] for c in order}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("order", [("cA", "cB"), ("cB", "cA")])
def test_each_clip_gets_its_own_geometry_in_either_order(monkeypatch, tmp_path, case, order):
    RS, recs, want = _two_clips(monkeypatch, tmp_path, case)
    scenes = _load(RS, tmp_path, order)
    assert {c: sc[1] for c, sc in scenes.items()} == want
    for c, sc in scenes.items():  # the GPU obj_key is the same key
        assert sc[2] == RS.object_cache_key("m", "s", recs[c][0], recs[c][1])


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_old_key_hands_the_second_clip_the_first_ones_geometry(monkeypatch, tmp_path, case):
    """Negative: with the key before the fix the order test fails — the clip loaded second gets the first one's."""
    RS, _recs, want = _two_clips(monkeypatch, tmp_path, case)
    monkeypatch.setattr(RS, "object_cache_key", _old_key)
    got = {c: sc[1] for c, sc in _load(RS, tmp_path, ("cA", "cB")).items()}
    assert got == {"cA": want["cA"], "cB": want["cA"]} != want
