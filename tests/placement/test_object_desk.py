"""Rule D24 in the judge (placement/scene/object_desk, CPU): a main-object hull vertex deeper than OBJ_BELOW_M below the
desk top inside the spawned desk footprint cuts the placement; the same motion beside the desk is free; the mask
equals the final check; the value has one owner and the height-only first-pass drop is gone."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from dexengine.data import desk_penetration as DP
from dexengine.placement.coords import core as VC
from dexengine.placement.scene import object_desk as OD

REPO = Path(__file__).resolve().parents[2]
BOX = ((0.0, 0.76), (-0.70, 0.70))  # scene footprint ((x0, x1), (y0, y1))
DESK_Z = 0.10  # capture desk plane; desk top in the scene = DESK_Z + Z_OFF
Z_OFF, TOP = 0.66, 0.76
CUBE = np.array([(x, y, z) for x in (-0.02, 0.02) for y in (-0.02, 0.02) for z in (-0.02, 0.02)], float)


def _tree(tmp_path: Path, xy_scene, dig_m: float) -> Path:
    """A one-object tree: a 4 cm cube resting on the desk plane, then pushed `dig_m` into it at scene xy `xy_scene`."""
    T = 30
    t = tmp_path / f"clip_{dig_m}_{xy_scene[0]}"
    (t / "0").mkdir(parents=True)
    cap = VC.scene_to_capture(np.array(xy_scene, float))
    q = np.zeros((T, 7))
    q[:, 3] = 1.0  # wxyz identity
    q[:, :2] = cap
    q[:, 2] = DESK_Z + 0.02
    q[10:20, 2] = DESK_Z + 0.02 - dig_m
    np.savez(t / "0/trajectory_kinematic.npz", qpos=q)
    (t / "task_info.json").write_text(json.dumps({"dataset_name": "taco", "objects": [{"name": "o", "qpos_start": 0}]}))
    return t


@pytest.fixture
def fake_hull(monkeypatch):
    """The hull reader faked (a cube per main object) and the main slots of the one-object tree; in-process path."""

    def hulls(ti, q, mains):
        from scipy.spatial.transform import Rotation as Rot

        R = Rot.from_quat(q[:, 3:7][:, [1, 2, 3, 0]]).as_matrix()
        return [(CUBE, R, q[:, :3])], "hull"

    monkeypatch.setattr(DP, "_main_hulls", hulls)
    monkeypatch.setattr(VC, "desk_mask_slots", lambda t, D=None: [(0, 7)])
    monkeypatch.setattr(OD, "_has_pxr", lambda: True)


def _cut(t: Path) -> bool:
    """Does the check cut the tree's own placement (no move, yaw 0)?"""
    deep = OD.deep_points(t, Z_OFF, TOP)
    return OD.hit(deep, (0.0, 0.0, 0.0, (0.0, 0.0)), BOX)


def test_a_planted_25mm_dig_over_the_desk_drops_and_the_same_motion_beside_it_keeps(tmp_path, fake_hull):
    assert _cut(_tree(tmp_path, (0.40, 0.0), 0.025))  # over the desk: cut
    assert not _cut(_tree(tmp_path, (-0.30, 0.0), 0.025))  # the same motion off the footprint: free space
    assert not _cut(_tree(tmp_path, (0.40, 0.0), 0.015))  # within the value: kept


def test_the_value_is_read_from_its_owner(tmp_path, fake_hull, monkeypatch):
    """Negative: with a 1 cm value the 15 mm dig over the desk is cut — the check holds no copy of the value."""
    t = _tree(tmp_path, (0.40, 0.0), 0.015)
    assert not _cut(t)
    monkeypatch.setattr(OD, "OBJ_BELOW_M", 0.01)
    assert _cut(t)


def test_the_mask_equals_the_final_check():
    rng = np.random.default_rng(1)
    deep = rng.uniform(-0.05, 0.05, size=(8, 2)) + VC.scene_to_capture(np.array([0.9, 0.0]))
    gx, gy = np.arange(-12, 13) * 0.035, np.arange(-21, 22) * 0.035
    ww = deep @ VC.RZW.T  # yaw 0
    grid = OD.grid_off_box(ww, gx, gy, BOX)
    for i in range(0, len(gx), 3):
        for j in range(0, len(gy), 3):
            d = VC.scene_to_capture(np.array([gx[i], gy[j]]))
            assert grid[i, j] == (not OD.hit(deep, (float(d[0]), float(d[1]), 0.0, (0.0, 0.0)), BOX))
    assert grid.any() and not grid.all()  # both outcomes occur


def _assigns(name: str) -> list[str]:
    def has(p: Path) -> bool:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        return any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
            for n in ast.walk(tree)
        )

    return [str(p.relative_to(REPO)) for p in sorted((REPO / "dexengine").rglob("*.py")) if has(p)]


def test_one_owner_and_the_first_pass_reads_it():
    """OBJ_BELOW_M is assigned in object_desk only; the first pass's height-only drop reads
    the cap from there (candidate_rules.d24_state) — no copy."""
    assert _assigns("OBJ_BELOW_M") == ["dexengine/placement/scene/object_desk.py"]
    rules = (REPO / "dexengine/data/candidate_rules.py").read_text(encoding="utf-8")
    assert "from dexengine.placement.scene.object_desk import OBJ_BELOW_M" in rules
    src = (REPO / "dexengine/data/candidates.py").read_text(encoding="utf-8")
    assert '("drop", "obj_below_desk(D24)")' in src


def _calls(path: str, func: str) -> set[str]:
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    return {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}


@pytest.mark.parametrize(
    "path,func,call",
    [
        ("dexengine/placement/judge/clip_inputs.py", "_clip_objects", "_OD.deep_points"),
        ("dexengine/placement/judge/clip_inputs.py", "_clip_desk_mask", "_OD.grid_off_box"),
        ("dexengine/placement/judge/search.py", "desk_ok_final", "_OD.hit"),
    ],
)
def test_the_judge_measures_masks_and_checks_with_this_rule(path, func, call):
    """Structural: the judge's clip prep, desk mask and final desk check each call the rule (deleting one fails)."""
    assert call in _calls(path, func)
