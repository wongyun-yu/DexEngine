"""Object quaternions are written unit and read unit (dexengine.retarget.quat_bounds).

- the converter (mano_lib.repair_object_quaternions) rescales every clip's object block to unit length, marks
  `quat_normalized` only beyond the corruption bound OBJ_QUAT_ATOL, and refuses a zero quaternion;
- require_unit passes unit quaternions and fails beyond UNIT_ATOL;
- the RL loader's main-object block (loader_clips._obj_block) dies on an off-unit quaternion;
- IK's input check (ik.inputs._check_poses) dies on an off-unit object block;
- every reader that builds a rotation from the stored object quaternion calls require_unit: f0's frame-0 load and
  window poses (repair_side._load_side · _window_poses), IK's input check and the RL loader's extra slots
  (loader._fill_slots) — read from the code."""

from __future__ import annotations

import ast

import numpy as np
import pytest


def _poses(norm: float, T: int = 5) -> np.ndarray:
    q = np.array([0.9, 0.1, -0.3, 0.2])
    p = np.zeros((T, 7))
    p[:, :3] = 0.1
    p[:, 3:] = q / np.linalg.norm(q) * norm
    return p


@pytest.mark.parametrize(("norm", "marked"), [(1.003, False), (0.9, True)])
def test_converter_writes_unit(norm, marked):
    from dexengine.data.readers import mano_lib

    arc = {"qpos_obj_right": _poses(norm), "qpos_obj_left": _poses(1.0)}
    worst = mano_lib.repair_object_quaternions(arc)
    for k in ("qpos_obj_right", "qpos_obj_left"):
        assert np.abs(np.linalg.norm(arc[k][:, 3:], axis=1) - 1.0).max() < 1e-12
        assert np.array_equal(arc[k][:, :3], _poses(1.0)[:, :3])  # positions untouched
    assert ("quat_normalized" in arc) is marked
    assert (worst is not None) is marked


def test_converter_refuses_zero():
    from dexengine.data.readers import mano_lib

    p = _poses(1.0)
    p[2, 3:] = 0.0
    with pytest.raises(ValueError, match="zero quaternion"):
        mano_lib.repair_object_quaternions({"qpos_obj_right": p, "qpos_obj_left": _poses(1.0)})


def test_require_unit():
    from dexengine.retarget.quat_bounds import UNIT_ATOL, require_unit

    require_unit(_poses(1.0)[:, 3:], "unit")
    require_unit(_poses(1.0 + 0.5 * UNIT_ATOL)[:, 3:], "inside")
    with pytest.raises(ValueError, match="QUAT_UNIT"):
        require_unit(_poses(1.0 + 2 * UNIT_ATOL)[:, 3:], "outside")
    with pytest.raises(ValueError, match="QUAT_UNIT"):
        require_unit(np.full((3, 4), np.nan), "nan")


def test_require_unit_names_the_rebuild_commands():
    """The refusal says how to rebuild (stage 1, then the base layer) — the same commands docs/datasets.md gives."""
    from dexengine.core.repo import REPO_ROOT
    from dexengine.retarget.quat_bounds import require_unit

    with pytest.raises(ValueError) as e:
        require_unit(_poses(1.5)[:, 3:], "old tree")
    msg = str(e.value)
    assert "scripts/stage1_retarget.py --bases" in msg and "dexengine/data/base_adapter.py --dataset" in msg
    docs = (REPO_ROOT / "docs" / "datasets.md").read_text(encoding="utf-8")
    assert "scripts/stage1_retarget.py --bases" in docs and "dexengine/data/base_adapter.py --dataset taco" in docs


def test_rl_loader_object_block_dies_off_unit():
    from dexengine.rl.tasks.mdp.trajectory import loader_clips as LC

    class Lay:
        def main_slice(self, side):
            return (0, 7)

        def is_active(self, side):
            return True

    ok = _poses(1.0)
    assert np.array_equal(LC._obj_block(Lay(), ok, 5, 7, "clip", "right"), ok)
    with pytest.raises(ValueError, match="QUAT_UNIT"):
        LC._obj_block(Lay(), _poses(1.003), 5, 7, "clip", "right")


def test_ik_input_check_dies_off_unit():
    from dexengine.retarget.ik import inputs as IKI

    def data(obj_norm):
        d = {}
        for k, tail in IKI.POSE_KEYS.items():
            a = np.zeros((5, *tail))
            a[..., 3] = 1.0
            d[k] = a
        d["qpos_obj_left"] = _poses(obj_norm)
        return d

    assert IKI._check_poses(data(1.0), "unit", 2) == 5
    with pytest.raises(ValueError, match="QUAT_UNIT"):
        IKI._check_poses(data(1.003), "off-unit", 2)


def _calls(module: str, fn: str) -> set:
    """Names called inside function `fn` of `module`, read from its source file without importing it (importing f0
    modules binds the hand configuration for the whole session)."""
    import importlib.util

    with open(importlib.util.find_spec(module).origin) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            return {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    raise AssertionError(f"{module}.{fn} not found")


def test_rotation_readers_require_unit():
    f0 = "dexengine.retarget.f0.repair_side"
    assert "require_unit" in _calls(f0, "_load_side")
    assert "require_unit" in _calls(f0, "_window_poses")
    assert "require_unit" in _calls("dexengine.retarget.ik.inputs", "_check_poses")
    assert "require_unit" in _calls("dexengine.rl.tasks.mdp.trajectory.loader", "_fill_slots")
