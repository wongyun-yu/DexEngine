"""S-float-settle in the training pin (rl/tasks/scene/float_lower, object_pin.pin_pose) and its scene-build statement
(rl/env_setup/float_settle): lift and lower are one reading, the rest-window gate, the cap, and judge gate D."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest
import torch

from dexengine.data.objects import float_settle as FS
from dexengine.rl.env_setup import float_settle as EFS
from dexengine.rl.tasks.scene import float_lower as FL
from dexengine.rl.tasks.scene import object_pin as op

DESK_TOP = 0.75
HZ = 0.05  # half height of the test box


def _env(monkeypatch, ref_z: float, first_touch=None, n: int = 3):
    """A one-clip env whose right object (a box, half height HZ) has its reference centre at ref_z for every env."""
    box = torch.tensor([[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (-HZ, HZ)])
    monkeypatch.setattr(op, "object_geometry", lambda env, obj: (box[None], torch.ones(1, 8, dtype=torch.bool), 1))
    monkeypatch.setattr(op, "desk_box", lambda env: (-0.5, 0.5, -0.5, 0.5, DESK_TOP, DESK_TOP - 0.05))
    monkeypatch.setattr(op, "support_disc_rows", lambda env, ids, side: None)

    def ref(ids, frames):
        p = torch.zeros(len(ids), 3)
        p[:, 2] = ref_z
        return {"right_obj_pos": p, "right_obj_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(len(ids), 1)}

    tm = types.SimpleNamespace(get_reference_state_for_envs=ref, clip_ids=torch.zeros(n, dtype=torch.long))
    cfg = types.SimpleNamespace()
    if first_touch is not None:
        setattr(cfg, FL.CFG_KEY, {"right": [first_touch], "left": [-1]})
    obj = types.SimpleNamespace(cfg=types.SimpleNamespace(prim_path="/o"))
    return types.SimpleNamespace(trajectory_manager=tm, cfg=cfg), obj


def _pin(env, obj, frames):
    ids = torch.arange(len(frames))
    return op.pin_pose(env, ids, torch.tensor(frames), "right", obj)


@pytest.mark.parametrize("g", [0.004, 0.012, 0.019])
def test_lift_and_lower_are_symmetric(monkeypatch, g):
    """A planted penetration of g is lifted by g; a planted gap of g is lowered by g: both end on the desk top."""
    env, obj = _env(monkeypatch, DESK_TOP + HZ - g, first_touch=20)
    pos, _q, lift, _ = _pin(env, obj, [0, 5, 19])
    assert torch.allclose(lift, torch.full((3,), g), atol=1e-6)
    assert torch.allclose(pos[:, 2] - HZ, torch.full((3,), DESK_TOP), atol=1e-6)
    env, obj = _env(monkeypatch, DESK_TOP + HZ + g, first_touch=20)
    pos, _q, lift, _ = _pin(env, obj, [0, 5, 19])
    assert float(lift.abs().max()) == 0.0
    assert torch.allclose(pos[:, 2] - HZ, torch.full((3,), DESK_TOP), atol=1e-6)


def test_no_lowering_from_the_first_touch_on(monkeypatch):
    env, obj = _env(monkeypatch, DESK_TOP + HZ + 0.012, first_touch=20)
    pos, *_ = _pin(env, obj, [19, 20, 40])
    assert pos[0, 2] == pytest.approx(DESK_TOP + HZ)
    assert torch.allclose(pos[1:, 2], torch.full((2,), DESK_TOP + HZ + 0.012))


def test_without_a_statement_the_pin_is_unchanged(monkeypatch):
    """No cfg.float_settle (rule off, or a clip with no settle object: -1): the reference pose, as before."""
    for ft in (None, -1):
        env, obj = _env(monkeypatch, DESK_TOP + HZ + 0.012, first_touch=ft)
        pos, *_ = _pin(env, obj, [0, 5, 19])
        assert torch.allclose(pos[:, 2], torch.full((3,), DESK_TOP + HZ + 0.012))


def test_gap_above_the_cap_on_an_eligible_env_raises(monkeypatch):
    """Negative: a gap > FLOAT_SETTLE_MAX_M reaching the pin is refused, not lowered."""
    env, obj = _env(monkeypatch, DESK_TOP + HZ + 0.025, first_touch=20)
    with pytest.raises(RuntimeError, match="float settle: gap 25.0 mm > cap 20 mm"):
        _pin(env, obj, [0])


def test_judge_gate_d_reads_the_lowered_pose(monkeypatch):
    """Judge path: gate D's support gap (object_pin.support_gap → env.object_support_gap) of a settled object is 0."""
    env, obj = _env(monkeypatch, DESK_TOP + HZ + 0.012, first_touch=20)
    gap = op.support_gap(env, torch.arange(2), torch.tensor([0, 30]), "right", obj)
    assert float(gap[0]) == pytest.approx(0.0, abs=1e-6) and float(gap[1]) == pytest.approx(0.012, abs=1e-6)


def test_non_finite_gap_is_not_lowered():
    low = FL.lower_amount(torch.tensor([float("inf"), -0.003, 0.01]), torch.tensor([0, 0, 0]), torch.tensor([5, 5, 5]))
    assert low.tolist() == pytest.approx([0.0, 0.0, 0.01])


def test_clip_statement_sides_and_drop():
    st = {
        "cup": {"verdict": "settle", "side": "right", "first_touch": 14, "gap_m": 0.008},
        "box": {"verdict": "rest", "side": "left", "first_touch": 3, "gap_m": 0.001},
    }
    assert EFS.clip_statement(st) == ({"right": 14, "left": -1}, [0.008])
    shared = {"tray": {"verdict": "settle", "side": "both", "first_touch": 9, "gap_m": 0.006}}
    assert EFS.clip_statement(shared)[0] == {"right": 9, "left": 9}
    with pytest.raises(RuntimeError, match="drop verdict"):
        EFS.clip_statement({"cup": {"verdict": "drop", "side": "right", "first_touch": 4, "gap_m": 0.03}})
    with pytest.raises(RuntimeError, match="re-stamp the tree"):
        EFS.clip_statement({"cup": {"verdict": "support", "side": "right", "first_touch": 4, "gap_m": 0.2}})
    carried = {"cap": {"verdict": "carried", "side": "left", "first_touch": 40, "gap_m": 0.2}}
    assert EFS.clip_statement(carried) == ({"right": -1, "left": -1}, [])


def test_build_reads_every_loader_clip_and_prints_the_marker(tmp_path, monkeypatch, capsys):
    for c in ("a", "b"):
        (tmp_path / c / "0").mkdir(parents=True)
        (tmp_path / c / "task_info.json").write_text(json.dumps({"dataset_name": "hot3d", "objects": []}))
        np.savez(tmp_path / c / "0" / "trajectory_kinematic.npz", qpos=np.zeros((4, 7)))
    said = {"a": {}, "b": {"cup": {"verdict": "settle", "side": "left", "first_touch": 7, "gap_m": 0.011}}}
    monkeypatch.setattr(FS, "statement", lambda tree, ti, q, rows_of: said[str(tree).rsplit("/", 1)[-1]])
    cfg = types.SimpleNamespace()
    EFS.build_float_settle(cfg, types.SimpleNamespace(sequence="a,b", sequence_dir=str(tmp_path)), "test")
    assert getattr(cfg, FL.CFG_KEY) == {"right": [-1, -1], "left": [-1, 7]}
    assert "[FLOAT_SETTLE] test: clips 2 · lowered 1 · max 11.0 mm · t " in capsys.readouterr().out
