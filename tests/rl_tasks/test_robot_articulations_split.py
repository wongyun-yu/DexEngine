"""Runtime and cfg writes of robot physics reach every robot articulation of the scene (split_artic.ROBOT_KEYS).

With the split articulation (DEXENGINE_BI_SPLIT_ARTIC=1) the left arm and hand are a second articulation,
``robot_left``. A write to ``robot`` alone leaves the left side at its USD value: hand friction 0.5 left · 2.0 right;
the mount height offset and the gravity-off audit on one fixed base only. Hand friction reads every shape back and
prints the shapes written against the shapes the scene holds; fewer dies."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dexengine.rl.env_setup import robot_gains
from dexengine.rl.tasks.scene import split_artic

N_ENV = 3


class View:
    """The PhysX articulation view's material calls (n_env, n_shapes, 3: static, dynamic, restitution)."""

    def __init__(self, n_shapes: int, mu: float, sticky: bool = True):
        self.max_shapes = n_shapes
        self.m = torch.full((N_ENV, n_shapes, 3), mu)
        self.m[:, :, 2] = 0.0
        self.sticky = sticky

    def get_material_properties(self):
        return self.m.clone()

    def set_material_properties(self, m, ids):
        if self.sticky:
            self.m[ids] = m[ids]


class Scene(dict):
    def keys(self):  # InteractiveScene.keys
        return list(super().keys())


def _env(**arts):
    return SimpleNamespace(
        scene=Scene({k: SimpleNamespace(root_physx_view=v) for k, v in arts.items()}), num_envs=N_ENV
    )


@pytest.fixture
def split(monkeypatch):
    monkeypatch.setenv(split_artic.BI_ENV, "1")


def test_split_scene_both_articulations_get_the_friction(split, capsys):
    right, left = View(5, 0.5), View(4, 0.5)
    robot_gains.apply_hand_friction(_env(robot=right, robot_left=left), 2.0)
    for v in (right, left):
        assert torch.all(v.m[:, :, :2] == 2.0) and torch.all(v.m[:, :, 2] == 0.0)
    out = capsys.readouterr().out
    assert "on 9 of 9 shapes (robot 5/5 · robot_left 4/4)" in out


def test_one_articulation_scene_writes_robot(monkeypatch, capsys):
    monkeypatch.setenv(split_artic.BI_ENV, "0")
    right = View(5, 0.5)
    robot_gains.apply_hand_friction(_env(robot=right), 2.0)
    assert torch.all(right.m[:, :, :2] == 2.0)
    assert "on 5 of 5 shapes (robot 5/5)" in capsys.readouterr().out


def test_a_left_articulation_the_flag_does_not_declare_dies(monkeypatch):
    """Negative: robot_left in the scene with the split off is a contradiction, not a scene to write half of."""
    monkeypatch.setenv(split_artic.BI_ENV, "0")
    with pytest.raises(RuntimeError, match="holds 'robot_left'"):
        robot_gains.apply_hand_friction(_env(robot=View(5, 0.5), robot_left=View(4, 0.5)), 2.0)


def test_the_old_robot_only_write_dies(split, monkeypatch):
    """Negative: the pre-fix write list (robot only) leaves robot_left at 0.5 — the shapes there are are counted
    from the scene, not from the write list, so the half-write dies instead of printing n of n."""
    right, left = View(5, 0.5), View(4, 0.5)
    env = _env(robot=right, robot_left=left)
    monkeypatch.setattr(split_artic, "robot_articulations", lambda e: [("robot", e.scene["robot"])])
    with pytest.raises(RuntimeError, match="5 of 9 robot shapes"):
        robot_gains.apply_hand_friction(env, 2.0)
    assert torch.all(left.m[:, :, :2] == 0.5)


def test_a_write_that_does_not_stick_dies(split):
    """Negative: a shape that reads back another friction is not counted as written — the run dies."""
    with pytest.raises(RuntimeError, match="5 of 9 robot shapes"):
        robot_gains.apply_hand_friction(_env(robot=View(5, 0.5), robot_left=View(4, 0.5, sticky=False)), 2.0)


def test_robot_articulations_is_the_split_owner(split):
    env = _env(robot=View(1, 0.5), robot_left=View(1, 0.5))
    assert [n for n, _ in split_artic.robot_articulations(env)] == ["robot", "robot_left"]


def test_no_friction_is_a_no_op(split):
    right, left = View(2, 0.5), View(2, 0.5)
    robot_gains.apply_hand_friction(_env(robot=right, robot_left=left), None)
    assert torch.all(right.m[:, :, :2] == 0.5) and torch.all(left.m[:, :, :2] == 0.5)


def test_a_friction_float32_cannot_hold_exactly_still_counts(split, capsys):
    """The view stores float32: 0.7 reads back as float32(0.7) and must still count as written."""
    right, left = View(3, 0.5), View(2, 0.5)
    robot_gains.apply_hand_friction(_env(robot=right, robot_left=left), 0.7)
    assert "on 5 of 5 shapes" in capsys.readouterr().out


def _zcfg(**arts):
    return SimpleNamespace(
        scene=SimpleNamespace(**{k: SimpleNamespace(init_state=SimpleNamespace(pos=p)) for k, p in arts.items()})
    )


def test_robot_z_offset_moves_every_robot_articulation(monkeypatch):
    """With the split, robot_left is a second fixed base: the mount-height offset moves both, or the two sides stand
    at different heights (negative: the pre-fix write moved scene.robot only)."""
    monkeypatch.setenv("DEXENGINE_ROBOT_Z_OFF", "0.05")
    cfg = _zcfg(robot=(0.0, 0.0, 0.1), robot_left=(0.0, 0.0, 0.1))
    robot_gains.apply_robot_z_off(cfg, "test")
    assert cfg.scene.robot.init_state.pos == pytest.approx((0.0, 0.0, 0.15))
    assert cfg.scene.robot_left.init_state.pos == pytest.approx((0.0, 0.0, 0.15))


def test_robot_z_offset_one_articulation_scene(monkeypatch):
    monkeypatch.setenv("DEXENGINE_ROBOT_Z_OFF", "-0.02")
    cfg = _zcfg(robot=(0.1, 0.2, 0.3))
    robot_gains.apply_robot_z_off(cfg, "test")
    assert cfg.scene.robot.init_state.pos == pytest.approx((0.1, 0.2, 0.28))


def _gcfg(*keys):
    return SimpleNamespace(
        scene=SimpleNamespace(
            **{
                k: SimpleNamespace(spawn=SimpleNamespace(rigid_props=SimpleNamespace(disable_gravity=False)))
                for k in keys
            }
        )
    )


def test_gravity_off_audit_reaches_every_robot_articulation():
    """Negative: the pre-fix write turned gravity off on scene.robot only."""
    cfg = _gcfg("robot", "robot_left")
    assert robot_gains.disable_robot_gravity(cfg, "test") == ["robot", "robot_left"]
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity and cfg.scene.robot_left.spawn.rigid_props.disable_gravity


def test_gravity_off_audit_one_articulation_scene():
    cfg = _gcfg("robot")
    assert robot_gains.disable_robot_gravity(cfg, "test") == ["robot"]
