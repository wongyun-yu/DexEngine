"""Arm gain overrides (env_setup.robot_gains): the per-joint keys come from the active arm spec.

A key is one arm joint index on both sides; Isaac Lab refuses to boot when a gain regex matches no joint, and a key
matching another index would silently apply the wrong gain, so each key must match exactly that index on every side.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from dexengine.core.arms import ARM_REGISTRY
from dexengine.rl.env_setup import robot_gains
from dexengine.rl.tasks import hooks


def _cfg():
    act = SimpleNamespace(stiffness=None, damping=None)
    return SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(actuators={"rby1_arm": act}))), act


def _args(**kw):
    base = {"arm_kp": None, "arm_kd": None, "arm_kp_per_joint": None, "arm_kd_per_joint": None, "arm_gains_given": True}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def rby1(monkeypatch):
    monkeypatch.setattr(hooks, "_REGISTRY", {})
    monkeypatch.setattr(hooks, "_LOADED", [])
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")
    monkeypatch.delenv("DEXENGINE_LEFT_ARM_KP_SCALE", raising=False)
    return ARM_REGISTRY["rby1"]


def test_per_joint_keys_match_one_index_on_every_side(rby1):
    cfg, act = _cfg()
    n = rby1.n_joints
    kp = ",".join(str(10.0 + i) for i in range(n))
    kd = ",".join(str(1.0 + i) for i in range(n))
    robot_gains.apply_arm_gain_overrides(cfg, _args(arm_kp_per_joint=kp, arm_kd_per_joint=kd), "test")
    names = {s: rby1.joint_names(s) for s in rby1.sides}
    for gains, base in ((act.stiffness, 10.0), (act.damping, 1.0)):
        assert isinstance(gains, dict) and len(gains) == n
        for i, (key, val) in enumerate(gains.items()):
            assert val == base + i
            for s in rby1.sides:
                assert [x for x in names[s] if re.fullmatch(key, x)] == [names[s][i]], (key, s)


def test_per_joint_wrong_count_fails(rby1):
    cfg, _ = _cfg()
    with pytest.raises(AssertionError, match="needs"):
        robot_gains.apply_arm_gain_overrides(cfg, _args(arm_kp_per_joint="1,2"), "test")


def test_scalar_gains_and_no_gains(rby1):
    cfg, act = _cfg()
    robot_gains.apply_arm_gain_overrides(cfg, _args(arm_kp=200.0, arm_kd=30.0), "test")
    assert (act.stiffness, act.damping) == (200.0, 30.0)
    cfg, act = _cfg()
    robot_gains.apply_arm_gain_overrides(cfg, _args(arm_gains_given=False), "test")
    assert (act.stiffness, act.damping) == (None, None)
