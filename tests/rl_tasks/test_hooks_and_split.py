"""Experiment hooks (no plugin = the arm body) and the two-articulation bimanual split rules."""

from __future__ import annotations

import sys

import pytest

from dexengine.rl.tasks import hooks
from dexengine.rl.tasks.scene import split_artic as sa


@pytest.fixture
def clean_hooks(monkeypatch):
    monkeypatch.setattr(hooks, "_REGISTRY", {})
    monkeypatch.setattr(hooks, "_LOADED", [])
    monkeypatch.delenv("DEXENGINE_FH_NAN_RESET", raising=False)
    monkeypatch.delenv("DEXENGINE_FH_PALM_POINT", raising=False)
    monkeypatch.delenv(hooks.PLUGINS_ENV, raising=False)
    yield


def test_no_plugin_means_arm_body(clean_hooks, monkeypatch):
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    assert hooks.floating_hand() is None
    assert hooks.floating_active() is False
    assert hooks.floating_nan_reset() is False
    assert hooks.floating_palm_point(None, "right") is None


def test_floating_arm_or_flag_without_plugin_fails(clean_hooks, monkeypatch):
    from dexengine.core.arms import FLOATING_ARM

    monkeypatch.setenv("DEXENGINE_ARM", FLOATING_ARM)
    with pytest.raises(RuntimeError, match="floating-hand plugin"):
        hooks.floating_hand()
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    monkeypatch.setenv("DEXENGINE_FH_NAN_RESET", "1")
    with pytest.raises(RuntimeError, match="floating-hand body only"):
        hooks.floating_nan_reset()
    monkeypatch.setenv("DEXENGINE_FH_NAN_RESET", "0")
    monkeypatch.setenv("DEXENGINE_FH_PALM_POINT", "arm_ee")
    with pytest.raises(RuntimeError, match="floating-hand body only"):
        hooks.floating_palm_point(None, "left")


def test_plugin_registers_on_import(clean_hooks, monkeypatch, tmp_path):
    (tmp_path / "fh_plugin_for_test.py").write_text(
        "from dexengine.rl.tasks import hooks\n"
        "class _FH:\n"
        "    @staticmethod\n"
        "    def active():\n        return True\n"
        "    @staticmethod\n"
        "    def nan_reset():\n        return False\n"
        "hooks.register(hooks.FLOATING_HAND, _FH)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv(hooks.PLUGINS_ENV, "fh_plugin_for_test")
    assert hooks.load_plugins() == ["fh_plugin_for_test"]
    assert hooks.floating_active() is True and hooks.floating_nan_reset() is False
    with pytest.raises(RuntimeError, match="already registered"):
        hooks.register(hooks.FLOATING_HAND, object())
    sys.modules.pop("fh_plugin_for_test", None)


def test_split_gate(monkeypatch):
    monkeypatch.delenv(sa.BI_ENV, raising=False)
    assert sa.bi_enabled() is False and sa.enabled() is False
    assert sa.left_robot_ns() == "{ENV_REGEX_NS}/Robot"
    monkeypatch.setenv(sa.BI_ENV, "1")
    assert sa.bi_enabled() and sa.enabled()
    assert sa.left_robot_ns() == "{ENV_REGEX_NS}/" + sa.LEFT_PRIM_LEAF


class _Act:
    def __init__(self, exprs, **kw):
        self.joint_names_expr = list(exprs)
        for k in ("stiffness", "damping", "armature", "effort_limit_sim", "velocity_limit_sim"):
            setattr(self, k, kw.get(k))


class _ArtCfg:
    def __init__(self, actuators, joint_pos):
        self.actuators = dict(actuators)
        self.init_state = type("S", (), {"joint_pos": dict(joint_pos), "joint_vel": {".*": 0.0}})()


def _spec():
    from dexengine.core.arms import ARM_REGISTRY
    from dexengine.core.hands import HAND_REGISTRY, mirror_right_to_left

    return ARM_REGISTRY["rby1"], HAND_REGISTRY["sharpa_wave"], mirror_right_to_left


def test_side_of_spec_names():
    arm, hand, to_left = _spec()
    cases = [(arm.joint_expr(s), s) for s in ("right", "left")]
    cases += [(arm.any_joint_expr(s), s) for s in ("right", "left")]
    cases += [(arm.ee_body_name(s), s) for s in ("right", "left")]
    cases += [(n, "right") for n in hand.finger_joint_names_right]
    cases += [(to_left(n), "left") for n in hand.finger_joint_names_right]
    cases += [(h, None) for h in arm.head_joint_names] + [("(?!(?:a|b)$).*", None)]
    assert [(n, sa.side_of(n)) for n, w in cases if sa.side_of(n) != w] == []
    h0 = arm.head_joint_names[0]
    assert sa.keep_for("right", h0) and not sa.keep_for("left", h0)
    assert sa.keep_for("right", "(?!(?:a)$).*") and sa.keep_for("left", "(?!(?:a)$).*")


@pytest.mark.parametrize("view", ["right", "left"])
def test_restrict_to_side_leaves_no_other_side_keys(view):
    arm, hand, to_left = _spec()
    fingers = list(hand.finger_joint_names_right) + [to_left(n) for n in hand.finger_joint_names_right]
    kp = dict.fromkeys(fingers, 1.0)
    cfg = _ArtCfg(
        {
            f"{arm.name}_arm": _Act(
                arm.joint_names_expr(),
                effort_limit_sim=arm.limit_map("effort"),
                velocity_limit_sim=arm.limit_map("velocity"),
            ),
            f"{arm.name}_head": _Act(list(arm.head_joint_names)),
            f"{hand.name}_fingers": _Act(fingers, stiffness=kp, damping=kp, armature=0.01),
        },
        {"(?!(?:x)$).*": 0.0, arm.head_joint_names[-1]: 0.52, fingers[0]: 0.1, to_left(fingers[0]): 0.1},
    )
    dropped = sa.restrict_actuators_to_side(cfg, view)
    sa.restrict_init_state_to_side(cfg, view)
    other = "left" if view == "right" else "right"
    leaked = [e for g in cfg.actuators.values() for e in g.joint_names_expr if sa.side_of(e) == other]
    for g in cfg.actuators.values():
        for fld in ("stiffness", "damping", "effort_limit_sim", "velocity_limit_sim"):
            v = getattr(g, fld)
            leaked += [k for k in (v if isinstance(v, dict) else {}) if sa.side_of(k) == other]
    leaked += [k for k in cfg.init_state.joint_pos if sa.side_of(k) == other]
    assert leaked == []
    assert all(g.joint_names_expr for g in cfg.actuators.values())
    assert dropped == ([f"{arm.name}_head"] if view == "left" else [])
