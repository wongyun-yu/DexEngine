"""Floating hand: the virtual-joint rules and the anchor rule minmid (pure numpy, no isaaclab, no torch)."""

import math
import pathlib

import numpy as np
import pytest

from experiments.ablations.floating_hand import body as F


def test_virtual_joint_and_palm_point_rules(monkeypatch):
    assert F.vj_joint_names("left") == [f"left_fh_wrist_{a}" for a in ("px", "py", "pz", "ry", "rx", "rz")]
    assert F.vj_link_names("right")[0] == "right_fh_wrist_anchor" and len(F.vj_link_names("right")) == 6
    arm = F.vj_armature(("right",))
    g = F.WRIST_GAINS
    assert [arm[n] for n in F.vj_joint_names("right")] == [0.0] * 3 + [g.rot_kd**2 / (16 * g.rot_kp)] * 3
    assert F.vj_armature_mark() == "[FH_WRIST] armature rot=0.375 pris=0 rule=kd2/16kp"
    assert F.vj_canon_path("/a/floating_x/floating_x.urdf") == pathlib.Path("/a/floating_x_vj/floating_x_vj.urdf")
    assert F.center_branch([5.0, 6.0]) == -1 and F.center_branch([0.1, -0.2]) == 0
    monkeypatch.setenv("DEXENGINE_ARM", "floating")
    monkeypatch.delenv(F.PALM_POINT_ENV, raising=False)
    assert F.palm_point_mode() == ""
    monkeypatch.setenv(F.PALM_POINT_ENV, "arm_ee")
    assert F.palm_point_mode() == "arm_ee"
    monkeypatch.setenv(F.PALM_POINT_ENV, "palm")
    with pytest.raises(ValueError):
        F.palm_point_mode()
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    monkeypatch.setenv(F.PALM_POINT_ENV, "arm_ee")
    with pytest.raises(ValueError):  # no consumer on the arm body
        F.palm_point_mode()


def test_anchor_minmid_on_a_yaw_only_path():
    # the hand only yaws about the world z axis: its z axis stays vertical, so the best anchor y is horizontal and the
    # middle angle after the rule is 0; the start-wrist anchor has the same y here, so both are 0
    t = np.linspace(0.0, 2.0, 50)
    q = np.stack([np.cos(t / 2), 0 * t, 0 * t, np.sin(t / 2)], -1)
    p = np.stack([0.1 * t, 0 * t, 0 * t], -1)
    st = F.anchor_minmid(q, p)
    assert abs(st["y"][2]) < 1e-3 and st["mid_after"] < 1e-3 and st["mid_before"] < 1e-9
    assert math.isclose(float(np.linalg.norm(st["extent"])), 0.2, rel_tol=1e-6)  # a straight 0.2 m path
    assert math.isclose(abs(float(np.linalg.norm(st["q"]))), 1.0, rel_tol=1e-9)
    F.anchor_check(st, "clip", "right")  # within every gate
