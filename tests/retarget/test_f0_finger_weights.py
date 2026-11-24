"""f0 repair weights: a joint's finger comes from the URDF tree (repair_cfg.finger_of_joints), not from its name.

A synthetic hand whose joint names say nothing about the finger: the rule must still find the thumb and index joints
through the fingertip each joint reaches, and die on a joint that reaches no fingertip or is not in the URDF."""

from __future__ import annotations

import pytest

from tests.imports import own_module

URDF = """<?xml version="1.0"?>
<robot name="h">
  <link name="palm"/>
  <link name="l1a"/><link name="tip_t"/>
  <link name="l2a"/><link name="l2b"/><link name="tip_i"/>
  <link name="l3a"/><link name="tip_m"/>
  <link name="side"/>
  <joint name="j_1_1" type="revolute"><parent link="palm"/><child link="l1a"/></joint>
  <joint name="j_1_2" type="revolute"><parent link="l1a"/><child link="tip_t"/></joint>
  <joint name="j_2_1" type="revolute"><parent link="palm"/><child link="l2a"/></joint>
  <joint name="j_2_2" type="revolute"><parent link="l2a"/><child link="l2b"/></joint>
  <joint name="j_2_3" type="fixed"><parent link="l2b"/><child link="tip_i"/></joint>
  <joint name="j_3_1" type="revolute"><parent link="palm"/><child link="l3a"/></joint>
  <joint name="j_3_2" type="revolute"><parent link="l3a"/><child link="tip_m"/></joint>
  <joint name="j_x" type="revolute"><parent link="palm"/><child link="side"/></joint>
</robot>
"""
TIPS = ("tip_t", "tip_i", "tip_m")


@pytest.fixture
def finger_of_joints(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")  # repair_cfg resolves the active hand and arm at import
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    return own_module("dexengine.retarget.f0.repair_cfg").finger_of_joints


@pytest.fixture
def urdf(tmp_path):
    p = tmp_path / "h.urdf"
    p.write_text(URDF)
    return p


def test_fingers_follow_the_tree_not_the_names(urdf, finger_of_joints):
    got = finger_of_joints(urdf, ["j_1_1", "j_1_2", "j_2_1", "j_2_2", "j_3_1", "j_3_2"], TIPS)
    assert got == {"j_1_1": 0, "j_1_2": 0, "j_2_1": 1, "j_2_2": 1, "j_3_1": 2, "j_3_2": 2}


def test_a_joint_reaching_no_fingertip_dies(urdf, finger_of_joints):
    with pytest.raises(SystemExit, match="reach no fingertip"):
        finger_of_joints(urdf, ["j_1_1", "j_x"], TIPS)


def test_a_joint_reaching_two_fingertips_dies(urdf, finger_of_joints):
    with pytest.raises(SystemExit, match="reach no fingertip or several"):
        finger_of_joints(urdf, ["j_1_1"], ("tip_t", "l1a"))


def test_a_joint_missing_from_the_urdf_dies(urdf, finger_of_joints):
    with pytest.raises(SystemExit, match="not in"):
        finger_of_joints(urdf, ["right_thumb_cmc"], TIPS)
