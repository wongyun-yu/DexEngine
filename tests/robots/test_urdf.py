"""URDF helpers of the merged-robot generator: the visual material rule, the posture override and the pad
measurement."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from dexengine.robots.urdf import materials as UM
from dexengine.robots.urdf import merge as MF

V = '<visual><geometry><box size="1 1 1"/></geometry>{}</visual>'
BLK = '<color rgba="0.02 0.02 0.02 1"/>'


def robot(*materials, top=""):
    body = "".join(f'<link name="l{i}">{V.format(m)}</link>' for i, m in enumerate(materials))
    return ET.fromstring(f'<robot name="t">{top}{body}</robot>')


def test_named_inline_colour_is_hoisted_once():
    r = robot(f'<material name="a">{BLK}</material>', f'<material name="a">{BLK}</material>')
    assert UM.hoist_named_materials(r) == 1
    assert len(r.findall("material")) == 1
    assert all(v.find("material/color") is None for v in r.iter("visual"))
    assert UM.visual_colors(r)["l0"] == [(0.02, 0.02, 0.02, 1.0)]
    assert UM.hoist_named_materials(r) == 0  # idempotent


def test_unnamed_or_colourless_materials_are_left_alone():
    r = robot(f'<material name="">{BLK}</material>')
    assert UM.hoist_named_materials(r) == 0 and r.find("link/visual/material/color") is not None
    r = robot('<material name="y"/>')
    assert UM.hoist_named_materials(r) == 0 and UM.visual_colors(r)["l0"] == [None]
    r = robot(f'<material name="x">{BLK}</material>', top=f'<material name="x">{BLK}</material>')
    assert UM.hoist_named_materials(r) == 0 and r.find("link/visual/material/color") is None


@pytest.mark.parametrize(
    "top,second",
    [
        ("", '<material name="x"><color rgba="1 1 1 1"/></material>'),
        ('<material name="x"><color rgba="1 1 1 1"/></material>', ""),
    ],
)
def test_conflicting_colours_fail(top, second):
    mats = [f'<material name="x">{BLK}</material>'] + ([second] if second else [])
    with pytest.raises(SystemExit):
        UM.hoist_named_materials(robot(*mats, top=top))


def test_posture_override_changes_only_listed_joints():
    from dexengine.core.arms import ARM_REGISTRY

    before = dict(ARM_REGISTRY["rby1"].skeleton.posture)
    got = MF.apply_posture_override(before, "torso_1=0.52, torso_2=-1.05")
    assert got["torso_1"] == 0.52 and got["torso_2"] == -1.05
    assert {k: v for k, v in got.items() if k not in ("torso_1", "torso_2")} == {
        k: v for k, v in before.items() if k not in ("torso_1", "torso_2")
    }
    assert ARM_REGISTRY["rby1"].skeleton.posture == before  # the spec's posture is not touched
    with pytest.raises(SystemExit):
        MF.apply_posture_override(before, "torso_9=1")


def test_pad_gap_on_a_box_tip():
    """A site at the centre of the far-end face of a box collider is on the pad: gap 0, lateral 0."""
    from dexengine.robots.rig import sites

    g = np.stack(np.meshgrid(*[np.linspace(-0.005, 0.005, 5)] * 2, np.linspace(0.0, 0.02, 9)), -1).reshape(-1, 3)
    gap, lat = sites.pad_gap(g, np.array([0.0, 0.0, 0.02]))
    assert gap == pytest.approx(0.0, abs=1e-12) and lat == pytest.approx(0.0, abs=1e-12)
    gap, _ = sites.pad_gap(g, np.array([0.0, 0.0, 0.015]))
    assert gap == pytest.approx(0.005)
