"""repair_root.scene_model compiles a tree's scene once per process for the same bytes: root_axes and turn_rows of
every written side read that one model; a scene whose bytes change is compiled again."""

from __future__ import annotations

import numpy as np

_XML = """<mujoco><worldbody><body name="b" pos="0 0 {z}"><joint name="x" type="slide" axis="1 0 0"/>
<joint name="y" type="slide" axis="0 1 0"/><joint name="z" type="slide" axis="0 0 1"/>
<joint name="rx" type="hinge" axis="1 0 0"/><joint name="ry" type="hinge" axis="0 1 0"/>
<joint name="rz" type="hinge" axis="0 0 1"/><geom type="sphere" size="0.01"/></body></worldbody></mujoco>"""


def test_same_bytes_one_model_new_bytes_new_model(tmp_path, monkeypatch):
    from dexengine.retarget.f0 import repair_root as RT

    monkeypatch.setattr(RT, "_SCENE", {})
    (tmp_path / "scene.xml").write_text(_XML.format(z=0.1))
    m1 = RT.scene_model(tmp_path)
    assert RT.scene_model(tmp_path) is m1
    (tmp_path / "scene.xml").write_text(_XML.format(z=0.2))
    m2 = RT.scene_model(tmp_path)
    assert m2 is not m1
    assert np.isclose(m2.body_pos[1][2], 0.2)


def test_root_axes_read_the_cached_model(tmp_path, monkeypatch):
    import mujoco

    from dexengine.retarget.f0 import repair_root as RT

    monkeypatch.setattr(RT, "_SCENE", {})
    (tmp_path / "scene.xml").write_text(_XML.format(z=0.1))
    calls = []
    real = mujoco.MjModel.from_xml_path
    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", lambda p: calls.append(p) or real(p))
    A1, A2 = RT.root_axes(tmp_path, 0), RT.root_axes(tmp_path, 0)
    assert len(calls) == 1
    assert np.array_equal(A1, A2)
