"""The f0 cache key (f0_in_v4) hashes only what f0 reads.

Positive and negative per part: a file or URDF element f0 never reads leaves the key as it is; one it reads moves it.
The measurement behind each negative case: on canonical shadows whose f0 output was byte-identical on rerun, v3
keys moved for a stray visual.obj beside surface_points.npz (TACO), a URDF material edit (wuji2) and the
reference time stamp in task_info (the task_info case is test_f0.py)."""

from __future__ import annotations

import json
import types

import pytest

from dexengine.retarget.f0 import shadow_key as SK

G = None  # repair_geom, imported under a hand (repair_cfg builds the hand config at import)


@pytest.fixture(autouse=True)
def _fresh_memo(monkeypatch):
    global G
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")  # any registered hand and arm; the hand tests replace HC
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    import importlib  # noqa: PLC0415

    G = importlib.import_module("dexengine.retarget.f0.repair_geom")
    monkeypatch.setattr(SK, "_GEOM_MEMO", {})


def _obj(tmp_path):
    mesh, sp = tmp_path / "objects" / "o1", tmp_path / "meshes" / "o1"
    mesh.mkdir(parents=True)
    sp.mkdir(parents=True)
    (mesh / "visual.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (sp / "surface_points.npz").write_bytes(b"points-1")
    return mesh, sp


def _okey(mesh, sp, art=None):
    SK._GEOM_MEMO.clear()
    return SK.object_key(mesh, sp, art)[0]


def test_object_key_reads_only_the_opened_files(tmp_path):
    mesh, sp = _obj(tmp_path)
    k0 = _okey(mesh, sp)
    (sp / "visual.obj").write_text("v 0 0 0\nvn 0 0 1\n")  # a stray copy beside the surface points (TACO)
    (mesh / "notes.txt").write_text("x")
    assert _okey(mesh, sp) == k0
    (mesh / "visual.obj").write_text("v 0 0 0\nv 2 0 0\nv 0 1 0\nf 1 2 3\n")
    k1 = _okey(mesh, sp)
    assert k1 != k0
    (sp / "surface_points.npz").write_bytes(b"points-2")
    k2 = _okey(mesh, sp)
    assert k2 != k1
    (sp / "surface_points.npz").unlink()  # optional: the loader falls back to mesh samples, the key records ABSENT
    assert _okey(mesh, sp) != k2
    assert SK.object_key(mesh, sp, None)[1] == {
        G.OBJ_MESH: SK.object_key(mesh, sp, None)[1][G.OBJ_MESH],
        G.OBJ_SP: "ABSENT",
    }


def test_articulated_object_key_is_its_link_parts(tmp_path):
    mesh, sp = _obj(tmp_path)
    for f in ("base.obj", "lid.obj", "lever.obj"):
        (mesh / f).write_text(f)
    art = {
        "parts": {
            "base": "base.obj",
            "moving": "lid.obj",
            "fixed": [{"link": "base", "mesh": "lever.obj", "pose": {"pos": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]}}],
        }
    }
    assert set(SK.object_key(mesh, sp, art)[1]) == {"mesh:base.obj", "mesh:lever.obj", "mesh:lid.obj"}
    k0 = _okey(mesh, sp, art)
    (mesh / "visual.obj").write_text("closed pose, not read")
    (sp / "surface_points.npz").write_bytes(b"closed-pose sidecar, not read")
    assert _okey(mesh, sp, art) == k0
    (mesh / "lever.obj").write_text("lever 2")
    assert _okey(mesh, sp, art) != k0


_URDF = """<robot name="r">
<material name="white"><color rgba="1 1 1 1"/></material>
<link name="torso"><collision><geometry><mesh filename="meshes/torso.stl"/></geometry></collision></link>
<link name="arm_6"><collision><geometry><mesh filename="meshes/arm.stl"/></geometry></collision></link>
<link name="right_palm"><visual><material name="white"/></visual>
  <collision><origin xyz="0 0 0.01" rpy="0 0 0"/><geometry><mesh filename="meshes/palm.stl" scale="1 1 1"/></geometry>
  </collision></link>
<link name="right_f1"><collision><geometry><mesh filename="meshes/f1.stl"/></geometry></collision></link>
<joint name="arm_j6" type="revolute"><parent link="torso"/><child link="arm_6"/><origin xyz="0 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/><limit lower="-1" upper="1"/></joint>
<joint name="right_mount" type="fixed"><parent link="arm_6"/><child link="right_palm"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/></joint>
<joint name="right_j1" type="revolute"><parent link="right_palm"/><child link="right_f1"/>
  <origin xyz="0.02 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="0" upper="1.5"/></joint>
</robot>
"""


@pytest.fixture
def hand(tmp_path, monkeypatch, _fresh_memo):
    d = tmp_path / "robot"
    (d / "meshes").mkdir(parents=True)
    for m in ("torso", "arm", "palm", "f1"):
        (d / "meshes" / f"{m}.stl").write_bytes(m.encode())
    u = d / "r.urdf"
    u.write_text(_URDF)
    hc = types.SimpleNamespace(
        name="fx",
        order=["right_j1"],
        urdf=lambda side: u,
        base_frame=lambda side: "right_palm",
        joint_name=lambda side, f: f,
    )
    rng = {"right_j1": (0.0, 1.5)}
    monkeypatch.setattr(G, "HC", hc)
    monkeypatch.setattr(G, "finger_limits", lambda: ({}, rng))
    from dexengine.placement import arm_ik as AIP

    # the test hand is not registered with the palm-offset owner: it stands without an offset (HandModel.palm_inv)
    monkeypatch.setattr(AIP, "palm_frame_offset", lambda name: (0.0, 0.0, 0.0))
    return types.SimpleNamespace(urdf=u, dir=d, rng=rng)


def _hkey():
    SK._GEOM_MEMO.clear()
    return SK.hand_key("right")[0]


def _edit(u, old, new):
    t = u.read_text()
    assert old in t
    u.write_text(t.replace(old, new, 1))


def test_hand_key_ignores_what_handmodel_does_not_read(hand):
    k0 = _hkey()
    assert set(SK.hand_key("right")[1]) == {"meshes/palm.stl", "meshes/f1.stl"}
    _edit(hand.urdf, '<color rgba="1 1 1 1"/>', '<color rgba="0.2 0.2 0.2 1"/>')  # a material (wuji2)
    (hand.dir / "meshes" / "arm.stl").write_bytes(b"arm 2")  # an arm mesh
    _edit(hand.urdf, '<origin xyz="0 0 0.3" rpy="0 0 0"/>', '<origin xyz="0 0 0.31" rpy="0 0 0"/>')  # above the base
    assert _hkey() == k0


@pytest.mark.parametrize(
    "change",
    [
        ("mesh", "meshes/f1.stl"),
        ("urdf", ('<origin xyz="0.02 0 0.05" rpy="0 0 0"/>', '<origin xyz="0.02 0 0.06" rpy="0 0 0"/>')),
        ("urdf", ('<axis xyz="0 1 0"/>', '<axis xyz="1 0 0"/>')),
        ("urdf", ('<origin xyz="0 0 0.01" rpy="0 0 0"/>', '<origin xyz="0 0 0.02" rpy="0 0 0"/>')),
        ("urdf", ('scale="1 1 1"', 'scale="1 1 1.1"')),
        ("range", None),
    ],
)
def test_hand_key_moves_with_what_handmodel_reads(hand, change):
    k0 = _hkey()
    kind, arg = change
    if kind == "mesh":
        (hand.dir / arg).write_bytes(b"f1 2")
    elif kind == "urdf":
        _edit(hand.urdf, *arg)
    else:
        hand.rng["right_j1"] = (0.0, 1.4)
    assert _hkey() != k0


def test_handmodel_reads_the_blocks_the_key_hashes(hand):
    """The loader and the key share one parser: the blocks HandModel loads are the key's collision lines."""
    blocks = G.hand_collision_blocks(hand.urdf.read_text(), "right")
    assert blocks == [
        ("right_palm", [("meshes/palm.stl", "0 0 0.01", "0 0 0", "1 1 1")]),
        ("right_f1", [("meshes/f1.stl", "", "", "")]),
    ]
    lines, _files = G.hand_input_lines("right")
    assert [ln for ln in lines if ln.startswith("joint|")] == [
        "joint|right_j1|revolute|right_palm|right_f1|0.02 0 0.05|0 0 0|0 1 0"
    ]


def test_stale_reason_names_the_moved_file():
    old = {"obj/right": "a", "files": {"obj/right": {G.OBJ_MESH: "1", G.OBJ_SP: "2"}}}
    new = {"obj/right": "b", "files": {"obj/right": {G.OBJ_MESH: "1", G.OBJ_SP: "3"}}}
    assert SK.input_key_diff(old, new) == f"obj/right[{G.OBJ_SP}]"
    assert SK.input_key_diff({"obj/right": "a"}, {"obj/right": "b"}) == "obj/right"  # a marker without file records
    assert json.dumps(SK.FP_DESIGN) == '"f0_in_v4"'
