"""The render body's head camera swap reads the camera visual — from the URDF the body was baked from and from the
stage rule — never from a name list: a renamed camera mesh (camera.stl → head_camera.stl) renames the visual and the
swap follows; a stage that disagrees with its URDF, or a render-only prim nobody declared, dies."""

from __future__ import annotations

import subprocess

import pytest

from dexengine.core.repo import FULL_ROBOT_URDF_DIR, REPO_ROOT
from dexengine.robots.usd import head_camera as HC

URDF = """<robot name="r">
  <link name="link_head_2"/>
  <link name="camera">
    <visual><geometry><mesh filename="../meshes/{stem}.stl"/></geometry></visual>
    <collision><geometry><mesh filename="../meshes/{stem}.stl"/></geometry></collision>
  </link>
  <joint name="{joint}" type="fixed"><parent link="{parent}"/><child link="camera"/></joint>
</robot>"""


def _urdf(tmp_path, stem="head_camera", parent="link_head_2", joint=None):
    joint = joint or HC.camera_link("rby1").joint
    p = tmp_path / "r.urdf"
    p.write_text(URDF.format(stem=stem, parent=parent, joint=joint), encoding="utf-8")
    return p


@pytest.mark.parametrize("stem", ["camera", "head_camera", "zedm"])
def test_stated_follows_the_mesh_file(tmp_path, stem):
    """The stated visual is the camera link's mesh file stem, whatever it is called."""
    assert HC.stated_camera_visuals(_urdf(tmp_path, stem), "rby1") == [stem]


@pytest.mark.parametrize("kw", [{"parent": "link_head_1"}, {"joint": "no_such_joint"}])
def test_stated_dies_without_the_skeleton_joint(tmp_path, kw):
    with pytest.raises(SystemExit):
        HC.stated_camera_visuals(_urdf(tmp_path, **kw), "rby1")


def test_every_tracked_merged_urdf_states_one_camera_visual():
    """Each merged URDF of the repo (every hand) states exactly one camera visual for the skeleton's head camera."""
    urdfs = sorted(FULL_ROBOT_URDF_DIR.glob("rby1_*/rby1_*.urdf"))
    assert urdfs
    for u in urdfs:
        assert len(HC.stated_camera_visuals(u, "rby1")) == 1, u


def test_body_urdf_reads_the_bake_record(tmp_path):
    """The URDF is read only while the bake record holds: the USD changed after the bake (record stale) dies, the URDF
    changed after the bake dies, and undoing either change reads it again; no record or no asset_path dies (re-bake)."""
    import hashlib
    import json

    from dexengine.robots.usd import build as B

    u = _urdf(tmp_path)
    usd = tmp_path / "b" / "b.usd"
    usd.parent.mkdir()
    usd.write_text("baked", encoding="utf-8")
    (usd.parent / "config.yaml").write_text(f"asset_path: {u}\nfix_base: true\n", encoding="utf-8")
    md5 = hashlib.md5(json.dumps({"fix_base": True}).encode())
    md5.update(u.read_bytes())
    (usd.parent / ".asset_hash").write_text(md5.hexdigest(), encoding="utf-8")
    with pytest.raises(SystemExit, match="no record"):
        HC.body_urdf(usd)
    B.write_bake_record(usd.parent, u, usd)
    assert HC.body_urdf(usd) == u
    usd.write_text("baked, then the USD alone re-written", encoding="utf-8")
    with pytest.raises(SystemExit, match="record stale"):
        HC.body_urdf(usd)
    usd.write_text("baked", encoding="utf-8")
    assert HC.body_urdf(usd) == u
    text = u.read_text(encoding="utf-8")
    u.write_text(text.replace("head_camera", "camera"), encoding="utf-8")
    with pytest.raises(SystemExit, match="other URDF bytes"):
        HC.body_urdf(usd)
    u.write_text(text, encoding="utf-8")
    assert HC.body_urdf(usd) == u
    other = usd.with_name("b_mi.usd")
    other.write_text("baked", encoding="utf-8")  # the same bytes, another file in the bound folder
    with pytest.raises(SystemExit, match="not the baked USD"):
        HC.body_urdf(other)
    (usd.parent / "config.yaml").write_text("fix_base: true\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown or missing"):
        HC.body_urdf(usd)


def test_no_name_list_left():
    """The swap and the parity check name no camera visual: the old list and its readers are gone."""
    out = subprocess.run(
        ["git", "grep", "-n", "-E", "RENDER_SWAP_VISUALS|CAMERA_PARTS", "--", "dexengine"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert out.stdout == "", out.stdout


# ── stage rule (pxr) ─────────────────────────────────────────────────────────────────────────────────────────────────
def _stage(tmp_path, cam: str, extra_vis=("head",)):
    """A minimal instanced head: /robot/link_head_2/{visuals,collisions} referencing one prototype each; the camera
    visual has a same-named collider (the importer's merged camera link)."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    st = Usd.Stage.CreateNew(str(tmp_path / "s.usda"))
    st.SetDefaultPrim(st.DefinePrim("/robot", "Xform"))
    for proto, kids in (("/Flattened_Prototype_1", (cam, *extra_vis)), ("/Flattened_Prototype_2", (cam,))):
        st.OverridePrim(proto)
        for k in kids:
            st.DefinePrim(f"{proto}/{k}", "Xform")
    for sub, proto in (("visuals", "/Flattened_Prototype_1"), ("collisions", "/Flattened_Prototype_2")):
        p = st.DefinePrim(f"/robot/link_head_2/{sub}", "Xform")
        # explicit items: what render_parts.single_user_prototype reads
        p.GetReferences().SetReferences([Sdf.Reference(primPath=Sdf.Path(proto))])
    return st


@pytest.mark.parametrize("cam", ["camera", "head_camera"])
def test_stage_rule_finds_the_camera_by_its_collider(tmp_path, cam):
    assert HC.camera_visual_names(_stage(tmp_path, cam)) == [cam]


def test_swap_dies_when_stage_and_urdf_disagree(tmp_path):
    st = _stage(tmp_path, "camera")
    with pytest.raises(SystemExit):
        HC.swap(st, ["head_camera"])


def test_swap_follows_a_renamed_camera(tmp_path):
    pytest.importorskip("trimesh")
    if not (REPO_ROOT / HC.ZEDM_MESH).is_file():
        pytest.skip("zedm mesh not in this checkout")
    st = _stage(tmp_path, "head_camera")
    line = HC.swap(st, ["head_camera"])
    assert "removed camera visual 'head_camera'" in line
    assert HC.problems(st) == []


def test_parity_allows_only_the_swap_roots(tmp_path):
    """camera_roots: the training plate's camera visual may be missing on the render plate and the ZED Mini root may
    be added there; any other one-plate visual root is not a swap prim (it fails parity as a changed prim)."""
    from dexengine.robots.usd import render_plate as RP

    st = _stage(tmp_path, "head_camera")
    cam_a, cam_b = RP.camera_roots(st, "rby1")
    assert cam_a == {"link_head_2": ("head_camera",)} and cam_b == {"link_head_2": (HC.ZEDM_ROOT,)}
    for path, roots, ok in (
        ("/robot/link_head_2/visuals/head_camera", cam_a, True),
        ("/robot/link_head_2/visuals/ZedMini", cam_b, True),
        ("/robot/link_head_2/visuals/foo", cam_b, False),
        ("/robot/link_head_2/visuals/head_camera", cam_b, False),
    ):
        st.DefinePrim(path, "Xform")
        assert RP._render_only_visual(st.GetPrimAtPath(path), roots) is ok, path
    assert RP.camera_roots(st, "") == ({}, {})


def test_training_body_zed_sits_at_the_calibrated_mount():
    """The training body's ZED Mini (zed_camera_link.xml visual and collision, in the camera link frame) composed with
    the skeleton joint is the calibrated mount (HC.ZEDM_MOUNT: link_head_2 -> ego_zed_m_camera, the render body's
    pose). Independent oracle of the numbers written in the link file."""
    import xml.etree.ElementTree as ET

    import numpy as np

    x = HC.camera_link("rby1")
    assert x is not None and x.parent == HC.ZEDM_MOUNT["parent_link"][0]
    Rj, tj = HC.rpy_to_mat(*x.rpy), np.array(x.xyz)
    Rm, tm = HC.rpy_to_mat(*HC.ZEDM_MOUNT["mount_rpy"][0]), np.array(HC.ZEDM_MOUNT["mount_xyz"][0])
    link = ET.parse(REPO_ROOT / x.link_xml).getroot()
    for tag in ("visual", "collision"):
        o = link.find(f"{tag}/origin")
        Rv = HC.rpy_to_mat(*map(float, o.get("rpy").split()))
        tv = np.array([float(v) for v in o.get("xyz").split()])
        assert np.abs(tj + Rj @ tv - tm).max() < 1e-6, tag
        assert np.abs(Rj @ Rv - Rm).max() < 1e-6, tag
