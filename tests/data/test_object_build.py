"""Object asset builder (dexengine/data/objects/build.py) on tiny synthetic meshes — no raw data needed."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from dexengine.data.objects import build as B

#: a 20 mm box in raw millimetres, written as quads with texture and normal indices (the OBJ parts the builder drops)
BOX_MM = np.array(
    [[0, 0, 0], [20, 0, 0], [20, 20, 0], [0, 20, 0], [0, 0, 20], [20, 0, 20], [20, 20, 20], [0, 20, 20]],
    dtype=np.float64,
)
BOX_QUADS = [(1, 4, 3, 2), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8)]


def box_obj(offset=(0.0, 0.0, 0.0)) -> str:
    v = BOX_MM + np.asarray(offset)
    lines = ["# raw", "mtllib box.mtl", "o box"]
    lines += [f"v {x} {y} {z} 0.5 0.5 0.5" for x, y, z in v]  # vertex colour columns are ignored
    lines += [f"vt {i / 8} {i / 8}" for i in range(8)] + ["vn 0 0 1"]
    lines += [f"f {a}/{a}/1 {b}/{b}/1 {c}/{c}/1 {d}/{d}/1" for a, b, c, d in BOX_QUADS]
    return "\n".join(lines) + "\n"


def write(tmp_path, name: str, text: str):
    p = tmp_path / name
    p.write_text(text)
    return p


def read_asset(path) -> tuple[np.ndarray, np.ndarray]:
    return B.parse_obj(path.read_bytes())


def test_parse_obj_keeps_raw_order_and_fans_polygons():
    v, f = B.parse_obj(box_obj().encode())
    np.testing.assert_array_equal(v, BOX_MM)  # raw order, no seam split from vt / vn
    assert f.shape == (12, 3)
    np.testing.assert_array_equal(f[:2], [[0, 3, 2], [0, 2, 1]])  # quad (1, 4, 3, 2) -> fan
    v2, f2 = B.parse_obj(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
    np.testing.assert_array_equal(f2, [[0, 1, 2]])  # relative indices
    with pytest.raises(ValueError):
        B.parse_obj(b"v 0 0 0\nf 1 2 3\n")


def test_unit_scale_and_layout(tmp_path):
    src = write(tmp_path, "box.obj", box_obj())
    ad = B.Adapter(refs={"box": {B.VISUAL: B.MeshRef(str(src))}}, units_per_m=1000.0)
    res = B.build_object("ds", "box", ad, str(tmp_path / "out"), visual_only=True)
    out = tmp_path / "out" / "box"
    assert sorted(p.name for p in out.iterdir()) == [B.VISUAL]  # no material, image or convex folder
    v, f = read_asset(out / B.VISUAL)
    np.testing.assert_allclose(v, BOX_MM / 1000.0, atol=1e-15)  # millimetres -> metres
    assert res["files"][B.VISUAL] == [8, 12, "written"]
    rv, rf = B.parse_obj(box_obj().encode())
    np.testing.assert_allclose(v[f], rv[rf] / 1000.0, atol=1e-15)  # face-corner positions = the raw mesh's


def test_frame_turn(tmp_path):
    src = write(tmp_path, "box.obj", box_obj())
    rx90 = ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))  # y-up -> z-up
    ad = B.Adapter(refs={"box": {B.VISUAL: B.MeshRef(str(src))}}, units_per_m=1000.0, rotation=rx90)
    B.build_object("ds", "box", ad, str(tmp_path / "out"), visual_only=True)
    v, _ = read_asset(tmp_path / "out" / "box" / B.VISUAL)
    np.testing.assert_allclose(v[3], [0.0, 0.0, 0.02], atol=1e-15)  # raw +y corner ends up on +z
    np.testing.assert_allclose(v, (BOX_MM / 1000.0) @ np.asarray(rx90).T, atol=1e-15)


def test_rerun_is_byte_stable_and_other_content_needs_overwrite(tmp_path):
    src = write(tmp_path, "box.obj", box_obj())
    ad = B.Adapter(refs={"box": {B.VISUAL: B.MeshRef(str(src))}})
    out = str(tmp_path / "out")
    B.build_object("ds", "box", ad, out, visual_only=True)
    assert B.build_object("ds", "box", ad, out, visual_only=True)["files"][B.VISUAL][2] == "same"
    write(tmp_path, "box.obj", box_obj(offset=(1.0, 0.0, 0.0)))
    with pytest.raises(RuntimeError, match="--overwrite"):
        B.build_object("ds", "box", ad, out, visual_only=True)
    assert B.build_object("ds", "box", ad, out, overwrite=True, visual_only=True)["files"][B.VISUAL][2] == "written"


def test_zip_ply_source_and_reader_names(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    zp = tmp_path / "meshes.zip"
    ply = trimesh.Trimesh(BOX_MM / 1000.0, [[0, 3, 2], [0, 2, 1]], process=False).export(file_type="ply")
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("contact_meshes/box.ply", ply)
    ref = B.MeshRef(str(zp), "contact_meshes/box.ply")
    v, f = B.read_mesh(ref)
    np.testing.assert_allclose(v, BOX_MM / 1000.0, atol=1e-7)
    assert f.shape == (2, 3)
    # OakInk2 naming comes from the reader: one asset per side and part id
    hf = tmp_path / "raw" / "hf"
    (hf / "anno_preview").mkdir(parents=True)
    rid = hf / "obj" / "object_repair" / "align_ds" / "O02@0001@00002"
    rid.mkdir(parents=True)
    (rid / "model.obj").write_text(box_obj())
    assert sorted(B.ADAPTERS["oakink2"](tmp_path / "raw").refs) == ["left_o02_0001_00002", "right_o02_0001_00002"]


def test_mano_names(tmp_path):
    for clip, right, left in (
        ("a__0", "processed/ds/assets/objects/cup", None),
        ("b__0", "processed/ds/assets/objects/cup", "processed/ds/assets/objects/lid"),
    ):
        (tmp_path / clip).mkdir()
        (tmp_path / clip / "task_info.json").write_text(
            json.dumps({"dataset_name": "ds", "right_object_mesh_dir": right, "left_object_mesh_dir": left})
        )
    assert B.mano_names("ds", tmp_path) == ["cup", "lid"]
    with pytest.raises(ValueError):
        B.mano_names("other", tmp_path)
    with pytest.raises(FileNotFoundError):
        B.mano_names("ds", tmp_path / "a__0")


def test_cli_selection(tmp_path, monkeypatch, capsys):
    src = write(tmp_path, "box.obj", box_obj())
    monkeypatch.setitem(B.ADAPTERS, "grab", lambda raw: B.Adapter(refs={"box": {B.VISUAL: B.MeshRef(str(src))}}))
    base = ["grab", "--raw", str(tmp_path), "--out", str(tmp_path / "out"), "--visual-only"]
    assert B.main(base + ["--objects", "box,nope"]) == 2
    assert B.main(base + ["--all"]) == 0
    assert "[OBJ_BUILD_DONE] requested=1 ok=1 failed=0" in capsys.readouterr().out


def test_convex_layout_rigid_and_articulated(tmp_path):
    pytest.importorskip("coacd")
    pytest.importorskip("threadpoolctl")
    D = B.decompose_module()
    rigid = write(tmp_path, "box.obj", box_obj())
    top = write(tmp_path, "top.obj", box_obj(offset=(0.0, 0.0, 20.0)))
    ad = B.Adapter(refs={"box": {B.VISUAL: B.MeshRef(str(rigid))}}, units_per_m=1000.0)
    out = tmp_path / "out"
    res = B.build_object("ds", "box", ad, str(out))
    n, action, _ = res["convex"]["convex"]
    assert action == "decomposed" and n >= 1
    assert D.stale_reason(str(out / "box" / B.VISUAL), str(out / "box" / "convex")) is None  # record written
    assert B.build_object("ds", "box", ad, str(out))["convex"]["convex"][1] == "reused"
    art = B.Adapter(
        refs={
            "hinge": {
                B.VISUAL: B.MeshRef(str(rigid)),
                "bottom.obj": B.MeshRef(str(rigid)),
                "top.obj": B.MeshRef(str(top)),
            }
        },
        units_per_m=1000.0,
        parts=("bottom.obj", "top.obj"),
    )
    B.build_object("ds", "hinge", art, str(out))
    names = sorted(p.name for p in (out / "hinge").iterdir())
    assert names == ["bottom.obj", "convex_bottom", "convex_top", "top.obj", B.VISUAL]  # parts only, no convex/
