"""S-decomp-fresh: the bake's decomposition stamp and the USD content oracle (data/objects/usd_decomp)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pxr")
trimesh = pytest.importorskip("trimesh")

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from dexengine.core import artifact_paths as AP  # noqa: E402
from dexengine.data.objects import usd_decomp as UD  # noqa: E402
from dexengine.retarget import decompose as D  # noqa: E402
from dexengine.robots.objects import usd_build as UB  # noqa: E402


def _parts(folder: Path, sizes: list[float], source: Path) -> Path:
    """A part folder with one box per size and the decomposition record of the canonical writer."""
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("*.obj"):
        f.unlink()
    for i, s in enumerate(sizes):
        box = trimesh.creation.box(extents=(s, s, s))
        box.apply_translation((0.1 * i, 0.0, 0.0))
        box.export(str(folder / f"{i}.obj"))
    D._write_record(str(folder), str(source), len(sizes))
    return folder


def _object(root: Path, oid: str, folder: str = "convex", sizes=(0.02, 0.03)) -> Path:
    src = root / oid / ("visual.obj" if folder == "convex" else f"{folder[len('convex_') :]}.obj")
    src.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(str(src))
    return _parts(root / oid / folder, list(sizes), src)


def _bake(path: Path, bodies: dict) -> Usd.Stage:
    """A USD with one rigid body per entry {body path: part folder}, authored by the bake's collision writer."""
    st = Usd.Stage.CreateNew(str(path))
    for body, folder in bodies.items():
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(st, Sdf.Path(body)).GetPrim())
        UB._add_collision_parts(st, body, str(folder), 1.0)
    st.GetRootLayer().Save()
    return Usd.Stage.Open(str(path))


def test_bake_stamps_each_body(tmp_path):
    """Wiring: the bake's collision writer stamps the folder it read (removing the write_stamp call fails here)."""
    cvx = _object(tmp_path / "obj", "a")
    st = _bake(tmp_path / "a.usd", {"/Object/base": cvx})
    assert UD.read_stamp(st) == {"/Object/base": {"dir": "a/convex", "fp": D.folder_fp(str(cvx)), "parts": 2}}


def test_fresh_usd_is_current(tmp_path):
    root = tmp_path / "obj"
    st = _bake(tmp_path / "a.usd", {"/Object/base": _object(root, "a")})
    assert UD.verdict(st, UD.convex_dirs(root, "a", False))[0] == "current"


def test_unstamped_equal_content_is_its_own_verdict(tmp_path):
    root = tmp_path / "obj"
    st = _bake(tmp_path / "a.usd", {"/Object/base": _object(root, "a")})
    layer = st.GetRootLayer()
    layer.customLayerData = {}
    assert UD.verdict(st, UD.convex_dirs(root, "a", False))[0] == "current_unstamped"


def test_usd_older_than_its_decomposition_is_stale(tmp_path):
    """Planted: the folder is decomposed again with another part count after the bake."""
    root = tmp_path / "obj"
    cvx = _object(root, "a")
    st = _bake(tmp_path / "a.usd", {"/Object/base": cvx})
    _parts(cvx, [0.02, 0.03, 0.04], root / "a" / "visual.obj")
    v, why = UD.verdict(st, UD.convex_dirs(root, "a", False))
    assert v == "stale_content" and "parts 2" in why and "a/convex': 3" in why


def test_same_count_other_vertices_is_stale(tmp_path):
    root = tmp_path / "obj"
    cvx = _object(root, "a")
    st = _bake(tmp_path / "a.usd", {"/Object/base": cvx})
    _parts(cvx, [0.02, 0.031], root / "a" / "visual.obj")
    assert UD.verdict(st, UD.convex_dirs(root, "a", False))[0] == "stale_content"


def test_non_canonical_folder_is_stale(tmp_path):
    root = tmp_path / "obj"
    cvx = _object(root, "a")
    st = _bake(tmp_path / "a.usd", {"/Object/base": cvx})
    rec = json.loads((cvx / D.RECORD_NAME).read_text())
    rec["params_fp"] = "0" * 16
    (cvx / D.RECORD_NAME).write_text(json.dumps(rec))
    v, why = UD.verdict(st, UD.convex_dirs(root, "a", False))
    assert v == "stale_folder" and "params_fp" in why


def test_stamp_naming_another_identity_fails(tmp_path):
    root = tmp_path / "obj"
    st = _bake(tmp_path / "a.usd", {"/Object/base": _object(root, "a")})
    layer = st.GetRootLayer()
    data = dict(layer.customLayerData)
    data[UD.STAMP_KEY] = {"/Object/base": {"dir": "a/convex", "fp": "f" * 16, "parts": 2}}
    layer.customLayerData = data
    assert UD.verdict(st, UD.convex_dirs(root, "a", False))[0] == "stamp_mismatch"


def test_no_folder_is_no_decomposition(tmp_path):
    root = tmp_path / "obj"
    st = _bake(tmp_path / "a.usd", {"/Object/base": _object(root, "a")})
    assert UD.convex_dirs(root, "b", False) == []
    assert UD.verdict(st, [])[0] == "no_decomposition"


def test_paired_asset_finds_parts_under_base_and_moving_ids(tmp_path):
    """A paired articulation keeps its parts in <base id>/convex_visual and <moving id>/convex_visual — there is no
    folder named after the asset id."""
    root = tmp_path / "obj"
    base, mov = _object(root, "left_o1", "convex_visual"), _object(root, "raw_o2", "convex_visual", (0.01,))
    _object(root, "left_o1")  # the base id's rigid folder is not an articulated candidate
    asset = AP.object_asset_id("left_o1", {"type": "slide", "parts": {"moving_id": "raw_o2"}})
    cands = UD.convex_dirs(root, asset, True)
    assert cands == [base, mov]
    st = _bake(tmp_path / "p.usd", {"/Object/base_link": base, "/Object/moving_link": mov})
    assert UD.verdict(st, cands)[0] == "current"


def test_asset_object_ids_inverts_the_asset_id():
    art = {"type": "hinge", "parts": {"moving_id": "m1", "fixed": [{"id": "f2"}, {"id": "f1"}]}}
    assert AP.asset_object_ids(AP.object_asset_id("b1", art)) == ["b1", "m1", "f1", "f2"]
    assert AP.asset_object_ids("plain_id") == ["plain_id"]
    for bad in ("a__b", "a__b__twist", "a____hinge"):
        with pytest.raises(ValueError):
            AP.asset_object_ids(bad)


def test_collision_mesh_outside_the_bake_layout_raises(tmp_path):
    st = Usd.Stage.CreateInMemory()
    m = UsdGeom.Mesh.Define(st, "/Object/base/hull")
    m.GetPointsAttr().Set([(0.0, 0.0, 0.0)])
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    with pytest.raises(ValueError, match="collision_<i>"):
        UD.usd_bodies(st)


def test_marker_counts_verdicts_and_cap():
    rows = [
        {"verdict": "current", "at_cap": 0},
        {"verdict": "current_unstamped", "at_cap": 1},
        {"verdict": "stale_content", "at_cap": 1},
        {"verdict": "no_decomposition", "at_cap": 0},
    ]
    line = UD.marker("oakink2", rows, frozen=False)
    assert line.startswith("[USD_DECOMP] oakink2 usds 4 · current 1 · current_unstamped 1 · not current 2")
    assert "'stale_content': 1" in line and "at cap 2" in line


def _export_without_stamp(path: Path) -> str:
    layer = Sdf.Layer.FindOrOpen(str(path))
    data = dict(layer.customLayerData)
    data.pop(UD.STAMP_KEY, None)
    if data:
        layer.customLayerData = data
    else:
        layer.ClearCustomLayerData()
    return layer.ExportToString()


def test_stamp_is_the_only_difference_from_an_unstamped_bake(tmp_path, monkeypatch):
    """A stamped bake and the same bake without the stamp writer differ only in customLayerData: every prim, the
    collision and visual geometry and the physics attributes are equal."""
    root = tmp_path / "obj"
    cvx = _object(root, "a")
    mesh = str(root / "a" / "visual.obj")
    UB.create_object_usd(mesh, str(tmp_path / "stamped.usd"), convex_dir=str(cvx))
    monkeypatch.setattr(UD, "write_stamp", lambda stage, body, d: {"dir": "", "fp": "", "parts": 0})
    UB.create_object_usd(mesh, str(tmp_path / "plain.usd"), convex_dir=str(cvx))
    plain = Sdf.Layer.FindOrOpen(str(tmp_path / "plain.usd"))
    assert UD.STAMP_KEY not in dict(plain.customLayerData)
    assert UD.STAMP_KEY in dict(Sdf.Layer.FindOrOpen(str(tmp_path / "stamped.usd")).customLayerData)
    assert _export_without_stamp(tmp_path / "stamped.usd") == plain.ExportToString()
