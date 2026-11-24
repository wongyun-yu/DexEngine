"""Articulated part masses follow one volume rule (dexengine/robots/objects/part_mass.py): a part's volume is its
mesh's volume when watertight, otherwise the summed volume of its convex pieces; the object total is split by volume.
The USD bake reads the rule for its link masses (no per-link cap, no open-mesh volume)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")

from dexengine.robots.objects import part_mass as PM  # noqa: E402

USD_BUILD = Path(PM.__file__).with_name("usd_build.py")


def _box(path: Path, extents, *, open_face: bool = False) -> None:
    m = trimesh.creation.box(extents=extents)
    if open_face:
        m = trimesh.Trimesh(m.vertices, m.faces[:-2], process=False)  # one side removed → not watertight
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def test_a_watertight_part_weighs_by_its_mesh_volume(tmp_path):
    f = tmp_path / "obj" / "lid.obj"
    _box(f, (0.1, 0.2, 0.05))
    v, src = PM.part_volume(str(f))
    assert src == "watertight" and v == pytest.approx(0.001)


def test_an_open_part_weighs_by_its_convex_pieces(tmp_path):
    f = tmp_path / "obj" / "body.obj"
    _box(f, (0.3, 0.3, 0.3), open_face=True)
    assert not trimesh.load(f, force="mesh", process=False).is_watertight
    cdir = tmp_path / "obj" / "convex_body"
    _box(cdir / "0.obj", (0.1, 0.1, 0.1))
    _box(cdir / "1.obj", (0.2, 0.1, 0.1))
    v, src = PM.part_volume(str(f))
    assert src == "convex(2)" and v == pytest.approx(0.003)


def test_an_open_part_without_pieces_dies(tmp_path):
    f = tmp_path / "obj" / "door.obj"
    _box(f, (0.3, 0.3, 0.02), open_face=True)
    with pytest.raises(FileNotFoundError, match=r"\[PART_MASS\].*no convex pieces"):
        PM.part_volume(str(f))


def test_a_link_sums_its_parts(tmp_path):
    a, b = tmp_path / "o" / "body.obj", tmp_path / "o" / "lever.obj"
    _box(a, (0.1, 0.1, 0.1))
    _box(b, (0.1, 0.1, 0.05))
    v, src = PM.link_volume([str(a), str(b)])
    assert v == pytest.approx(0.0015) and src == "watertight+watertight"


def test_the_total_splits_by_volume_and_sums_back():
    m = PM.split_by_volume(0.2, {"base": 0.0231, "moving": 0.00276})
    assert m["base"] + m["moving"] == pytest.approx(0.2)
    assert m["moving"] / m["base"] == pytest.approx(0.00276 / 0.0231)


@pytest.mark.parametrize(("total", "vols"), [(0.0, {"a": 1.0}), (0.2, {"a": 1.0, "b": 0.0}), (0.2, {})])
def test_a_split_without_a_positive_total_or_volume_dies(total, vols):
    with pytest.raises(ValueError, match=r"\[PART_MASS\]"):
        PM.split_by_volume(total, vols)


def _fn(name: str) -> ast.FunctionDef:
    tree = ast.parse(USD_BUILD.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_the_link_body_takes_its_mass_and_weighs_nothing_itself():
    """Structure: _part_body receives the mass; it reads no mesh volume and applies no cap of its own."""
    f = _fn("_part_body")
    assert "mass" in [a.arg for a in f.args.args]
    assert not [n for n in ast.walk(f) if isinstance(n, ast.Attribute) and n.attr == "volume"]
    assert not [n for n in ast.walk(f) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "min"]


def _calls(fn: str) -> set:
    return {ast.unparse(n.func) for n in ast.walk(_fn(fn)) if isinstance(n, ast.Call)}


def test_the_articulated_bake_splits_with_the_rule():
    """Structure: the articulated USD builder weighs its links with part_mass (link_volume → split_by_volume) and keeps
    the bake total (_bake_total)."""
    assert "_link_masses" in _calls("create_articulated_object_usd")
    assert {"PM.link_volume", "PM.split_by_volume", "_bake_total"} <= _calls("_link_masses")


def test_the_bake_total_is_the_per_link_capped_sum():
    """The total every articulated USD carried: per link min(density × mesh volume, cap), summed."""
    src = ast.unparse(_fn("_bake_total"))
    assert "min(density * sum(" in src and "mass_cap)" in src and "for k in links" in src


def test_the_rule_module_imports_without_third_party_packages():
    """The bake imports it in the Isaac interpreter: top-level imports are standard library only."""
    tree = ast.parse(Path(PM.__file__).read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    mods = {a.name for n in top if isinstance(n, ast.Import) for a in n.names} | {
        n.module for n in top if isinstance(n, ast.ImportFrom)
    }
    assert mods <= {"__future__", "os"}, mods
    assert os.path.basename(PM.__file__) == "part_mass.py"
