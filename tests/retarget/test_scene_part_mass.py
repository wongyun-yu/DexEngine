"""The MuJoCo retarget scene splits an articulated object's mass by the part_mass rule: the total stays what the scene
gave the parts (OBJECT_DENSITY on their meshes) and is split by the parts' volumes (their convex pieces when the mesh
is open), not by MuJoCo's volume of each open mesh."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
trimesh = pytest.importorskip("trimesh")

from dexengine.retarget.scene import build as B  # noqa: E402


def _box(path: Path, extents, *, open_face: bool = False) -> None:
    m = trimesh.creation.box(extents=extents)
    if open_face:
        m = trimesh.Trimesh(m.vertices, m.faces[:-2], process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(path)


def _parts(tmp_path):
    """bottom: open 0.2 m cube (pieces = 0.1 × 0.2 × 0.1 m) · top: closed 0.1 m cube."""
    bottom, top = tmp_path / "o" / "bottom.obj", tmp_path / "o" / "top.obj"
    _box(bottom, (0.2, 0.2, 0.2), open_face=True)
    _box(tmp_path / "o" / "convex_bottom" / "0.obj", (0.1, 0.2, 0.1))
    _box(top, (0.1, 0.1, 0.1))
    return {"bottom": str(bottom), "top": str(top)}


def test_the_total_stays_the_scenes_and_splits_by_volume(tmp_path, capsys):
    files = _parts(tmp_path)
    m = B.artic_part_masses("right_object", files)
    total = B._mesh_mass(files["bottom"]) + B._mesh_mass(files["top"])
    assert m["bottom"] + m["top"] == pytest.approx(total)
    assert m["bottom"] / m["top"] == pytest.approx(0.002 / 0.001)
    assert "[PART_MASS] right_object: bottom V=2000.0cm3 (convex(1))" in capsys.readouterr().out


def test_the_geom_carries_its_share(tmp_path):
    files = _parts(tmp_path)
    m = B.artic_part_masses("o", files)
    s = mujoco.MjSpec()
    s.add_mesh(name="mb", file=files["bottom"])
    b = s.worldbody.add_body(name="b")
    b.add_freejoint(name="j")
    B._part_collision_geom(b, "g", "mb", m["bottom"])
    assert float(s.compile().body_mass[1]) == pytest.approx(m["bottom"])


def test_a_missing_part_dies(tmp_path):
    files = _parts(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"\[PART_MASS\].*'top'"):
        B.artic_part_masses("o", {"bottom": files["bottom"], "top": None})


def test_both_articulated_parts_go_through_the_rule():
    """Structure: add_primary_body weighs the parts with artic_part_masses and gives both collision geoms a share."""
    tree = ast.parse(Path(B.__file__).read_text(encoding="utf-8"))
    f = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "add_primary_body")
    calls = [c for c in ast.walk(f) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
    assert [ast.unparse(c.args[1]) for c in calls if c.func.id == "artic_part_masses"] == [
        "{'bottom': art.bottom_file, 'top': art.top_file}"
    ]
    assert [ast.unparse(c.args[3]) for c in calls if c.func.id == "_part_collision_geom"] == [
        "pm['bottom']",
        "pm['top']",
    ]
