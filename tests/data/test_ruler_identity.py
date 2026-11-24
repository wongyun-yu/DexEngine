"""The triangle ruler's identity (sdf_grid.ruler_id) covers every module the ruler is made of
(collider_shape.RULER_MODULES): a changed byte in any of them changes it, and the set cannot leave one out."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import sdf_grid as SG

PKG = "dexengine.data.objects"


def _file(name: str) -> str:
    return importlib.import_module(f"{PKG}.{name}").__file__


def _imports(name: str) -> set:
    """The modules of PKG that module `name` imports at module level (function-level imports are not the ruler's)."""
    out = set()
    for n in ast.parse(Path(_file(name)).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.ImportFrom) and n.module == PKG:
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and (n.module or "").startswith(PKG + "."):
            out.add(n.module.split(".")[3])
        elif isinstance(n, ast.Import):
            out |= {a.name.split(".")[3] for a in n.names if a.name.startswith(PKG + ".")}
    return out


def _missing(modules) -> set:
    """collider_shape and every PKG module reached from it by module-level imports, less `modules`."""
    need, todo = {"collider_shape"}, ["collider_shape"]
    while todo:
        for m in _imports(todo.pop()) - need:
            need.add(m)
            todo.append(m)
    return need - set(modules)


def test_the_set_names_every_ruler_module():
    assert not _missing(CS.RULER_MODULES)
    assert _missing(tuple(m for m in CS.RULER_MODULES if m != "triangle_geom")) == {"triangle_geom"}  # negative


def test_one_changed_byte_in_any_ruler_module_changes_the_identity(tmp_path):
    files = [_file(m) for m in CS.RULER_MODULES]
    assert SG.ruler_id() == SG.blobs_id(files)
    for i in range(len(files)):
        copies = []
        for j, f in enumerate(files):
            b = bytearray(Path(f).read_bytes())
            if j == i:
                b[len(b) // 2] ^= 1
            (tmp_path / f"{i}_{j}.py").write_bytes(bytes(b))
            copies.append(tmp_path / f"{i}_{j}.py")
        assert SG.blobs_id(copies) != SG.blobs_id(files)
    assert SG.blobs_id(files[::-1]) != SG.blobs_id(files)  # the order is part of it
