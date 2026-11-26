"""Asset rule of the record's render-only object list (render_only_build module doc 'Asset'): the asset folder of an
original object outside the tree has the same object id and the same mesh (vertex and face counts, triangle positions
within MESH_TOL_M), a training USD and surface samples. The generator (scene.render_only_build) calls resolve_asset."""

from __future__ import annotations

import os

MESH_TOL_M = 1e-6

_MESH: dict = {}


def _verts_faces(path: str):
    import numpy as np
    import trimesh

    if path not in _MESH:
        m = trimesh.load(path, force="mesh", process=False)
        _MESH[path] = (np.asarray(m.vertices, float), np.asarray(m.faces, int))
    return _MESH[path]


def same_mesh(a: str, b: str) -> bool:
    """Same vertex and face counts, and the three vertex **positions** of each face in order within MESH_TOL_M (vertex
    indices a face points at may differ because of duplicate seam vertices; measured O02@0055@00001: 589/5000 faces
    with different indices, triangle position difference 0)."""
    import numpy as np

    va, fa = _verts_faces(a)
    vb, fb = _verts_faces(b)
    return (
        va.shape == vb.shape
        and fa.shape == fb.shape
        and bool(np.abs(va - vb).max() <= MESH_TOL_M)
        and bool(np.abs(va[fa] - vb[fb]).max() <= MESH_TOL_M)
    )


_ASSET: dict = {}


def _id_is(key_of, name: str, rid: str) -> bool:
    """Is this asset folder the object rid (a folder not in that dataset's object id form is not a candidate)."""
    try:
        return key_of(name) == rid
    except ValueError:
        return False


def resolve_asset(ds: str, rid: str, raw_mesh: str, key_of) -> dict:
    """Same object id and same mesh rule (module doc) -> {obj_id | None, matches[], candidates[]}."""
    from dexengine.core import paths as P
    from dexengine.data.contact import fingerprint as SF
    from dexengine.data.objects import hull as OH

    k = (ds, rid)
    if k in _ASSET:
        return _ASSET[k]
    root = P.layer_path("assets_objects", ds)
    cands = [d.name for d in (sorted(root.iterdir()) if root.is_dir() else []) if _id_is(key_of, d.name, rid)]
    matches = []
    if os.path.isfile(raw_mesh):
        for c in cands:
            vis = root / c / "visual.obj"
            if (
                vis.is_file()
                and same_mesh(raw_mesh, str(vis))
                and OH.usd_file(ds, c, False).is_file()
                and SF.surface_path(P.main_root(), ds, {"obj_id": c}).is_file()
            ):
                matches.append(c)
    _ASSET[k] = out = {"obj_id": matches[0] if matches else None, "matches": matches, "candidates": cands}
    return out
