"""Tree reading: object meshes and articulated parts, the placed-vertex function, frame maps and tree records."""

from __future__ import annotations

import numpy as np

from dexengine.data.artic.fk import (  # noqa: F401 — re-exported (owner: data.artic.fk)
    artic_moving_vertices,
    pose_points,
)
from dexengine.placement.coords.frames import _paths

#: Object mesh vertices by absolute file path.
_MESH_V_CACHE: dict = {}
#: The stage server (placement/bake/server.py) keeps these globals between clips (keys do not depend on the clip).
_SERVER_KEEP = ("_MESH_V_CACHE",)


def _object_mesh_dir(tree_dir, ti: dict, o: dict):
    """A stored object reference (task_info `objects[].mesh_dir` = logical name) → its folder under the data root
    (`data_layout.object_path`). The dataset is the tree's `dataset_name`; missing → raises."""
    ds = ti.get("dataset_name")
    if not ds:
        raise ValueError(
            f"{tree_dir}: task_info has no dataset_name — cannot resolve the object reference (objects[].mesh_dir)"
        )
    return _paths().main_root() / _paths().data_layout.object_path(ds, o["mesh_dir"])


def _object_part_path(tree_dir, ti: dict, o: dict, rel: str):
    """The file of an articulated part (stored part path `rel`, relative to mesh_dir): resolved to a logical name by
    `data_layout.object_part_ref` (same-folder file · `../<sibling id>/…`), then opened through `object_path` — the same
    names setup_inputs declares."""
    ds = ti.get("dataset_name")
    if not ds:
        raise ValueError(f"{tree_dir}: task_info has no dataset_name — cannot resolve the articulated part path")
    dl = _paths().data_layout
    return _paths().main_root() / dl.object_path(ds, dl.object_part_ref(ds, o["mesh_dir"], rel))


def _taco_paths():
    from dexengine.core import artifact_paths  # noqa: PLC0415

    return artifact_paths


def artic_link_vertices(tree_dir, ti: dict, o: dict, link: str) -> np.ndarray:
    """Vertices of an articulated link (base|moving) in the link frame — the link part ∪ its fixed attached parts
    (moved by their pose; the list comes from artifact_paths.artic_link_files)."""
    parts = (o.get("articulation") or {}).get("parts") or {}
    return np.concatenate(
        [
            pose_points(_mesh_vertices_file(_object_part_path(tree_dir, ti, o, rel)), pose)
            for rel, pose in _taco_paths().artic_link_files(parts, link)
        ],
        0,
    )


def _mesh_vertices(mesh_dir) -> np.ndarray:
    """Object mesh vertices (N,3) — the `v` lines of visual.obj (cached by path). Raises if missing."""
    import pathlib as _pl

    return _mesh_vertices_file(_pl.Path(mesh_dir) / "visual.obj")


def _mesh_vertices_file(p) -> np.ndarray:
    """Vertices (N,3) of one OBJ file (cached by path); articulated parts are read through this too. A relative path
    raises (stored object references are resolved by `_object_mesh_dir` first); a missing file raises."""
    import pathlib as _pl

    p = _pl.Path(p)
    if not p.is_absolute():
        raise ValueError(
            f"relative mesh path {p} — task_info object references must be resolved to an absolute path "
            "by _object_mesh_dir (data_layout.object_path)"
        )
    if not p.is_file():
        raise FileNotFoundError(f"object mesh missing: {p}")
    key = str(p)
    if key not in _MESH_V_CACHE:
        vs = []
        with open(p, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.startswith("v "):
                    t = ln.split()
                    vs.append((float(t[1]), float(t[2]), float(t[3])))
        if not vs:
            raise ValueError(f"mesh has no vertices: {p}")
        _MESH_V_CACHE[key] = np.asarray(vs, float)
    return _MESH_V_CACHE[key]


def _tree_key(tree_dir) -> tuple:
    """Cache key of a tree = path + mtimes of task_info.json and the kinematic npz (a re-bake of the same path must
    never be served the previous bake's numbers)."""
    import pathlib as _pl

    p = _pl.Path(tree_dir)
    key = [str(p)]
    for f in (p / "task_info.json", p / "0/trajectory_kinematic.npz"):
        try:
            key.append(f.stat().st_mtime_ns)
        except FileNotFoundError:  # noqa: PERF203 — a missing file is part of the key
            key.append(None)
    return tuple(key)


def _rest_rule():
    """`dexengine.data.support.rest` — the rest-interval definition (Z_STAB), window slicing, rest-object identity."""
    from dexengine.data.support import rest  # noqa: PLC0415

    return rest


def _qpos_layout_mod():
    """`dexengine.core.qpos_layout` — the one reader of stamped tree statements (the env's trajectory loader reads
    through it too)."""
    from dexengine.core import qpos_layout  # noqa: PLC0415

    return qpos_layout


def _placers(tree_dir, ti: dict, q):
    """(_placed(o, f) → world vertices, _main_for(side) → main object or None) — shared by the aerial support, the
    rest discs and the start-embed check. `_main_for` has **no fallback** (objs[0] for a hand with no main object
    would put a support under an object nobody manipulates)."""
    from scipy.spatial.transform import Rotation as _R

    objs = [o for o in (ti.get("objects") or []) if o.get("qpos_start") is not None and o.get("mesh_dir")]
    if not objs:
        raise ValueError(f"{tree_dir}: no object in objects[] has mesh_dir/qpos_start — cannot place supports")

    def _placed(o, f):
        s = int(o["qpos_start"])
        p = q[f, s : s + 3]
        R = _R.from_quat(q[f, [s + 4, s + 5, s + 6, s + 3]]).as_matrix()  # wxyz → xyzw
        art = o.get("articulation")
        if art:
            # articulated: base part + moving part at that frame's joint value (visual.obj is the closed pose;
            #   many arctic clips start open)
            Vb = artic_link_vertices(tree_dir, ti, o, "base")
            Vm = artic_moving_vertices(
                art, artic_link_vertices(tree_dir, ti, o, "moving"), float(q[f, int(art["qpos_index"])])
            )
            return np.concatenate([Vb, Vm], 0) @ R.T + p
        return _mesh_vertices(_object_mesh_dir(tree_dir, ti, o)) @ R.T + p

    return _placed, lambda side: _main_of(objs, side)


def _main_of(objs: list, side: str) -> dict | None:
    """The main object of a hand — the first of `objs` with role main and that side or "both"; None = the hand has
    none (no fallback)."""
    return next((o for o in objs if o.get("role") == "main" and o.get("side") in (side, "both")), None)


def tree_f0_pad(tree_dir, ti: dict) -> int:
    """The f0 pad P of a tree = frames prepended in front of the base clip. Read, never recomputed: a baked tree's
    `task_info.f0_prepad` (written before the stamp) · otherwise `F0_REPAIR.json:n_blend_effective` (the f0 shadow
    staging tree the judge reads) · otherwise 0 (a base-layer tree)."""
    import json as _json
    import pathlib as _pl

    if ti.get("f0_prepad") is not None:
        return int(ti["f0_prepad"])
    f = _pl.Path(tree_dir) / "F0_REPAIR.json"
    if f.is_file():
        return int(_json.loads(f.read_text()).get("n_blend_effective", 0) or 0)
    return 0


def _base_to_tree(tree_dir, ti: dict | None = None, T: int | None = None, src_idx="tree"):
    """base frame → tree frame mapping. `support.rest_intervals` start/end are **base frames** (the tree's own
    base-layer window). A tree may carry ① an f0 pad P in front (pre-retime frame = base + P) and ② a retime map
    `retime_src_idx` (tree frame → pre-retime fractional index). Both are applied; identity when neither.
    `src_idx`: "tree" = read the map from the tree npz · None = the qpos in hand has no retime map (a pre-retime
    snapshot) · array = that map."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    if ti is None:
        ti = _json.loads((tree_dir / "task_info.json").read_text())
    P = tree_f0_pad(tree_dir, ti)
    if isinstance(src_idx, str):
        z = np.load(tree_dir / "0/trajectory_kinematic.npz")
        Tt = int(T if T is not None else len(z["qpos"]))
        src_idx = np.asarray(z["retime_src_idx"], float) if "retime_src_idx" in z.files else None
    else:
        Tt = int(T)
    if src_idx is None:
        return lambda b: int(min(Tt - 1, max(0, int(b) + P)))
    src = np.asarray(src_idx, float)
    return lambda b: int(min(Tt - 1, max(0, int(np.searchsorted(src, float(int(b) + P))))))


def _base_T(ti: dict) -> int | None:
    """The tree's base-layer length (segment.T) — the frame range `support.rest_intervals` must live in."""
    T = (ti.get("segment") or {}).get("T")
    return int(T) if T is not None else None


def tree_dt(tree_dir, ti: dict) -> float:
    """One env step on this tree's clock (s) = `task_info.ref_dt` (50 Hz on base, f0 shadow and baked trees). Missing →
    raises: the render's frame plan needs the clock."""
    dt = ti.get("ref_dt")
    if dt is None:
        raise ValueError(f"{tree_dir}: task_info has no ref_dt — the tree clock is unknown")
    return float(dt)


def is_baked_tree(ti: dict) -> bool:
    """A post-bake tree (placement applied by the bake — it stamps `placement_v2`)."""
    return bool(ti.get("placement_v2"))


def tree_placement(ti: dict) -> tuple | None:
    """The placement a tree's qpos carries — (dx, dy, yaw_deg, pivot) as the bake recorded it (`xy_shift_traj ·
    yaw_deg · pivot_traj` with `placement_v2`) · None for an unplaced tree (base · f0 shadow). A baked tree missing one
    of the three raises (an identity guess would compare two frames)."""
    if not is_baked_tree(ti):
        return None
    miss = [k for k in ("xy_shift_traj", "yaw_deg", "pivot_traj") if ti.get(k) is None]
    if miss:
        raise ValueError(f"baked tree (placement_v2) without placement record {miss} — the tree frame is unknown")
    dx, dy = ti["xy_shift_traj"]
    return float(dx), float(dy), float(ti["yaw_deg"]), ti["pivot_traj"]
