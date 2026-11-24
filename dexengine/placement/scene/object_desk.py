"""Main object ↔ desk check of the judge (rule D24): rule · threshold · measure · judge mask · final check.

Rule      at every frame of the placed layout, no collision-hull vertex of a main object (the objects the env spawns,
          coords.desk_mask_slots) lies deeper than OBJ_BELOW_M below the desk top inside the desk footprint — the desk
          the env spawns: footprint coords.desk_bounds_scene_for(tree, margin 0) (the judge's DESK_XY, the box
          wrist_desk.desk_of reads) · top = the judge's desk height. A vertex below the top beside the desk is free
          space. The placement is planar (coords.place_objects moves x · y only), so which vertices are deep does not
          depend on it; where they land does.
Measure   desk_penetration.deep_vertices — the D24 hull reader (objects.hull: rigid authored hull · articulated root
          link), every frame of the tree the judge places. The pin lift (rl/tasks/scene/object_pin.pin_pose) raises
          only the pose it writes (reset · hold · park) and leaves the reference trajectory as it is, so the reference
          is measured as recorded. An interpreter without pxr reads them through desk_penetration's CLI in the retarget
          python (paths.retarget_py), the same function.
Mask      `grid_off_box`: per yaw and cell, no deep vertex inside the footprint (a pre-filter); final check `hit` (the
          bake's placement formula, the same box) decides.
Reason    DROP_WORD: the judge's place line counts the cells this check cuts; a clip with no cell left drops with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Deepest a main-object collision-hull vertex may sit below the desk top inside the desk footprint (m).
OBJ_BELOW_M = 0.02
DROP_WORD = "obj_desk_pen"
TAG = "[OBJ_DESK]"
OWNER = "placement/scene/object_desk.py:OBJ_BELOW_M"


class ObjectDeskUnmeasured(RuntimeError):
    """A main object of a judged tree has no readable collision hull: the desk check cannot run (never a pass)."""


def _has_pxr() -> bool:
    import importlib.util  # noqa: PLC0415

    return importlib.util.find_spec("pxr") is not None


def _deep(tree_dir: Path, ti: dict, q: np.ndarray, mains: set, below_z: float) -> np.ndarray:
    """desk_penetration.deep_vertices in this interpreter when it has pxr, else through its CLI (module doc)."""
    from dexengine.data import desk_penetration as DP  # noqa: PLC0415 — the D24 hull reader

    if _has_pxr():
        pts, basis = DP.deep_vertices(ti, q, mains, below_z)
        if pts is None:
            raise ObjectDeskUnmeasured(f"{TAG} {tree_dir.name}: basis {basis} — no readable main-object hull")
        return pts
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from dexengine.core import paths as P  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "deep.npz"
        cmd = [P.retarget_py(), DP.__file__, "--deep", str(tree_dir), "--mains", ",".join(sorted(mains))]
        r = subprocess.run(
            [*cmd, "--below", repr(below_z), "--out", str(out)], capture_output=True, text=True, timeout=600
        )
        if r.returncode != 0 or not out.is_file():
            raise ObjectDeskUnmeasured(f"{TAG} {tree_dir.name}: desk_penetration --deep rc {r.returncode}: {r.stdout}")
        with np.load(out) as z:
            return np.asarray(z["deep"], float)


def deep_points(tree_dir, z_off: float, desk_top: float) -> np.ndarray:
    """Capture xy (N, 2) of the main-object hull vertices deeper than OBJ_BELOW_M below the desk top at any frame of the
    tree (scene z = capture z + z_off; the main objects = coords.desk_mask_slots)."""
    from dexengine.placement.coords import core as VC  # noqa: PLC0415

    t = Path(tree_dir)
    ti = json.loads((t / "task_info.json").read_text(encoding="utf-8"))
    q = np.load(t / "0" / "trajectory_kinematic.npz")["qpos"]
    starts = {s for s, _e in VC.desk_mask_slots(t, q.shape[1])}
    mains = {str(o["name"]) for o in ti.get("objects") or [] if int(o["qpos_start"]) in starts}
    if not mains:
        return np.zeros((0, 2))
    return _deep(t, ti, q, mains, desk_top - z_off - OBJ_BELOW_M)[:, :2]


def grid_off_box(ww: np.ndarray, gx: np.ndarray, gy: np.ndarray, box) -> np.ndarray:
    """Judge mask — ww (N,2) scene xy (already rotated) · ascending cell offsets gx (X,) · gy (Y,) → (X,Y) bool: per
    cell no point of ww + (gx, gy) inside the box ((x0, x1), (y0, y1)) (edges inside). Each point covers one rectangle
    of cells; the union is counted on a difference grid."""
    (x0, x1), (y0, y1) = box
    X, Y = len(gx), len(gy)
    ix0, ix1 = np.searchsorted(gx, x0 - ww[:, 0], "left"), np.searchsorted(gx, x1 - ww[:, 0], "right")
    iy0, iy1 = np.searchsorted(gy, y0 - ww[:, 1], "left"), np.searchsorted(gy, y1 - ww[:, 1], "right")
    on = (ix0 < ix1) & (iy0 < iy1)
    D = np.zeros((X + 1, Y + 1), np.int64)
    for a, b, s in ((ix0, iy0, 1), (ix1, iy0, -1), (ix0, iy1, -1), (ix1, iy1, 1)):
        np.add.at(D, (a[on], b[on]), s)
    return D.cumsum(0).cumsum(1)[:X, :Y] == 0


def hit(deep_cap: np.ndarray, place, box) -> bool:
    """Does a deep vertex land inside the desk footprint under the placement (capture dx · dy · yaw · pivot — the
    bake's formula coords.place_objects)? The judge's final check."""
    from dexengine.placement.coords import core as VC  # noqa: PLC0415

    if not len(deep_cap):
        return False
    (x0, x1), (y0, y1) = box
    w = VC.capture_to_scene(VC.place_objects(deep_cap, *place))
    return bool(np.any((w[:, 0] >= x0) & (w[:, 0] <= x1) & (w[:, 1] >= y0) & (w[:, 1] <= y1)))


def line(clip: str, deep: np.ndarray, box) -> str:
    """The judge's marker line of this check (value and owner stated)."""
    (x0, x1), (y0, y1) = box
    return (
        f"{TAG} {clip} deep hull vertices {len(deep)} (main objects, deeper than OBJ_BELOW_M={OBJ_BELOW_M} m below "
        f"the desk top, owner {OWNER}) · footprint x=({x0:+.3f},{x1:+.3f}) y=({y0:+.3f},{y1:+.3f})"
    )
