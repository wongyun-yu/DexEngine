"""Coordinate frames, scene mode, desk boxes, object slots, the placement formula and the robot's standing spot.

Two frames: the qpos in `trajectory_kinematic.npz` is in **capture coordinates**; scene (env-local) coordinates are
Rz(−90°)·capture (scene_x = capture_y, scene_y = −capture_x). The robot base and the desk bounds are scene values.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rot

from dexengine.placement.coords.thresholds import DESK_EDGE_MARGIN, SCENE_MODES

# capture → scene rotation (the same Rz(−90°) as the arm IK)
RZ_CAP2SCENE = _Rot.from_euler("z", -90, degrees=True)
# 2D matrix form (identical to the judge's RZW: ww = pp @ RZW.T)
RZW = np.array([[0.0, 1.0], [-1.0, 0.0]])

#: Old file-path names of package modules still asked for through `_pkg_module` (record/store/scene_layout.py).
_PKG_MODULES = {"_shared/arm_specs.py": "dexengine.core.arms"}


def _paths():
    """`dexengine.core.paths`, imported lazily (paths sits above coords)."""
    from dexengine.core import paths  # noqa: PLC0415

    return paths


def _pkg_module(rel: str):
    """The package module behind an old `tasks/manager_based/<rel>` file name. Unknown name → KeyError."""
    import importlib  # noqa: PLC0415

    return importlib.import_module(_PKG_MODULES[rel])


def desk_height() -> float:
    """Desk top z in the robot world (m) — `DESK_HEIGHT` of the task constants (the value the env and arm IK read)."""
    from dexengine.rl.tasks.cfg import constants  # noqa: PLC0415

    return float(constants.DESK_HEIGHT)


def desk_placement():
    """The scene desk top's owner module (`dexengine.core.desk` — centre x · size per workspace profile)."""
    from dexengine.core import desk  # noqa: PLC0415

    return desk


def desk_top_for(dataset: str | None = None) -> tuple:
    """A dataset's scene desk top (x_off, half_x, half_y) [m] from `dexengine.core.desk`. `dataset` None = the tabletop
    top, for callers that have no tree (arm-only probes · the IK pool's own sidecar); an unregistered dataset raises."""
    dp = desk_placement()
    if dataset is None:
        return (dp.DESK_X_TASK, dp.DESK_TOP_HALF_X, dp.DESK_TOP_HALF_Y)
    return dp.desk_top_for(str(dataset), layout=_paths().data_layout)


def desk_size_mark(dataset: str) -> str:
    """The applied-state marker `[DESK_SIZE] profile=… dataset=… x_off=… half=(…)` of a dataset's desk top — whoever
    judges or spawns at a desk prints it."""
    return desk_placement().desk_size_mark(str(dataset), layout=_paths().data_layout)


def desk_check_on(scene: str) -> bool:
    """Does this scene mode run the desk checks (the judge's desk mask and final desk check, the desk static /
    place-rule masks, the pool desk clearance, the collection desk mask)? Only `table` has a desk; `aerial` turns them
    all off, whatever any flag says. Unknown mode raises."""
    if scene not in SCENE_MODES:
        raise ValueError(f"scene {scene!r} — only {SCENE_MODES} are allowed")
    return scene == "table"


def scene_env(scene: str):
    """Context that sets env `DEXENGINE_SCENE` to `scene` and restores it afterwards. scene_mode() called without a tree
    reads only the env, so a caller that knows the tree's scene passes it through this. Unknown scene raises."""
    import contextlib as _contextlib
    import os as _os

    if scene not in SCENE_MODES:
        raise ValueError(f"scene {scene!r} — only {SCENE_MODES} are allowed")

    @_contextlib.contextmanager
    def _cm():
        old = _os.environ.get("DEXENGINE_SCENE")
        _os.environ["DEXENGINE_SCENE"] = scene
        try:
            yield scene
        finally:
            if old is None:
                _os.environ.pop("DEXENGINE_SCENE", None)
            else:
                _os.environ["DEXENGINE_SCENE"] = old

    return _cm()


def scene_mode(tree_dir=None) -> str:
    """The scene mode: env `DEXENGINE_SCENE` (stage 2 `--scene` exports it; absent = table). If the tree's task_info
    carries a `scene` written at bake time it is cross-checked — a mismatch raises. table|aerial only."""
    import json as _json
    import os as _os
    import pathlib as _pl

    env = _os.environ.get("DEXENGINE_SCENE", "")
    if env and env not in SCENE_MODES:
        raise ValueError(f"DEXENGINE_SCENE={env!r} — only {SCENE_MODES} are allowed")
    if tree_dir is not None:
        p = _pl.Path(tree_dir) / "task_info.json"
        st = _json.loads(p.read_text()).get("scene") if p.is_file() else None
        if st and st not in SCENE_MODES:
            raise ValueError(f"{tree_dir}: task_info.scene={st!r} — only {SCENE_MODES} are allowed")
        if st and env and st != env:
            raise ValueError(f"{tree_dir}: task_info.scene={st} but DEXENGINE_SCENE={env} — the scenes disagree")
        if st and not env:
            return st
    return env or "table"


def obj_slots(D: int, n_obj: int = 2) -> list[tuple[int, int]]:
    """Object slots [(s,e), ...] for a qpos of width D — the legacy TACO convention (n_obj objects of 7 columns,
    contiguous at the very end of qpos). It does not hold on the base layer (arctic appends hinge columns, hocap mixes
    extras in, a shared-object dataset has 1 object); `tree_object_slots(tree_dir)` is the reader for trees and uses
    this only as an oracle on legacy trees."""
    base = D - 7 * n_obj
    if base < 0:
        raise ValueError(f"qpos width {D} is too small for {n_obj} objects")
    return [(base + 7 * i, base + 7 * i + 7) for i in range(n_obj)]


def tree_object_slots(tree_dir, D: int | None = None) -> list[tuple[int, int]]:
    """The object blocks the tree declares [(s,e), …] — main objects and extras, in position order, read through
    `qpos_layout`. On a legacy tree (no qpos_layout key) the reader reproduces the `obj_slots(D)` arithmetic; given D
    that is verified."""
    lay = _paths().qpos_layout.read(str(tree_dir), nq=D)  # given the width, a legacy tree's hand width is derived
    seen, out = set(), []
    for s in list(lay.main.values()) + list(lay.extras):
        if s not in seen:
            seen.add(s)
            out.append((int(s), int(s) + 7))
    out.sort()
    if D is not None and not lay.source.startswith("task_info.layout") and out != obj_slots(D, len(out)):
        raise ValueError(
            f"{tree_dir}: legacy tree, but the tree-derived slots {out} differ from the last-14 "
            f"convention {obj_slots(D, len(out))}"
        )
    return out


def desk_mask_slots(tree_dir, D: int | None = None) -> list[tuple[int, int]]:
    """The object blocks the judge holds on the desk at the first frame (its desk mask and final desk check) — the
    objects the env spawns (main objects only, `Layout.main`), in position order. A scene prop is not spawned, so
    where it lands after placement neither narrows the placement nor needs handling off the desk."""
    lay = _paths().qpos_layout.read(str(tree_dir), nq=D)
    return [(int(s), int(s) + 7) for s in sorted(set(lay.main.values()))]


def capture_to_scene(p: np.ndarray) -> np.ndarray:
    """capture → scene. Takes both (...,2) and (...,3) (z passes through unchanged)."""
    p = np.asarray(p, float)
    if p.shape[-1] == 2:
        return p @ RZW.T
    return RZ_CAP2SCENE.apply(p.reshape(-1, 3)).reshape(p.shape)


def scene_to_capture(p: np.ndarray) -> np.ndarray:
    """scene → capture (the inverse of capture_to_scene)."""
    p = np.asarray(p, float)
    if p.shape[-1] == 2:
        return p @ RZW
    return RZ_CAP2SCENE.inv().apply(p.reshape(-1, 3)).reshape(p.shape)


def _desk_box(top, shrink_robot: float, margin: float) -> tuple:
    x, hx, hy = top
    x0 = x - hx + margin + shrink_robot
    x1 = x + hx - margin
    y0 = -hy + margin
    y1 = hy - margin
    return (x0, x1), (y0, y1)


def desk_bounds_scene(shrink_robot: float = 0.0, margin: float = DESK_EDGE_MARGIN):
    """**Tabletop** desk bounds (scene coordinates) → ((x0,x1), (y0,y1)) — for a caller that has no tree; with a tree,
    `desk_bounds_scene_for(tree)`. `shrink_robot` trims that much more off the robot-side (−x) edge."""
    return _desk_box(desk_top_for(None), shrink_robot, margin)


def desk_bounds_dataset(dataset: str | None, shrink_robot: float = 0.0, margin: float = DESK_EDGE_MARGIN) -> tuple:
    """A dataset's scene desk bounds → ((x0, x1), (y0, y1)) — `desk_top_for(dataset)` as a box, trimmed by `margin`
    on every edge and `shrink_robot` more on the robot-side (−x) edge. For a caller that knows its dataset but holds
    no tree yet; with a tree, `desk_bounds_scene_for`."""
    return _desk_box(desk_top_for(dataset), shrink_robot, margin)


def desk_box_for(dataset: str | None) -> tuple:
    """The desk top's scene box of a dataset → ((x0, x1), (y0, y1)), margin 0."""
    return desk_bounds_dataset(dataset, 0.0, 0.0)


def tree_dataset(tree_dir=None, ti: dict | None = None) -> str | None:
    """A tree's dataset (task_info `dataset_name`) — None only when there is no tree at all. A tree without the field
    raises (the desk has no default dataset)."""
    if ti is None:
        if tree_dir is None:
            return None
        import json as _json
        from pathlib import Path as _Path

        ti = _json.loads((_Path(tree_dir) / "task_info.json").read_text())
    ds = ti.get("dataset_name")
    if not ds:
        raise ValueError(f"{tree_dir}: task_info has no dataset_name — cannot pick the desk top (workspace)")
    return str(ds)


def desk_bounds_scene_for(
    tree_dir=None,
    shrink_robot: float = 0.0,
    margin: float = DESK_EDGE_MARGIN,
    ti: dict | None = None,
):
    """Per-tree **scene desk bounds**: the one box of the tree's dataset's workspace profile (`desk_top_for`), trimmed
    by `margin` on every edge and `shrink_robot` more on the robot-side (−x) edge. No tree (tree_dir · ti both None) =
    the tabletop box. The env spawns that same top (one desk per run), so judging, baking, gates and rendering see the
    desk the policy trains at."""
    return desk_bounds_dataset(tree_dataset(tree_dir, ti), shrink_robot, margin)


def _arm_registry() -> dict:
    """The arm registry (`dexengine.core.arms.ARM_REGISTRY`)."""
    from dexengine.core.arms import ARM_REGISTRY  # noqa: PLC0415

    return ARM_REGISTRY


def robot_base_scene(arm: str, xy_only: bool = True) -> np.ndarray:
    """Robot base (**scene coordinates**) of ``arm`` from its ArmSpec — never parsed or hardcoded. ArmSpec honours the
    `DEXENGINE_<ARM>_BASE_POS` env, so the placement solver, IK, training and rendering all see the same value."""
    reg = _arm_registry()
    spec = reg.get(arm)
    if spec is None:
        raise KeyError(
            f"arm '{arm}' is not in ARM_REGISTRY (known: {sorted(reg)}) — do not work around it with a hardcoded base"
        )
    if spec.base_pos is None:
        raise ValueError(
            f"arm '{arm}' has no base (base_pos) — a floating hand; ask the scene's robot spot with the "
            f"data-world arm (robot_keepout does)"
        )
    b = np.asarray(spec.base_pos, dtype=float)
    return b[:2] if xy_only else b


def robot_keepout(tree_dir=None, arm: str | None = None) -> tuple:
    """The robot's standing spot in scene xy → (cx, cy, r): the disc around the robot base (robot_base_scene) whose rim
    is the desk's robot-side edge (desk_bounds_scene_for(tree, margin 0)). Nothing else stands inside the disc — the
    one definition of "an object overlaps the robot" (the spawn rule extra_slots · the judge's mask static_fixed).
    The arm = `arm`, else env DEXENGINE_ARM, else the registered arm the tree's combo folder is named after
    (paths.seq_dir_identity of the tree root); none of the three → raises (no default arm)."""
    import os as _os
    import pathlib as _pl

    a = arm or _os.environ.get("DEXENGINE_ARM")
    if not a and tree_dir is not None:
        combo = _paths().seq_dir_identity(_pl.Path(tree_dir).resolve().parent)[0]
        a = next((n for n in sorted(_arm_registry(), key=len, reverse=True) if combo.startswith(n + "_")), None)
    if not a:
        raise ValueError(
            f"{tree_dir}: unknown arm (no argument, no DEXENGINE_ARM, no combo folder on the tree) — "
            f"cannot place the robot's standing spot"
        )
    # The standing spot is a question about the scene (the tree), asked with the arm of the data world: a floating
    # hand has no base but reads the rby1 world's trees and placements, so it must see the same scene.
    from dexengine.core.floating import data_world_arm  # noqa: PLC0415

    a = data_world_arm(a)
    b = robot_base_scene(a)
    (x0, _x1), _ = desk_bounds_scene_for(tree_dir, margin=0.0)
    r = float(x0) - float(b[0])
    if not r > 0.0:
        raise ValueError(
            f"robot base x {float(b[0]):.3f} lies on the desk side of the desk's robot edge "
            f"{float(x0):.3f} — cannot place the robot's standing spot"
        )
    return float(b[0]), float(b[1]), r


def in_robot_keepout(xy_scene, keepout: tuple) -> bool:
    """Any of the scene xy points (N, 2) lies inside the robot's standing spot (robot_keepout)."""
    cx, cy, r = keepout
    xy = np.asarray(xy_scene, float)
    return bool((np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) < r).any())


def place_objects(obj_xy_cap: np.ndarray, dx: float, dy: float, yaw_deg: float, pivot) -> np.ndarray:
    """Object xy with the placement applied (**capture coordinates**) — the bake's `_tf_points` formula:

        q = (p − pivot) @ Rz.T + pivot + (dx,dy)

    Every placement check goes through this function so judging and baking cannot diverge."""
    Rz = _Rot.from_euler("z", yaw_deg, degrees=True).as_matrix()[:2, :2]
    pv = np.asarray(pivot, float)[:2]
    return (np.asarray(obj_xy_cap, float) - pv) @ Rz.T + pv + np.array([dx, dy], float)


def place_pose(pose, dx: float, dy: float, yaw_deg: float, pivot) -> list:
    """An object's 7-pose (capture x·y·z · wxyz) with the placement applied — position through place_objects (z is
    untouched: the placement is planar), orientation Rz(yaw)·R (the bake's left-composition). The judge places a base
    tree's fixed objects with this, so its guard sees them where the bake puts them."""
    p = np.asarray(pose, float)
    if p.shape != (7,):
        raise ValueError(f"pose {list(p.ravel())[:7]} — must be 7 values x·y·z · wxyz")
    xy = place_objects(p[None, :2], dx, dy, yaw_deg, pivot)[0]
    r = (_Rot.from_euler("z", yaw_deg, degrees=True) * _Rot.from_quat([p[4], p[5], p[6], p[3]])).as_quat()
    return [float(xy[0]), float(xy[1]), float(p[2]), float(r[3]), float(r[0]), float(r[1]), float(r[2])]
