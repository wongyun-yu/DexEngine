"""Desk placement — the names and the code that applies them, in one place.

Why
    The collision top, the four legs and the visual desk must move and resize as one piece. A move that leaves the
    legs behind leaves **invisible (visible=False) collision pillars** standing away from the top.

Moving the legs with the top changes no physics
    A URDF FK scan of both wrists' reachable space (4000 samples) never comes within **42.0 cm** of the nearest leg
    center (leg half-width 2 cm). Objects sit near the trajectory (within robot reach), so they never meet the legs
    either.

Geometry (defined only here)
    Top collision box center z=0.745, thickness 0.03 -> **top surface 0.76 m** = traj_offset_z.
    Legs at x ±0.32, y ±0.65 from the top center, center z=0.365 (height 0.73).

Visual desk (one: the Willow table — dexengine.rl.tasks.scene.desk_visual)
    The desk collides only through the cuboids above. The visual is the unit-scale Willow asset scaled per axis.
"""

from __future__ import annotations

# --- X placement: two values, each named for its use ---------------------------
DESK_X_HANDTRACK = 0.50
"""Hand-tracking (arm-IK) scene. The trajectory is at X≈0 and the desk is **visual only**, so it may sit far."""

DESK_X_TASK = 0.03
"""Center x of the tabletop top in task scenes — used by eval and also by **residual (task) training**. Objects
must rest on the top. A run reads its value through `desk_top()`, not from here (each workspace has its own top)."""

# --- Height / size -------------------------------------------------------------
DESK_TOP_CENTER_Z = 0.745
DESK_TOP_SURFACE_Z = 0.76  # = DESK_TOP_CENTER_Z + thickness 0.03/2, equals traj_offset_z
DESK_TOP_THICK = 0.03
DESK_TOP_HALF_X = 0.38  # world half-width x of the top for front-of-table capture (workspace tabletop)
DESK_TOP_HALF_Y = 0.70  # world half-width y (the scene cfg CuboidCfg size is rotated 90°: (2*hy, 2*hx, thick))
DESK_LEG_DX = 0.32
DESK_LEG_DY = 0.65
DESK_LEG_CENTER_Z = 0.365
DESK_LEG_HALF = 0.02  # leg half-width — how close a leg center may get to the top center
DESK_VISUAL_UNIT = 0.01  # unit scale of the Willow desk USD (cm) — at this scale its footprint = tabletop top
#: Name of the desk-top prim under {ENV}/Desk. Look D and the visual augmentation bind the desk material to it
#: (dexengine.record.looks.materials.desk_woods, DEXENGINE_AUG_DESK_MDL); the Willow asset names its top so.
DESK_TOP_PRIM = "Wood"
#: The removed desk choice (scene section · --set scene.desk / scene.desk_usd): given at all = error (the launcher ·
#: desk_visual.desk_cfg).
REMOVED_DESK_ENV = ("DEXENGINE_DESK", "DEXENGINE_DESK_USD")

# --- Top (center x · size): one per workspace ------------------------------------------
#: Top of room-scale capture (room) — **the same value as the tabletop top**. Placement proxy over 4,627 HOT3D clips
#: with 49 candidate sizes (depth 0.30–0.76 × width 0.40–1.40 m, robot-side edge fixed): a smaller depth never helps
#: (0.76 is always best); a 1.00 m width gains +4 … +55 clips (0.1–1.2 %) whose hands/objects going below the
#: surface can stay off the desk, but 164 clips then get static objects the desk cannot hold (−774 … −1,300 if
#: narrower). Changing the value means re-measuring the IK pool's arm/hand clearances against that top — until then
#: dexengine.placement.reach_map.clearance.check_desk_top stops pool loading.
ROOM_DESK_TOP = (DESK_X_TASK, DESK_TOP_HALF_X, DESK_TOP_HALF_Y)

#: Workspace (`dexengine.core.layout.DatasetSpec.workspace`) → top (x_off, half_x, half_y) [m · scene frame] — the
#: one owner of the scene desk's **center x and size** (center y = 0). Judge · bake · gate (placement coords read this
#: module) and env (collision top · legs · visual desk) all read `desk_top_for`. A top must lie inside the tabletop
#: top — the arm IK pool and the visual asset are valid only there (checked at import). The desk–robot relative
#: height does not depend on the workspace (DESK_TOP_SURFACE_Z).
DESK_TOP_BY_WORKSPACE: dict[str, tuple[float, float, float]] = {
    "tabletop": (DESK_X_TASK, DESK_TOP_HALF_X, DESK_TOP_HALF_Y),
    "room": ROOM_DESK_TOP,
}
_EPS = 1e-9
if (
    abs(DESK_TOP_CENTER_Z + DESK_TOP_THICK / 2 - DESK_TOP_SURFACE_Z) > _EPS
    or abs(DESK_TOP_SURFACE_Z - DESK_TOP_THICK - 2 * DESK_LEG_CENTER_Z) > _EPS
):
    raise ValueError(
        "dexengine.core.desk: the top (center z, thickness, surface) and the legs (center z) do not "
        "meet — the legs must reach from the floor to the underside of the top"
    )
if _bad := {
    k: v
    for k, v in DESK_TOP_BY_WORKSPACE.items()
    if not (
        v[1] > DESK_LEG_HALF
        and v[2] > DESK_LEG_HALF
        and v[2] <= DESK_TOP_HALF_Y + _EPS
        and v[0] - v[1] >= DESK_X_TASK - DESK_TOP_HALF_X - _EPS
        and v[0] + v[1] <= DESK_X_TASK + DESK_TOP_HALF_X + _EPS
    )
}:
    raise ValueError(
        f"dexengine.core.desk.DESK_TOP_BY_WORKSPACE: tops outside the tabletop top (x {DESK_X_TASK} ± "
        f"{DESK_TOP_HALF_X}, y ± {DESK_TOP_HALF_Y}) or with a half-width at or below the leg half-width "
        f"{DESK_LEG_HALF}: {_bad}"
    )

DATASET_ENV = "DEXENGINE_DATASET"  # the run's dataset — an explicit input (the canonical entry point requires it)
DESK_SIZE_MARK = "[DESK_SIZE]"


def _layout(layout=None):
    """dexengine.core.layout (the dataset table), imported inside the package. A caller that loaded this file by
    path passes the layout module it already loaded as `layout` — the file is not loaded again here."""
    if layout is not None:
        return layout
    try:
        from dexengine.core import layout as _lay
    except ImportError:
        raise ImportError(
            "[desk] the desk module loaded outside the package does not read the layout by itself — "
            "pass layout=<the layout module> (dexengine.core.paths.data_layout)"
        ) from None
    return _lay


def desk_workspace(dataset: str, layout=None) -> str:
    """The dataset's workspace (= key of the top table). An unregistered dataset or a workspace with no top in the
    table raises — an unknown dataset never silently gets the tabletop desk."""
    specs = _layout(layout).DATASETS
    if dataset not in specs:
        raise KeyError(
            f"[desk] unknown dataset {dataset!r} — not in dexengine.core.layout.DATASETS (registered: {sorted(specs)})"
        )
    ws = specs[dataset].workspace
    if ws not in DESK_TOP_BY_WORKSPACE:
        raise KeyError(f"[desk] workspace {ws!r} ({dataset}) has no top in DESK_TOP_BY_WORKSPACE")
    return ws


def desk_top_for(dataset: str, layout=None) -> tuple[float, float, float]:
    """The dataset's scene desk top (x_off, half_x, half_y) [m] — what judge · bake · env read (`layout`: see
    `_layout`)."""
    return DESK_TOP_BY_WORKSPACE[desk_workspace(dataset, layout)]


def run_dataset(env=None) -> str:
    """This run's dataset (env `DEXENGINE_DATASET`). Missing → raises — the desk has no default dataset."""
    import os

    ds = (os.environ if env is None else env).get(DATASET_ENV, "").strip()
    if not ds:
        raise RuntimeError(
            f"[desk] {DATASET_ENV} is not set — the run's dataset decides its desk (no default). "
            "Canonical entry point: scripts/stage3_rl.sh"
        )
    return ds


def desk_top(env=None) -> tuple[float, float, float]:
    """This run's top = `desk_top_for(this run's dataset)`. One desk per run — a bundle tree from a dataset with
    another top makes the loader fail (`check_bundle`)."""
    return desk_top_for(run_dataset(env))


def desk_half(env=None) -> tuple[float, float]:
    """This run's top half-widths (half_x, half_y) — the size part of `desk_top`."""
    return desk_top(env)[1:]


def desk_size_mark(dataset: str, layout=None) -> str:
    """Marker `[DESK_SIZE] profile=<workspace> dataset=<ds> x_off=<x> half=(x,y) src=<owner>` — printed by whoever
    lays the desk."""
    x, hx, hy = desk_top_for(dataset, layout)
    return (
        f"{DESK_SIZE_MARK} profile={desk_workspace(dataset, layout)} dataset={dataset} x_off={x:.3f} "
        f"half=({hx:.3f},{hy:.3f}) src=desk_placement.desk_top_for"
    )


def check_bundle(tree_datasets, env=None) -> str:
    """Check every tree of the bundle: the top of the dataset it records (task_info `dataset_name`) must equal this
    run's top. Any other top raises — the judge measured that clip at another desk. A tree with no dataset raises too.
    Returns one marker line."""
    run_ds = run_dataset(env)
    want = desk_top_for(run_ds)
    rows = list(tree_datasets)
    if not rows:
        raise ValueError("[desk] empty bundle — no tree to check the top against")
    if blank := [i for i, ds in enumerate(rows) if not ds]:
        raise ValueError(
            f"[desk] {len(blank)}/{len(rows)} trees of the bundle have no dataset_name (first indices "
            f"{blank[:3]}) — the desk cannot be decided"
        )
    other: dict[str, int] = {}
    for ds in rows:
        if desk_top_for(str(ds)) != want:
            other[str(ds)] = other.get(str(ds), 0) + 1
    if other:
        raise ValueError(
            f"[desk] the bundle mixes in datasets with another top: run {run_ds} top {want} ↔ "
            f"{ {k: (n, desk_top_for(k)) for k, n in other.items()} } — one bundle, one desk"
        )
    seen = sorted({str(ds) for ds in rows})
    return (
        f"{DESK_SIZE_MARK} bundle check: datasets {seen} of trees {len(rows)}/{len(rows)} top = run ({run_ds}) "
        f"x_off={want[0]:.3f} half=({want[1]:.3f},{want[2]:.3f})"
    )


def _top_half(scene_cfg):
    """The half-widths (half_x, half_y) this scene cfg's collision top carries — None without a top."""
    top = getattr(scene_cfg, "desk_collision_top", None)
    if top is None:
        return None
    size_y, size_x, _t = top.spawn.size  # frame rotated 90°
    return size_x / 2, size_y / 2


def _leg_offsets(half_x: float, half_y: float) -> tuple[float, float]:
    """Leg center (dx, dy) from the top center — keeps the inset from the edges (DESK_LEG_DX/DY at the tabletop
    half-widths)."""
    return (
        max(DESK_LEG_HALF, DESK_LEG_DX - (DESK_TOP_HALF_X - half_x)),
        max(DESK_LEG_HALF, DESK_LEG_DY - (DESK_TOP_HALF_Y - half_y)),
    )


def _place_legs(scene_cfg, x: float, half_x: float, half_y: float) -> int:
    dx, dy = _leg_offsets(half_x, half_y)
    n = 0
    for key, (sx, sy) in zip(_LEG_KEYS, _LEG_SIGNS, strict=True):
        leg = getattr(scene_cfg, key, None)
        if leg is None:
            continue
        leg.init_state.pos = (sx * dx + x, sy * dy, DESK_LEG_CENTER_Z)
        n += 1
    return n


def apply_desk_half(scene_cfg, half_x: float, half_y: float) -> str:
    """Fit the desk (collision top + legs + visual) to the half-widths **as one piece** — the visible desk and the
    colliding desk have the same size. Only what exists is fitted; returns a string naming what was. The Willow desk
    scales the unit-scale asset per axis (asset x = world y)."""
    done = []
    top = getattr(scene_cfg, "desk_collision_top", None)
    if top is not None:
        top.spawn.size = (2 * half_y, 2 * half_x, DESK_TOP_THICK)  # frame rotated 90°
        done.append("top")
        if n := _place_legs(scene_cfg, top.init_state.pos[0], half_x, half_y):
            done.append(f"legs×{n}")
    desk = getattr(scene_cfg, "desk", None)
    if desk is not None:
        desk.spawn.scale = (
            DESK_VISUAL_UNIT * (half_y / DESK_TOP_HALF_Y),
            DESK_VISUAL_UNIT * (half_x / DESK_TOP_HALF_X),
            DESK_VISUAL_UNIT,
        )
        done.append("visual")
    return f"desk_half=({half_x:.3f},{half_y:.3f}) ({', '.join(done) if done else 'nothing'})"


_LEG_KEYS = ("desk_leg_0", "desk_leg_1", "desk_leg_2", "desk_leg_3")
_LEG_SIGNS = ((-1, +1), (-1, -1), (+1, +1), (+1, -1))  # order of legs 0..3 in the scene cfg


def apply_desk_x(scene_cfg, x: float) -> str:
    """Move the desk (visual USD + collision top + 4 legs) to x **as one piece**.

    Only what exists is moved, so scenes without legs (other dataset cfgs) are safe; returns a string naming what
    moved — a caller that logs it records the placement. The legs follow the edges of the half-widths the top
    carries (as written by `apply_desk_half`).
    """
    moved = []
    if hasattr(scene_cfg, "desk"):
        scene_cfg.desk.init_state.pos = (x, 0.0, 0.0)
        moved.append("desk")
    if hasattr(scene_cfg, "desk_collision_top"):
        scene_cfg.desk_collision_top.init_state.pos = (x, 0.0, DESK_TOP_CENTER_Z)
        moved.append("top")
    half = _top_half(scene_cfg)
    if half is None and any(getattr(scene_cfg, k, None) is not None for k in _LEG_KEYS):
        raise ValueError("[desk] legs but no collision top — no half-width to place the legs by")
    n_legs = _place_legs(scene_cfg, x, *half) if half is not None else 0
    if n_legs:
        moved.append(f"legs×{n_legs}")
    return f"desk_x={x} ({', '.join(moved) if moved else 'nothing'}), top surface z={DESK_TOP_SURFACE_Z}"


def apply_desk_top(scene_cfg, top: tuple[float, float, float]) -> str:
    """Fit the whole desk to the top statement (x_off, half_x, half_y) — size first, then center (the legs follow
    the new edges)."""
    x, half_x, half_y = top
    size = apply_desk_half(scene_cfg, half_x, half_y)
    return f"{apply_desk_x(scene_cfg, x)} | {size}"
