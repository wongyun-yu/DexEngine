"""Desk statement <-> measurement: is the desk laid out the desk the judge reads (statement owner
`dexengine.core.desk.desk_top`).

An independent oracle comparing a measurement (the prim bounds on the live stage) with the statement (top x_off ·
half widths); values are not restated.
Caller: `ArmResidualTrainEnv.load_managers` (before the collision audit; one-hand and bimanual, training, judge and
render alike).
"""

from __future__ import annotations

from dexengine.core import desk as _dp
from dexengine.rl.tasks.scene.object_pin import DESK_PRIM

DESK_VISUAL_PRIM = "Desk"
DESK_AUDIT_TOL_M = (
    2e-3  # allowed statement <-> live bounds difference (covers ≈ 0.2 mm from rounding a 90° rotation quaternion)
)


def _prim_xy_box(env, leaf: str):
    """Live bounds (x0, x1, y0, y1) of env 0's prim `leaf` in env-local coordinates (invisible prims measured too). None
    without it."""
    from pxr import Usd, UsdGeom

    prim = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[0]}/{leaf}")
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], False, True)  # visibility ignored
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    o = [float(v) for v in env.scene.env_origins[0].tolist()]
    lo, hi = r.GetMin(), r.GetMax()
    return (lo[0] - o[0], hi[0] - o[0], lo[1] - o[1], hi[1] - o[1])


def audit_desk(env) -> None:
    """Desk statement <-> measurement: is the desk laid out the desk the judge reads (owner
    `dexengine.core.desk.desk_top`).
    (1) Bundle: the top of every tree's dataset = this run's top (all of them; a mix fails). (2) Measurement: the live
    x·y bounds of env 0's collision top and visual desk equal the statement box within `DESK_AUDIT_TOL_M` (otherwise it
    fails: the visible desk and the colliding desk split). A scene without a desk entity prints that. Marker
    `[DESK_SIZE]`."""
    mark = _dp.DESK_SIZE_MARK
    loader = getattr(getattr(env, "trajectory_manager", None), "_loader", None)
    datasets = getattr(loader, "clip_datasets", None)
    if datasets is None:
        raise RuntimeError(f"{mark} cannot read the bundle's tree datasets (trajectory_manager._loader.clip_datasets)")
    print(_dp.check_bundle(datasets), flush=True)
    x, hx, hy = _dp.desk_top()
    want = (x - hx, x + hx, -hy, hy)

    def _xy(b):
        return f"x[{b[0]:.4f},{b[1]:.4f}] y[{b[2]:.4f},{b[3]:.4f}]"

    said = []
    for label, key, leaf in (
        ("collision top", "desk_collision_top", DESK_PRIM),
        ("visual desk", "desk", DESK_VISUAL_PRIM),
    ):
        if getattr(env.cfg.scene, key, None) is None:
            said.append(f"{label} missing")
            continue
        box = _prim_xy_box(env, leaf)
        if box is None:
            raise RuntimeError(f"{mark} {label}: in the scene cfg but prim {leaf} is missing")
        err = max(abs(a - b) for a, b in zip(box, want, strict=True))
        if err > DESK_AUDIT_TOL_M:
            raise RuntimeError(
                f"{mark} {label} measured {_xy(box)} ≠ statement {_xy(want)} (difference {err * 1000:.1f} mm > "
                f"{DESK_AUDIT_TOL_M * 1000:.0f} mm) — the desk laid out is not the judge's desk"
            )
        said.append(f"{label} {_xy(box)} difference {err * 1000:.1f} mm")
    print(
        f"{mark} measured = statement {_xy(want)}: "
        + " · ".join(said)
        + f" (allowed {DESK_AUDIT_TOL_M * 1000:.0f} mm)",
        flush=True,
    )
