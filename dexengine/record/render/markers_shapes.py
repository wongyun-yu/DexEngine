"""Prim geometry of the place marker under ROOT (render.markers_scene builds the scene and the record): plate meshes
(prism, band_mesh), the plate height rules (pad_base_z, check_pad_top) and the outline/contour, X tape and pad plate
builders."""

from __future__ import annotations

import math

from dexengine.core.errors import InputMissingError
from dexengine.record.render.markers import LINE_GAP, LINE_W, MARK
from dexengine.record.render.markers_geom import convex_hull, radial, ray_exit, support_top

PLATE_T = 0.001  # thickness (m)
LIFT = 0.001  # lift above the end-pose object bottom: top = supporting surface + 2 mm
# X marker (the shape taped with electrical tape on the real robot): two strips crossing at 90°, ±45° to the robot's
# front (env +x)
X_TAPE_W = 0.019  # strip width (m) = real electrical tape width 19 mm
# arm end = centre + r + X_PAD: the same rule as the ring's outer radius (r = half the longer object xy extent)
X_PAD = LINE_GAP + LINE_W
# stay inside the support top (tape cannot be stuck in the air on the real robot):
#   arm length = min(rule above, length inside the support top - X_SUPPORT_MARGIN). Support top = the xy convex hull of
#   the smallest scene rigid body with vertices at end-pose object bottom z ± X_SUPPORT_ZTOL (robot and main object
#   excluded, centre inside that top) (markers_geom.support_top). On a desk (no top vertices within the search
#   radius) the rule above as it is.
X_SUPPORT_MARGIN = 0.01
PAD_R = 0.06  # support pad radius (m): the same for every clip (one size and shape)
ROOT = "/World/CollectLook/place_marker"


def _mesh(stage, path: str, pts, counts, idx):
    import numpy as np

    from pxr import UsdGeom, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pts, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(list(map(int, counts))))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(list(map(int, idx))))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    return m


def _prism(stage, path: str, poly, z0: float, z1: float):
    """Convex polygon (counter-clockwise xy) plate: bottom z0, top z1 (marker Xform frame)."""
    n = len(poly)
    pts = [(x, y, z0) for x, y in poly] + [(x, y, z1) for x, y in poly]
    counts, idx = [n, n], list(range(n - 1, -1, -1)) + list(range(n, 2 * n))  # bottom (-z side) · top (+z side)
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [i, j, n + j, n + i]
    return _mesh(stage, path, pts, counts, idx)


def band_mesh(stage, path: str, inner, outer, z0: float, z1: float):
    """Closed band plate (inner ring inner, outer ring outer: same count, same direction): top, bottom, inner and outer
    walls."""
    n = len(inner)
    # 0 inner bottom · n outer bottom · 2n inner top · 3n outer top
    L = [(x, y, z) for z in (z0, z1) for ring in (inner, outer) for x, y in ring]
    counts, idx = [], []
    for i in range(n):
        j = (i + 1) % n
        for q in (
            (2 * n + i, 3 * n + i, 3 * n + j, 2 * n + j),  # top
            (i, j, n + j, n + i),  # bottom
            (n + i, n + j, 3 * n + j, 3 * n + i),  # outer wall
            (i, 2 * n + i, 2 * n + j, j),
        ):  # inner wall
            counts.append(4)
            idx += list(q)
    return _mesh(stage, path, L, counts, idx)


def pad_base_z(zb: float, desk_z: float) -> float:
    """Base height of a desk plate (plate = base + LIFT … + LIFT + PLATE_T).
    Rule: when a plate placed on the object's end bottom would have its top below the desk top (buried in the desk),
    base = desk top, else the object's end bottom. Aerial scenes have the desk sunk (desk_z << zb), so unchanged.
    place_target and judge values are neither used nor changed."""
    if not (math.isfinite(zb) and math.isfinite(desk_z)):
        raise SystemExit(f"{MARK} plate height inputs are not numbers zb={zb} desk={desk_z}")
    return float(desk_z) if zb + LIFT + PLATE_T < desk_z else float(zb)


def check_pad_top(top_z: float, surface_z: float, what: str) -> None:
    """A plate top below the surface it lies on is invisible in the image: the session dies instead of passing."""
    if not top_z >= surface_z:
        raise SystemExit(f"{MARK} plate top {top_z:.4f} below {what} {surface_z:.4f}: the plate is invisible")


def _desk_top_z(raw_env, stage) -> float:
    """Current world top z of the desk visual prim (sunk on aerial scenes). Owner of the prim name =
    dexengine.rl.tasks.scene.desk_check.DESK_VISUAL_PRIM."""
    from pxr import Usd, UsdGeom

    from dexengine.rl.tasks.scene.desk_check import DESK_VISUAL_PRIM

    prim = stage.GetPrimAtPath(f"{raw_env.scene.env_prim_paths[0]}/{DESK_VISUAL_PRIM}")
    if not prim.IsValid():
        raise InputMissingError(
            f"{MARK} desk prim {DESK_VISUAL_PRIM} missing: the plate bottom rule cannot be measured"
        )
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    return float(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2])


def outline_contour(stage, s, n, foot, band, zmin, z0, z1):
    """Outline (filled hull plate) or contour (band around the hull) of the bottom-band footprint -> (xy, geo)."""
    import numpy as np

    poly = convex_hull(foot)
    if len(poly) < 3:
        raise SystemExit(f"{MARK} {n}: bottom band points {len(foot)} -> hull vertices {len(poly)} < 3")
    if s == "outline":
        _prism(stage, f"{ROOT}/{n}/plate", poly, z0, z1)
        xy = np.array(poly)
    else:
        pc = np.array(poly).mean(0)
        xy = radial(poly, pc, LINE_GAP + LINE_W)
        band_mesh(stage, f"{ROOT}/{n}/line", radial(poly, pc, LINE_GAP), xy, z0, z1)
    geo = {"hull_vertices": len(poly), "band_m": round(band - zmin, 4), "foot_points": int(len(foot))}
    return xy, geo


def x_tape(raw_env, stage, n, wp, zmin, c, r, rp, z0, z1):
    """X marker: two tape strips at ±45° with arm = centre + r + X_PAD, cut to stay inside the support top ->
    (xy, geo)."""
    import numpy as np

    h0, w = r + X_PAD, 0.5 * X_TAPE_W
    cw = wp[:2] + c
    sup, sup_path = support_top(raw_env, stage, float(wp[2] + zmin), cw, (rp,))
    h_fit = None
    if sup is not None:
        dirs = [
            np.array([math.cos(g), math.sin(g)]) for g in (math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4)
        ]
        h_fit = min(ray_exit(sup, cw + o * np.array([-d_[1], d_[0]]) * w, d_) for d_ in dirs for o in (1, -1))
        h_fit -= X_SUPPORT_MARGIN
    h = h0 if h_fit is None else min(h0, h_fit)
    if h <= w:
        raise SystemExit(
            f"{MARK} {n}: arm inside the support top {h:.3f} m <= band half width {w}: the X does not fit({sup_path})"
        )
    corners = []
    for i, ang in enumerate((math.pi / 4, -math.pi / 4)):
        d_ = np.array([math.cos(ang), math.sin(ang)])
        nrm = np.array([-d_[1], d_[0]])
        poly = [c + h * d_ + w * nrm, c - h * d_ + w * nrm, c - h * d_ - w * nrm, c + h * d_ - w * nrm]
        _prism(stage, f"{ROOT}/{n}/tape{i}", poly, z0, z1)
        corners += poly
    xy = np.array(corners)
    geo = {
        "radius_m": round(r, 4),
        "arm_m": round(h, 4),
        "arm_rule_m": round(h0, 4),
        "arm_fit_m": None if h_fit is None else round(h_fit, 4),
        "support": sup_path,
        "tape_w_m": X_TAPE_W,
        "center_off_m": [round(float(x), 4) for x in c],
    }
    return xy, geo


def pad_plate(raw_env, stage, n, wp, zmin, c, r, u, sup, sup_path, sup_z, sup_tops):
    """Pad: one-size cylinder plate on the support top (or on the desk, raised out of it by pad_base_z) ->
    (xy, geo, z0, z1)."""
    from pxr import Gf, UsdGeom

    zb = float(wp[2] + zmin)  # end-pose object bottom (world)
    # on a support: plate = highest support top below the plate centre + LIFT
    if sup is not None:
        z0 = (sup_z - zb) + LIFT
        z1 = z0 + PLATE_T
        check_pad_top(zb + z1, sup_z, "support top")
    # on the desk: buried -> raise to the desk top (pad_base_z)
    else:
        desk_z = _desk_top_z(raw_env, stage)
        base = pad_base_z(zb, desk_z)
        z0 = (base - zb) + LIFT
        z1 = z0 + PLATE_T
        print(
            f"{MARK} desk plate bottom rule applied · object end bottom {zb:.4f} · desk top "
            f"{desk_z:.4f} · plate base {base:.4f} · raised {(base - zb) * 1000:.1f} mm",
            flush=True,
        )
        check_pad_top(zb + z1, desk_z, "desk top")
    cyl = UsdGeom.Cylinder.Define(stage, f"{ROOT}/{n}/pad")
    cyl.CreateRadiusAttr(PAD_R)
    cyl.CreateHeightAttr(PLATE_T)
    cyl.CreateAxisAttr("Z")
    UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(float(c[0]), float(c[1]), z0 + 0.5 * PLATE_T))
    xy = c + PAD_R * u
    over = None
    # distance outside our support pad top (one size, not clamped)
    if sup is not None:
        cwp = wp[:2] + c
        over = round(max(0.0, PAD_R - min(ray_exit(sup, cwp, ud) for ud in u[::8])), 4)
    geo = {
        "pad_r_m": PAD_R,
        "radius_obj_m": round(r, 4),
        "support": sup_path,
        "overhang_m": over,
        "support_top_z": None if sup is None else round(sup_z, 4),
        "support_tops": sup_tops,
        # plate support top - object bottom (+ = object sunk by that much)
        "sink_m": None if sup is None else round(z0 - LIFT, 4),
        "center_off_m": [round(float(x), 4) for x in c],
    }
    return xy, geo, z0, z1
