"""Pure geometry and stage queries of the place marker (outline hull, radial offsets, support top)."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys

from dexengine.core.errors import InputMissingError
from dexengine.record.render.markers import MARK, ref_file

REF_MD5 = "5ea78940bb9ec9fccbe97531b8990a9b"
REF_MODULE = "e5_place_ref"  # module name of the frozen copy (the name the old sys.path import used)


def load_ref():
    """The frozen place single source REF_DIR/e5_place_ref.py, md5-checked against REF_MD5. Loaded by file location
    under its module name and cached in sys.modules (no sys.path change)."""
    f = ref_file()
    if not f.is_file():
        raise InputMissingError(f"{MARK} place single source {f} is missing (no fallback)")
    md5 = hashlib.md5(f.read_bytes()).hexdigest()
    if md5 != REF_MD5:
        raise SystemExit(f"{MARK} {f} md5 {md5} != frozen value {REF_MD5}")
    sys.dont_write_bytecode = True  # no __pycache__ in the frozen copy (read only)
    PR = sys.modules.get(REF_MODULE)
    if PR is None:
        spec = importlib.util.spec_from_file_location(REF_MODULE, f)
        PR = importlib.util.module_from_spec(spec)
        sys.modules[REF_MODULE] = PR
        spec.loader.exec_module(PR)
    return PR


NSEG = 96  # contour and ring perimeter segments (one vertex per direction)
# pad support top search: highest support point up to this height above the object bottom (the highest top of
#   overlapping supports)
PAD_TOP_REACH = 0.05
X_SUPPORT_REACH = 0.6  # xy radius (m) to look for support vertices
X_SUPPORT_ZTOL = 0.006


def convex_hull(xy):
    """2D convex hull (counter-clockwise, monotone chain)."""
    pts = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in xy})
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def radial(poly, c, d: float):
    """Points of the perimeter of convex polygon poly rounded outward by d, measured from c in NSEG directions
    ([NSEG, 2], counter-clockwise). Widening = hull of a radius-d circle at every vertex (Minkowski sum); per direction
    the intersection of that hull with the ray from c."""
    import numpy as np

    P = np.array(poly, float)
    if d > 0:
        a = np.linspace(0, 2 * math.pi, 32, endpoint=False)
        P = np.array(convex_hull((P[:, None, :] + d * np.stack([np.cos(a), np.sin(a)], -1)[None]).reshape(-1, 2)))
    out = []
    for th in np.linspace(0, 2 * math.pi, NSEG, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)])
        best = None
        for i in range(len(P)):
            a0, b0 = P[i] - c, P[(i + 1) % len(P)] - c
            e = b0 - a0
            den = u[0] * (-e[1]) - u[1] * (-e[0])
            if abs(den) < 1e-12:
                continue
            t = (a0[0] * (-e[1]) - a0[1] * (-e[0])) / den
            s = (u[0] * a0[1] - u[1] * a0[0]) / den
            if t > 0 and -1e-9 <= s <= 1 + 1e-9:
                best = t if best is None else min(best, t)
        if best is None:
            raise InputMissingError(f"{MARK} no perimeter ray intersection (centre outside the footprint?)")
        out.append(c + best * u)
    return np.array(out)


def object_meshes(stage, root_path: str):
    """Visible visual meshes under the object root (purposes default and render, inside instances too) -> [(path,
    points [n,3] in the object body frame, root scale included)]."""
    import numpy as np

    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise InputMissingError(f"{MARK} object root {root_path} missing")
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    M = UsdGeom.Xformable(root).GetLocalTransformation()
    # root scale (read the same way as render.visibility object_boxes)
    sc = np.array([np.linalg.norm([M[i][0], M[i][1], M[i][2]]) for i in range(3)])
    out = []
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not p.IsA(UsdGeom.Mesh):
            continue
        im = UsdGeom.Imageable(p)
        if im.ComputeVisibility() == UsdGeom.Tokens.invisible or im.ComputePurpose() not in (
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
        ):
            continue
        pts = UsdGeom.Mesh(p).GetPointsAttr().Get()
        if not pts:
            continue
        rel, _ = xc.ComputeRelativeTransform(p, root)
        R = np.array([[rel[i][j] for j in range(4)] for i in range(4)])
        out.append((str(p.GetPath()), (np.asarray(pts, float) @ R[:3, :3] + R[3, :3]) * sc))  # row-vector form (Gf)
    if not out:
        raise SystemExit(f"{MARK} 0 visible visual meshes under {root_path}")
    return out


def ray_exit(poly, c, d) -> float:
    """Distance from point c inside a convex polygon along direction d to the edge (0 if none)."""
    best = None
    for i in range(len(poly)):
        a_, e = poly[i], poly[(i + 1) % len(poly)] - poly[i]
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a_ - c
        tt, ss = (w[0] * e[1] - w[1] * e[0]) / den, (w[0] * d[1] - w[1] * d[0]) / den
        if tt > 0 and -1e-9 <= ss <= 1 + 1e-9:
            best = tt if best is None else min(best, tt)
    return float(best or 0.0)


def _inside(poly, c) -> bool:
    import numpy as np

    n = len(poly)
    s = []
    for i in range(n):
        a_, b_ = poly[i], poly[(i + 1) % n]
        s.append(np.sign((b_[0] - a_[0]) * (c[1] - a_[1]) - (b_[1] - a_[1]) * (c[0] - a_[0])))
    return all(x >= 0 for x in s) or all(x <= 0 for x in s)


def _root_local_points(root, xc):
    """Mesh points (and a 64-point rim for cylinders) under one rigid object, in its root frame."""
    import numpy as np

    from pxr import Usd, UsdGeom

    pts = []
    for q in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if q.IsA(UsdGeom.Mesh):
            v = UsdGeom.Mesh(q).GetPointsAttr().Get()
            P = np.asarray(v, float) if v else None
        elif q.IsA(UsdGeom.Cylinder):
            cy = UsdGeom.Cylinder(q)
            rr, hh = float(cy.GetRadiusAttr().Get()), float(cy.GetHeightAttr().Get())
            a_ = np.linspace(0, 2 * math.pi, 64, endpoint=False)
            P = np.stack([rr * np.cos(a_), rr * np.sin(a_), np.full(64, 0.5 * hh)], -1)
        else:
            continue
        if P is None or not len(P):
            continue
        rel, _ = xc.ComputeRelativeTransform(q, root)
        Rm = np.array([[rel[i][j] for j in range(4)] for i in range(4)])
        pts.append(P @ Rm[:3, :3] + Rm[3, :3])
    return pts


def support_top(raw_env, stage, wz: float, cw, skip, want_z: bool = False) -> tuple:
    """Support top under the end-pose object -> (xy convex hull (world) | None, asset name). Candidates = scene rigid
    bodies (robot and main object excluded; support discs and static objects included); poses from env data
    (root_pos_w, root_quat_w; USD transforms do not follow the physics pose) · points = visible meshes and cylinders
    under the root (body frame). The smallest-area one whose vertex z is within wz ± X_SUPPORT_ZTOL, xy within
    X_SUPPORT_REACH of cw, and whose hull contains cw. Not found (desk) -> None; the marker line records the nearest
    candidates.
    want_z (pad branch only): third value = the highest top z among **all** supports containing the plate centre cw
    (per support, the max z of points with z <= wz + PAD_TOP_REACH) · fourth value = a table of (top, name) of those
    supports. Using only the top of the smallest-area support buries the plate under overlapping supports (measured:
    2/4 supports visible, tops 0.8825 and 0.8856, plate 0.8835 -> inside the upper disc, only a side band visible).
    Other branches pass want_z=False (return values and behaviour unchanged)."""
    import numpy as np

    from pxr import Usd, UsdGeom

    PR = load_ref()
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    best, seen, tops = (None, None, None, None), [], []
    from dexengine.rl.eval.traj_record import all_obj_body

    for name, ast in raw_env.scene.rigid_objects.items():
        path = ast.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
        # static bodies are not support candidates (off = always false)
        if path in skip or "/Robot" in path or all_obj_body(name):
            continue
        root = stage.GetPrimAtPath(path)
        if not root.IsValid():
            continue
        pts = _root_local_points(root, xc)
        if not pts:
            continue
        M = UsdGeom.Xformable(root).GetLocalTransformation()
        sc = np.array([np.linalg.norm([M[i][0], M[i][1], M[i][2]]) for i in range(3)])
        pos = ast.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        Rq = PR.quat_R(ast.data.root_quat_w[0].detach().cpu().numpy().astype(float))
        W = (np.concatenate(pts) * sc) @ Rq.T + pos
        near = (np.abs(W[:, 2] - wz) <= X_SUPPORT_ZTOL) & (np.linalg.norm(W[:, :2] - cw, axis=1) <= X_SUPPORT_REACH)
        seen.append((round(float(np.abs(W[:, 2] - wz).min()), 4), int(near.sum()), name))
        sel = W[near][:, :2]
        if len(sel) < 3:
            continue
        hull = convex_hull(sel)
        if len(hull) < 3:
            continue
        hull = np.array(hull)
        if not _inside(hull, cw):
            continue
        area = 0.5 * abs(
            sum(
                hull[i][0] * hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0] * hull[i][1]
                for i in range(len(hull))
            )
        )
        lo_h, hi_h = hull.min(0), hull.max(0)
        box = np.all((W[:, :2] >= lo_h - 1e-4) & (W[:, :2] <= hi_h + 1e-4), axis=1) & (W[:, 2] <= wz + PAD_TOP_REACH)
        tops.append((float(W[box][:, 2].max()), name))
        if best[0] is None or area < best[0]:
            best = (area, hull, name, None)
    if best[1] is None:
        print(f"{MARK} support top not found (z {wz:.4f}) · nearest rigid bodies {sorted(seen)[:4]}", flush=True)
    if not want_z:
        return best[1], best[2]
    # several supports under the plate centre (measured: tops 0.8825 and 0.8856, plate 0.8835 inside the upper disc):
    # the highest top
    return best[1], best[2], (max(tops)[0] if tops else None), [(round(z, 4), n) for z, n in sorted(tops, reverse=True)]
