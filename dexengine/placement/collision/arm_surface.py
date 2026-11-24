#!/usr/bin/env python3
"""The arm collision ruler — the **surface of the colliders the sim spawns** (the only ruler). Judging, baking, the gate
and the final guard read it through collision.core, so "the guard passes" reads as "the sim colliders do not overlap
beyond the tolerance".

Shapes are read, never generated here: the arm bodies are sim_collider.bodies (vendor capsules with their radius ·
revision hulls), the torso is sim_collider.torso_bodies (via collision.torso). Which shape stands for a revision hull is
arm_proxy's decision (COLL_PROXY): "capsule" = its fitted capsule from arm_proxy's table (the body list then holds
capsules only) · "mesh" = the hull itself.

What is measured (thresholds are collision.core's named constants):
  desk   per capsule: COLL_PROXY "capsule" = the exact depth of the lowest point of (capsule ∩ footprint prism)
         (arm_proxy.capsule_desk_pen) · "mesh" = ztop + r − z(axis sample), counted inside the desk footprint grown
         by r; per hull (mesh only): ztop − the lowest point of (hull ∩ prism) (clip_min_z). Threshold DESK_SURF_PEN_MM.
  cross  signed surface distance over every arm body pair of the two arms: capsule–capsule = axis distance − r1 − r2;
         hull–capsule = min over hull surface samples of (distance to the axis − r); hull–hull = −(depth of the deepest
         surface sample of one hull inside the other). Threshold ARM_OVERLAP_MAX_MM (allowed overlap).
  torso  arm bodies + palm ↔ torso hulls (collision.torso) — TORSO_RULER below states that the torso is measured
         with this same surface ruler.

Marker: ``[ARM_RULER] surface digest=<sim_collider.digest> torso=<TORSO_RULER> …`` printed once per process (per
resident-worker request) when the first context is built. A run whose log lacks it did not run with this ruler.
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.collision import arm_proxy as AP
from dexengine.placement.collision.geom import _seg_seg_dist, capsule_endpoints

#: The arm ↔ torso check uses the surface ruler (collision.torso). Shown by the [ARM_RULER] marker.
TORSO_RULER = "surface"

HULL_SAMPLE_M = 0.015  # surface-sample spacing on hull triangles (cross only); the undetected sagitta for a
#                         35 mm capsule is r − sqrt(r² − (h/2)²) ≈ 0.8 mm, under the tolerance used with this ruler
NEAR_M = 0.010  # bounding-sphere prefilter: poses farther apart than this keep the sphere bound as their value
_SAID: dict = {}
_DIGEST: dict = {}  # URDF → sim_collider.digest of its arm shapes (clip-independent)
_SERVER_KEEP = ("_DIGEST",)


def hull_samples(v: np.ndarray, tris: np.ndarray, h: float) -> np.ndarray:
    """Hull vertices + barycentric grid points on every triangle (spacing ≤ h along the longest edge)."""
    pts = [v]
    for t in tris:
        a, b, c = v[t]
        n = int(np.ceil(max(np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)) / h))
        if n <= 1:
            continue
        ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float)
        w = np.stack([ij[:, 0], ij[:, 1], n - ij[:, 0] - ij[:, 1]], 1) / n
        pts.append(w @ np.stack([a, b, c]))
    return np.unique(np.round(np.concatenate(pts), 7), axis=0)


def hull_record(link: str, k: int, v: np.ndarray) -> dict:
    """One hull body: link-frame vertices + facet planes + bounding sphere + surface samples."""
    from scipy.spatial import ConvexHull

    from dexengine.robots import wrist_hull as G

    ctr, tri = v.mean(0), G.hull_triangles(v)
    edges = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1), axis=0)
    return {
        "link": link,
        "k": int(k),
        "v": v,
        "eq": ConvexHull(v).equations,
        "ctr": ctr,
        "edges": edges,
        "rad": float(np.linalg.norm(v - ctr, axis=1).max()),
        "smp": hull_samples(v, tri, HULL_SAMPLE_M),
    }


def build_surface(urdf: str, arm: str, side: str, joints: list) -> dict:
    """Surface bodies of one arm side from the one generator: {caps: [k · p · R · r · L · link], hulls: [...], links}.
    k = index of the link's joint in ``joints`` (the FK chain of collision.core.build_arm_ctx). COLL_PROXY "capsule": a
    revision hull enters as its table capsule (``proxy`` True) — hulls stays empty."""
    from dexengine.core import arms as AR
    from dexengine.placement.collision import sim_collider as SC

    spec = AR.ARM_REGISTRY[arm]
    kmap = {spec.body_name(side, spec.index_base + k): k for k in range(len(joints))}
    caps, hulls = [], []
    for b in SC.bodies(arm, side, urdf):
        if isinstance(b, SC.Capsule):
            caps.append(
                {
                    "k": kmap[b.link],
                    "p": np.asarray(b.xyz, float),
                    "R": np.asarray(b.R, float),
                    "r": float(b.r),
                    "L": float(b.L),
                    "link": b.link,
                }
            )
        elif AP.capsule():
            caps.append(dict(AP.lookup(AP.arm_key(arm, b.link), b.verts), k=kmap[b.link], link=b.link, proxy=True))
        else:
            hulls.append(hull_record(b.link, kmap[b.link], np.asarray(b.verts, float)))
    AP.say(
        ("arm", arm, side, str(urdf), AP.kind()),
        f"{AP.marker()} · {arm}/{side} arm: "
        + (
            " ".join(
                f"{c['link'].split('_')[-1]}(r{c['r'] * 1e3:.1f} L{c['L'] * 1e3:.1f} {c['axis']})"
                for c in caps
                if c.get("proxy")
            )
            + " · search desk·torso filter = capsule (guard = mesh)"
            if AP.capsule()
            else "hulls as they are (original mesh)"
        ),
    )
    if not _SAID:
        _SAID[1] = True
        if urdf not in _DIGEST:
            _DIGEST[urdf] = SC.digest(arm, [urdf])
        print(
            f"[ARM_RULER] surface digest={_DIGEST[urdf]} torso={TORSO_RULER} · desk = surface (capsule r · hull "
            f"vertices) · cross = signed surface distance (every arm body pair) · torso = surface (collision.torso)"
            f" · {arm}/{side} capsules {len(caps)} · hulls "
            + (", ".join(f"{h['link']}({len(h['v'])} vertices)" for h in hulls) or "none"),
            flush=True,
        )
    return {"caps": caps, "hulls": hulls, "links": [c["link"] for c in caps] + [h["link"] for h in hulls]}


# ── desk (numpy) ──────────────────────────────────────────────────────────────────────────────────────────────


def _support_pen_r(pts, ztop: float, desk_xy, r):
    """Penetration of a sphere of radius r at each point pts(…,3) into the desk top (m, positive = below); footprint
    grown by r. Aerial scene = no desk (−1e9): support discs touch objects only, so no robot check reads them."""
    from dexengine.placement.coords import core as VC

    if VC.scene_mode() == "aerial":
        return np.full(pts.shape[:-1], -1e9)
    (x0, x1), (y0, y1) = desk_xy if desk_xy is not None else VC.desk_bounds_scene(0.0, margin=0.0)
    inb = (pts[..., 0] >= x0 - r) & (pts[..., 0] <= x1 + r) & (pts[..., 1] >= y0 - r) & (pts[..., 1] <= y1 + r)
    return np.where(inb, ztop + r - pts[..., 2], -1e9)


def clip_min_z(xp, V, E, N, D, box):
    """Lowest z of (convex hull ∩ the footprint's vertical prism) per pose, exact — xp = numpy | torch (one formula
    for both twins). V (…, nv, 3) world vertices · E (ne, 2) edges · N (…, F, 3) · D (…, F) world facets (N·x + D ≤ 0
    inside). The minimum sits at a vertex of the clipped polytope: a hull vertex inside the prism, a hull edge crossing
    a side wall, or a prism corner line crossing a facet. 1e9 where the hull misses the prism."""
    (x0, x1), (y0, y1) = box
    big = 1e9
    inb = (V[..., 0] >= x0) & (V[..., 0] <= x1) & (V[..., 1] >= y0) & (V[..., 1] <= y1)
    z = xp.amin(xp.where(inb, V[..., 2], big), -1)
    P0, Dl = V[..., E[:, 0], :], V[..., E[:, 1], :] - V[..., E[:, 0], :]
    for ax, val, ot, lo, hi in ((0, x0, 1, y0, y1), (0, x1, 1, y0, y1), (1, y0, 0, x0, x1), (1, y1, 0, x0, x1)):
        s = (val - P0[..., ax]) / Dl[..., ax]
        o = P0[..., ot] + s * Dl[..., ot]
        ok = (s >= 0) & (s <= 1) & (o >= lo) & (o <= hi)
        z = xp.minimum(z, xp.amin(xp.where(ok, P0[..., 2] + s * Dl[..., 2], big), -1))
    nz = N[..., 2]
    neg, pos = nz < -1e-12, nz > 1e-12
    for cx, cy in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
        rhs = -(N[..., 0] * cx + N[..., 1] * cy + D)  # N_z · z ≤ rhs
        b = rhs / xp.where(neg | pos, nz, 1.0)
        low, up = xp.amax(xp.where(neg, b, -big), -1), xp.amin(xp.where(pos, b, big), -1)
        side = xp.amin(xp.where(neg | pos, big, rhs), -1) >= -1e-12  # vertical facets hold the line
        z = xp.minimum(z, xp.where((low <= up + 1e-12) & side & (low > -big / 2), low, big))
    return z


def _hull_desk_pen(h, R, t, ztop: float, desk_xy) -> np.ndarray:
    """Depth (m) of the deepest point of the hull inside the footprint prism below the desk top, per pose (T',);
    −1e9 where it misses the prism or the scene is aerial."""
    from dexengine.placement.coords import core as VC

    if VC.scene_mode() == "aerial":
        return np.full(len(t), -1e9)
    box = desk_xy if desk_xy is not None else VC.desk_bounds_scene(0.0, margin=0.0)
    N = np.einsum("tij,fj->tfi", R, h["eq"][:, :3])
    with np.errstate(divide="ignore", invalid="ignore"):
        z = clip_min_z(
            np,
            np.einsum("tij,vj->tvi", R, h["v"]) + t[:, None, :],
            h["edges"],
            N,
            h["eq"][:, 3][None] - (N * t[:, None, :]).sum(-1),
            box,
        )
    return np.where(z < 1e8, ztop - z, -1e9)


def _cap_desk_pen(xp, ctr, axw, h, r, ztop: float, desk_xy):
    """COLL_PROXY "capsule": exact capsule desk depth (arm_proxy.capsule_desk_pen) of capsules (ctr, unit axis axw,
    half length h, radius r) — −1e9 in the aerial scene (no desk)."""
    from dexengine.placement.coords import core as VC

    if VC.scene_mode() == "aerial":
        return ctr[..., 0] * 0.0 - 1e9
    box = desk_xy if desk_xy is not None else VC.desk_bounds_scene(0.0, margin=0.0)
    return AP.capsule_desk_pen(xp, ctr - axw * h, ctr + axw * h, r, ztop, box)


def desk_pen_bodies(ctx: dict, Ts, base, ts, desk_xy) -> list:
    """[(link, pen (T', S))] per surface body: capsule exact (COLL_PROXY capsule · S = 1) or axis samples ts (mesh ·
    footprint grown by r) · hull exact (S = 1)."""
    out = []
    for c in ctx["surf"]["caps"]:
        Tk = Ts[:, c["k"]]
        Rw, tw = Tk[:, :3, :3], Tk[:, :3, 3]
        ctr = tw + np.einsum("tij,j->ti", Rw, c["p"]) + base
        axw = np.einsum("tij,jk,k->ti", Rw, c["R"], np.array([0.0, 0.0, 1.0]))
        if AP.capsule():
            pen = _cap_desk_pen(np, ctr, axw, c["L"] / 2.0, float(c["r"]), ctx["ztop"], desk_xy)
            out.append((c["link"], pen[:, None]))
            continue
        pts = ctr[:, None, :] + axw[:, None, :] * (ts * (c["L"] / 2.0))
        out.append((c["link"], _support_pen_r(pts, ctx["ztop"], desk_xy, float(c["r"]))))
    for h in ctx["surf"]["hulls"]:
        Tk = Ts[:, h["k"]]
        out.append((h["link"], _hull_desk_pen(h, Tk[:, :3, :3], Tk[:, :3, 3] + base, ctx["ztop"], desk_xy)[:, None]))
    return out


# ── desk (torch twin) ─────────────────────────────────────────────────────────────────────────────────────────


def torch_surface(ctx: dict, dev) -> dict:
    """numpy surface bodies → torch float64 tensors on dev (capsule radii · hull vertices). The capsule tensors of
    collision.core_torch.torch_ctx are built from the same ctx["caps"] list."""
    import torch

    sf = ctx["surf"]
    f64 = {"dtype": torch.float64, "device": dev}
    return {
        "r": torch.tensor([float(c["r"]) for c in sf["caps"]], **f64),
        "links": list(sf["links"]),
        "hulls": [
            (
                int(h["k"]),
                torch.tensor(h["v"], **f64),
                torch.tensor(h["edges"], dtype=torch.long, device=dev),
                torch.tensor(h["eq"], **f64),
            )
            for h in sf["hulls"]
        ],
    }


def _support_pen_r_t(pts, ztop: float, desk_xy, r):
    """torch twin of _support_pen_r (r: float or tensor broadcastable to pts[..., 0])."""
    import torch

    from dexengine.placement.coords import core as VC

    neg = torch.full(pts.shape[:-1], -1e9, dtype=pts.dtype, device=pts.device)
    if VC.scene_mode() == "aerial":
        return neg
    (x0, x1), (y0, y1) = desk_xy if desk_xy is not None else VC.desk_bounds_scene(0.0, margin=0.0)
    inb = (pts[..., 0] >= x0 - r) & (pts[..., 0] <= x1 + r) & (pts[..., 1] >= y0 - r) & (pts[..., 1] <= y1 + r)
    return torch.where(inb, ztop + r - pts[..., 2], neg)


def desk_pen_bodies_t(tc: dict, Ts, ctr, axw, base, ts, desk_xy):
    """Per-pose penetration of every surface body (…, nb) — torch twin of the per-frame max of desk_pen_bodies.
    ctr (…, nc, 3) already carries base (as in collision.desk_penetration_t); axw (…, nc, 3) unit axes."""
    import torch

    st = tc["surf"]
    if AP.capsule():
        cols = [_cap_desk_pen(torch, ctr, axw, tc["cap_h"][:, None], st["r"], tc["ztop"], desk_xy)]  # (…, nc)
    else:
        pts = ctr[..., None, :] + axw[..., None, :] * (ts[:, None] * tc["cap_h"][:, None, None])  # (…, nc, S, 3)
        cols = [_support_pen_r_t(pts, tc["ztop"], desk_xy, st["r"][:, None]).amax(-1)]  # (…, nc)
    from dexengine.placement.coords import core as VC

    aerial = VC.scene_mode() == "aerial"
    box = desk_xy if desk_xy is not None else VC.desk_bounds_scene(0.0, margin=0.0)
    bt = base[..., 0, :] if base.dim() > 1 else base  # base carries a body axis (…, 1, 3) for the capsules
    for k, V, E, EQ in st["hulls"]:
        Tk = Ts[..., k, :, :]
        R, t = Tk[..., :3, :3], Tk[..., :3, 3] + bt
        if aerial:
            cols.append(torch.full(t.shape[:-1] + (1,), -1e9, dtype=t.dtype, device=t.device))
            continue
        N = torch.einsum("...ij,fj->...fi", R, EQ[:, :3])
        z = clip_min_z(
            torch,
            torch.einsum("...ij,vj->...vi", R, V) + t[..., None, :],
            E,
            N,
            EQ[:, 3] - (N * t[..., None, :]).sum(-1),
            box,
        )
        cols.append(torch.where(z < 1e8, tc["ztop"] - z, -1e9)[..., None])
    return torch.cat(cols, -1)


# ── two-arm crossing (numpy; the guard is numpy-only) ─────────────────────────────────────────────────────────


def pt_seg(p, a, b):
    """Distance of points p (m, P, 3) to segments a, b (m, 1, 3)."""
    ab = b - a
    den = np.maximum((ab * ab).sum(-1), 1e-12)
    t = np.clip(((p - a) * ab).sum(-1) / den, 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _hull_pose(h, Ts):
    """(R, t) per pose of a hull body's link."""
    Tk = Ts[:, h["k"]]
    return Tk[:, :3, :3], Tk[:, :3, 3]


def hull_cap_sd(h, R, t, a, b, r):
    """Signed surface distance hull ↔ capsule per pose (T',); the sphere bound where clearly apart."""
    cw = np.einsum("tij,j->ti", R, h["ctr"]) + t
    sd = pt_seg(cw[:, None, :], a[:, None, :], b[:, None, :])[:, 0] - h["rad"] - r
    near = np.nonzero(sd < NEAR_M)[0]
    for i0 in range(0, len(near), 64):
        ix = near[i0 : i0 + 64]
        pw = np.einsum("tij,pj->tpi", R[ix], h["smp"]) + t[ix, None, :]
        sd[ix] = pt_seg(pw, a[ix, None, :], b[ix, None, :]).min(-1) - r
    return sd


def _inside_sd(hA, RA, tA, hB, RB, tB, ix):
    """min over A's surface samples of B's max plane value (negative = the deepest sample inside B)."""
    pw = np.einsum("tij,pj->tpi", RA[ix], hA["smp"]) + tA[ix, None, :]
    pb = np.einsum("tji,tpj->tpi", RB[ix], pw - tB[ix, None, :])  # world → B link frame (R^T)
    d = pb @ hB["eq"][:, :3].T + hB["eq"][:, 3]
    return d.max(-1).min(-1)


def hull_hull_sd(hA, RA, tA, hB, RB, tB):
    """Signed surface distance hull ↔ hull per pose (T',); the sphere bound where clearly apart."""
    ca = np.einsum("tij,j->ti", RA, hA["ctr"]) + tA
    cb = np.einsum("tij,j->ti", RB, hB["ctr"]) + tB
    sd = np.linalg.norm(ca - cb, axis=-1) - hA["rad"] - hB["rad"]
    near = np.nonzero(sd < NEAR_M)[0]
    for i0 in range(0, len(near), 32):
        ix = near[i0 : i0 + 32]
        sd[ix] = np.minimum(_inside_sd(hA, RA, tA, hB, RB, tB, ix), _inside_sd(hB, RB, tB, hA, RA, tA, ix))
    return sd


def cross_surface_each(ctxR: dict, ctxL: dict, TsR, TsL) -> np.ndarray:
    """Signed surface distance between the two arms per (aligned) pose (T',) [m] — negative = overlap depth."""
    sR, sL = ctxR["surf"], ctxL["surf"]
    out = np.full(TsR.shape[0], 1e9)
    segR = [(c, *capsule_endpoints(c, TsR[:, c["k"]])) for c in sR["caps"]]
    segL = [(c, *capsule_endpoints(c, TsL[:, c["k"]])) for c in sL["caps"]]
    for c1, a1, b1 in segR:
        for c2, a2, b2 in segL:
            out = np.minimum(out, _seg_seg_dist(a1, b1, a2, b2) - c1["r"] - c2["r"])
    hR = [(h, *_hull_pose(h, TsR)) for h in sR["hulls"]]
    hL = [(h, *_hull_pose(h, TsL)) for h in sL["hulls"]]
    for hs, segs in ((hR, segL), (hL, segR)):
        for h, R, t in hs:
            for c, a, b in segs:
                out = np.minimum(out, hull_cap_sd(h, R, t, a, b, float(c["r"])))
    for hA, RA, tA in hR:
        for hB, RB, tB in hL:
            out = np.minimum(out, hull_hull_sd(hA, RA, tA, hB, RB, tB))
    return out
