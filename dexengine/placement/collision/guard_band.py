#!/usr/bin/env python3
"""Guard band — the mesh guard (collision.core.side_guard: desk · torso) skips the (hull, pose) pairs whose verdict the
proxy capsule already proves, without changing one guard output (verdict · reason · its depth · the [TORSO_GUARD]
marker).

Bound: a revision / finger hull A lies inside its table capsule (arm_proxy.lookup — the search filter's) grown by A's
outside error e_out = max over A's vertices of (axis distance − r), ≥ 0. The axis distance is convex, so its max over
the hull sits at a vertex: e_out is exact, not sampled (the rulers' surface samples, rounded to 1e-7 m by
arm_surface.hull_samples, lie ≤ 0.09 µm off A — inside the margins below). The grown capsule C⁺ ⊇ A gives per pose:
  desk   depth(A) ≤ depth(C⁺): A ∩ prism ⊂ C⁺ ∩ prism and arm_proxy.capsule_desk_pen is exact. A pose with
         depth(C⁺) ≤ tolerance − EPS_M can neither fail nor be the maximum of a failing guard (> tolerance), so it is
         not clipped (clip_min_z) and its bound stands in.
  torso  dist(A, H) ≥ |axis − ctr_H| − rad_H − (r + e_out) per masked torso part H. Above CLEAR_M for every part, both
         mesh rulers read > 0 there: the filter grid (the 8 nodes around any hull point lie outside the union — a node
         outside every hull holds a positive value and a trilinear read of positives is positive; no torso vertex is
         inside A) and the exact ruler (no sample of either inside the other). The pair can neither fail nor hold the
         minimum of a failing guard (< −tolerance): skipped. The palm and vendor capsules carry no band.
A proven-fail band (capsule overlap + error < −tolerance) would not save the mesh: the reason carries the extremum's
depth (exact), so every pair not proven clear is measured on the mesh.
Marker: ``[GUARD_BAND] <side> skipped=<pairs> exact=<pairs> (desk s/x · torso s/x)`` per side guard.
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.collision import arm_proxy as AP
from dexengine.placement.collision import sdf_grid as HG

#: torso: clearance of the grown capsule above which a pair is skipped — the grid cell diagonal (√3·GRID_M: every node
#: around a point this far out lies outside the union) + 0.1 mm for rounding.
CLEAR_M = float(np.sqrt(3.0)) * HG.GRID_M + 1e-4
#: desk: rounding margin under the tolerance (m).
EPS_M = 1e-6


def band(key: str, v) -> dict:
    """The band of hull v (body frame) keyed ``key`` in the capsule table: axis centre p · unit axis d · half length h ·
    grown radius rad = r + e_out (dies like arm_proxy.lookup when the table does not hold these vertices)."""
    c = AP.lookup(key, v)
    v = np.asarray(v, float)
    d = c["R"][:, 2]
    e = float((_seg_dist(v, c["p"] - 0.5 * c["L"] * d, c["p"] + 0.5 * c["L"] * d) - c["r"]).max())
    return {"p": c["p"], "d": d, "h": 0.5 * c["L"], "r": c["r"], "e_out": max(e, 0.0), "rad": c["r"] + max(e, 0.0)}


def _seg_dist(p, a, b):
    """Distance of points p (…, 3) to segments a–b (…, 3) (broadcast)."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-30), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def _world(bd: dict, R, t):
    """(centre, unit axis) of the band capsule per pose."""
    return np.einsum("tij,j->ti", R, bd["p"]) + t, np.einsum("tij,j->ti", R, bd["d"])


def desk_max(ctx: dict, Ts, base, ts, desk_xy, cnt: dict) -> float:
    """The side guard's desk term = collision.core.desk_penetration's desk component (frames + chords in Ts): the vendor
    capsules through arm_surface.desk_pen_bodies; a hull clipped (_hull_desk_pen) only on the poses its band bound puts
    above cnt["tol"] — elsewhere the bound (≤ tol) stands in: the value is exact whenever it exceeds the tolerance."""
    from dexengine.placement.collision import arm_surface as AS

    m = max(
        [
            float(p.max())
            for _l, p in AS.desk_pen_bodies(dict(ctx, surf=dict(ctx["surf"], hulls=[])), Ts, base, ts, desk_xy)
        ]
        + [-1e9]
    )
    for h in ctx["surf"]["hulls"]:
        Tk = Ts[:, h["k"]]
        R, t = Tk[:, :3, :3], Tk[:, :3, 3] + base
        pen = desk_ub(h["band"], R, t, ctx["ztop"], desk_xy)
        ix = np.nonzero(pen > cnt["tol"])[0]
        if len(ix):
            pen[ix] = AS._hull_desk_pen(h, R[ix], t[ix], ctx["ztop"], desk_xy)
        add(cnt, "desk", len(pen) - len(ix), len(ix))
        m = max(m, float(pen.max()))
    return m


def desk_ub(bd: dict, R, t, ztop: float, desk_xy) -> np.ndarray:
    """(P,) desk depth of the grown capsule — ≥ the hull's (arm_surface.desk_pen_bodies' convention; t carries base)."""
    from dexengine.placement.collision import arm_surface as AS

    c, ax = _world(bd, R, t)
    return AS._cap_desk_pen(np, c, ax, bd["h"], bd["rad"], ztop, desk_xy)


def torso_skip(tz: dict, b: dict, Ts):
    """(P,) True where body b's grown capsule clears every masked torso part by > CLEAR_M · None without a band."""
    bd = b["hull"].get("band") if b["kind"] == "hull" else None
    if bd is None:
        return None
    Tk = Ts[:, b["k"]]
    c, ax = _world(bd, Tk[:, :3, :3], Tk[:, :3, 3])
    lo, hi = c - ax * bd["h"], c + ax * bd["h"]
    lb = np.full(len(c), np.inf)
    for tn in b["mask"]:
        H = tz["torso"][tn]
        lb = np.minimum(lb, _seg_dist(H["ctr"], lo, hi) - H["rad"])
    return lb - bd["rad"] > CLEAR_M


def torso_terms(tz: dict, Ts, TA, count) -> tuple[float, float, float]:
    """(exact min arm + palm, exact min fingers, filter min) of torso.torso_guard on Ts · TA (+ finger frames)
    — banded; measured again unbanded when the exact verdict and the filter's disagree (its marker prints both minima
    over every pair)."""
    from dexengine.placement.collision import torso as TS

    if TS.tol_m() < 0:  # a skipped pair is proven > 0 — that passes only a tolerance ≥ 0
        raise SystemExit(
            f"[GUARD_BAND] torso tolerance {TS.TORSO_OVERLAP_MAX_MM} mm < 0 — a skipped pair (> 0) would miss a failure"
        )
    for on in (True, False):
        sa = [torso_skip(tz, b, Ts) if on else None for b in tz["bodies"]]
        sf = [torso_skip(tz, b, TA) if on else None for b in tz["fingers"]]
        (ea, fa), (ef, _) = TS._each(tz, Ts, tz["bodies"], sa), TS._each(tz, TA, tz["fingers"], sf)
        out = float(ea.min()), float(ef.min()), float(fa.min())
        if on:
            n = sum(int(x.sum()) for x in sa + sf if x is not None)
            add(count, "torso", n, len(Ts) * len(sa) + len(TA) * len(sf) - n)
        if (min(out[:2]) >= -TS.tol_m()) == (out[2] >= -TS.tol_m()):
            break
    return out


def counter(desk_tol_mm: float) -> dict:
    """A side guard's band state: the desk skip level and the (skipped, exact) pair counts per term."""
    return {"tol": desk_tol_mm / 1000.0 - EPS_M, "desk": [0, 0], "torso": [0, 0]}


def add(cnt: dict | None, term: str, skipped: int, exact: int) -> None:
    """Count skipped / exact pairs of one term ("desk" | "torso"); no counter = nothing."""
    if cnt is not None:
        cnt[term][0] += skipped
        cnt[term][1] += exact


def say(side: str, cnt: dict) -> None:
    """The [GUARD_BAND] marker of one side guard."""
    s, x = cnt["desk"][0] + cnt["torso"][0], cnt["desk"][1] + cnt["torso"][1]
    print(
        f"[GUARD_BAND] {side} skipped={s} exact={x} (desk {cnt['desk'][0]}/{cnt['desk'][1]} · torso "
        f"{cnt['torso'][0]}/{cnt['torso'][1]})",
        flush=True,
    )
