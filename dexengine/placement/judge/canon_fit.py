"""Canonical-cell fit for the augmentation region cells — the one owner of the match tolerance and of rule A.

The judge side (judge.aug_cells) calls index_cache; the augmentation side (record.augment.policy) only reads the
tolerance. The judge does not import augmentation code, so the shared tolerance lives on the judge side. This module
does not import numpy (it only does arithmetic on the arrays the judge passes), so the augmentation side reads it
cheaply.

CANON_MATCH_FRAC: tolerance for matching the canonical cell = this fraction of the grid spacing (position: grid step ·
  yaw: the smallest gap between the region's yaw values). It covers the 4th-decimal rounding of the adopted record
  (≈ 0.0014 step) and never confuses a neighbouring cell.

Rule A (canonical-cell fit):
  canonical residual = canonical offset − offset of the nearest grid point (folded per capture axis into (−½, ½] step).
  Only for a clip whose residual exceeds CANON_MATCH_FRAC × step on either axis, the residual is added to that clip's
  judge search origin d0, which puts the canonical placement on a grid point. Only xy moves — the yaw must already be a
  yaw grid point, otherwise the run dies. A clip within the tolerance changes nothing (values unchanged).
  Why: placements adopted before the clip-centred search origin (d0 = target point − pivot per clip) sit on the d0 = 0
  grid, and those adoptions stay valid. The clip-centred grid offsets such a placement from its cell by the fractional
  part, so judge mode D and cell picking would find no canonical cell.
Rule A, whole-cell shift (at most a whole-cell move):
  when the canonical placement lies outside the judge window (window cell offset |kx − Wx| ≤ Wx · |kyy − Wy| ≤ Wy), the
  origin d0 moves by whole cells so the canonical placement sits on the window's edge cell (window_shift — the fewest
  cells per window axis). A canonical placement inside the window changes nothing. Used together with the fractional
  fit above.
  Why: a canonical placement transplanted from another hand's canon was not found inside this window and can sit one
  cell outside it (window x offset −13 at half-width 12 was measured). Folding the fraction cannot fix a whole-cell
  offset.
Canonical-cell volume exemption (volume_exempt): a canonical cell that passes the desk and robot-spot
  mask but drops out of the volume filter (two-arm covered frames ≥ n95) is put back into the volume, that cell only —
  also when the whole volume is empty (the region is then the canonical cell alone). The canonical placement passed
  adoption (FULL · guard), but the judge's volume index clips pool-outside frames to the pool edge before adding the
  window offset, so the coverage of the same placement depends on the origin d0 (an XHand grab clip whose origin rule A
  moved lost its canonical cell and judge mode D died). Augmentation candidate cells keep the filter. Marker
  [CANON_EXEMPT] … volume.
  The exemption covers the canonical cell only. Clipping after the window offset in Judge._clip_volume would make it
  unnecessary, but changes judge values (every built bundle would be built again).
"""

from __future__ import annotations

CANON_MATCH_FRAC = 0.01
TAG = "[CANON_FIT]"


def fold(v: float, step: float) -> float:
    """Residual of v measured from the nearest point of the step grid — magnitude ≤ ½ step."""
    return float(v) - float(step) * round(float(v) / float(step))


def residual(e_cap, step: float) -> tuple[float, float]:
    """Canonical offset − offset of one grid point (capture x, y in m) → per-axis folded residual (rx, ry) in m."""
    return fold(e_cap[0], step), fold(e_cap[1], step)


def fit(e_cap, step: float) -> tuple[float, float] | None:
    """Rule A — residual (rx, ry) m to add to d0 (capture frame). None when both axes are within the tolerance
    (CANON_MATCH_FRAC × step): no fit."""
    r = residual(e_cap, step)
    if max(abs(r[0]), abs(r[1])) <= CANON_MATCH_FRAC * float(step):
        return None
    return r


def window_shift(J, e_cap):
    """Rule A, whole-cell shift — canonical offset − offset of the window's centre cell (capture, m) → the capture
    offset (sx, sy) m to add to the origin and the window cell counts (nx, ny); None when the canonical placement is
    inside the window. Window cell offset = rz · e / step (the inverse of Judge._cell_dcap — cell offset = rz⁻¹ · bxy +
    d0cap)."""
    w = J.rz.apply([float(e_cap[0]), float(e_cap[1]), 0.0])[:2] / float(J.step)
    s = []
    for v, half in zip(w, (int(J.Wx), int(J.Wy)), strict=True):
        n = int(round(float(v)))
        s.append(n - (half if n > 0 else -half) if abs(n) > half else 0)
    if s == [0, 0]:
        return None
    c = J.rz.inv().apply([s[0] * float(J.step), s[1] * float(J.step), 0.0])
    return (float(c[0]), float(c[1])), (s[0], s[1])


def _wrap(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def yaw_on_grid(yaw: float, yaws) -> bool:
    """Whether the yaw is a yaw grid point — tolerance = CANON_MATCH_FRAC × the smallest gap between yaw values
    (360° for a single yaw)."""
    u = sorted({round(float(y), 9) for y in yaws})
    gaps = [b - a for a, b in zip(u, u[1:], strict=False) if b - a > 0]
    tol = CANON_MATCH_FRAC * (min(gaps) if gaps else 360.0)
    return min(abs(_wrap(float(yaw) - y)) for y in u) <= tol


def fit_origin(J, ck, cn: dict, die):
    """Rule A — fitted origin (d0s, d0cap, residual, window cell shift) of a clip whose canonical residual is outside
    the tolerance or whose canonical placement is outside the window (window_shift); None when neither (nothing
    changes). J = the judge (Judge — _origin_shift · _pivot_swap · rz · step · yaws · Wx · Wy),
    ck = the clip (clip · cpiv · canon_pivot — the canonical record's pivot, AugCells._canon), cn = the canonical
    placement.
    The residual is measured at the canonical yaw from the offset of the window's centre cell (bxy 0; d0cap + the pivot
    swap term — Judge._pivot_swap, the same function as adoption). One window cell is one step along one capture axis
    (rz = Rz(−90°)), so each axis is folded separately; dies when that relation or the d0s ↔ d0cap relation of
    Judge._origin_shift does not hold."""
    d0s, d0cap = J._origin_shift(ck.cpiv)
    yaw = float(cn["yaw_deg"])
    d_c, _p2 = J._pivot_swap(ck.canon_pivot, ck.cpiv, d0cap, yaw)
    e_cap = (cn["dx"] - float(d_c[0]), cn["dy"] - float(d_c[1]))
    r = fit(e_cap, float(J.step))
    sh = window_shift(J, e_cap)
    if r is None and sh is None:
        return None
    r = r if r is not None else (0.0, 0.0)
    if not yaw_on_grid(yaw, J.yaws):
        die(
            f"{ck.clip}: canonical yaw {yaw}° is not a yaw grid point — an xy fit (rule A) cannot make a canonical cell"
        )
    for b in ((float(J.step), 0.0, 0.0), (0.0, float(J.step), 0.0)):
        c = J.rz.inv().apply(b)
        if abs(abs(c[0]) + abs(c[1]) - float(J.step)) > 1e-9 or min(abs(c[0]), abs(c[1])) > 1e-9:
            die(
                f"{ck.clip}: one window cell {b[:2]} is capture {c[:2]} — not an axis-aligned step (the residual "
                "cannot be folded per axis)"
            )
    back = J.rz.inv().apply([float(d0s[0]), float(d0s[1]), 0.0])
    if max(abs(back[0] - float(d0cap[0])), abs(back[1] - float(d0cap[1]))) > 1e-12:
        die(
            f"{ck.clip}: origin d0s {d0s} · d0cap {d0cap} are not related by rz — the fitted origin cannot be given "
            "to both frames"
        )
    (sx, sy), cells = sh if sh is not None else ((0.0, 0.0), (0, 0))
    d0cap_fit = d0cap + [r[0] + sx, r[1] + sy, 0.0]
    d0s_fit = J.rz.apply([float(d0cap_fit[0]), float(d0cap_fit[1]), 0.0])[:2]
    return d0s_fit, d0cap_fit, r, cells


def index_cache(J, ck, cn: dict, geom, die) -> None:
    """Judge._clip_index_cache — with a rule-A fit, Judge._origin_shift is replaced by the fitted origin for this clip
    only (_clip_index_cache is the one caller of _origin_shift; the desk mask and the cell offsets read the ck.d0s ·
    ck.d0cap it writes, so judge modes D and L share the origin). geom = cell → (yaw, dcap, d, p2) (AugCells._geom).
    Dies when, after the fit, the nearest cell is farther from the canonical placement than the tolerance."""
    got = fit_origin(J, ck, cn, die)
    if got is None:
        J._clip_index_cache(ck)
        return
    d0s_fit, d0cap_fit, r, cells = got
    J._origin_shift = lambda _cpiv: (d0s_fit.copy(), d0cap_fit.copy())
    try:
        J._clip_index_cache(ck)
    finally:
        del J._origin_shift
    yd = [abs(_wrap(float(y) - cn["yaw_deg"])) for y in J.yaws]
    ky = yd.index(min(yd))
    best = min(
        max(abs(float(d[0]) - cn["dx"]), abs(float(d[1]) - cn["dy"]))
        for kx in range(J.NBx)
        for kyy in range(J.NBy)
        for d in [geom(ck, (ky, kx, kyy, cn["dz"]))[2]]
    )
    if best > CANON_MATCH_FRAC * float(J.step):
        die(
            f"{ck.clip}: after the rule-A fit the nearest cell is still {best:.3e} m from the canonical placement — "
            "the canonical placement is not a grid point"
        )
    print(
        f"{TAG} {ck.clip} canonical residual ({r[0] / J.step:+.3f}, {r[1] / J.step:+.3f}) step (tolerance "
        f"{CANON_MATCH_FRAC:g}) · window cell shift {cells} → origin d0 capture +({r[0] * 1e3:+.1f}, "
        f"{r[1] * 1e3:+.1f}) mm + cells {cells} · distance after fit {best:.1e} m "
        f"(rule A{' whole-cell' if cells != (0, 0) else ''})",
        flush=True,
    )


def volume_exempt(J, ck, cn: dict, grid, err: float, tag: str) -> bool:
    """Canonical-cell volume exemption (module doc) — grid · err = the window cell nearest to the canonical placement
    (ky, kx, kyy) and its distance (m). True when the cell was put into the volume."""
    ky, kx, kyy = grid
    lay = next((z for z in J.DZ if float(z) == float(cn["dz"])), None)
    if lay is None or err > J.step / 2.0 or lay not in getattr(ck, "V", {}):
        return False  # off the grid · no dz layer — the canonical-cell check dies
    if ck.desk_m is not None and not ck.desk_m[ky, kx, kyy]:
        return False  # dropped by the desk · robot-spot mask — outside this exemption
    if ck.V[lay][ky, kx, kyy]:
        return False
    cov = int(ck.COV[lay][ky, kx, kyy]) if lay in getattr(ck, "COV", {}) else -1
    ck.V[lay][ky, kx, kyy] = True
    print(
        f"{tag} [CANON_EXEMPT] {ck.clip} canonical cell ({ky},{kx},{kyy}) dz={float(lay):g} exempt from the volume "
        f"filter — covered frames {cov} < n95 {ck.n95} · desk · robot spot unchanged · augmentation candidate cells "
        "keep the filter (the judge volume index clips pool-outside frames before the window offset)",
        flush=True,
    )
    return True
