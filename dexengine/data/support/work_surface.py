#!/usr/bin/env python3
"""Work surface rule: desk height and desk footprint from **placement evidence** (dataset-independent, single owner).

Inputs are only one recording's object pose samples `{uid: (N,8) [t_ns x y z qw qx qy qz]}` (+z-up world) · mesh points
`mesh(uid) → (k,3)` · sample rate hz (the sample step for measuring rest, stated by the caller) · the clip window
[t0, t1] (ns) · the main-object uids. Reading the source (readers) is the caller's job (per-dataset desk recoverers).

Branches:
  A placement evidence  desk height = the **lowest surface** among those on which the handled objects (main objects +
                        objects moved within the window) and the objects **receiving** them (lying under that placement:
                        supports, containers) rest within the clip or within ADJ_GAP_S before/after it (work_surface).
  B no evidence         (in hand throughout) the caller's desk is used but kept **from burying** things: when the desk
      is
                        above the lowest underside of a lifted object or the lowest point of a holding hand inside the
                        footprint, it is lowered to that lowest point − BURY_MARGIN_M (not_buried). Without a measured
                        desk for the dataset, the given desk = the surface the main object rested on most recently
                            (nearest_rest).
  footprint             = the convex hull of object footprints placed on the desk surface (± SAME_SURFACE_M) in that
                        recording: places where a table is **confirmed**. Hand or object XY passing more than DEEP_M
                            below
                        the surface inside it counts as a **conflict** (passing through where the desk is). Without
                        placements on that surface there is no footprint (None).
Pedestals for objects placed above this height and exclusion from spawning of untouched static objects under the desk
are decided by the tree-layer owners reading this height (coords.compute_desk_support · extra_slots).
Motion and jump thresholds come from the scene requirement statement (scene_requirement); the 'held from below'
geometry is coords.support_contact alone.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as Rot

from dexengine.data.objects.scene_requirement import JUMP_M, MOVE_M, ROT_DEG

V_REST_MPS = 0.02  # rest: origin speed < this value
REST_MIN_S = 0.5  # minimum rest window length
# placement attached to the clip: only rest windows within this of the window boundary are the clip's placed surfaces
ADJ_GAP_S = 2.0
# branch B's given desk (datasets without a measured desk): the surface the main object rested on most recently within
# this
NEAR_GAP_S = 60.0
SAME_SURFACE_M = 0.03  # same surface = bottom height within ± this of the desk surface
DEEP_M = (
    0.02  # passing below the surface = deeper than this (object hull penetration, the same 2 cm as the hand tolerance)
)
# branch B: when buried, lower to the lowest point − this value (the same as the support audit's 'lowest point − 3 cm')
BURY_MARGIN_M = 0.03
BRANCH_PLACED, BRANCH_GIVEN = "A_placed", "B_given"


def _idx(a: np.ndarray, t_ns) -> int:
    return min(int(np.searchsorted(a[:, 0], t_ns)), len(a) - 1)


def _span(a: np.ndarray, t0_ns, t1_ns) -> tuple[int, int]:
    i0 = _idx(a, t0_ns)
    return i0, max(_idx(a, t1_ns), i0)


def verts_at(a: np.ndarray, pts: np.ndarray, i: int) -> np.ndarray:
    """World vertices of sample i."""
    return pts @ Rot.from_quat(a[i, [5, 6, 7, 4]]).as_matrix().T + a[i, 1:4]


def still_mask(a: np.ndarray, hz: float) -> np.ndarray:
    """Whether each sample is at rest: origin displacement to the next sample × hz < V_REST_MPS (hz = the sample rate
    stated by the caller). The last sample counts as at rest."""
    return np.r_[np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1) * hz, 0.0] < V_REST_MPS


def rest_win(hz: float) -> int:
    """Minimum number of samples in a rest window (REST_MIN_S)."""
    return max(2, int(round(REST_MIN_S * hz)))


def rest_runs(a: np.ndarray, hz: float, lo: int = 0, hi: int | None = None) -> list:
    """Rest windows [(i, j)] within [lo, hi] (inclusive, length ≥ REST_MIN_S)."""
    hi = len(a) - 1 if hi is None else hi
    m = still_mask(a, hz)[lo : hi + 1]
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    win = rest_win(hz)
    return [
        (lo + int(i), lo + int(j))
        for i, j in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1, strict=True)
        if j - i + 1 >= win
    ]


def window_motion(a: np.ndarray, i0: int, i1: int) -> tuple[float, float, float]:
    """(max displacement from the first sample m, max rotation °, max displacement between samples m) over the window
    [i0, i1]."""
    p = a[i0 : i1 + 1, 1:4]
    R = Rot.from_quat(a[i0 : i1 + 1, [5, 6, 7, 4]])
    step = float(np.linalg.norm(np.diff(p, axis=0), axis=1).max()) if len(p) > 1 else 0.0
    return float(np.linalg.norm(p - p[0], axis=1).max()), float(np.degrees((R * R[0].inv()).magnitude()).max()), step


def _low_z(a: np.ndarray, pts: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Lowest vertex z per sample i0..i1 (n,), using only the z row of the rotation (same values as verts_at's z)."""
    Rz = Rot.from_quat(a[i0 : i1 + 1, [5, 6, 7, 4]]).as_matrix()[:, 2, :]
    return (pts @ Rz.T).min(0) + a[i0 : i1 + 1, 3]


def _floor(a: np.ndarray, pts: np.ndarray, i: int, j: int) -> float:
    return float(np.median([float(verts_at(a, pts, k)[:, 2].min()) for k in range(i, j + 1, 3)]))


def _edge_rests(a: np.ndarray, pts: np.ndarray, hz: float, i0: int, i1: int, max_gap_s: float) -> list:
    """Bottoms of the nearest rest windows before and after the window [(z, 'before'|'after', time gap s, representative
    sample)], within max_gap_s of the window boundary. The bottom is measured on the rest_win samples at the window side
    (the far end of a long rest window is not the clip's placement)."""
    win, out = rest_win(hz), []
    before = rest_runs(a, hz, 0, i0)
    if before:
        i, j = before[-1]
        gap = float((a[i0, 0] - a[j, 0]) / 1e9)
        if gap <= max_gap_s:
            out.append((_floor(a, pts, j - win + 1, j), "before", round(gap, 2), j))
    after = rest_runs(a, hz, i1, len(a) - 1)
    if after:
        i, j = after[0]
        gap = float((a[i + win - 1, 0] - a[i1, 0]) / 1e9)
        if gap <= max_gap_s:
            out.append((_floor(a, pts, i, i + win - 1), "after", round(gap, 2), i))
    return out


def placed_surfaces(a: np.ndarray, pts: np.ndarray, t0_ns, t1_ns, hz: float) -> list:
    """Surfaces the object **rested on** attached to the clip [(bottom z, 'still'|'before'|'after'|'inside', time gap s,
    representative sample)]. Unmoved throughout the window (MOVE_M · ROT_DEG): the one bottom at the window start.
    Moved:
    the nearest rest windows within ADJ_GAP_S before and after the window (the surface it was picked from and put down
    on) and one per rest window inside it. None → []."""
    i0, i1 = _span(a, t0_ns, t1_ns)
    mv, rot, _ = window_motion(a, i0, i1)
    if mv < MOVE_M and rot < ROT_DEG:
        return [(float(verts_at(a, pts, i0)[:, 2].min()), "still", 0.0, i0)]
    return _edge_rests(a, pts, hz, i0, i1, ADJ_GAP_S) + [
        (_floor(a, pts, i, j), "inside", 0.0, (i + j) // 2) for i, j in rest_runs(a, hz, i0, i1)
    ]


def nearest_rest(a: np.ndarray, pts: np.ndarray, t0_ns, t1_ns, hz: float, max_gap_s: float = NEAR_GAP_S):
    """Bottom (z, 'before'|'after', time gap s) of the nearest rest window before the clip (else the nearest after) or
    None: the surface the object rested on when the clip has no attached placement (starting or ending in hand). Within
    one recording objects move between desk, floor and shelf, so the nearest rest window is read, not a per-object
    mode."""
    rows = _edge_rests(a, pts, hz, *_span(a, t0_ns, t1_ns), max_gap_s)
    return rows[0][:3] if rows else None


def interaction_uids(objs: dict, main_uids: list, t0_ns, t1_ns) -> list:
    """Objects the hand handles: main objects + objects moved within the window (at least MOVE_M · ROT_DEG, with steps
    below JUMP_M between samples; jumps are tracking defects)."""
    out = list(dict.fromkeys(u for u in main_uids if u in objs))
    for uid, a in objs.items():
        if uid in out:
            continue
        mv, rot, step = window_motion(a, *_span(a, t0_ns, t1_ns))
        if (mv >= MOVE_M or rot >= ROT_DEG) and step < JUMP_M:
            out.append(uid)
    return out


def receivers_of(objs: dict, mesh, uid: str, t_ns, hz: float) -> list:
    """Objects **receiving** object uid at time t [(receiver uid, its bottom z)]: at rest at that time (inside a rest
    window) and touching under uid's bottom (coords.support_contact; centre inside the contact polygon, the receiver's
    centre of mass lower)."""
    from dexengine.data.support.bands import SUPPORT_POLY_PAD_M  # noqa: PLC0415
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of the 'held from below' geometry

    a = objs[uid]
    Vu = verts_at(a, mesh(uid), _idx(a, t_ns))
    out = []
    for v, b in objs.items():
        if v == uid:
            continue
        k = _idx(b, t_ns)
        if not any(i <= k <= j for i, j in rest_runs(b, hz)):
            continue
        Vv = verts_at(b, mesh(v), k)
        c = C.support_contact(Vu, Vv)
        if c is not None and c["poly_out_m"] <= SUPPORT_POLY_PAD_M and not c["above"]:
            out.append((v, float(Vv[:, 2].min())))
    return out


def work_surface(objs: dict, mesh, main_uids: list, t0_ns, t1_ns, hz: float, receivers: bool = True) -> dict | None:
    """Branch A: desk surface = the lowest surface among those the handled objects and their receivers rest on. Based on
    handled objects, so background objects on the floor cannot pull the desk down (in a clip picking something up from
    the floor, the floor is the work surface). Without placement evidence None (branch B).
    Returns {branch, z, uid, how, gap_s, n_interaction, surfaces: {uid: [(z, how, time gap)]}}; receivers' surfaces have
    how 'recv:<how of the object they held>'. receivers=False measures without receivers (comparison only)."""
    hands_on = interaction_uids(objs, main_uids, t0_ns, t1_ns)
    surf: dict = {u: [] for u in hands_on}
    for u in hands_on:
        for z, how, gap, k in placed_surfaces(objs[u], mesh(u), t0_ns, t1_ns, hz):
            surf[u].append((z, how, gap))
            under = receivers_of(objs, mesh, u, objs[u][k, 0], hz) if receivers else []
            for v, zv in under:  # receivers and the objects receiving them (two levels)
                surf.setdefault(v, []).append((zv, f"recv:{how}", gap))
                for w, zw in receivers_of(objs, mesh, v, objs[u][k, 0], hz):
                    surf.setdefault(w, []).append((zw, f"recv:{how}", gap))
    cand = [(z, u, how, gap) for u, rows in surf.items() for z, how, gap in rows]
    if not cand:
        return None
    z, u, how, gap = min(cand)
    return {
        "branch": BRANCH_PLACED,
        "z": z,
        "uid": u,
        "how": how,
        "gap_s": gap,
        "n_interaction": len(hands_on),
        "surfaces": {k: sorted({(round(a, 4), b, c) for a, b, c in v}) for k, v in surf.items()},
    }


def _hull(xy: np.ndarray):
    """(hull vertices (k,2), half-plane equations (m,3)); degenerate (point, segment) gives (None, None)."""
    try:
        h = ConvexHull(np.asarray(xy, float))
    except (QhullError, ValueError):
        return None, None
    return np.asarray(xy, float)[h.vertices], h.equations


def inside(eq: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Mask of points inside the convex hull (half-plane equations)."""
    return (np.asarray(xy, float) @ eq[:, :2].T + eq[:, 2]).max(1) <= 0.0


def rests(objs: dict, mesh, hz: float) -> list:
    """Every placement of the recording [(uid, bottom z, footprint XY hull points (k,2))], one per rest window (measured
    once per recording and reused for each clip)."""
    out = []
    for u, a in objs.items():
        for i, j in rest_runs(a, hz):
            V = verts_at(a, mesh(u), (i + j) // 2)
            hull, _ = _hull(V[:, :2])
            out.append((u, float(V[:, 2].min()), V[:, :2] if hull is None else hull))
    return out


def below_points(objs: dict, mesh, uids: list, t0_ns, t1_ns, z: float, hands: np.ndarray | None) -> tuple[list, list]:
    """Points passing more than DEEP_M below the surface: ((k,3) per object frame, (k,3) per hand frame). uids = the
    handled objects, hands = hand keypoints (T, K, 3) in the world (None without them)."""
    ob = []
    for u in uids:
        a = objs[u]
        i0, i1 = _span(a, t0_ns, t1_ns)
        for k in i0 + np.nonzero(_low_z(a, mesh(u), i0, i1) < z - DEEP_M)[0]:
            V = verts_at(a, mesh(u), int(k))
            ob.append(V[V[:, 2] < z - DEEP_M])
    hb = []
    if hands is not None:
        m = hands[..., 2] < z - DEEP_M
        hb = [hands[f][m[f]] for f in np.nonzero(m.any(1))[0]]
    return ob, hb


def footprint(
    rest_rows: list, objs: dict, mesh, main_uids: list, t0_ns, t1_ns, z: float, hands: np.ndarray | None = None
) -> dict:
    """Desk footprint: convex hull of the footprints placed on desk surface z (± SAME_SURFACE_M) in that recording
    (rest_rows = rests()) + this clip's conflict counts. Returns {hull: (k,2) world XY | None, eq, n_rests, n_objects,
    obj_below · hand_below: frames passing below the surface, obj_conflict · hand_conflict: those whose deep points lie
    inside the footprint (passing where the desk is)}."""
    on = [(u, xy) for u, zr, xy in rest_rows if abs(zr - z) <= SAME_SURFACE_M]
    hull, eq = _hull(np.concatenate([xy for _, xy in on])) if on else (None, None)
    ob, hb = below_points(objs, mesh, interaction_uids(objs, main_uids, t0_ns, t1_ns), t0_ns, t1_ns, z, hands)
    conf = (lambda rows: sum(bool(inside(eq, r[:, :2]).any()) for r in rows)) if eq is not None else (lambda rows: 0)
    return {
        "hull": hull,
        "eq": eq,
        "n_rests": len(on),
        "n_objects": len({u for u, _ in on}),
        "obj_below": len(ob),
        "hand_below": len(hb),
        "obj_conflict": conf(ob),
        "hand_conflict": conf(hb),
    }


def not_buried(objs: dict, mesh, main_uids: list, t0_ns, t1_ns, z: float, eq, hands: np.ndarray | None) -> tuple:
    """Branch B: when the given desk z is above the lowest underside of lifted objects or the lowest point of holding
    hands inside the footprint (eq; None = no XY limit), lower it to that lowest point − BURY_MARGIN_M. Returns (desk z,
    lowered depth m; 0 = unchanged)."""
    lows = []
    for u in interaction_uids(objs, main_uids, t0_ns, t1_ns):
        a = objs[u]
        i0, i1 = _span(a, t0_ns, t1_ns)
        for k in i0 + np.nonzero(_low_z(a, mesh(u), i0, i1) < z)[0]:  # XY is checked only on frames lower than the desk
            V = verts_at(a, mesh(u), int(k))
            V = V[inside(eq, V[:, :2])] if eq is not None else V
            if len(V):
                lows.append(float(V[:, 2].min()))
    if hands is not None:
        H = hands.reshape(-1, 3)
        H = H[inside(eq, H[:, :2])] if eq is not None else H
        if len(H):
            lows.append(float(H[:, 2].min()))
    if not lows or min(lows) >= z:
        return z, 0.0
    return min(lows) - BURY_MARGIN_M, z - min(lows)
