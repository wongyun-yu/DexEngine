"""The frame-0 functions of one side's desk solve (repair_side._side_steps): the class masks and initial values of the
three solve classes, the penetration counters, the contact-link test and the finger box. repair_side imports these
names (the same objects; its tests patch them there)."""

from __future__ import annotations

import numpy as np

from dexengine.retarget.f0.repair_cfg import CONTACT_BAND, CONTACT_MIN_PTS, HC
from dexengine.retarget.f0.repair_geom import _aa_to_R, hulls_contain


def _solve_inits(hand, wpos0, wR0, indep0, desk_z, desk_margin, opos):
    """_side_steps step: class masks and initial values of the three solve classes (fingers · wrist · free)."""
    _pts0 = np.vstack([lk[1] for lk in hand.link_points_world(wpos0, wR0, hand.fingers12(indep0))])
    lift = np.array([0.0, 0.0, max(0.0, (desk_z - desk_margin) - _pts0[:, 2].min()) + 1e-3])
    away = wpos0 - opos
    away = away / max(np.linalg.norm(away), 1e-9)
    NF = HC.n_indep
    M_FIN = np.array([0] * 6 + [1] * NF, bool)
    M_WRI = np.array([1] * 6 + [0] * NF, bool)
    M_ALL = np.ones(HC.nd, bool)
    fin_inits = [np.concatenate([np.zeros(6), np.full(NF, v)]) for v in (-0.15, -0.35, -0.6, 0.0)]
    wri_inits = [
        np.concatenate([lift, np.zeros(3 + NF)]),
        np.concatenate([lift + away * 0.01, np.zeros(3 + NF)]),
        np.concatenate([lift + away * 0.03, np.zeros(3 + NF)]),
    ]
    # The free class needs BOTH finger directions. Opening-only inits cannot reach the "pull the
    # wrist out AND curl the fingers back around the handle" basin, which is the only free-class
    # motion that re-establishes a grasp after the palm is evicted. Measured on one left side:
    # opening-only reports INFEASIBLE, adding the curling inits finds a solution keeping 39 % of
    # contact vs 31 % for wrist-only.
    all_inits = [
        np.concatenate([lift, np.zeros(3), np.full(NF, -0.20)]),
        np.concatenate([lift + away * 0.02, np.zeros(3), np.full(NF, -0.35)]),
        np.concatenate([away * 0.020, np.zeros(3), np.full(NF, 0.20)]),
        np.concatenate([away * 0.025, np.zeros(3), np.full(NF, 0.35)]),
        np.concatenate([away * 0.030, np.zeros(3), np.full(NF, 0.50)]),
        np.zeros(HC.nd),
    ]
    return M_FIN, M_WRI, M_ALL, fin_inits, wri_inits, all_inits


def _penetration_fns(
    _ctr, _clr, hand, obj, oR, opos, wpos0, wR0, indep0, desk_z, desk_margin, pen_margin, _dm_s, _pm_s
):
    """The penetration counters of one side. `n_pen` counts desk, object and reverse-hull overlaps at the current
    clearance (`_clr`) and margins and drives the evaluation meter (`_ctr`); `_n_pen_strict` evaluates it at the
    stricter solve margins by swapping them in for one call, so both live in this one scope."""

    def n_pen(dq, split=False):
        _ctr[0] += 1  # the budget meter; this line defines "how much work was done"
        wp = wpos0 + dq[:3]
        wRm = _aa_to_R(dq[3:6]) @ wR0
        f12 = hand.fingers12(indep0 + dq[6:])
        links = hand.link_points_world(wp, wRm, f12)
        allpts = np.vstack([lk[1] for lk in links])
        # DESK is a HARD constraint alongside the object: neither may be collided with.
        # The desk plane sits at trajectory-frame z = desk_z (TACO staging: 0 — VERIFIED, not assumed: every object's
        # mesh bottom at frame 0 lands within 0.6 mm of zero across the control panel; base
        # layer trees declare it in task_info.support.desk_z, read via coords). The penetrating
        # geometry is real and specific: fingertip links only (little/ring/index/thumb
        # tips 1-2 cm under), never the palm, and a clean clip shows 0/27
        # links below — so this is a retargeting artifact, not a frame-convention bug.
        # It survives in some clips (one still trains to 100 %) because PhysX can push
        # free fingertips up; that makes it benign-but-real, and we still remove it.
        # NOTE: a desk failure caused by the arm and its mount (robot_z_off) is NOT this:
        # it lies outside any hand-frame test and belongs to placement.
        #
        # `_clr` (nonlocal, set by the solve loop) demands a positive GAP instead of merely
        # forbidding overlap. Needed because `allpts` samples the hand surface: pushing the
        # sampled points to exactly z = 0 leaves the mesh between them still crossing. Solve
        # with a gap, then verify with every vertex at gap = 0.
        _floor = desk_z - desk_margin + _clr[0]
        desk_depth = np.clip(_floor - allpts[:, 2], 0.0, None).sum()
        n_desk = int(np.count_nonzero(allpts[:, 2] < _floor))
        n_obj, obj_depth = obj.penetration_stats(oR, opos, allpts, pen_margin, _clr[0])
        n_rev = hulls_contain(links, obj.surface_world(oR, opos))
        if split:
            # Return the depth as well: the count alone cannot tell "pressed in a
            # little" from "the hand has gone through the desk". desk_depth is the sum,
            # _desk_max is the depth of the deepest point (m).
            _desk_max = float(np.clip(_floor - allpts[:, 2], 0.0, None).max()) if len(allpts) else 0.0
            return (n_desk, n_obj, n_rev, float(desk_depth), float(obj_depth), _desk_max)
        return n_desk + n_obj + n_rev, desk_depth + obj_depth

    def _n_pen_strict(dq):
        nonlocal desk_margin, pen_margin
        dm, pm = desk_margin, pen_margin
        desk_margin, pen_margin = _dm_s, _pm_s
        try:
            return n_pen(dq)
        finally:
            desk_margin, pen_margin = dm, pm

    return n_pen, _n_pen_strict


def _contact_links_fn(hand, obj, oR, opos, wpos0, wR0, indep0, band: float = CONTACT_BAND):
    """The contact-link test of one side at frame 0 (see contact_links); `band` = the distance that counts as touching
    (CONTACT_BAND; the object repair's frame-0 set uses repair_obj.OBJ_TOUCH_MM)."""

    def contact_links(dq):
        """Set of links TOUCHING the object: >= CONTACT_MIN_PTS points within CONTACT_BAND of
        the surface, or inside it (a penetrating point is meant to end up ON the surface, so
        it still counts as touching).

        LINKS, NOT POINTS. Point counts scale with contact AREA,
        which legitimately COLLAPSES when a deep overlap is resolved into a surface touch —
        scoring points would reject exactly the repair we want. Measured: DBK 156 left keeps
        only 37 % of its contact POINTS but all 3 of its contact LINKS; the hand is still on
        the object with the same fingers, just no longer buried in it. Link membership is
        invariant to overlap depth, so 'deep contact -> light contact' passes and only a
        finger actually LEAVING the object fails."""
        wp = wpos0 + dq[:3]
        wRm = _aa_to_R(dq[3:6]) @ wR0
        out = set()
        for ln, pts, _, _, _ in hand.link_points_world(wp, wRm, hand.fingers12(indep0 + dq[6:])):
            local = (pts - opos) @ oR
            d, ix = obj.kdt.query(local)
            inside = (np.einsum("ij,ij->i", local - obj._s_pts[ix], obj._s_nrm[ix]) < 0) & (
                obj.delaunay_local.find_simplex(local) >= 0
            )
            if np.count_nonzero((d < band) | inside) >= CONTACT_MIN_PTS:
                out.add(ln)
        return out

    return contact_links


def _finger_box(hand, indep0):
    """BOX CONSTRAINT on the independent finger joints, applied by CLIPPING inside the evaluation (standard bound
    handling for Nelder-Mead: the optimiser cannot gain by leaving the box because the clipped point is what gets
    scored, and the returned dq is re-derived from the clip). Effective bounds widen to include the ORIGINAL angle
    so a source pose that already sits marginally outside the URDF range (measured: index_1 = 1.5306 vs upper
    1.52856) is preserved rather than "corrected" — this tool repairs penetration, it does not silently edit
    retargeting output."""
    _INDEP_F = [HC.order[i] for i in HC.indep_idx]
    _lo = np.array([min(hand.limits[f][0], indep0[i]) for i, f in enumerate(_INDEP_F)])
    _hi = np.array([max(hand.limits[f][1], indep0[i]) for i, f in enumerate(_INDEP_F)])
    return _lo, _hi
