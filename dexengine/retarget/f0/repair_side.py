"""f0 repair of one side (_side_steps): check the frame-0 pose, build the solve request for the GPU engine and turn
its answer into the repaired pose and the per-side report.

Shallow penetration is not repaired. Measured (47 clips × 2 sides): of the 65 sides that do penetrate, the median
max depth is 5.9mm and 71% are at or under 10mm — the fingertip is "pressed" into the desk, within what PhysX can
push out. Running Nelder-Mead whenever a single vertex is inside, regardless of depth, can spend 190 seconds on a
1.6mm one. This threshold measures desk depth only. `_dmax` is n_pen(split=True)'s _desk_max and excludes object
depth, so a side with zero desk penetration and only object penetration is skipped here whatever its depth. That is
intended; the evidence is the measurement in the CONTACT PRESERVATION note of repair_solve: repairs that remove object
penetration lost >= 50 % of contact links on 5 of 12 sides, and the 50–100 % range is empty, so there is no
threshold to tune. The desk is a static hard constraint (penetrating it goes through the table); the object is the
task target and must be touched, and penetration is unavoidable when retargeting MANO to a robot hand, so it cannot
be an exclusion criterion. The real fix for sides with large object penetration is the retargeting stage (fingertip
error threshold 20 mm). To change this, revise that decision first rather than this value.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from dexengine.core.paths import qpos_layout as QL
from dexengine.placement.coords.core import desk_height, scene_z_offset
from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0.repair_cfg import (
    CONTACT_BAND,
    CONTACT_MIN_PTS,
    HC,
    SIDE_BUDGET_EVALS,
    STRICT_MARGIN_SCALE,
    main_object_dirs,
)
from dexengine.retarget.f0.repair_geom import (
    _HAND_CACHE,
    _OBJ_CACHE,
    HandModel,
    ObjectGeom,
    _aa_to_R,
    _quat_wxyz_to_R,
    hulls_contain,
)
from dexengine.retarget.f0.repair_solve import solve_loop


def _load_side(seq: str, side: str, data_root: Path, wrist_root: Path, pen_margin: float, dataset: str):
    """_side_steps step 1: the tree, its layout, the main object and the frame-0 wrist; the hand and object models
    (cached). Returns (t0, t_load, art, theta0, pose0, scene) — pose0 / scene are unpacked by the caller."""
    t0 = time.time()
    tree = data_root / seq
    d = tree / "0"
    npz = np.load(d / "trajectory_kinematic.npz", allow_pickle=True)
    qpos = npz["qpos"]
    # The tree states the column addresses (qpos_layout is the owner). For the legacy TACO staging
    # the reader's legacy branch reproduces the legacy `off=0|H · obj=2H|2H+7` arithmetic (all 2,317
    # trees compared, 0 differences). The hand width is checked against the spec (HC.H).
    lay = QL.read(str(tree), nq=int(qpos.shape[1]), expect_hand_qpos_dim=HC.H)
    H = lay.hand_qpos_dim
    hs, ms = lay.hand_slice(side), lay.main_slice(side)
    if hs is None or ms is None:
        raise ValueError(f"{tree}: {side} hand/main-object block not in the layout (hands={lay.hands} main={lay.main})")
    off = hs[0]
    obj_sl = slice(*ms)
    fingers0 = qpos[0, off + 6 : off + H].copy()
    mesh_dir, sp_dir, art = main_object_dirs(tree, lay, side, dataset)
    # Desk plane (capture z) = world desk height − the capture→world z offset. TACO staging is 0
    # (z=0 is the desk plane); the base layer uses support.desk_z.
    desk_z = desk_height() - scene_z_offset(tree)
    wr = np.load(wrist_root / f"wrist_poses_{side}" / f"{seq}.npz")
    wpos0 = wr["positions"][0].astype(np.float64)
    wquat0 = wr["quaternions"][0].astype(np.float64)
    wR0 = _quat_wxyz_to_R(wquat0)
    hand = _HAND_CACHE.get((HC.name, side))
    if hand is None or hand.pen_margin != pen_margin:
        hand = HandModel(side, pen_margin=pen_margin)
        _HAND_CACHE[(HC.name, side)] = hand
    # Articulated object (2c): the joint value at frame 0 fixes the geometry — put θ0 (rounded to
    # 1e-4 rad) into the cache key. A rigid body keeps the previous key.
    theta0 = float(qpos[0, int(art["qpos_index"])]) if art else 0.0
    _ck = (mesh_dir, sp_dir) if not art else (mesh_dir, sp_dir, "artic", round(theta0, 4))
    obj = _OBJ_CACHE.get(_ck)
    if obj is None:
        obj = ObjectGeom(mesh_dir, sp_dir=sp_dir, articulation=art, theta=theta0)
        _OBJ_CACHE[_ck] = obj
    opos, oquat = qpos[0, obj_sl][:3], qpos[0, obj_sl][3:]
    oR = _quat_wxyz_to_R(oquat)
    t_load = time.time() - t0

    indep0 = fingers0[HC.indep_idx]
    pose0 = (wpos0, wquat0, wR0, indep0, fingers0, off, obj_sl)
    scene = (hand, obj, _ck, oR, opos, desk_z)
    return t0, t_load, art, theta0, pose0, scene


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
    # motion that re-establishes a grasp after the palm is evicted. Measured on rpan 059 left:
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
        # tips 1-2 cm under), never the palm, and a clean clip (DRP 034) shows 0/27
        # links below — so this is a retargeting artifact, not a frame-convention bug.
        # It survives in some clips (EBP 139 R trains to 100 %) because PhysX can push
        # free fingertips up; that makes it benign-but-real, and we still remove it.
        # NOTE: SOK 298's desk failure is NOT this — its offender is the arm+mount
        # (robot_z_off = -5.5 cm), outside any hand-frame test (see O2 / placement).
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


def _contact_links_fn(hand, obj, oR, opos, wpos0, wR0, indep0):
    """The contact-link test of one side at frame 0 (see contact_links)."""

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
            if np.count_nonzero((d < CONTACT_BAND) | inside) >= CONTACT_MIN_PTS:
                out.add(ln)
        return out

    return contact_links


def _initial_report(seq, side, n_pen, art, theta0, t_load):
    """The strict check of the unrepaired frame 0 (clearance 0) and the report it starts; returns the overlap count,
    the deepest desk penetration (m) and the report."""
    t1 = time.time()
    nd, no, nr, _dsum, _osum, _dmax = n_pen(np.zeros(HC.nd), split=True)
    n0 = nd + no + nr  # hard constraint = desk + object
    t_check = time.time() - t1
    rep = {
        "sequence": seq,
        "side": side,
        "n_pen_before": n0,
        "n_desk": nd,
        "n_obj": no,
        "n_rev": nr,
        "desk_depth_sum_mm": round(_dsum * 1000, 3),
        "desk_depth_max_mm": round(_dmax * 1000, 3),
        "obj_depth_sum_mm": round(_osum * 1000, 3),
        "obj_articulated": bool(art),
        "obj_theta0": (round(theta0, 5) if art else None),  # 2c marker: the check used the articulated geometry (θ0)
        "t_load_s": round(t_load, 3),
        "t_check_s": round(t_check, 3),
    }
    return n0, _dmax, rep


def _finger_box(hand, indep0):
    """BOX CONSTRAINT on the six independent finger joints, applied by CLIPPING inside the evaluation (standard bound
    handling for Nelder-Mead: the optimiser cannot gain by leaving the box because the clipped point is what gets
    scored, and the returned dq is re-derived from the clip). Effective bounds widen to include the ORIGINAL angle
    so a source pose that already sits marginally outside the URDF range (rpan 059 left index_1 = 1.5306 vs upper
    1.52856) is preserved rather than "corrected" — this tool repairs penetration, it does not silently edit
    retargeting output."""
    _INDEP_F = [HC.order[i] for i in HC.indep_idx]
    _lo = np.array([min(hand.limits[f][0], indep0[i]) for i, f in enumerate(_INDEP_F)])
    _hi = np.array([max(hand.limits[f][1], indep0[i]) for i, f in enumerate(_INDEP_F)])
    return _lo, _hi


def _shallow_skip(rep, _dmax, _skip_mm, t0):
    """The report of a side whose deepest desk penetration is within DEXENGINE_F0_SKIP_DEPTH_MM (not repaired)."""
    rep["status"] = (
        f"{PH.SKIP}: max depth {_dmax * 1000:.1f}mm ≤ {_skip_mm:.0f}mm (shallow penetration — repair skipped)"
    )
    rep["skipped_shallow"] = True
    rep["t_total_s"] = round(time.time() - t0, 3)
    return rep


def _side_steps(
    seq: str,
    side: str,
    data_root: Path,
    wrist_root: Path,
    desk_margin: float,
    do_repair: bool,
    seed: int = 0,
    pen_margin: float = 0.0,
    clearance: float = 0.0,
    verify_retries: int = 3,
    side_budget_evals: int = SIDE_BUDGET_EVALS,
    dataset: str = "taco",
    *,
    _ctr: list,
):
    """Returns dict report. Never writes; caller handles output assembly.

    A generator: it yields a `SolveRequest` per solve and receives (x, c). The control flow (class order, budget, strict
    verification, acceptance) is owned here alone; the solver engine (GPU) only solves requests."""
    t0, t_load, art, theta0, pose0, scene = _load_side(seq, side, data_root, wrist_root, pen_margin, dataset)
    wpos0, wquat0, wR0, indep0, fingers0, off, obj_sl = pose0
    hand, obj, _ck, oR, opos, desk_z = scene

    _clr = [0.0]  # required clearance (m). The initial check is 0 = strict; raised above 0 only while optimising.

    # HYSTERESIS: optimise against margins x0.7 (stricter) so the accepted pose clears
    # the CHECK margins with slack. Without it the optimiser parks vertices exactly on
    # the boundary and float rounding flips single points on recheck (n_pen=1 at the
    # desk plane after a REPAIRED write).
    _dm_s, _pm_s = desk_margin * STRICT_MARGIN_SCALE, pen_margin * STRICT_MARGIN_SCALE

    n_pen, _n_pen_strict = _penetration_fns(
        _ctr, _clr, hand, obj, oR, opos, wpos0, wR0, indep0, desk_z, desk_margin, pen_margin, _dm_s, _pm_s
    )
    contact_links = _contact_links_fn(hand, obj, oR, opos, wpos0, wR0, indep0)

    n0, _dmax, rep = _initial_report(seq, side, n_pen, art, theta0, t_load)
    if n0 == 0 or not do_repair:
        rep["t_total_s"] = round(time.time() - t0, 3)
        return rep
    # Shallow penetrations are not repaired: see the module docstring.
    _skip_mm = float(os.environ.get("DEXENGINE_F0_SKIP_DEPTH_MM", "10") or 0)
    if _skip_mm > 0 and _dmax * 1000.0 <= _skip_mm:
        return _shallow_skip(rep, _dmax, _skip_mm, t0)

    t2 = time.time()
    _ctr[1] = _ctr[0]  # the budget counts from here on (the one initial check is excluded)

    def _add_evals(n):
        """Add the batch evaluations done outside by the GPU engine to the meter. Host calls of n_pen (final feasibility
        checks etc.) count themselves through `_ctr[0] += 1` inside n_pen; the sum is the side's evaluation count."""
        _ctr[0] += int(n)

    _lo, _hi = _finger_box(hand, indep0)

    def _clip(dq):
        dq = np.asarray(dq, float).copy()
        dq[6:] = np.clip(indep0 + dq[6:], _lo, _hi) - indep0
        return dq

    M_FIN, M_WRI, M_ALL, fin_inits, wri_inits, all_inits = _solve_inits(
        hand, wpos0, wR0, indep0, desk_z, desk_margin, opos
    )
    return (
        yield from solve_loop(
            side,
            rep,
            pose0,
            scene,
            (M_FIN, M_WRI, M_ALL, fin_inits, wri_inits, all_inits),
            (_clip, _add_evals, _n_pen_strict, contact_links),
            (_dm_s, _pm_s, _lo, _hi),
            (verify_retries, side_budget_evals, _ctr, _clr, clearance),
            (desk_margin, pen_margin),
            t0,
            t2,
        )
    )
