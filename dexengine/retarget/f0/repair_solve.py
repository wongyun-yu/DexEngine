"""f0 repair of one side, last step: the class-wise solve loop with strict verification, and its cost.

The loop yields one `SolveRequest` per class (fingers, wrist, free) to the solver engine, accepts the first
feasible class, verifies it on every collision-mesh vertex and retries with a doubled clearance.

WHY CLASS-WISE, NOT ONE JOINT SOLVE OVER ALL nd DIMENSIONS (measured).
The penalty terms dominate the landscape (LAMBDA=1e3 per point, DEPTH_W=5e4 per metre:
1 mm of depth costs 50, while the ENTIRE finger-only solution costs 0.126). Nelder-Mead
therefore expands along whatever direction kills penetration fastest, and a wrist
translation moves all ~1620 hand points at once while a finger angle moves one
fingertip. So the search reaches feasibility through the wrist and the movement
weights never arbitrate — even after rescaling the variables to make the cost metric
isotropic (tried; one clip still chose a 2.31 cm wrist shift).

Measured on one right side (penetration = thumb/index TIPS only, 19 pts):
  fingers-only, wrist frozen -> FEASIBLE at cost 0.126 (max 0.286 rad)  <- what we want
  wrist-only,  fingers frozen -> INFEASIBLE (18 pts remain)
  free joint solve            -> feasible at cost ~1.9 (wrist 2.31 cm)  <- hand leaves
                                                                           the handle
So: solve each STRUCTURAL CLASS separately and take the cheapest feasible one. Fingers
first (reshapes the grasp in place), then wrist-only, then the free solve as fallback.
Cheapest-feasible-wins is the same rule the weighted L2 already expresses — this just
stops one basin from swallowing the search before the comparison can happen. No new
tuned constant is introduced.

STRICT PRIORITY, fingers before wrist. Wrist translation
displaces the whole hand relative to the object and therefore breaks the grasp; finger
flexion reshapes it in place. So a feasible fingers-only repair is ACCEPTED OUTRIGHT,
even where some small wrist nudge would score cheaper on the weighted L2 — preserving
the grasp is the intent the weights were standing in for.

CONTACT IS MEASURED, NOT ENFORCED. For every accepted repair the links that frame 0 was
touching are checked again (contact_links; link membership ignores overlap depth) and the
retained fraction is recorded. Losing a touching link never rejects a repair and never
SKIPs a side (see Acceptance below). A side whose penetration is a property of the WHOLE
trajectory, not of frame 0, can lose a finger's contact here; such clips need the fix one
stage earlier, in retargeting.

STRICT VERIFICATION LOOP. The solve runs on a decimated point set with a
positive gap; acceptance is then re-tested against EVERY collision-mesh vertex at gap 0,
which is the acceptance rule (not a single vertex may overlap). If the
full-vertex test still finds overlap, the gap doubles and we solve again — the sampled
solution was simply not conservative enough for the geometry between sample points.

Acceptance. The goal is RESET STABILITY, not grasp preservation: the first feasible class is accepted outright.
Contact-link retention is still MEASURED and RECORDED (diagnostic only, no veto); the strict zero-overlap
verification (which IS the stability guarantee) stays hard.

Per-side budget. An infeasible side exhausts every class x init x clearance
retry; unbounded, it starves the OTHER side of the sequence-level timeout, so a
repairable left never gets written because an unrepairable right burned the clock.
Soft check between attempts/classes — worst overshoot is one class solve.
The unit is n_pen evaluations (not seconds): the same purpose (a side without a solution cannot hold out
forever), counted in a quantity independent of machine speed. Rationale: the SIDE_BUDGET_EVALS comment.
"""

from __future__ import annotations

import time

import numpy as np

from dexengine.retarget.f0.repair_cfg import HC, SolveRequest, _no_solution_status, budget_skip_status
from dexengine.retarget.f0.repair_geom import _aa_to_R, _strict_model
from dexengine.retarget.f0.solve_classes import class_loop


def solve_cost(dq):
    """The solve cost: weighted L2 of the step (HC.w_dq)."""
    return float(np.sum((HC.w_dq * dq) ** 2))


def _strict_overlap_fn(side, wpos0, wR0, indep0, desk_z, desk_margin, obj, oR, opos):
    """The exact acceptance test of a step dq: overlap counts of every collision-mesh vertex of the strict hand
    model against the desk plane and the object (winding number, contains_exact)."""
    strict = _strict_model(side)

    def strict_overlap(dq):
        """Every collision-mesh vertex × the exact test. The desk is z<desk_z (exact to begin
        with), the object is the winding-number test (contains_exact)."""
        wp = wpos0 + dq[:3]
        wRm = _aa_to_R(dq[3:6]) @ wR0
        links = strict.link_points_world(wp, wRm, strict.fingers12(indep0 + dq[6:]))
        allp = np.vstack([lk[1] for lk in links])
        nd = int(np.count_nonzero(allp[:, 2] < desk_z - desk_margin))
        no = obj.contains_exact(oR, opos, allp)
        return nd, no

    return strict_overlap


def _budget_exhausted(rep, side_budget_evals, _ctr, t0, t2):
    """The report of a side whose evaluation budget ran out before the next attempt."""
    rep["budget_exhausted"] = True
    rep["status"] = budget_skip_status(side_budget_evals, _ctr[0] - _ctr[1])
    rep["t_opt_s"] = round(time.time() - t2, 3)
    rep["t_total_s"] = round(time.time() - t0, 3)
    return rep


def _solve_geom(
    side, hand, obj, _ck, opos, oR, desk_z, _dm_s, _clr, _pm_s, wpos0, wR0, indep0, _lo, _hi, _n_pen_strict, pen_margin
):
    """The geometry a `SolveRequest` carries to the solver engine (hand and object models, desk floor at the current
    clearance, solve margins, finger box, and the host strict check). The GPU engine batch-evaluates through it and
    adds the evaluations used to the meter via add_evals; the final feasibility check is geom["n_pen"] =
    _n_pen_strict, run on the host and counting itself."""
    return {
        "hand": hand,
        "hand_key": (HC.name, side, pen_margin),
        "obj": obj,
        "obj_key": _ck,
        "opos": opos,
        "oR": oR,
        "floor": desk_z - _dm_s + _clr[0],
        "clr": _clr[0],
        "pm": _pm_s,
        "wpos0": wpos0,
        "wR0": wR0,
        "indep0": indep0,
        "expand": hand.fingers12,
        "nd": HC.nd,
        "w_dq": HC.w_dq,
        "lo": _lo,
        "hi": _hi,
        "n_pen": _n_pen_strict,
    }


def _record_contact_links(rep, side, L0, contact_links, cand):
    """Record how many of the frame-0 contact links the accepted step keeps (diagnostic only, no veto)."""
    kept = L0 & contact_links(cand[0])
    rep["contact_links_kept"] = len(kept)
    if len(kept) < len(L0):
        lost = sorted(lk.split("hand_")[-1] for lk in (L0 - kept))
        rep["contact_links_lost"] = lost
        print(
            f"    [record] {side} contact links {len(kept)}/{len(L0)} kept "
            f"(lost {','.join(lost[:4])}) — v2 policy: no effect on acceptance"
        )


def _close_report(rep, chosen, verify_hist, t0, t2):
    """Solve class, verification history and timings of the finished side."""
    rep["solve_class"] = chosen
    rep["verify"] = " / ".join(verify_hist)
    rep["t_opt_s"] = round(time.time() - t2, 3)
    rep["t_total_s"] = round(time.time() - t0, 3)


def _repaired_fields(rep, best):
    """Status and step summary of a repaired side."""
    rep["status"] = "REPAIRED"
    rep["dq"] = [round(float(v), 5) for v in best]
    rep["wrist_shift_cm"] = round(float(np.linalg.norm(best[:3])) * 100, 2)
    rep["wrist_rot_deg"] = round(float(np.degrees(np.linalg.norm(best[3:6]))), 2)
    rep["max_finger_rad"] = round(float(np.abs(best[6:]).max()), 4)


def solve_loop(side, rep, pose0, scene, classes, fns, bounds, budget, margins, t0, t2):
    """_side_steps last step (a generator): solve class by class, verify strictly, retry with a doubled clearance
    (`class_loop`), and finish the report. Yields `SolveRequest`s and returns the report."""
    wpos0, wquat0, wR0, indep0, fingers0, off, obj_sl = pose0
    hand, obj, _ck, oR, opos, desk_z = scene
    M_FIN, M_WRI, M_ALL, fin_inits, wri_inits, all_inits = classes
    _clip, _add_evals, _n_pen_strict, contact_links = fns
    _dm_s, _pm_s, _lo, _hi = bounds
    verify_retries, side_budget_evals, _ctr, _clr, clearance = budget
    desk_margin, pen_margin = margins
    strict_overlap = _strict_overlap_fn(side, wpos0, wR0, indep0, desk_z, desk_margin, obj, oR, opos)
    L0 = contact_links(np.zeros(HC.nd))
    rep["contact_links_before"] = len(L0)

    def request(tag, mask, inits):
        geom = _solve_geom(
            side,
            hand,
            obj,
            _ck,
            opos,
            oR,
            desk_z,
            _dm_s,
            _clr,
            _pm_s,
            wpos0,
            wR0,
            indep0,
            _lo,
            _hi,
            _n_pen_strict,
            pen_margin,
        )
        return SolveRequest(
            tag=tag, mask=mask, inits=inits, clip=_clip, cost=solve_cost, add_evals=_add_evals, geom=geom
        )

    status, cand, verify_hist = yield from class_loop(
        (("fingers", M_FIN, fin_inits), ("wrist", M_WRI, wri_inits), ("free", M_ALL, all_inits)),
        request,
        lambda x: sum(strict_overlap(x)),
        verify_retries,
        clearance,
        _clr,
        lambda: _ctr[0] - _ctr[1] > side_budget_evals,
    )
    if status == "budget":
        return _budget_exhausted(rep, side_budget_evals, _ctr, t0, t2)
    if status == "budget_class":
        rep["budget_exhausted"] = True  # cut off between classes; the final status below reads this
    best, chosen = (cand[0], cand[2]) if cand else (None, None)
    if cand:
        _record_contact_links(rep, side, L0, contact_links, cand)
    _close_report(rep, chosen, verify_hist, t0, t2)
    if best is None:
        rep["status"] = _no_solution_status(rep, side_budget_evals, _ctr[0] - _ctr[1])
        return rep
    _repaired_fields(rep, best)
    rep["_internal"] = (best, wpos0, wquat0, wR0, indep0, fingers0, off, obj_sl)
    return rep
