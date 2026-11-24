"""f0 repair of one side (_side_steps): check the frame-0 pose, build the solve request for the GPU engine and turn
its answer into the repaired pose and the per-side report.

Shallow penetration is not repaired. Measured (47 clips × 2 sides): of the 65 sides that do penetrate, the median
max depth is 5.9mm and 71% are at or under 10mm — the fingertip is "pressed" into the desk, within what PhysX can
push out. Running Nelder-Mead whenever a single vertex is inside, regardless of depth, can spend 190 seconds on a
1.6mm one. This threshold measures desk depth only (`_dmax` = n_pen(split=True)'s _desk_max, no object depth).

Object penetration has its own rule (repair_obj, default on, opt-out DEXENGINE_F0_OBJ_REPAIR=0): a side
with no desk repair whose fingers start more than OBJ_REPAIR_MM inside the main object's cooked hull is moved out to
OBJ_RESIDUAL_MM (not to zero) while every link touching the object at frame 0 stays in the contact band; a frame 0
that cannot reach it keeps its least-SAT attempt at or under OBJ_REPAIR_MM (marked obj_kept), and a side the rule
cannot repair even that way is not written (SKIP_OBJ; the prejudge holds the clip, HOLD_OBJ). It also measures the
other main object and the hand-on kinematic slots, each with its own thresholds (repair_bodies). The earlier decision
skipped such sides at any depth because repairs to zero overlap lost >= 50 % of the contact links on 5 of 12 sides
(CONTACT PRESERVATION note of repair_solve); the residual and the contact keep answer that, and the pen audit (a deep
frame-0 overlap kicks the object or never resolves) is the reason to repair at all.
Every reader takes the tree's qpos rows from `_qpos`: the frame-0 separation (repair_sep, rule 1d) applied, so the hand
is measured against the moved object.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from dexengine.core.paths import qpos_layout as QL
from dexengine.placement.coords.core import desk_height, scene_z_offset
from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0 import repair_bodies as RB
from dexengine.retarget.f0 import repair_obj as RO
from dexengine.retarget.f0 import repair_sep as RSEP
from dexengine.retarget.f0 import repair_still as RST
from dexengine.retarget.f0.repair_cfg import (
    HC,
    SIDE_BUDGET_EVALS,
    STRICT_MARGIN_SCALE,
    main_object_dirs,
    main_object_entry,
)
from dexengine.retarget.f0.repair_geom import (
    _HAND_CACHE,
    _OBJ_CACHE,
    HandModel,
    ObjectGeom,
    _aa_to_R,
    _quat_wxyz_to_R,
    _strict_model,
    object_cache_key,
)
from dexengine.retarget.f0.repair_obj_solve import Projection, obj_solve_window
from dexengine.retarget.f0.repair_side_fns import _contact_links_fn, _finger_box, _penetration_fns, _solve_inits
from dexengine.retarget.f0.repair_solve import solve_loop
from dexengine.retarget.quat_bounds import require_unit


def _load_side(seq: str, side: str, data_root: Path, wrist_root: Path, pen_margin: float, dataset: str):
    """_side_steps step 1: the tree, its layout, the main object and the frame-0 wrist; the hand and object models
    (cached). Returns (t0, t_load, art, theta0, pose0, scene) — pose0 / scene are unpacked by the caller."""
    t0 = time.time()
    tree = data_root / seq
    qpos = _qpos(seq, data_root)
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
    # Articulated object (2c): the joint value at frame 0 and the articulation record fix the geometry — both are in
    # the cache key (repair_geom.object_cache_key). A rigid body keeps the previous key.
    theta0 = float(qpos[0, int(art["qpos_index"])]) if art else 0.0
    _ck = object_cache_key(mesh_dir, sp_dir, art, theta0)  # also the GPU evaluator's obj_key (repair_solve)
    obj = _OBJ_CACHE.get(_ck)
    if obj is None:
        obj = _OBJ_CACHE[_ck] = ObjectGeom(mesh_dir, sp_dir=sp_dir, articulation=art, theta=theta0)
    opos, oquat = qpos[0, obj_sl][:3], qpos[0, obj_sl][3:]
    require_unit(oquat, f"f0 {seq} [{side}] main object frame 0")  # _quat_wxyz_to_R does not normalise
    oR = _quat_wxyz_to_R(oquat)
    t_load = time.time() - t0

    indep0 = fingers0[HC.indep_idx]
    pose0 = (wpos0, wquat0, wR0, indep0, fingers0, off, obj_sl)
    scene = (hand, obj, _ck, oR, opos, desk_z)
    return t0, t_load, art, theta0, pose0, scene


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


def _shallow_skip(rep, _dmax, _skip_mm, t0):
    """The report of a side whose deepest desk penetration is within DEXENGINE_F0_SKIP_DEPTH_MM (not repaired)."""
    rep["status"] = (
        f"{PH.SKIP}: max depth {_dmax * 1000:.1f}mm ≤ {_skip_mm:.0f}mm (shallow penetration — repair skipped)"
    )
    rep["skipped_shallow"] = True
    rep["t_total_s"] = round(time.time() - t0, 3)
    return rep


def _cooked_main_object(seq: str, side: str, data_root: Path, dataset: str):
    """(CookedObject | ArticObject | BodySet | None, why) of the side's main object and the other bodies it is measured
    with (repair_bodies) — the shapes the object repair measures (repair_obj); `.at(qpos row)` gives the pieces."""
    tree = data_root / seq
    o = main_object_entry(tree, QL.read(str(tree), nq=None), side, dataset)
    main, why = RO.cooked(o, dataset, tree)
    return (main, why) if main is None else (RB.with_bodies(main, o, tree, side, dataset), why)


def _qpos(seq: str, data_root: Path) -> np.ndarray:
    """The tree's qpos rows as every f0 reader sees them: the frame-0 separation applied (repair_sep.applied)."""
    tree = data_root / seq
    return RSEP.applied(tree, np.load(tree / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"])


def _geom_at(scene, art, qrow):
    """The check geometry (ObjectGeom) of the main object at a qpos row: rigid = the frame-0 one; articulated = the one
    at that row's joint value (frame 0's when the two round the same to 1e-4 rad — a rule within one clip; the cache key
    holds the exact frame-0 value) — the touch measure follows the joint as the SAT does."""
    obj, ck = scene[1], scene[2]
    if not art:
        return obj
    th = float(qrow[int(art["qpos_index"])])
    return obj if round(th, 4) == round(ck[3], 4) else ObjectGeom(ck[0], sp_dir=ck[1], articulation=art, theta=th)


def _sat_fn(side, pose0, scene, cobj):
    """dq -> repair_obj.Sat of the strict hand: the cooked bodies (main (depth [m], link), the others) and stills."""
    wpos0, _wq, wR0, indep0 = pose0[:4]
    _hand, _obj, _ck, oR, opos, _dz = scene
    strict = _strict_model(side)

    def sat_fn(dq):
        wRm = _aa_to_R(dq[3:6]) @ wR0
        f12 = strict.fingers12(indep0 + dq[6:])
        sat = RO.sat_bodies(RO.hand_parts_world(strict, wpos0 + dq[:3], wRm, f12), cobj, oR, opos)
        return RST.with_stills(sat, cobj.stills, strict, (wpos0 + dq[:3], wRm, f12))

    return sat_fn


def _obj_gate(rep, seq, side, data_root, dataset, pose0, scene, desk_deep):
    """The object rule's gate of a side (repair_obj.gate): (SAT measure of the frame-0 hand | None when the object is
    not measured, the cooked object, opened). Records the measure and the cooked hull digest in the report."""
    cobj, why = _cooked_main_object(seq, side, data_root, dataset)
    rep["obj_cooked_digest"] = cobj.digest if cobj is not None else None
    rep["obj_slots"] = [n for n, _c, _o, _s in getattr(cobj, "others", ())]
    rep.update({"obj_stills": [b.name for b in cobj.stills]} if getattr(cobj, "stills", ()) else {})
    c0 = cobj.at(_qpos(seq, data_root)[0]) if cobj is not None else None
    sat_fn = _sat_fn(side, pose0, scene, c0) if c0 is not None else None
    opened = RO.gate(rep, sat_fn(np.zeros(HC.nd)) if sat_fn else None, why, desk_deep)
    return sat_fn, cobj, opened


def _window_poses(seq, side, data_root, wrist_root, pose0, n):
    """Per window frame f (at most the clip's length): (wrist position, wrist R, independent fingers, object R, object
    position, qpos row) from the tree's qpos and the side's wrist cache — the same sources as the frame-0 load
    (_load_side)."""
    qpos = _qpos(seq, data_root)
    wr = np.load(wrist_root / f"wrist_poses_{side}" / f"{seq}.npz")
    off, obj_sl = pose0[5], pose0[6]
    for f in range(min(n, len(qpos), len(wr["positions"]))):
        o = qpos[f, obj_sl]
        require_unit(o[3:], f"f0 {seq} [{side}] main object frame {f}")  # _quat_wxyz_to_R does not normalise
        yield (
            wr["positions"][f].astype(np.float64),
            _quat_wxyz_to_R(wr["quaternions"][f].astype(np.float64)),
            qpos[f, off + 6 : off + HC.H][HC.indep_idx].copy(),
            _quat_wxyz_to_R(o[3:]),
            np.asarray(o[:3], np.float64),
            qpos[f],
        )


def _frame_problem(side, cobj, scene, fpose):
    """The projection problem of one window frame and its acceptance measures (repair_obj_solve.solve_frame); the
    object (cooked pieces and touch geometry) at that frame's joint values."""
    hand, _obj, _ck, _oR, _opos, desk_z = scene
    wpos, wR, indep, oR, opos, qrow = fpose
    obj, cobj = _geom_at(scene, cobj.articulation, qrow), cobj.at(qrow)
    strict = _strict_model(side)
    sat_fn = _sat_fn(side, (wpos, None, wR, indep), (hand, obj, _ck, oR, opos, desk_z), cobj)

    def desk_fn(dq):
        wRm = _aa_to_R(dq[3:6]) @ wR
        return RO.desk_max_depth(strict, wpos + dq[:3], wRm, strict.fingers12(indep + dq[6:]), desk_z)

    d0 = desk_fn(np.zeros(HC.nd))
    touch = _contact_links_fn(hand, obj, oR, opos, wpos, wR, indep, RO.OBJ_TOUCH_MM / 1000.0)(np.zeros(HC.nd))
    proj = Projection(strict, cobj, oR, opos, (wpos, wR, indep), desk_z - d0 - 1e-6, _finger_box(hand, indep), touch)
    return {"proj": proj, "sat_fn": sat_fn, "desk_fn": desk_fn, "d0": d0, "touch": touch}


def _obj_steps(seq, side, data_root, wrist_root, rep, pose0, scene, cobj, run, t0):
    """The object branch of a side (rule repair_obj): the projection over the spawn-hold window
    (repair_obj_solve.obj_solve_window), one problem per window frame, built as the solve reaches it. run =
    (verify_retries, _ctr, desk_margin, clearance). Returns the report (no solve requests: the projection runs here)."""
    verify_retries, _ctr, *desk = run
    _ctr[1] = _ctr[0]
    fps = list(_window_poses(seq, side, data_root, wrist_root, pose0, RO.window_frames()))
    RST.side_grids(seq, side, _strict_model(side), [fp[:3] for fp in fps], cobj.stills, rep)
    problems = (_frame_problem(side, cobj, scene, fp) for fp in fps)
    rep = obj_solve_window(rep, pose0, problems, verify_retries, _ctr, t0, RO.desk_start(rep, scene[5], *desk))
    if rep.get("status") == "REPAIRED":
        rep["obj_window_n"] = len(rep["obj_window"])
        rep["obj_window"] = _with_fade(rep["obj_window"], scene[0], seq, side, data_root, wrist_root, pose0)
    rep["t_total_s"] = round(time.time() - t0, 3)
    return rep


def _with_fade(rows, hand, seq, side, data_root, wrist_root, pose0):
    """The window rows + OBJ_FADE_FRAMES rows fading the last change to 0 (each clipped to that frame's finger box)."""
    n, B = len(rows), RO.OBJ_FADE_FRAMES
    out = [np.asarray(r, np.float64) for r in rows]
    for f, fp in enumerate(_window_poses(seq, side, data_root, wrist_root, pose0, n + B)):
        if f < n:
            continue
        dq = rows[-1] * (1.0 - (f - n + 1) / (B + 1))
        lo, hi = _finger_box(hand, fp[2])
        dq[6:] = np.clip(fp[2] + dq[6:], lo, hi) - fp[2]
        out.append(dq)
    return np.array(out)


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
    # Shallow desk penetrations are not repaired: see the module docstring.
    _skip_mm = float(os.environ.get("DEXENGINE_F0_SKIP_DEPTH_MM", "10") or 0)
    desk_deep = not (_skip_mm > 0 and _dmax * 1000.0 <= _skip_mm)
    sat_fn, cobj, opened = _obj_gate(rep, seq, side, data_root, dataset, pose0, scene, desk_deep)
    if opened and do_repair:
        run = (verify_retries, _ctr, desk_margin, clearance)
        return _obj_steps(seq, side, data_root, wrist_root, rep, pose0, scene, cobj, run, t0)
    if n0 == 0 or not do_repair:
        rep["t_total_s"] = round(time.time() - t0, 3)
        return rep
    if not desk_deep:
        return _shallow_skip(rep, _dmax, _skip_mm, t0)

    rep = yield from _desk_steps(
        side,
        rep,
        pose0,
        scene,
        (_n_pen_strict, contact_links, _dm_s, _pm_s),
        (desk_margin, pen_margin, verify_retries, side_budget_evals, _ctr, _clr, clearance),
        t0,
    )
    return RO.after_desk(rep, sat_fn)


def _desk_steps(side, rep, pose0, scene, fns, run, t0):
    """_side_steps' desk branch (a generator; the statements as they were): the budget meter, the finger box, the class
    inits and solve_loop. fns = (_n_pen_strict, contact_links, desk solve margin, object solve margin); run =
    (desk_margin, pen_margin, verify_retries, side_budget_evals, _ctr, _clr, clearance)."""
    wpos0, _wq, wR0, indep0 = pose0[:4]
    hand, _obj, _ck, _oR, opos, desk_z = scene
    _n_pen_strict, contact_links, _dm_s, _pm_s = fns
    desk_margin, pen_margin, verify_retries, side_budget_evals, _ctr, _clr, clearance = run
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
