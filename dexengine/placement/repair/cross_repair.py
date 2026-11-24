#!/usr/bin/env python3
"""Cross repair — the stage-2 step that opens up penetrations of the canonical collision sets (① crossed hand/arm,
② hand <-> opposite object, ④ arm/hand <-> torso) with the minimum displacement.

The robot hand is bigger than the human hand, so following the human hand spacing makes the hands overlap (audit of
128 reference clips: 47 with hand-hand penetration, up to 28.9 mm, 23 of them from the spawn frame).

Position in the chain (`DEXENGINE_CROSS_REPAIR=1`): bake -> f0 -> body -> wrist -> ik -> cross repair -> retime -> body
-> wrist. The distances need the arm joints (hand links are placed by FK), so the repair runs after the one IK solve,
and IK is never re-solved (a full DP re-solve leaves the valley the judge passed); instead the 7 arm joints are
corrected locally around the fixed solution by damped least squares. As in the f0 desk repair, the reference is made
physically feasible with minimal displacement and the time axis is unchanged.

Repairable pairs = only pairs with a hand link (`cross_sets.is_hand_pair`): this step owns the 6D wrist target and the
finger abduction joints, and both move hand links only. Arm-arm and arm-torso are decided by the arm joints (and their
large depths include the visual-hull overshoot at the elbow), so they are reported only. ③ object <-> object is
measured and reported (`obj_obj_max_depth_mm`); its rule belongs to the gate. Moving held objects by different amounts
changes their relative pose — reported as `obj_rel_shift_max_mm` (marker `[XR_OBJ_REL]`; 0 when nothing moved) and
carried into the gate table (obj_rel_shift_reported).

Algorithm (per frame, only the frames P whose hand-pair penetration > --repair-mm; frame 0 joins P whenever its
clearance < --clearance-mm):
  1. finger abduction: the fingers of the penetrating pair (thumb CMC_AA, the others MCP_AA) are spread by +-step, only
     when that frame has no contact (joint limits, --max-finger-deg);
  2. wrist shift along the deepest hand pair's normal (and more directions, see xr_solver.BatchSolver) — half each when
     both ends are hands, all of it on the free hand when only one hand holds an object; a holding hand carries its
     object (grasp invariant) and is never pushed down; shares follow the headroom under each hand's cap
     (--max-wrist-mm; 10 mm for a holding hand with an empty-handed partner), and a proposal over the cap is shortened
     to a partial step;
  3. time: frames are solved in order, taking the better of the cold solve and the one warm-started from t-1, plus the
     smallest alpha back-off that keeps clearance; then 3-point smoothing inside an episode (a contiguous run of P) and
     a C¹ (cos) ramp over +- --window frames outside it; frame 0 is never reduced by the ramp;
  4. guard-aware alpha: every per-frame alpha candidate is re-measured with the canonical arm guard
     (collision.desk_penetration_each · cross_min_dist_each) and an alpha that violates it more than the original is
     dropped (so final_guard does not reject the bake);
  5. application: the arm joints (IK npz) get the local correction, and the tree qpos wrist moves by the displacement
     actually achieved (fingers, held object and the matching qvel channels follow). Writing, the time profile and the
     backups are wrist_shift's: one backup per file (`0/trajectory_kinematic.pre_xr.npz`, `<ik>.pre_xr.npz`; a re-run
     starts from it — the state just before retiming, so re-run the trailing retime -> body -> wrist too). After the
     repair the IK npz `wrist_pos_error_all` is recomputed.
Contact: per side and per finger, from the canonical MANO contact sidecars mapped onto the tree
(cross_steps.contact_frames); a missing sidecar or another length dies.
Distance ruler: `cross_sets.CrossSetModel(robust_zero=True)`.

Output: `0/xr_report.json` (REPORT_REL — per-set min distance before/after, per-frame displacement, statistics,
unresolved frames) · one TSV line in --ledger · rc 0 (REPAIRED / CLEAN / SKIPPED_SHALLOW) · rc 3 (FAILED = unresolved
penetration left; the tree is updated anyway — record it and let placement continue) · rc 2 (input error) · rc 1 = an
uncaught exception. --dry-run writes no tree, needs an explicit --report, and writes its ledger line to the mode-tagged
ledger (paths.mode_record ``.dryrun``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# isort: split
from dexengine.placement.collision import cross_sets as CC
from dexengine.placement.coords import core as _VC  # owner of the desk z offset (scene_z_offset)
from dexengine.placement.repair import wrist_shift as WS  # moving the hand and writing it back
from dexengine.placement.repair import xr_gpu as XG  # batch evaluation and the greedy pass
from dexengine.placement.repair.cross_steps import (
    apply_shift,
    contact_frames,
    per_frame_repair,
    report_after,
    smooth_and_guard,
    state_before,
    write_tree,
)
from dexengine.placement.repair.cross_util import (  # noqa: F401 — re-exported (cross_repair.<name> keeps working)
    LEDGER_HEADER,
    OBJ_REL_KEY,
    SET_REPAIR,
    _ledger,
    _palm_tracking_error,
    _set_stat,
    is_hand_link,
    obj_rel_shift_mm,
    side_of_link,
)

HELP = (
    "cross repair — opens up penetrations of the canonical collision sets (crossed hand/arm, hand <-> opposite "
    "object, arm/hand <-> torso) with the minimum displacement (a stage-2 step)."
)
REPORT_REL = "0/xr_report.json"  # the tree's cross-repair report (the chain's default; --report overrides one run)


def obj_rel_shift_reported(tree) -> float | None:
    """The gate table's object relative-shift column: the value obj_rel_shift_mm gave this tree's cross-repair run, read
    from its report. None = the tree has no report (cross repair off). A report without the value dies (stale)."""
    p = Path(tree) / REPORT_REL
    if not p.is_file():
        return None
    rep = json.loads(p.read_text())
    if OBJ_REL_KEY not in rep:
        raise KeyError(f"[XR_OBJ_REL] {p}: the report has no {OBJ_REL_KEY} — a stale cross-repair report (re-bake)")
    return float(rep[OBJ_REL_KEY])


def build_parser():
    ap = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ik-right", required=True)
    ap.add_argument("--ik-left", required=True)
    ap.add_argument("--assets-root", required=True)
    ap.add_argument(
        "--urdf",
        default="",
        help="empty = picked from full_robot_gen/ by the tree's arm+hand and the canonical variant (HandLayout rule)",
    )
    ap.add_argument("--clearance-mm", type=float, default=2.0, help="repair target clearance (signed distance >= this)")
    ap.add_argument("--repair-mm", type=float, default=2.0, help="only penetrations deeper than this are repaired")
    ap.add_argument("--window", type=int, default=5, help="width [frames] of the C¹ (cos) ramp outside an episode")
    ap.add_argument(
        "--max-wrist-mm",
        type=float,
        default=30.0,
        help="cap on the wrist target shift (mm per hand). Measured on 13 clips: 20 -> 30 raises the removed "
        "hand-pair union from 72 to 86%% and hand-object from 58 to 80%% with no guard loss (arm <= "
        "10.2°, tracking p99 within +13 mm, r max 1.09 -> 1.31 absorbed by the retiming)",
    )
    ap.add_argument(
        "--max-arm-deg",
        type=float,
        default=6.0,
        help="cap on the arm joint increment per damped least-squares step (deg)",
    )
    ap.add_argument("--max-finger-deg", type=float, default=12.0)
    ap.add_argument("--finger-step-deg", type=float, default=3.0)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--distmax", type=float, default=0.03)
    ap.add_argument(
        "--no-carry-object",
        action="store_true",
        help="a holding hand does not carry its object (breaks the grasp geometry — diagnostic)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write no tree — needs --report (the in-tree report is the write run's record) · the ledger "
        "line goes to <ledger>.dryrun.tsv",
    )
    ap.add_argument(
        "--report", default="", help="default <tree>/0/xr_report.json (write runs only — --dry-run must name one)"
    )
    ap.add_argument("--ledger", default="", help="one TSV line per clip (created with a header if missing)")
    # Arm and hand are explicit: with a default, a caller that passes no hand would run every hand in the default
    # hand's guard context. They are checked against the tree identity (task_info combo, robot_type).
    ap.add_argument("--arm", required=True, help="arm id — must equal the arm of the tree's task_info combo")
    ap.add_argument("--hand", required=True, help="hand id — must equal the hand of the tree's task_info robot_type")
    return ap


def _args(ap):
    """Parse, then refuse a dry run without an explicit report: the default <tree>/0/xr_report.json is the write run's
    record (carry_record carries it into the ledger), so a run that writes no tree never writes it."""
    a = ap.parse_args()
    if a.dry_run and not a.report:
        ap.error(
            "--dry-run needs an explicit --report — the default <tree>/0/xr_report.json is the write run's record "
            "(carry_record carries it into the ledger), so a run that writes no tree must not overwrite it"
        )
    return a


def _model_limits(z, L, tree, a, ikR, ikL):
    """Frame time, the cross-set model, the repair thresholds and the arm joint limits."""
    if "frequency" not in z.files or not float(z["frequency"]) > 0:  # speeds and the retime need it — no default rate
        raise ValueError(f"{tree.name}: trajectory_kinematic.npz has no frequency")
    dt = 1.0 / float(z["frequency"])
    # desk_h = the tree→world z offset (coords.scene_z_offset: base layer DESK_HEIGHT − support.desk_z; on TACO staging
    # 0.76 = the former default)
    model = CC.CrossSetModel(
        L,
        tree_dir=tree,
        assets_root=a.assets_root,
        robust_zero=True,  # robust_zero: re-query sporadic 0.0 distances
        desk_h=_VC.scene_z_offset(tree),
    )
    n_zero_fixed = model.zero_requeried
    arm = {"right": ikR[0], "left": ikL[0]}
    base = ikR[2]
    clearance = a.clearance_mm / 1000.0
    repair_thr = -a.repair_mm / 1000.0
    step = np.deg2rad(a.finger_step_deg)
    fmax = np.deg2rad(a.max_finger_deg)
    wmax = a.max_wrist_mm / 1000.0
    carry = not a.no_carry_object

    IK1 = {"right": ikR[1], "left": ikL[1]}
    qlo = np.asarray(model.model.lowerPositionLimit, float)
    qhi = np.asarray(model.model.upperPositionLimit, float)
    lim = {s: (np.array([qlo[model.jq[n]] for n in arm[s]]), np.array([qhi[model.jq[n]] for n in arm[s]])) for s in arm}
    dqmax = np.deg2rad(a.max_arm_deg)
    return dt, model, n_zero_fixed, arm, base, clearance, repair_thr, step, fmax, wmax, carry, IK1, lim, dqmax


def _guard_and_batch(a, base, tree, model, L, qpos, IK1, arm, lim, dqmax, n_dls, carry):
    """The canon arm guard context (desk, torso and two-arm surface distance) and the pose-to-distance batch."""
    # ── The canon arm guard (the same primitives as dexengine/placement/collision/core.py pen_cross_guard: the search
    # scalar desk·torso and the two-arm surface distance) — per-frame α candidates are re-measured with it and any α
    # that violates it more than the original is dropped (blocks final_guard exposure). Violation > 0 = beyond the named
    # tolerance.
    from dexengine.placement.collision import core as SCL

    GCTX = {s_: SCL.build_arm_ctx(a.arm, a.hand, s_) for s_ in ("right", "left")}
    g_zoff = float(base[2] - GCTX["right"]["base"][2])  # IK npz base already holds the z offset
    G_DESK_XY = _VC.desk_bounds_scene_for(tree, margin=0.0)  # the desk top of the tree's dataset
    g_thr, g_cross = SCL.DESK_SURF_PEN_MM / 1000.0, -SCL.ARM_OVERLAP_MAX_MM / 1000.0  # cross: sd ≥ g_cross

    # ── pose -> distance evaluation: the xr_gpu batch. The values are CPU mj_geomDistance (float64); the GPU only
    # filters. The [XR_GPU] marker is printed on every run ("[XR…" lines are kept verbatim: augment record text).
    PB = XG.PoseBatch(model, L, qpos, IK1, arm, base, lim, dqmax, n_dls, carry, XG.GpuDist(model), a.distmax)
    print(
        f"[XR_GPU] exact=1 filter={'cuda' if PB.G.use_gpu else 'cpu'} tol={PB.G.tol:g} iters={PB.G.iters} "
        f"pairs={len(model.pairs)} margin_mm={XG.EXACT_MARGIN * 1000:g} mjw={PB.G.mjw} "
        "(owner xr_gpu · values = CPU float64)",
        flush=True,
    )
    return SCL, GCTX, g_zoff, G_DESK_XY, g_thr, g_cross, PB


def _report_clean(hand_min, report, t0, a, tree, arm_only, obj_obj_mm):
    """Report of a tree with no frame to repair (CLEAN or SKIPPED_SHALLOW), written to the report and the ledger."""
    clean = not (hand_min < 0).any()
    report.update(
        after=report["before"],
        after_hand=report["before_hand"],
        applied=False,
        unresolved_frames=[],
        spawn_after_mm=report["spawn_before_mm"],
        stats={},
        status="CLEAN" if clean else "SKIPPED_SHALLOW",
        **{OBJ_REL_KEY: 0.0},  # nothing moved — the two objects keep their relative pose
        sec=round(time.time() - t0, 1),
    )
    _write(report, a, tree)
    _ledger(report, a)
    print(
        f"[cross_repair] {tree.name}: {report['status']} — 0 frames with hand-pair penetration > {a.repair_mm}mm "
        f"(hand-pair min {hand_min.min() * 1000:.2f}mm · arm-only max {arm_only:.1f}mm · obj-obj max "
        f"{obj_obj_mm:.1f}mm)",
        flush=True,
    )


def _print_summary(report, tree, P, T, residual, n_guard, arm_only, obj_obj_mm):
    """One summary line of the repair (frames, penetration before → after, spawn, wrist, finger and arm moves,
    residual)."""
    bh, ah = report["before_hand"], report["after_hand"]
    bp = sum(bh[s]["pen_frames"] for s in SET_REPAIR)
    apn = sum(ah[s]["pen_frames"] for s in SET_REPAIR)
    print(
        f"[cross_repair] {tree.name}: {report['status']} repair frames {len(P)}/{T} (with ramp "
        f"{report['stats']['frames_touched']}) · hand-pair penetration "
        f"{bp}f/{max(bh[s]['max_depth_mm'] for s in SET_REPAIR)}mm -> "
        f"{apn}f/{max(ah[s]['max_depth_mm'] for s in SET_REPAIR)}mm · "
        f"spawn {report['spawn_before_mm']}->{report['spawn_after_mm']}mm · "
        f"wrist max R {report['stats']['wrist_mm']['right']['max']} L {report['stats']['wrist_mm']['left']['max']}mm · "
        f"finger max R {report['stats']['finger_deg']['right']['max']} "
        f"L {report['stats']['finger_deg']['left']['max']}° · "
        f"arm max R {report['stats']['arm_deg']['right']['max']} L {report['stats']['arm_deg']['left']['max']}° · "
        f"residual {len(residual)}f (guard {n_guard}f) · arm-only {arm_only:.1f}mm · obj-obj max {obj_obj_mm:.1f}mm"
        f" ({report['sec']}s)",
        flush=True,
    )


def _repair_tree(z, L, tree, a, ikR, ikL, qpos, n_dls, T, contact_source, t0, FREEZE, contact, nF, kin_p):
    """Repair one loaded tree: model and limits, the arm guard and the pose batch, the before state, then (when any
    frame needs it) the per-frame repair, smoothing and guard, the shift and the after report, and the written tree.
    Returns rc."""
    dt, model, n_zero_fixed, arm, base, clearance, repair_thr, step, fmax, wmax, carry, IK1, lim, dqmax = _model_limits(
        z, L, tree, a, ikR, ikL
    )
    SCL, GCTX, g_zoff, G_DESK_XY, g_thr, g_cross, PB = _guard_and_batch(
        a, base, tree, model, L, qpos, IK1, arm, lim, dqmax, n_dls, carry
    )

    P, _fqb, arm_only, before, before_h, hand_min, need, obj_obj_mm, report = state_before(
        L, PB, T, a, carry, clearance, contact_source, dt, n_dls, n_zero_fixed, qpos, repair_thr, tree
    )
    if len(P) == 0:
        _report_clean(hand_min, report, t0, a, tree, arm_only, obj_obj_mm)
        return 0

    _BS, dF, dW, fq0, grasp, grasp_all, unresolved = per_frame_repair(
        FREEZE, L, P, PB, T, a, carry, clearance, contact, fmax, nF, qpos, report, step, wmax
    )
    n_guard, n_guard8, w = smooth_and_guard(
        GCTX,
        G_DESK_XY,
        IK1,
        L,
        PB,
        SCL,
        _fqb,
        a,
        clearance,
        dF,
        dW,
        g_cross,
        g_thr,
        g_zoff,
        grasp_all,
        hand_min,
        need,
        qpos,
    )
    after, after_h, armq, dp_ach, new = apply_shift(
        L, PB, T, a, carry, dF, dW, fq0, grasp, grasp_all, ikL, ikR, nF, qpos, report, tree, w
    )
    residual = report_after(
        P,
        PB,
        _BS,
        a,
        after,
        after_h,
        armq,
        before,
        before_h,
        dF,
        dp_ach,
        hand_min,
        ikL,
        ikR,
        model,
        n_guard,
        n_guard8,
        need,
        report,
        t0,
        tree,
        unresolved,
        w,
    )
    write_tree(FREEZE, L, a, arm, armq, base, carry, dp_ach, dt, ikL, ikR, kin_p, model, new, tree, z)
    _write(report, a, tree)
    _ledger(report, a)
    _print_summary(report, tree, P, T, residual, n_guard, arm_only, obj_obj_mm)
    # A residual has its own rc 3 — Python gives rc 1 to an uncaught exception, so a residual reported as rc 1 would
    # make the consumer read "died" as "residual" and continue. rc 1 = exception · rc 2 = input error · rc 3 = residual.
    return 3 if residual else 0


def main() -> int:
    ap = build_parser()
    a = _args(ap)
    n_dls = 2
    t0 = time.time()
    tree = Path(a.tree)
    kin_p = tree / "0/trajectory_kinematic.npz"
    if not kin_p.is_file():
        print(f"[cross_repair] no tree: {kin_p}", flush=True)
        return 2
    # The hand/object qpos addresses and the finger joints are derived from the tree + URDF (HandLayout). A mismatch in
    # width, hand or URDF dies here with an exception.
    L = CC.HandLayout(tree, urdf=a.urdf or None)
    # explicit input <-> tree identity: a mismatch would run in another hand's context, so it dies.
    if (a.arm, a.hand) != (L.arm, L.hand):
        print(
            f"[cross_repair] ✗ --arm/--hand = {a.arm}/{a.hand} differs from the tree identity {L.arm}/{L.hand} — "
            f"not running in another hand's context",
            flush=True,
        )
        return 2
    print(f"[XR_HAND] arm={a.arm} hand={a.hand} (tree identity matches)", flush=True)
    nF = L.nF
    # One-hand rule: when the tree names an idle side (task_info.idle_sides, owner collision.idle_sides_of_tree)
    # that hand and arm are **frozen** — neither finger abduction nor wrist displacement is given to that side (the
    # resting pose is owned by ArmSpec.idle_qpos alone). With no idle side, behaviour is unchanged.
    from dexengine.placement.collision import idle_arm as IA

    FREEZE = set(IA.idle_sides_of_tree(tree))
    if FREEZE:
        print(
            f"[IDLE_SIDE] cross_repair {tree.name}: frozen sides {sorted(FREEZE)} (excluded from the finger and "
            "wrist movers)",
            flush=True,
        )
    z = np.load(WS.source_of(kin_p, "pre_xr"))
    qpos = np.array(z["qpos"], float)
    T = len(qpos)
    ikR = CC.load_ik(WS.source_of(Path(a.ik_right), "pre_xr"))
    ikL = CC.load_ik(WS.source_of(Path(a.ik_left), "pre_xr"))
    if len(ikR[1]) != T or len(ikL[1]) != T:
        print(
            f"[cross_repair] length mismatch tree {T} · ikR {len(ikR[1])} · ikL {len(ikL[1])} — tree and IK are not "
            "the same generation",
            flush=True,
        )
        return 2
    contact, contact_source = contact_frames(T, tree, z)
    return _repair_tree(z, L, tree, a, ikR, ikL, qpos, n_dls, T, contact_source, t0, FREEZE, contact, nF, kin_p)


def _write(report: dict, a, tree: Path):
    p = Path(a.report) if a.report else tree / REPORT_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report, f, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
