"""The steps of the cross repair (repair/cross_repair.py main): contact flags, state before, per-frame repair, time
smoothing with the non-regression guard, application through the arm joints, the after-state report and the tree
write. Each step takes the values it reads and returns the values main reads after it.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np

from dexengine.placement.collision import cross_sets as CC
from dexengine.placement.collision import det_math as D
from dexengine.placement.coords import core as _VC
from dexengine.placement.repair import wrist_shift as WS
from dexengine.placement.repair import xr_gpu as XG
from dexengine.placement.repair.cross_report import after_fields
from dexengine.placement.repair.cross_util import (
    OBJ_REL_KEY,
    SET_REPAIR,
    _palm_tracking_error,
    _set_stat,
    is_hand_link,
    obj_rel_shift_mm,
    side_of_link,
)


def state_before(L, PB, T, a, carry, clearance, contact_source, dt, n_dls, n_zero_fixed, qpos, repair_thr, tree):
    """State before the repair — per-set minimum distances, the frames that need it and the report head."""
    # ── State before ────────────────────────────────────────────────────────
    before = {s: np.full(T, a.distmax) for s in CC.SETS}  # all pairs of the set
    before_h = {s: np.full(T, a.distmax) for s in CC.SETS}  # only the pairs with a hand link (the repair target)
    _fqb = {s: np.stack([L.hand_blocks(qpos[t])[0][s] for t in range(T)]) for s in ("right", "left")}
    _, _, _mn, _hm, _, _ = PB.eval(np.arange(T), _fqb, None, None, want_pairs=False)
    for _k, s in enumerate(CC.SETS):
        before[s][:] = _mn[:, _k]
        before_h[s][:] = _hm[:, _k]
    hand_min = np.min([before_h[s] for s in SET_REPAIR], axis=0)
    need = hand_min < repair_thr
    spawn_forced = bool(hand_min[0] < clearance and not need[0])
    if spawn_forced:
        need[0] = True  # spawn is always opened to clearance (blocks instant A1 blow-up)
    P = np.where(need)[0]
    obj_obj_mm = float(-before["3_obj_obj"].min() * 1000) if (before["3_obj_obj"] < 0).any() else 0.0
    arm_only = max((float(-before[s].min() * 1000) for s in SET_REPAIR if (before[s] < 0).any()), default=0.0)
    report = {
        "tree": str(tree),
        "clip": tree.name,
        "T": T,
        "params": {
            "clearance_mm": a.clearance_mm,
            "repair_mm": a.repair_mm,
            "window": a.window,
            "max_wrist_mm": a.max_wrist_mm,
            "max_finger_deg": a.max_finger_deg,
            "sets": list(SET_REPAIR),
            "carry_object": carry,
            "dt": dt,
        },
        "before": {s: _set_stat(before[s]) for s in CC.SETS},
        "before_hand": {s: _set_stat(before_h[s]) for s in SET_REPAIR},
        "obj_obj_max_depth_mm": round(obj_obj_mm, 1),
        "arm_only_max_depth_mm": round(arm_only, 1),
        "spawn_before_mm": round(float(hand_min[0] * 1000), 2),
        "spawn_forced": spawn_forced,
        "n_repair_frames": int(len(P)),
        "frames": [],
    }
    report["params"].update(
        mode="v2", contact_source=contact_source, zero_requeried=n_zero_fixed, max_arm_deg=a.max_arm_deg, n_dls=n_dls
    )

    return P, _fqb, arm_only, before, before_h, hand_min, need, obj_obj_mm, report


def _round_robin(_eps, _BS, _prevs, _cold, clearance, _chosen):
    """Frame rounds across the episodes: cold solutions, warm starts from the previous frame, back-off; fills _chosen
    and _prevs."""
    for _r in range(max((len(e) for e in _eps), default=0)):
        _fr = [(ei, e[_r]) for ei, e in enumerate(_eps) if len(e) > _r]
        _wj = [(ei, t) for ei, t in _fr if _r > 0]  # warm = starts from the previous frame's solution
        _wres = {}
        if _wj:
            for (_ei, t), res in zip(_wj, _BS.solve([_BS.new(t, _prevs[ei]) for ei, t in _wj]), strict=True):
                _wres[t] = res
        _sel = {}
        for _ei, t in _fr:
            shift, fq, fq0, final, rec, grasp = _cold[t]
            if t in _wres:
                w_shift, w_fq, _, w_final, w_rec, _ = _wres[t]
                w_rec["warm"] = True
                if w_final >= clearance - 1e-6 or w_final > final + 1e-4:
                    shift, fq, final, rec = w_shift, w_fq, w_final, w_rec
            _sel[t] = [shift, fq, fq0, final, rec, grasp]
        _bj = [
            t
            for ei, t in _fr
            if _sel[t][3] >= clearance - 1e-6 and any(float(D.norm(_sel[t][0][k])) > 1e-9 for k in _sel[t][0])
        ]
        for t, o in zip(
            _bj, _BS.backoff([(t, _sel[t][0], _sel[t][1], _sel[t][2], _sel[t][5]) for t in _bj]), strict=True
        ):
            if o is not None:
                _sel[t][0], _sel[t][1], _sel[t][3] = o[0], o[1], o[2]
                _sel[t][4]["backoff"] = o[3]
        for ei, t in _fr:
            shift, fq, fq0, final, rec, grasp = _sel[t]
            _prevs[ei] = (int(t), {k: shift[k].copy() for k in shift}, {k: fq[k] - fq0[k] for k in fq})
            _chosen[t] = _sel[t]


def per_frame_repair(FREEZE, L, P, PB, T, a, carry, clearance, contact, fmax, nF, qpos, report, step, wmax):
    """Per-frame greedy repair of the frames in P (xr_gpu.BatchSolver) — wrist shift dW and finger change dF per
    side."""
    # ── Per-frame repair ──────────────────────────────────────────────────────
    dW = {"right": np.zeros((T, 3)), "left": np.zeros((T, 3))}
    dF = {"right": np.zeros((T, nF)), "left": np.zeros((T, nF))}
    grasp_all = {"right": contact[:, :5].any(axis=1), "left": contact[:, 5:10].any(axis=1)}
    unresolved = []
    # The greedy is xr_gpu.BatchSolver: (1) the cold solutions of every frame of P in one lock-step batch; (2) the warm
    # solves depend on the previous frame of the same episode (a contiguous run of P), so the r-th frames of all
    # episodes are solved together round by round — the order dependence is kept; (3) the three alpha back-off
    # candidates in one batch too.
    _BS = XG.BatchSolver(
        PB,
        {
            "L": L,
            "qpos": qpos,
            "grasp_all": grasp_all,
            "contact": contact,
            "fmax": fmax,
            "step": step,
            "FREEZE": FREEZE,
            "carry": carry,
            "wmax": wmax,
            "clearance": clearance,
            "distmax": a.distmax,
            "iters": a.iters,
            "SET_REPAIR": SET_REPAIR,
            "CC": CC,
            "side_of_link": functools.partial(side_of_link, arm=PB.M.arm_spec),
            "is_hand_link": functools.partial(is_hand_link, arm=PB.M.arm_spec),
        },
    )
    _cold = {int(t): r for t, r in zip(P, _BS.solve([_BS.new(t) for t in P]), strict=True)}
    _eps = []
    for t in P:
        if _eps and _eps[-1][-1] == t - 1:
            _eps[-1].append(int(t))
        else:
            _eps.append([int(t)])
    _chosen = {}
    _prevs = {}
    _round_robin(_eps, _BS, _prevs, _cold, clearance, _chosen)
    for t in P:  # records in P order
        shift, fq, fq0, final, rec, grasp = _chosen[int(t)]
        rec.update(
            final_dist_mm=round(final * 1000, 2),
            resolved=bool(final >= clearance - 1e-6),
            wrist_mm={k: round(float(D.norm(shift[k]) * 1000), 1) for k in shift},
            finger_deg={k: round(float(np.rad2deg(np.abs(fq[k] - fq0[k]).max())), 1) for k in fq},
        )
        if not rec["resolved"]:
            unresolved.append(int(t))
        for k in shift:
            dW[k][t] = shift[k]
            dF[k][t] = fq[k] - fq0[k]
        report["frames"].append(rec)

    return _BS, dF, dW, fq0, grasp, grasp_all, unresolved


def smooth_and_guard(
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
):
    """Time smoothing and ramp of the shifts, then the per-frame non-regression guard on the touched frames."""
    # ── Time smoothing + ramp (wrist_shift: 3-point in the episode, C¹ ramp outside, spawn frame kept) ─────
    w = WS.smooth_ramp([dW["right"], dF["right"], dW["left"], dF["left"]], need, a.window, det=True)

    # ── Per-frame non-regression guard ──────────────────────────────────────── 3-point smoothing and the ramp change
    # the per-frame optimum and can make the max depth **worse** (measured on brush_eraser_bowl__20231103_042:
    # 22.1 → 30.3mm). No frame may end up worse than before the repair, so for those frames only we shrink the
    # amplitude α (α=0 is the original = always safe) and pick the best α. Only touched frames are examined, so the cost
    # is small.
    touched_idx = np.where(w > 0)[0]
    n_guard = 0
    n_guard8 = 0
    if len(touched_idx):
        # touched frames x alpha (1, .75, .5, .25) are evaluated speculatively in one batch; the choice goes by
        # descending alpha with a break once wide enough. An alpha that violates the arm guard (the primitives of
        # collision pen_cross_guard) more than the original is discarded.
        _AL = (1.0, 0.75, 0.5, 0.25)
        _it = [(int(t), al) for t in touched_idx for al in _AL]
        _ts = np.array([x[0] for x in _it])
        _fq0 = {int(t): L.hand_blocks(qpos[t])[0] for t in touched_idx}
        _fqb = {k: np.stack([_fq0[t][k] + dF[k][t] * al for t, al in _it]) for k in ("right", "left")}
        _shb = {k: np.stack([dW[k][t] * al for t, al in _it]) for k in ("right", "left")}
        _grb = {k: np.array([bool(grasp_all[k][t]) for t, _ in _it]) for k in ("right", "left")}
        _aqb, _, _, _, _rows, _ = PB.eval(_ts, _fqb, _shb, _grb, want_pairs=True, worst_sets=SET_REPAIR)
        _v = _rows.worst_v  # = the worst-hand-pair rule (xr_gpu.RowSet)

        def _gv(aqR, aqL):
            v_ = np.full(len(aqR), -1e9)
            for s_, q_ in (("right", aqR), ("left", aqL)):
                v_ = np.maximum(v_, SCL.desk_penetration_each(GCTX[s_], q_, z_off=g_zoff, desk_xy=G_DESK_XY) - g_thr)
            return np.maximum(v_, g_cross - SCL.cross_min_dist_each(GCTX["right"], GCTX["left"], aqR, aqL, subdiv=1))

        _gva = _gv(_aqb["right"], _aqb["left"])
        _gv0 = np.maximum(
            0.0,
            _gv(
                np.stack([np.asarray(IK1["right"][t], float) for t in touched_idx]),
                np.stack([np.asarray(IK1["left"][t], float) for t in touched_idx]),
            ),
        )
        for _i, t in enumerate(touched_idx):
            cands = []
            for _j, alpha in enumerate(_AL):
                _n = _i * len(_AL) + _j
                v = float(_v[_n])
                if _gva[_n] > _gv0[_i] + 1e-6:
                    n_guard8 += 1
                    continue
                cands.append((v, alpha))
                if v >= clearance - 1e-9:
                    break
            cands.append((float(hand_min[t]), 0.0))
            _, best_a = max(cands, key=lambda x: (x[0], x[1]))
            if best_a < 1.0:
                for k in dW:
                    dW[k][t] *= best_a
                    dF[k][t] *= best_a
                n_guard += 1

    return n_guard, n_guard8, w


def apply_shift(L, PB, T, a, carry, dF, dW, fq0, grasp, grasp_all, ikL, ikR, nF, qpos, report, tree, w):
    """Apply the shifts through the arm joints and the tree (wrist, fingers, held object); the after state."""
    # ── Application: arm joints (local correction) + tree (wrist, fingers, held object) ──────── The wrist displacement
    # is realised through the **arm joints** (arm_at → cross_sets.CrossSetModel.arm_dq). That is why the "after state"
    # below is not an approximation but the state of the joint values training will actually see. The arm IK is never
    # re-solved (the reason is in the arm_dq docstring).
    new = np.array(qpos)
    armq = {"right": np.array(ikR[1], float), "left": np.array(ikL[1], float)}
    dp_ach = {"right": np.zeros((T, 3)), "left": np.zeros((T, 3))}
    after = {s: np.full(T, a.distmax) for s in CC.SETS}
    after_h = {s: np.full(T, a.distmax) for s in CC.SETS}
    # all T frames at once (untouched frames have shift 0, which arm_at skips)
    _fqa = {k: np.zeros((T, nF)) for k in ("right", "left")}
    for t in range(T):
        fq0, _ = L.hand_blocks(qpos[t])
        for k in _fqa:
            _fqa[k][t] = fq0[k] + dF[k][t]
        for k in _fqa:
            for aa in L.aa[k].values():
                if aa is not None:
                    _fqa[k][t, aa[0]] = float(np.clip(_fqa[k][t, aa[0]], aa[1], aa[2]))
    _shA = {k: np.where((w > 0)[:, None], dW[k], 0.0) for k in dW}
    _gra = {k: np.asarray(grasp_all[k], bool) for k in ("right", "left")}
    _aqa, _acha, _mna, _hma, _, _ = PB.eval(np.arange(T), _fqa, _shA, _gra, want_pairs=False)
    for _k, s in enumerate(CC.SETS):
        after[s][:] = _mna[:, _k]
        after_h[s][:] = _hma[:, _k]
    dobj = {k: np.zeros((T, 3)) for k in L.hands}
    for t in range(T):
        grasp = {k: bool(grasp_all[k][t]) for k in grasp_all}
        for k, off in L.hands.items():
            armq[k][t] = _aqa[k][t]
            dp_ach[k][t] = _acha[k][t]
            dobj[k][t] = WS.shift_row(
                new[t],
                off,
                L.finger_off,
                L.H,
                D.matvec3(CC.RZ_POS90, _acha[k][t]),
                _fqa[k][t],
                L.obj[k] if carry and grasp[k] else None,
            )
    rel = obj_rel_shift_mm(dobj, L.obj, L.shared_object)
    report[OBJ_REL_KEY] = round(float(rel), 2)
    print(
        f"[XR_OBJ_REL] {tree.name}: max relative motion of the two objects {rel:.1f} mm (report only — "
        f"the object-object rule is the gate's)",
        flush=True,
    )
    return after, after_h, armq, dp_ach, new


def report_after(
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
):
    """Residual frames and the after-state report fields."""
    hand_min_after = np.min([after_h[s] for s in SET_REPAIR], axis=0)
    # The verdict is taken on the **final state**. A "clearance not reached" during the per-frame optimisation
    # (unresolved_opt) is changed afterwards by the ramp and the non-regression guard, so using it as-is reads a clip
    # with penetration 0 as FAILED (measured: 20231027_006).
    residual = np.where(hand_min_after < 0)[0].tolist()
    wr = {k: D.norm(dp_ach[k]) * 1000 for k in dp_ach}  # achieved wrist displacement (mm)
    fd = {k: np.rad2deg(np.abs(dF[k]).max(axis=1)) for k in dF}
    ad = {k: np.rad2deg(np.abs(armq[k] - {"right": ikR[1], "left": ikL[1]}[k]).max(axis=1)) for k in armq}
    touched = w > 0
    after_fields(
        report,
        after,
        after_h,
        a,
        residual,
        unresolved,
        need,
        hand_min_after,
        touched,
        hand_min,
        n_guard,
        n_guard8,
        wr,
        fd,
        ad,
        t0,
    )
    report["params"]["xr_gpu"] = {
        "device": PB.G.dev,
        "calls": PB.G.ncall,
        "queries": PB.G.nq_total,
        "exact": list(PB.G.nexact),
        "batches": (_BS.nbatch if len(P) else 0),
        "poses": (_BS.nposes if len(P) else 0),
    }
    print(
        f"[XR_GPU] {tree.name}: calls={PB.G.ncall} queries={PB.G.nq_total} exact={PB.G.nexact} "
        f"zeros={model.zero_requeried} "
        f"solver_batches={(_BS.nbatch if len(P) else 0)} solver_poses={(_BS.nposes if len(P) else 0)}",
        flush=True,
    )
    report["frame_min_before_mm"] = {s: np.round(before[s] * 1000, 2).tolist() for s in CC.SETS}
    report["frame_min_after_mm"] = {s: np.round(after[s] * 1000, 2).tolist() for s in CC.SETS}
    report["frame_hand_before_mm"] = {s: np.round(before_h[s] * 1000, 2).tolist() for s in SET_REPAIR}
    report["frame_hand_after_mm"] = {s: np.round(after_h[s] * 1000, 2).tolist() for s in SET_REPAIR}
    return residual


def write_tree(FREEZE, L, a, arm, armq, base, carry, dp_ach, dt, ikL, ikR, kin_p, model, new, tree, z):
    """Write the repaired tree (qpos, IK npz, wrist cache) unless --dry-run."""
    if not a.dry_run:
        vel = [
            c
            for off in L.hands.values()
            for c in ((off, off, 3), (off + L.finger_off, off + L.finger_off, L.H - L.finger_off))
        ]
        vel += [(L.obj[k], L.obj_vel[k], 3) for k in ("right", "left") if carry and L.obj[k] is not None]
        out = WS.apply_delta(z, new, vel, dt, "gradient")
        out["xr_wrist_shift_world_right"] = dp_ach["right"].astype(np.float32)
        out["xr_wrist_shift_world_left"] = dp_ach["left"].astype(np.float32)
        WS.save_npz(kin_p, out, "pre_xr")
        # The trajectory changed, so the support stamp is recomputed from it (carrying a held object moves its rest
        # and start positions).
        if tree.joinpath("task_info.json").is_file() and _VC.is_baked_tree(
            json.loads(tree.joinpath("task_info.json").read_text())
        ):
            _VC.stamp_scene(tree, keep_height=True)
        # Arm joints — written together with the tree, as **the same generation**. If only one side got written it would
        # be a half-write, hence atomic replace + backup (fingerprinted).
        for side, p in (("right", Path(a.ik_right)), ("left", Path(a.ik_left))):
            bk = p.with_suffix(".pre_xr.npz")  # WS.source_of has already made the backup and its fingerprint
            zz = dict(np.load(bk, allow_pickle=True))
            if side in FREEZE and np.abs(armq[side] - {"right": ikR[1], "left": ikL[1]}[side]).max() > 0:
                raise RuntimeError(
                    f"[cross_repair] {tree.name}: the arm joints of frozen side {side} changed — "
                    f"frozen-side rule violated"
                )
            zz["arm_joint_positions_all"] = armq[side].astype(np.asarray(zz["arm_joint_positions_all"]).dtype)
            if "arm_joint_positions" in zz:
                zz["arm_joint_positions"] = armq[side][0].astype(np.asarray(zz["arm_joint_positions"]).dtype)
            if side not in FREEZE:  # frozen side keeps error 0 (still arm)
                # The arm joints changed, so the tracking-error arrays in the IK npz are re-measured too
                # (so the gate's diagnostic column err_p99 does not show a stale value).
                # Target = the same expression as the arm IK: Rz(−90)·(tree wrist position (after the repair) +
                #          R_palm·palm_off) + (0,0,DESK) − base.
                # The orientation R_palm is not changed by the repair, so the palm body quaternion from
                # 0/body_positions.npz (= the source wrist_poses was extracted from) is used.
                errs = _palm_tracking_error(model, arm[side], armq[side], new, side, tree, base, a.arm, a.hand)
                if errs is not None:
                    pe, re_ = errs
                    zz["wrist_pos_error_all"] = pe.astype(np.float32)
                    zz["wrist_rot_error_deg_all"] = re_.astype(np.float32)
                    zz["wrist_pos_error"] = np.float32(pe[0])
                    zz["wrist_rot_error_deg"] = np.float32(re_[0])
                    zz["xr_tracking_recomputed"] = np.array("cross_repair", dtype="<U16")
                else:
                    print(
                        f"[cross_repair] {side}: body_positions.npz missing or mismatched — wrist_pos_error_all not "
                        "recomputed (the stale value stays)",
                        flush=True,
                    )
            WS.save_npz(p, zz, "pre_xr")


def contact_frames(T, tree, z):
    """Per-frame, per-finger contact of both hands (T, 10: right 5, then left 5) for the holding-hand and finger-lock
    decisions, read per side from the canonical MANO contact sidecars mapped onto this tree
    (coords.contacts.finger_contacts), and its source label. The repair runs before retiming: a qpos in hand that
    carries a retime map dies."""
    from dexengine.placement.coords import contacts as CT  # noqa: PLC0415

    if "retime_src_idx" in z.files:
        raise ValueError(f"{tree}: the trajectory in hand is retimed — the cross repair reads the pre-retime state")
    ti = json.loads((tree / "task_info.json").read_text(encoding="utf-8"))
    return CT.finger_contacts(tree, ti, T), "contact_mano_base"
