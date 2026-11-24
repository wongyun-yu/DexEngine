"""The solve half of the bake's arm IK stage (coarse_path.dp): reuse of the judge trajectory or collection of
candidates, the coarse-path DP, polish, the arm-flip fix and the npz write. dp.main calls reuse_or_collect_then_dp;
the guards, the stage state (desk box, load time) and the CLI stay in dp.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from dexengine.placement import arm_flip as _AF
from dexengine.placement.coarse_path import path as _CP
from dexengine.placement.coarse_path.dp import (
    DLS_LAMBDA,
    F0_BLEND_FRAMES,
    JUDGE_STRIDE,
    POLISH_OK_POS_M,
    POLISH_OK_ROT_DEG,
    RC_PIN_RETRY,
    REPOLISH_ITERS_MULT,
    arm_flip_fix,
    solver_stamp,
    stage_guards,
)


def _report_low_coverage(a, cov, T):
    """The coverage-gate failure line."""
    print(
        f"[stage_dp] {a.sequence}/{a.side}: candidate cover {cov}/{T} "
        f"({cov / max(T, 1) * 100:.0f}% < {a.min_cov * 100:.0f}%) — failed",
        flush=True,
    )


def _report_err_gate(pef, a, T, _errp99):
    """The wrist-error gate failure line, with where the bad frames sit."""
    # Where it went wrong: clustered at the front = the span the f0 repair inserted (the judge never sees it);
    # scattered = the placement itself is unreachable. Also how many error frames lie on the judge's lattice
    # (stride JUDGE_STRIDE after the F0_BLEND_FRAMES inserted frames).
    _bad = np.flatnonzero(pef * 1000 > a.max_err_mm)
    _loc = (
        f"front{np.mean(_bad < 25) * 100:.0f}%/mid{np.mean((_bad >= 25) & (_bad < T * 0.8)) * 100:.0f}%"
        f"/back{np.mean(_bad >= T * 0.8) * 100:.0f}% total{len(_bad) * 100 / max(T, 1):.0f}%frames"
    )
    _P = F0_BLEND_FRAMES
    _on = np.mean(((_bad - _P) % JUDGE_STRIDE == 0) & (_bad >= _P)) * 100 if len(_bad) else 0.0
    _loc += f" · on the judge lattice (stride5) {_on:.0f}%"
    print(
        f"[stage_dp] {a.sequence}/{a.side}: tracking error p99 {_errp99:.0f}mm "
        f"> {a.max_err_mm:.0f}mm — failed (unreachable placement) [{_loc}]",
        flush=True,
    )


def _report_stretch_gate(a, _stretch, _rmax, stepmax):
    """The stretch-gate failure line."""
    # Stretching would fix it, but the clip would become too long (e.g. a single 228° valley jump needs 63×).
    print(
        f"[stage_dp] {a.sequence}/{a.side}: stretch {_stretch:.1f}x > {a.max_stretch} "
        f"(max speed ratio {_rmax:.0f}, maxΔq={stepmax:.0f}°) — failed",
        flush=True,
    )


def _write_dp_result(a, base, joints, qf, pef, redf, T, _flip, pin, cov, stepmax, t0):
    """Write the solved trajectory's npz and its marker line."""
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{a.sequence}.npz",
        base_xyz=np.asarray(base, dtype=np.float64),  # the robot base this IK was solved against
        arm_joint_names=np.array(joints),
        arm_joint_positions=qf[0].astype(np.float32),
        arm_joint_positions_all=qf.astype(np.float32),
        wrist_pos_error=np.float32(pef[0]),
        wrist_pos_error_all=pef.astype(np.float32),
        wrist_rot_error_deg=np.float32(redf[0]),
        wrist_rot_error_deg_all=redf.astype(np.float32),
        base_z_off=np.float32(a.base_z_off),
        num_frames=np.int32(T),
        solver=solver_stamp("pool_dp_v5", _flip + ("+pin" if pin is not None else "")),
    )
    print(
        f"[stage_dp] {a.sequence}/{a.side}: T={T} coverage {cov}/{T} "
        f"err_p99={np.percentile(pef, 99) * 1000:.1f}mm maxΔq={stepmax:.1f}° "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )


def finish_dp_and_write(
    cand_q, cand_of, a, T, torch, DT, DEV, ik, Tt, BUDGET, FF, VST, base, joints, t0, pin=None, ref=None, rescue=False
):
    """Polish the candidates, run the DP, polish the path, check it and write the npz → stage rc."""
    n_all = len(cand_q)
    cov = sum(1 for _, c in cand_of if c > 0)
    # Coverage gate: frames with no candidate are filled by interpolating the neighbouring joints, and with many
    # gaps that interpolation strays far from the target — so a real gate.
    if cov < a.min_cov * T:
        _report_low_coverage(a, cov, T)
        return 3
    # Batch polish (candidate → that frame's target)
    if n_all:
        q0 = torch.tensor(np.stack(cand_q), dtype=DT, device=DEV)
        idxf = torch.tensor(np.repeat(np.arange(T), [c for _, c in cand_of]), dtype=torch.long, device=DEV)
        qp, pe, red, mg = ik._dls(Tt[idxf], a.polish_iters, DLS_LAMBDA, 0.0, q0)
        ok = ((pe < POLISH_OK_POS_M) & (red < POLISH_OK_ROT_DEG)).cpu().numpy()
        qp = qp.cpu().numpy()
        mgc = mg.cpu().numpy()
    if n_all == 0:
        print(f"[stage_dp] {a.sequence}/{a.side}: no valid candidate — failed", flush=True)
        return 3
    q_dp = _CP.dp_path(qp, ok, mgc, cand_of, T, BUDGET, a.budget_mult, ik.n)
    if q_dp is None:
        print(f"[stage_dp] {a.sequence}/{a.side}: no valid candidate after polish — failed", flush=True)
        return 3
    # Final polish (warm start from the path) + errors
    qf, pef, redf, mgf = ik._dls(Tt, a.final_iters, DLS_LAMBDA, 0.0, torch.tensor(q_dp, dtype=DT, device=DEV))
    qf = qf.cpu().numpy()
    pef = pef.cpu().numpy()
    redf = redf.cpu().numpy()
    _flip = ""
    if T > 1:
        qf, pef, redf, _flip, _bad = arm_flip_fix(
            a,
            ik,
            Tt,
            qf,
            pef,
            redf,
            torch,
            VST,
            BUDGET,
            FF,
            lambda e, q: VST.weighted_p99(e, q, BUDGET, a.safety, floor=FF),
            ref=ref,
        )
        if _bad and rescue:
            return RC_PIN_RETRY
        if _bad:
            print(f"[stage_dp] {a.sequence}/{a.side}: arm flipped — failed (own DP)", flush=True)
            return 3
    stepmax = float(np.rad2deg(np.abs(np.diff(qf, axis=0)).max())) if T > 1 else 0.0
    if pin is not None and T > 1:
        _jmax, _l2 = _AF.judge_jump_max(), float(np.linalg.norm(np.diff(qf, axis=0), axis=1).max())
        print(f"[DP_PIN] {a.sequence}/{a.side}: max single step {_l2:.2f} rad · judge --jump-max {_jmax:g}", flush=True)
        if _l2 > _jmax:
            print(
                f"[stage_dp] {a.sequence}/{a.side}: arm flipped — failed (jointly solved DP step {_l2:.2f} rad)",
                flush=True,
            )
            return 3
    # Shipping check = the judge's and the gate's two criteria, on the **post-retiming** values: retiming only
    # stretches the time axis (the error is unchanged, the speed ratio shrinks by 1/k) but stretches the fast,
    # high-error spans most, so measuring before retiming underestimates.
    _errp99 = (
        VST.weighted_p99(pef * 1000, qf, BUDGET, a.safety, floor=FF) if T > 1 else float(np.percentile(pef, 99) * 1000)
    )
    if _errp99 > a.max_err_mm:
        _report_err_gate(pef, a, T, _errp99)
        return 3
    _stretch = VST.total_stretch(qf, BUDGET, a.safety, floor=FF)
    _rmax = VST.raw_ratio_max(qf, BUDGET)
    if _stretch > a.max_stretch:
        _report_stretch_gate(a, _stretch, _rmax, stepmax)
        return 3
    if not stage_guards(qf, a, a.output_dir):
        return 4
    _write_dp_result(a, base, joints, qf, pef, redf, T, _flip, pin, cov, stepmax, t0)
    return 0


def _reuse_seed(P_ins, qr, ik, Tt, a, torch, DT, DEV):
    """The judge's joints, with the f0-inserted head frames solved from the first posture (same valley)."""
    if P_ins == 0:
        q_seed = qr
    else:
        # Start the inserted span from the original first posture so it stays in the same valley.
        q_head = np.repeat(qr[0:1], P_ins, axis=0)
        qh, _peh, _redh, _ = ik._dls(
            Tt[:P_ins], a.final_iters, DLS_LAMBDA, 0.0, torch.tensor(q_head, dtype=DT, device=DEV)
        )
        q_seed = np.vstack([qh.cpu().numpy(), qr])
    return q_seed


def _borderline_repolish(ik, Tt, a, torch, qf, DT, DEV, VST, BUDGET, FF, _e, _st2, pef, redf, _reuse_ok):
    """One more polish with REPOLISH_ITERS_MULT × the iterations, adopted only when it passes; else the clip goes to the
    DP path."""
    # only on success (it never adds rejections).
    qf3, pef3, redf3, _ = ik._dls(
        Tt, a.final_iters * REPOLISH_ITERS_MULT, DLS_LAMBDA, 0.0, torch.tensor(qf, dtype=DT, device=DEV)
    )
    qf3 = qf3.cpu().numpy()
    pef3 = pef3.cpu().numpy()
    _e3 = float(np.percentile(pef3, 99) * 1000)
    _st3 = VST.total_stretch(qf3, BUDGET, a.safety, floor=FF)
    if _e3 <= a.max_err_mm and _st3 <= a.max_stretch:
        print(
            f"[stage_dp] {a.sequence}/{a.side}: reuse recovered by the borderline re-polish "
            f"({_e:.0f}→{_e3:.0f}mm, {_st2:.2f}→{_st3:.2f})",
            flush=True,
        )
        qf, pef, redf = qf3, pef3, redf3.cpu().numpy()
        _e, _st2 = _e3, _st3
    else:
        # Hand the clip to the DP path: the placement is usually fine, only the reuse seed (repeated first
        # posture + a shallow DLS) cannot follow the f0-inserted span.
        print(
            f"[stage_dp] {a.sequence}/{a.side}: reuse check failed "
            f"(err p99={_e:.0f}mm stretch={_st2:.2f} · after re-polish "
            f"{_e3:.0f}mm/{_st3:.2f}) — retry on the general path",
            flush=True,
        )
        _reuse_ok = False
    return _reuse_ok, qf, pef, redf, _e, _st2


def _write_reused(a, base, joints, qf, pef, redf, T, _flip, P_ins, _e, _st2, t0):
    """Write the reused trajectory's npz and its marker line."""
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / f"{a.sequence}.npz",
        # The robot base this IK was solved against: a consumer standing the robot on another base
        # misses by exactly that much even with correct joint angles.
        base_xyz=np.asarray(base, dtype=np.float64),
        arm_joint_names=np.array(joints),
        arm_joint_positions=qf[0].astype(np.float32),
        arm_joint_positions_all=qf.astype(np.float32),
        wrist_pos_error=np.float32(pef[0]),
        wrist_pos_error_all=pef.astype(np.float32),
        wrist_rot_error_deg=np.float32(redf[0]),
        wrist_rot_error_deg_all=redf.astype(np.float32),
        base_z_off=np.float32(a.base_z_off),
        num_frames=np.int32(T),
        solver=solver_stamp("pool_dp_v5_reuse", _flip),
    )
    print(
        f"[stage_dp] {a.sequence}/{a.side}: reuse T={T} (solved only the {P_ins} inserted frames) "
        f"err_p99={_e:.1f}mm stretch={_st2:.2f} ({time.time() - t0:.1f}s)",
        flush=True,
    )


def _collect_then_dp(
    _load_pool, q_pin, T, a, gi, nx, ny, nz, aiN, torch, DT, DEV, ik, Tt, BUDGET, FF, VST, base, joints, t0
):
    """The DP path: collect candidates from the reach map and solve; one pinned re-solve on RC_PIN_RETRY."""
    K, VQ, VOK = _load_pool()
    # First solve unpinned; if the bake end gate stops it, solve once more pinned around the judge trajectory.
    _ref = q_pin if (q_pin is not None and len(q_pin) == T) else None
    cand_q, cand_of = _CP.collect(a, T, K, VQ, VOK, gi, nx, ny, nz, aiN, None)
    rc = finish_dp_and_write(
        cand_q,
        cand_of,
        a,
        T,
        torch,
        DT,
        DEV,
        ik,
        Tt,
        BUDGET,
        FF,
        VST,
        base,
        joints,
        t0,
        ref=_ref,
        rescue=a.dp_pin_rad > 0 and _ref is not None,
    )
    if rc == RC_PIN_RETRY:
        print(
            f"[DP_PIN] {a.sequence}/{a.side}: solving again pinned within {a.dp_pin_rad:g} rad of the judge trajectory",
            flush=True,
        )
        cand_q, cand_of = _CP.collect(a, T, K, VQ, VOK, gi, nx, ny, nz, aiN, _ref)
        rc = finish_dp_and_write(
            cand_q, cand_of, a, T, torch, DT, DEV, ik, Tt, BUDGET, FF, VST, base, joints, t0, pin=_ref, ref=_ref
        )
        if rc:
            print(
                f"[stage_dp] {a.sequence}/{a.side}: arm flipped — failed (rc {rc} even after re-solving jointly)",
                flush=True,
            )
    return rc


def reuse_or_collect_then_dp(
    a, T, ik, Tt, torch, DT, DEV, VST, BUDGET, FF, base, joints, t0, _load_pool, gi, nx, ny, nz, aiN
):
    """Reuse the judge's joints when possible, else collect candidates and run the DP (pinned retry included)."""
    _reuse_ok = bool(a.reuse_npz) and Path(a.reuse_npz).exists()
    q_pin = None
    _canon = a.arm_flip_fail == _AF.BAKE_GATE_DEFAULT and a.dp_pin_rad == _AF.BAKE_PIN_RAD
    print(
        f"[ARM_FLIP_BAKE] {a.sequence}/{a.side}: fail={a.arm_flip_fail} pin_rad={a.dp_pin_rad:g} "
        f"({'canonical' if _canon else 'explicit'})",
        flush=True,
    )
    if _reuse_ok:
        qr = np.load(a.reuse_npz)["arm_joint_positions_all"].astype(np.float64)
        # The judge looks at the wrist cache before the tail trim, the bake at the trimmed tree, so the judge's
        # joints can be 1~2 frames longer: that much is reused with the tail cut off.
        if 0 < len(qr) - T <= a.max_trim:
            print(
                f"[stage_dp] {a.sequence}/{a.side}: reused trajectory is {len(qr) - T} frames longer "
                f"— tail cut (staging trim)",
                flush=True,
            )
            qr = qr[:T]
        P_ins = T - len(qr)
        # Reuse is only an optimisation: when it cannot be used, fall back to the DP path (never kill the clip).
        if not (0 <= P_ins <= a.max_insert):
            print(
                f"[stage_dp] {a.sequence}/{a.side}: inserted frames {P_ins} out of range "
                f"(0~{a.max_insert}) — reuse skipped, solving on the DP path",
                flush=True,
            )
            _reuse_ok = False
    if _reuse_ok:
        # Reuse holds only on the wrist **before retiming** (a retimed tree would give tens of "inserted" frames).
        q_seed = _reuse_seed(P_ins, qr, ik, Tt, a, torch, DT, DEV)
        q_pin = np.array(q_seed, dtype=np.float64)  # the judge trajectory — read by the pinned re-solve
        qf, pef, redf, _mg = ik._dls(Tt, a.final_iters, DLS_LAMBDA, 0.0, torch.tensor(q_seed, dtype=DT, device=DEV))
        qf = qf.cpu().numpy()
        pef = pef.cpu().numpy()
        redf = redf.cpu().numpy()
        _flip = ""
        if T > 1:
            qf, pef, redf, _flip, _ = arm_flip_fix(
                a, ik, Tt, qf, pef, redf, torch, VST, BUDGET, FF, lambda e, q: float(np.percentile(e, 99)), site="reuse"
            )
        _e = float(np.percentile(pef, 99) * 1000)
        _st2 = VST.total_stretch(qf, BUDGET, a.safety, floor=FF)
        if _e > a.max_err_mm or _st2 > a.max_stretch:
            # Borderline re-polish: a trajectory the judge approved can exceed the threshold by a few mm after the
            # f0 insertion and the batch polish. One more polish with REPOLISH_ITERS_MULT × the iterations, adopted
            _reuse_ok, qf, pef, redf, _e, _st2 = _borderline_repolish(
                ik, Tt, a, torch, qf, DT, DEV, VST, BUDGET, FF, _e, _st2, pef, redf, _reuse_ok
            )
        if _reuse_ok and not stage_guards(qf, a, a.output_dir):
            print(
                f"[stage_dp] {a.sequence}/{a.side}: the reused result hit the penetration / cross guards "
                "— retrying on the DP path",
                flush=True,
            )
            _reuse_ok = False
        if _reuse_ok:
            _write_reused(a, base, joints, qf, pef, redf, T, _flip, P_ins, _e, _st2, t0)
            return 0
        # Reuse failed — fall back to the DP path below.
    return _collect_then_dp(
        _load_pool, q_pin, T, a, gi, nx, ny, nz, aiN, torch, DT, DEV, ik, Tt, BUDGET, FF, VST, base, joints, t0
    )
