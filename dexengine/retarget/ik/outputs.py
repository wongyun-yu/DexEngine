"""Output archives and diagnostics of the hand tracker."""

from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np

from dexengine.retarget.ik.posture import PosturePrior

PALM_VALID_M = 0.05
VIOLATION_VALID = 1e-6
ROLLOUT_SEMANTICS = (
    "qpos_rollout / trajectory_ikrollout.npz hold one deterministic debug rollout of the scene model "
    "(position actuators tracking the output rows, zero control noise); it is a sanity sample, not a "
    "physics-optimised trajectory"
)


def _nan_stat(a: np.ndarray, fn) -> float:
    a = np.asarray(a, dtype=float)
    return float(fn(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan")


def _savez(path: str, arrays: dict) -> None:
    """`np.savez` through a temporary file and an atomic rename (a crash never leaves a half-written archive)."""
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def extra_keys(kp, qpos: np.ndarray, extra_cols: dict[int, int]) -> dict:
    """Keys shared by the kinematic and the rollout archive: extra-object rows and hinge angles of the output rows."""
    n = qpos.shape[0]
    out: dict = {}
    if kp.n_extra > 0:
        out["n_extra_objects"] = np.int64(kp.n_extra)
        for i in range(kp.n_extra):
            a = extra_cols.get(i)
            out[f"extra_obj_{i}_qpos"] = qpos[:, a : a + 7].copy() if a is not None else kp.ref[:n, 12 + i].copy()
    for side, angles in kp.arti.items():
        out[f"qpos_obj_arti_{side}"] = angles[2 : 2 + n]
    return out


def _stats(x: np.ndarray) -> tuple[float, float, float]:
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(x)), float(np.percentile(x, 95)), float(np.max(x))


def prior_report(prior: PosturePrior, config_path: str, qpos: np.ndarray, dt: float) -> tuple[dict, dict]:
    """(diagnostic npz arrays, `posture_prior` JSON block) of a prior run, measured on the output rows."""
    syn = prior.synergy_residual(qpos)
    pq = qpos[:, prior.qpos_adr]
    vel = np.diff(pq, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    # the three per-joint/per-finger time series are stored column-major; values are equal
    # either way, only the npy header flag and payload order change
    arrays = {
        "synergy_residual": np.asfortranarray(syn),
        "synergy_finger_names": np.array([f.name for f in prior.fingers]),
        "prior_weights": prior.weights,
        "prior_joint_qpos_addr": prior.qpos_adr.astype(np.int64),
        "prior_joint_vel": np.asfortranarray(vel),
        "prior_joint_acc": np.asfortranarray(acc),
    }
    col = {int(a): k for k, a in enumerate(prior.qpos_adr)}
    fv, fa, flex, contact = {}, {}, {}, {}
    for k, fg in enumerate(prior.fingers):
        joints = [fg.mcp, fg.pip] + ([fg.dip] if fg.has_dip else []) + fg.abd
        cols = [col[j.qadr] for j in joints]
        v, a = np.abs(vel[:, cols]).ravel(), np.abs(acc[:, cols]).ravel()
        fv[fg.name] = dict(zip(("mean_rad_s", "p95_rad_s", "max_rad_s"), _stats(v), strict=True))
        fa[fg.name] = dict(zip(("mean_rad_s2", "p95_rad_s2", "max_rad_s2"), _stats(a), strict=True))
        st = {}
        for role in ("mcp", "pip", "dip"):
            if role == "dip" and not fg.has_dip:
                continue
            j = getattr(fg, role)
            phi = np.degrees(j.sign * (qpos[:, j.qadr] - j.neutral))
            st[f"{role}_mean_deg"], st[f"{role}_std_deg"] = float(phi.mean()), float(phi.std())
        flex[fg.name] = st
        contact[fg.name] = int(prior.contact_mask[:, k].sum())
    slew = prior.slew_limit
    jumps = int((np.abs(np.diff(pq, axis=0)) > slew).any(axis=1).sum()) if slew > 0 and len(pq) > 1 else 0
    c = prior.conf
    block = {
        "config_path": config_path,
        "lambda_synergy": c["lambda_synergy"],
        "lambda_mcp_share": c["lambda_mcp_share"],
        "lambda_rest": c["lambda_rest"],
        "lambda_abduction": c["lambda_abduction"],
        "lambda_hyperextension": c["lambda_hyperextension"],
        "lambda_limit_avoid": c["lambda_limit_avoid"],
        "limit_margin_deg": c["limit_margin_deg"],
        "limit_avoid_roles": list(c["limit_avoid_roles"]),
        "limit_avoid_joints": [j.name for j in prior.barrier_joints],
        "limit_avoid_bands_deg": {
            j.name: [math.degrees(j.band[0]), math.degrees(j.band[1])] for j in prior.barrier_joints
        },
        "lambda_vel": c["lambda_vel"],
        "damping": c["damping"],
        "contact_weight_scale": c["contact_weight_scale"],
        "slew_rate_limit_rad_per_frame": slew,
        "prior_weight_noncontact": 1.0,
        "prior_weight_contact": c["contact_weight_scale"],
        "contact_frames_per_finger": contact,
        "synergy_residual_mean_rad": float(syn.mean()) if syn.size else float("nan"),
        "synergy_residual_max_rad": float(syn.max()) if syn.size else float("nan"),
        "per_finger_synergy_residual_mean_rad": {f.name: float(syn[:, k].mean()) for k, f in enumerate(prior.fingers)},
        "finger_velocity": fv,
        "finger_acceleration": fa,
        "jump_frames_over_slew_after_filter": jumps,
        "flexion_stats_deg": flex,
    }
    return arrays, block


def diagnostics(
    raw,
    sm_qpos: np.ndarray,
    sm_qvel: np.ndarray,
    post_palm: dict,
    post_tips: np.ndarray,
    *,
    T: int,
    h: float,
    ref_dt: float,
    settle: int,
    init_settle: int,
    multistart: int,
    limit_hx: bool,
    clipped: list[str],
    rollout_peak: float,
    rollout_diverged: bool,
) -> tuple[dict, dict]:
    """(diagnostics npz arrays, diagnostics JSON) of one run; also logs the warning conditions."""
    palm_r, palm_l = raw.palm_err["right"], raw.palm_err["left"]
    palm_ok = np.ones(T, dtype=bool)
    for p in (palm_r, palm_l):
        palm_ok &= ~(np.isfinite(p) & (p > PALM_VALID_M))
    valid = palm_ok & (raw.viol_final <= VIOLATION_VALID)
    arrays = {
        "palm_error_right": palm_r,
        "palm_error_left": palm_l,
        "fingertip_error": raw.tip_err,
        "joint_violation_preclamp": raw.viol_pre,
        "joint_violation_final": raw.viol_final,
        "valid_mask": valid.astype(bool),
        "post_filter_palm_error_right": post_palm["right"],
        "post_filter_palm_error_left": post_palm["left"],
        "post_filter_fingertip_error": post_tips,
    }
    has_bad = not (np.isfinite(sm_qpos).all() and np.isfinite(sm_qvel).all())
    far = np.zeros(T, dtype=bool)
    for p in (palm_r, palm_l):
        far |= np.isfinite(p) & (p > PALM_VALID_M)
    gt5 = int(far.sum())
    info = {
        "num_raw_frames": T,
        "num_filtered_frames": int(sm_qpos.shape[0]),
        "settle_steps": settle,
        "ik_multistart": multistart,
        "init_settle_steps": init_settle,
        "ik_physical_mode": True,
        "limit_hyperextension": bool(limit_hx),
        "hyperext_limited_joints": list(clipped),
        "average_frame_size": 3,
        "sim_dt": h,
        "ref_dt": ref_dt,
        "has_nan_or_inf": bool(has_bad),
        "palm_error_right_mean_m": _nan_stat(palm_r, np.mean),
        "palm_error_right_max_m": _nan_stat(palm_r, np.max),
        "palm_error_left_mean_m": _nan_stat(palm_l, np.mean),
        "palm_error_left_max_m": _nan_stat(palm_l, np.max),
        "fingertip_error_mean_m": _nan_stat(raw.tip_err, np.mean),
        "fingertip_error_max_m": _nan_stat(raw.tip_err, np.max),
        "post_filter_palm_error_right_mean_m": _nan_stat(post_palm["right"], np.mean),
        "post_filter_palm_error_left_mean_m": _nan_stat(post_palm["left"], np.mean),
        "post_filter_fingertip_error_mean_m": _nan_stat(post_tips, np.mean),
        "joint_violation_preclamp_max_rad": float(raw.viol_pre.max()),
        "joint_violation_final_max_rad": float(raw.viol_final.max()),
        "num_valid_frames": int(valid.sum()),
        "num_frames_palm_error_gt_5cm": gt5,
        "rollout_noise_std": 0.0,
        "rollout_max_abs_qpos": rollout_peak,
        "rollout_diverged": bool(rollout_diverged),
        "rollout_semantics": ROLLOUT_SEMANTICS,
    }
    _warn(info)
    return arrays, info


def _warn(info: dict) -> None:
    def w(msg: str) -> None:
        warnings.warn(f"[IK] {msg}", stacklevel=3)

    if info["has_nan_or_inf"]:
        w("output has NaN/Inf")
    if info["joint_violation_final_max_rad"] > VIOLATION_VALID:
        w(f"final joint violation {info['joint_violation_final_max_rad']:.3g} > {VIOLATION_VALID}")
    for k in ("palm_error_right_mean_m", "palm_error_left_mean_m", "fingertip_error_mean_m"):
        if np.isfinite(info[k]) and info[k] > 0.03:
            w(f"{k} = {info[k] * 1e3:.1f} mm > 30 mm")
    if info["num_frames_palm_error_gt_5cm"]:
        w(f"{info['num_frames_palm_error_gt_5cm']} frames with palm error > 5 cm")
    for raw_k, post_k in (
        ("palm_error_right_mean_m", "post_filter_palm_error_right_mean_m"),
        ("palm_error_left_mean_m", "post_filter_palm_error_left_mean_m"),
        ("fingertip_error_mean_m", "post_filter_fingertip_error_mean_m"),
    ):
        if np.isfinite(info[raw_k]) and np.isfinite(info[post_k]) and info[post_k] - info[raw_k] > 0.002:
            w(f"{post_k} worse than raw by {(info[post_k] - info[raw_k]) * 1e3:.1f} mm")


def write_all(out_dir: str, kin: dict, roll: dict, diag_npz: dict, diag_json: dict) -> None:
    """Write the kinematic, rollout and diagnostics archives and the diagnostics JSON (each atomically)."""
    os.makedirs(out_dir, exist_ok=True)
    _savez(os.path.join(out_dir, "trajectory_kinematic.npz"), kin)
    _savez(os.path.join(out_dir, "trajectory_ikrollout.npz"), roll)
    _savez(os.path.join(out_dir, "retarget_diagnostics.npz"), diag_npz)
    tmp = os.path.join(out_dir, f"retarget_diagnostics.json.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(diag_json, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, "retarget_diagnostics.json"))
