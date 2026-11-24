"""Post-processing of the raw solve: unwrap, tangent-space smoothing, re-projection,
velocities, object passthrough, and the single debug rollout on the scene model."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from dexengine.retarget.ik.inputs import FINGERS, Keypoints, object_col, tip_col, wrist_col
from dexengine.retarget.ik.model import LimitSets

WINDOW = 3
ROLLOUT_DIVERGED_QPOS = 100.0


@dataclass
class Smoothed:
    qpos: np.ndarray  # (T-3, nq)
    qvel: np.ndarray  # (T-3, nv)


def free_joints(model: mujoco.MjModel) -> list[tuple[int, int, str]]:
    """(qpos address, dof address, body name) of every free joint; each must sit on an object body."""
    out = []
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])) or ""
        if "object" not in body:
            raise ValueError(f"[IK] free joint on non-object body {body!r}")
        out.append((int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]), body))
    return out


def _diff(model: mujoco.MjModel, q1: np.ndarray, q2: np.ndarray, dt: float) -> np.ndarray:
    v = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, v, dt, q1, q2)
    return v


def smooth(model: mujoco.MjModel, raw: np.ndarray, limits: LimitSets, dt: float) -> Smoothed:
    """Unwrap the wrapping hinges, average each forward window of `WINDOW` frames in the tangent space, re-project,
    and difference successive rows into velocities (the first smoothed row is dropped)."""
    q = raw.copy()
    if limits.wrap_adr.size:
        q[:, limits.wrap_adr] = np.unwrap(q[:, limits.wrap_adr], axis=0)
    T = q.shape[0]
    rows = []
    free = [a for a, _, _ in free_joints(model)]
    for k in range(T - WINDOW + 1):
        mean = np.mean([_diff(model, q[k], q[i], 1.0) for i in range(k, k + WINDOW)], axis=0)
        out = q[k].copy()
        mujoco.mj_integratePos(model, out, mean, 1.0)
        mujoco.mj_normalizeQuat(model, out)
        rows.append(out)
    s = np.array(rows)
    if limits.wrap_adr.size:
        s[:, limits.wrap_adr] = limits.wrap_lo + np.mod(s[:, limits.wrap_adr] - limits.wrap_lo, limits.wrap_w)
    for a in free:
        quat = s[:, a + 3 : a + 7]
        s[:, a + 3 : a + 7] = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    if limits.clip_adr.size:
        s[:, limits.clip_adr] = np.clip(s[:, limits.clip_adr], limits.clip_lo, limits.clip_hi)
    qvel = np.array([_diff(model, s[i - 1], s[i], dt) for i in range(1, s.shape[0])])
    return Smoothed(qpos=s[1:], qvel=qvel)


def object_passthrough(model: mujoco.MjModel, sm: Smoothed, kp: Keypoints, dt: float) -> None:
    """In place: object free joints take the raw reference rows 0..T-4 (front-aligned); their velocities are
    finite differences of those rows (row 0 copies row 1)."""
    n = sm.qpos.shape[0]
    for qa, da, body in free_joints(model):
        ref = kp.ref[:n, object_col(body, kp.n_extra)]
        sm.qpos[:, qa : qa + 7] = ref
        vel = np.zeros((n, 6))
        quat = ref[:, 3:7] / np.linalg.norm(ref[:, 3:7], axis=1, keepdims=True)
        for j in range(1, n):
            vel[j, :3] = (ref[j, :3] - ref[j - 1, :3]) / dt
            w = np.zeros(3)
            mujoco.mju_subQuat(w, quat[j], quat[j - 1])
            vel[j, 3:] = w / dt
        if n > 1:
            vel[0] = vel[1]
        sm.qvel[:, da : da + 6] = vel


def rollout(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, ref_dt: float) -> tuple[np.ndarray, float, bool]:
    """Single debug rollout on the scene model: actuators track the output rows.

    Returns (rows, max |qpos|, diverged). Row 0 is the start; row i >= 1 is the state after (i+1)*n_sub steps;
    on a MuJoCo fatal error the rows not yet reached stay zero and the maximum is infinite.
    """
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    n_sub = max(1, int(ref_dt / model.opt.timestep))
    trn = np.array([int(model.actuator_trnid[a][0]) for a in range(model.nu)], dtype=int)
    lim = model.actuator_ctrllimited.astype(bool)
    rng = model.actuator_ctrlrange
    out = np.zeros_like(qpos)
    out[0] = qpos[0]
    d.qpos[:] = qpos[0]
    d.qvel[:] = qvel[0]
    fatal = False

    def control(i: int) -> None:
        c = qpos[i][trn].copy()
        c[lim] = np.clip(c[lim], rng[lim, 0], rng[lim, 1])
        d.ctrl[:] = c

    try:
        control(0)
        for _ in range(n_sub):
            mujoco.mj_step(model, d)
        for i in range(1, qpos.shape[0]):
            control(i)
            for _ in range(n_sub):
                mujoco.mj_step(model, d)
            out[i] = d.qpos
    except mujoco.FatalError:
        fatal = True
    peak = float("inf") if fatal else float(np.abs(out).max())
    diverged = fatal or not np.isfinite(out).all() or peak > ROLLOUT_DIVERGED_QPOS
    return out, peak, diverged


def post_filter_errors(
    model: mujoco.MjModel, qpos: np.ndarray, kp: Keypoints
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Palm and fingertip errors of each output row on the scene model against reference frame min(j+2, T-1)."""
    d = mujoco.MjData(model)
    n = qpos.shape[0]
    palm = {s: np.full(n, np.nan) for s in ("right", "left")}
    tips = np.full((n, 10), np.nan)
    sid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_palm") for s in ("right", "left")}
    tid = {
        (s, k): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_{f}_tip")
        for s in ("right", "left")
        for k, f in enumerate(FINGERS)
    }
    for j in range(n):
        src = min(j + WINDOW // 2 + 1, kp.T - 1)
        d.qpos[:] = qpos[j]
        mujoco.mj_kinematics(model, d)
        for s in ("right", "left"):
            if sid[s] >= 0:
                palm[s][j] = np.linalg.norm(d.site_xpos[sid[s]] - kp.ref[src, wrist_col(s), :3])
            for k in range(5):
                if tid[(s, k)] >= 0:
                    tips[j, (0 if s == "right" else 5) + k] = np.linalg.norm(
                        d.site_xpos[tid[(s, k)]] - kp.ref[src, tip_col(s, k), :3]
                    )
    return palm, tips
