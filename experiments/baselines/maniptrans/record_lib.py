"""ManipTrans rollout -> the scorer's motion-form npz (owner of the gym -> CHORD world map and of the object body
names). gym = mujoco2gym · data and world = world_from_data · data, so world = world_from_data · inv(mujoco2gym) ·
gym.
Isaac Gym root-state quaternions are xyzw; the npz stores wxyz. Pure numpy/scipy (runs in ManipTrans' Python 3.8)."""

import numpy as np
from scipy.spatial.transform import Rotation as R


def gym_to_world(world_from_data, mujoco2gym, pos_gym, quat_xyzw_gym):
    W = np.asarray(world_from_data) @ np.linalg.inv(np.asarray(mujoco2gym))
    p = pos_gym @ W[:3, :3].T + W[:3, 3]
    q = (R.from_matrix(W[:3, :3]) * R.from_quat(quat_xyzw_gym.reshape(-1, 4))).as_quat().reshape(quat_xyzw_gym.shape)
    return p, q[..., [3, 0, 1, 2]]


def last_progress(seq_len):
    """ManipTrans' own horizon: an episode SUCCEEDS at the progress p with p + 1 + 3 >= seq_len ('+3 for max future 3
    steps' — dexhandmanip_bih.py compute_imitation_reward 'succeeded'), so the last state it evaluates is p = seq_len
    − 4. State p (after the physics step, compared by its reward with demo[p]) is recorded as the p-th entry → t = p /
    fps."""
    return seq_len - 4  # int or tensor


def method_reference(obj_traj_gym, world_from_data, mujoco2gym, fps_out, seq_len):
    """The object reference ManipTrans is GIVEN (demo obj_trajectory 4×4 — already in the gym frame: ManipData applies
    mujoco2gym on load) → CHORD world with the same transform as the rollout (gym_to_world), up to its own horizon.
    Returns ref_t (K,), pos (K,3), quat wxyz (K,4), horizon_t — the scorer checks it against the original motion
    (≤ 1 mm) and scores only to horizon_t."""
    T = np.asarray(obj_traj_gym, float)[: int(seq_len)]
    pos, q = gym_to_world(world_from_data, mujoco2gym, T[:, :3, 3], R.from_matrix(T[:, :3, :3]).as_quat())
    lp = last_progress(int(seq_len))
    return np.arange(len(T)) / float(fps_out), pos, q, lp / float(fps_out)


def write_npz(path, t, per_body, ref=None):
    """per_body: {body_name: (pos_world (N,T,3), quat_wxyz (N,T,4))} · ref: {body: (ref_t, pos, quat, horizon_t)} from
    method_reference — the scorer requires it (method horizon + reference oracle)."""
    out = {"t": np.asarray(t, float)}
    for name, (p, q) in per_body.items():
        out[f"obj_pos_{name}"] = p
        out[f"obj_quat_{name}"] = q
    for name, (rt, rp, rq, ht) in (ref or {}).items():
        out["ref_t"] = rt
        out[f"ref_obj_pos_{name}"] = rp
        out[f"ref_obj_quat_{name}"] = rq
        out["horizon_t"] = np.float64(ht)
    np.savez(path, **out)
