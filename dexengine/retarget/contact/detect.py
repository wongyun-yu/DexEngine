"""Per-frame fingertip contact evaluation and the contact keys it writes."""

from __future__ import annotations

import json
import os
import warnings

import mujoco
import numpy as np

from dexengine.retarget import mjutil
from dexengine.retarget.contact.scene import SIDES, ContactScene, build_contact_scene, hand_body_names

#: per-axis spread of a finger's contact points (population std, m) above which the finger is disabled
MAX_CONTACT_SPREAD = 0.1


def frame_poses(z: dict, t: int) -> list[np.ndarray]:
    """The 14 body poses of frame `t` in body order: wrists and tips per side, then the two objects."""
    out = []
    for side in SIDES:
        out.append(z[f"qpos_wrist_{side}"][t])
        out.extend(z[f"qpos_finger_{side}"][t])
    out.append(z["qpos_obj_right"][t])
    out.append(z["qpos_obj_left"][t])
    return out


def evaluate(scene: ContactScene, z: dict) -> tuple[np.ndarray, np.ndarray]:
    """(raw contact (T, 10) bool, fingertip positions in their object frame (T, 10, 3)) of every frame."""
    m = scene.model
    d = mujoco.MjData(m)
    T = z["qpos_wrist_right"].shape[0]
    n_free = len(hand_body_names()) + 2
    raw = np.zeros((T, 10), dtype=bool)
    pos = np.zeros((T, 10, 3))
    hinge_angle = dict.fromkeys(scene.hinge_qadr, 0.0)
    for t in range(T):
        poses = np.concatenate([np.asarray(p, dtype=np.float64) for p in frame_poses(z, t)])
        q = np.zeros(m.nq)
        free_adr = [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free_adr) != n_free:
            raise RuntimeError(f"[CONTACT] scene has {len(free_adr)} free joints, expected {n_free}")
        for k, a in enumerate(free_adr):
            q[a : a + 7] = poses[7 * k : 7 * k + 7]
        for side, adr in scene.hinge_qadr.items():
            key = f"qpos_obj_arti_{side}"
            if key in z:
                hinge_angle[side] = float(z[key][t, 0])
            q[adr] = hinge_angle[side]
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_step(m, d)
        sens = d.sensordata
        raw[t] = sens[:10] > 0
        pos[t] = sens[10:40].reshape(10, 3)
    return raw, pos


def postprocess(raw: np.ndarray, pos: np.ndarray, tag: str = "") -> tuple[np.ndarray, np.ndarray]:
    """(smoothed contact (T, 10) float 0/1, mean contact point per fingertip (10, 3)) with spread disabling."""
    T = raw.shape[0]
    c = np.zeros(raw.shape, dtype=np.float64)
    for t in range(T):
        c[t] = raw[max(0, t - 1) : min(T, t + 2)].any(axis=0)
    mean = np.zeros((10, 3))
    for k in range(10):
        sel = raw[:, k]
        if not sel.any():
            continue
        pts = pos[sel, k]
        spread = pts.std(axis=0)
        if (spread > MAX_CONTACT_SPREAD).any():
            warnings.warn(
                f"[CONTACT] {tag} fingertip {k}: contact points spread {np.round(spread, 3).tolist()} m "
                f"> {MAX_CONTACT_SPREAD} m on some axis; finger disabled",
                stacklevel=2,
            )
            c[:, k] = 0.0
            continue
        mean[k] = pts.mean(axis=0)
    return c, mean


def contact_keys(emb: str, c: np.ndarray, mean: np.ndarray) -> dict[str, np.ndarray]:
    """The four output arrays laid out per embodiment.

    Insertion order is the archive's member order for keys not yet present: left before
    right, each flag array before its mean positions. Keys already in the archive keep their place.
    """
    T = c.shape[0]
    if emb == "bimanual":
        right, left, pos_right, pos_left = c[:, 0:5], c[:, 5:10], mean[0:5], mean[5:10]
    elif emb == "right":
        right, left, pos_right, pos_left = c, np.zeros((T, 10)), mean, np.zeros((10, 3))
    elif emb == "left":
        right, left, pos_right, pos_left = np.zeros((T, 10)), c, np.zeros((10, 3)), mean
    else:
        raise ValueError(f"[CONTACT] unknown embodiment {emb!r}")
    return {"contact_left": left, "contact_pos_left": pos_left, "contact_right": right, "contact_pos_right": pos_right}


def detect(dataset_dir: str, dataset_name: str, embodiment_type: str, task: str, data_id: int) -> str:
    """Detect fingertip contacts of one clip and write the four contact keys into its keypoint archive."""
    from dexengine.retarget.ik.prior import emb_sides  # noqa: PLC0415

    sides = emb_sides(embodiment_type)
    mano_dir = mjutil.processed_data_dir(dataset_dir, dataset_name, "mano", embodiment_type, task, data_id)
    archive = os.path.join(mano_dir, "trajectory_keypoints.npz")
    if not os.path.isfile(archive):
        raise FileNotFoundError(f"[CONTACT] keypoint archive missing: {archive}")
    info_file = os.path.join(os.path.dirname(mano_dir), "task_info.json")
    task_info = None
    if os.path.isfile(info_file):
        with open(info_file) as f:
            task_info = json.load(f)
    else:
        warnings.warn(f"[CONTACT] {info_file} missing; no object meshes", stacklevel=2)
    with np.load(archive, allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}
    scene = build_contact_scene(dataset_dir, task_info, sides)
    raw, pos = evaluate(scene, z)
    c, mean = postprocess(raw, pos, tag=f"{dataset_name}/{task}")
    z.update(contact_keys(embodiment_type, c, mean))
    tmp = f"{archive}.tmp{os.getpid()}.npz"
    np.savez(tmp, **z)
    os.replace(tmp, archive)
    n_geoms = {s: len(g) for s, g in scene.object_geoms.items()}
    print(
        f"[CONTACT] {dataset_name}/{task} {embodiment_type}: frames {raw.shape[0]}, object geoms {n_geoms}, "
        f"contact frames per tip {raw.sum(axis=0).tolist()} -> {archive}"
    )
    return archive
