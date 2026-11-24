"""Frame and wrist-joint helpers of posthoc_refs.py (numpy / scipy only).

The MuJoCo hand trees carry each palm as a free body under the world with 6 wrist joints (3 slides, then 3 hinges, all
on signed basis axes at the body origin). `wrist_chain` reads that chain from a model and `wrist_dofs` inverts it: the
6 joint values that put the palm body at a given world pose, hinge angles kept continuous with the previous frame.
"""

from __future__ import annotations

import numpy as np


def quat_R(q):
    """wxyz quaternion → rotation matrix (normalised first)."""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def ang_deg(R) -> float:
    """Rotation angle of a rotation matrix (deg)."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def wrist_chain(model, mujoco, body: str, start: int):
    """The 6 wrist joints of a palm body: (base R, base p, slide [(qadr, axis)], hinge [(qadr, axis)])."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if b < 0 or model.body_parentid[b] != 0:
        raise RuntimeError(f"[FHPOST] palm body {body}: missing or not a child of the world")
    js = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
    sl = [
        (int(model.jnt_qposadr[j]), model.jnt_axis[j].copy())
        for j in js
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
    ]
    hi = [
        (int(model.jnt_qposadr[j]), model.jnt_axis[j].copy())
        for j in js
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if len(js) != 6 or len(sl) != 3 or len(hi) != 3 or [q for q, _ in sl + hi] != list(range(start, start + 6)):
        raise RuntimeError(
            f"[FHPOST] palm body {body}: wrist chain is not 3 slides + 3 hinges at qpos {start}..{start + 5}"
        )
    if np.abs(model.jnt_pos[js]).max() > 1e-9:
        raise RuntimeError(f"[FHPOST] palm body {body}: a wrist joint has an anchor off the body origin")
    for _, ax in sl + hi:
        if sorted(np.abs(ax).round(9).tolist()) != [0.0, 0.0, 1.0]:
            raise RuntimeError(f"[FHPOST] palm body {body}: wrist axis {ax} is not a signed basis vector")
    R0 = np.zeros(9)
    mujoco.mju_quat2Mat(R0, model.body_quat[b])
    return b, R0.reshape(3, 3), model.body_pos[b].copy(), sl, hi


def wrist_dofs(R0, p0, sl, hi, Rm, pm, prev):
    """Wrist dofs that put the palm body at (Rm, pm) in the MuJoCo world; hinge angles continuous with `prev`."""
    from scipy.spatial.transform import Rotation

    pl, Rl = R0.T @ (pm - p0), R0.T @ Rm
    q = np.zeros(6)
    for i, (_, ax) in enumerate(sl):
        k = int(np.argmax(np.abs(ax)))
        q[i] = pl[k] * np.sign(ax[k])
    ks = [int(np.argmax(np.abs(ax))) for _, ax in hi]
    sg = np.array([np.sign(ax[k]) for (_, ax), k in zip(hi, ks, strict=True)])
    e = Rotation.from_matrix(Rl).as_euler("".join("XYZ"[k] for k in ks))  # intrinsic, the joint order
    if sorted(ks) != [0, 1, 2]:
        raise RuntimeError(f"[FHPOST] wrist hinge axes {ks} are not three different axes")
    alt = np.array([e[0] + np.pi, np.pi - e[1], e[2] + np.pi])  # the second solution of a Tait–Bryan triple
    best = None
    for cand in (e, alt):
        c = cand * sg
        c = c + 2 * np.pi * np.round((prev - c) / (2 * np.pi))
        d = float(np.abs(c - prev).max())
        if best is None or d < best[0]:
            best = (d, c)
    q[3:] = best[1]
    return q
