"""The single source of the P&P place target: the object poses at the **last frame** of the variant's
trajectory_kinematic.npz. The P&P judge (pnp) and the place marker of the renderer read this one function. numpy only.
Reading = the frozen judge_v1.kin_scene: objects[i].qpos_start of <tree_clip>/task_info.json → qpos[:, s:s+3] position,
qpos[:, s+3:s+7] quaternion (wxyz).
Receiver = the frozen put() rule: the non-background object nearest to the moved object at the reference end ('body'
frame when it is a main object, else 'world').
Frame = tree (kinematic) frame; the session env frame is reached with judge_v1.env_from_tree's (R, t) (a receiver-body
place does not depend on the frame)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# reference value (the frozen put's min_ref_disp_m); P&P also counts moves < 2 cm (put back in place) as a place
MIN_DISP_M = 0.02


def quat_R(q: np.ndarray) -> np.ndarray:
    """wxyz (…,4) → (…,3,3) — the same formula as judge_v1.quat_R (kept here so this module stays numpy-only)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def kin_objects(tree_clip: str | Path) -> dict:
    """{name: dict(meta, p (T,3), R (T,3,3))} — kinematic trajectories in the tree frame (same reading as
    kin_scene).
    """
    tc = Path(tree_clip)
    ti = json.loads((tc / "task_info.json").read_text())
    z = np.load(tc / "0" / "trajectory_kinematic.npz")
    q = z["qpos"]
    out = {}
    for x in ti["objects"]:
        s = x["qpos_start"]
        out[x["name"]] = {
            "meta": x,
            "p": q[:, s : s + 3].astype(np.float64),
            "R": quat_R(q[:, s + 3 : s + 7].astype(np.float64)),
        }
    return out


def place_target(tree_clip: str | Path) -> dict:
    """Place target per main object → {name: dict(p_start, p_end (3,), R_end (3,3), disp_m, receiver, frame, rel_end
    (3,) | None)}. p_end, R_end = last frame (tree frame). With frame='body', rel_end = the place in the receiver's end
    body frame
    (frame independent; the judge uses it)."""
    o = kin_objects(tree_clip)
    out = {}
    for n, v in o.items():
        if v["meta"].get("role") != "main":
            continue
        pe, ps = v["p"][-1], v["p"][0]
        oth = [
            (float(np.linalg.norm(w["p"][-1] - pe)), m)
            for m, w in o.items()
            if m != n and w["meta"].get("tag") != "background"
        ]
        recv = min(oth)[1] if oth else None
        body = recv is not None and o[recv]["meta"].get("role") == "main"
        rel = (o[recv]["R"][-1].T @ (pe - o[recv]["p"][-1])) if body else None
        out[n] = {
            "p_start": ps,
            "p_end": pe,
            "R_end": v["R"][-1],
            "disp_m": float(np.linalg.norm(pe - ps)),
            "receiver": recv,
            "frame": "body" if body else "world",
            "rel_end": rel,
        }
    return out
