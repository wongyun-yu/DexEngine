"""Keypoint archive intake for the hand tracker: validation, the entity reference table and the contact reference.

Entity table layout (fixed for every embodiment, so single-hand runs index the same columns):
`0` right wrist, `1..5` right tips, `6` left wrist, `7..11` left tips, `12..12+n-1` extra objects,
`12+n` right object, `13+n` left object.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from dexengine.retarget.quat_bounds import HAND_QUAT_ATOL, OBJ_QUAT_ATOL

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("right", "left")
POSE_KEYS: dict[str, tuple[int, ...]] = {
    "qpos_wrist_right": (7,),
    "qpos_finger_right": (5, 7),
    "qpos_wrist_left": (7,),
    "qpos_finger_left": (5, 7),
    "qpos_obj_right": (7,),
    "qpos_obj_left": (7,),
}
HAND_KEYS = ("qpos_wrist_right", "qpos_finger_right", "qpos_wrist_left", "qpos_finger_left")
OBJECT_KEYS = ("qpos_obj_right", "qpos_obj_left")


def wrist_col(side: str) -> int:
    return 0 if side == "right" else 6


def tip_col(side: str, finger_index: int) -> int:
    return wrist_col(side) + 1 + finger_index


def object_col(body: str, n_extra: int) -> int:
    """Entity column of an object body name (`right_object`, `left_object`, `extra_object_<i>`)."""
    if body == "right_object":
        return 12 + n_extra
    if body == "left_object":
        return 13 + n_extra
    prefix = "extra_object_"
    if body.startswith(prefix) and body[len(prefix) :].isdigit():
        i = int(body[len(prefix) :])
        if i < n_extra:
            return 12 + i
        raise ValueError(f"[IK] scene body {body!r} has no extra-object trajectory (archive has {n_extra})")
    raise ValueError(f"[IK] {body!r} is not an object body name")


@dataclass
class Keypoints:
    """Validated archive content the solve needs."""

    T: int
    ref: np.ndarray  # (T, 14 + n_extra, 7) entity reference table
    n_extra: int
    extra_rows: list[np.ndarray]  # qpos_extra_obj_<i> as stored
    arti: dict[str, np.ndarray]  # side -> (T, 1) hinge angles, input dtype
    contact_raw: dict[str, np.ndarray]  # side -> (T, K) float64, input flags (all-ones when absent)
    contact_ref: np.ndarray  # (T, K_right + K_left) aggregated: each row = any(row) in every column


def _quat_dev(a: np.ndarray) -> float:
    return float(np.abs(np.linalg.norm(a[..., 3:7], axis=-1) - 1.0).max()) if a.size else 0.0


def load_keypoints(path: str, window: int) -> Keypoints:
    """Load and check `trajectory_keypoints.npz`; raises on any contract violation."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    T = _check_poses(data, path, window)
    extra_rows = _extra_rows(data, T)
    n = len(extra_rows)

    ref = np.zeros((T, 14 + n, 7))
    for side in SIDES:
        ref[:, wrist_col(side)] = data[f"qpos_wrist_{side}"]
        ref[:, wrist_col(side) + 1 : wrist_col(side) + 6] = data[f"qpos_finger_{side}"]
    for i, a in enumerate(extra_rows):
        ref[:, 12 + i] = a
    ref[:, 12 + n] = data["qpos_obj_right"]
    ref[:, 13 + n] = data["qpos_obj_left"]

    arti = {}
    for side in SIDES:
        key = f"qpos_obj_arti_{side}"
        if key in data:
            a = data[key]
            if a.shape != (T, 1) or not np.isfinite(a).all():
                raise ValueError(f"[IK] {key}: shape {a.shape} (expected ({T}, 1)) or non-finite values")
            arti[side] = a

    raw, contact_ref = _contact(data, path, T)
    return Keypoints(
        T=T, ref=ref, n_extra=n, extra_rows=extra_rows, arti=arti, contact_raw=raw, contact_ref=contact_ref
    )


def _check_poses(data: dict, path: str, window: int) -> int:
    """Check the pose keys (present, shaped, finite, unit quaternions, longer than the smoothing window); returns T."""
    missing = [k for k in POSE_KEYS if k not in data]
    if missing:
        raise KeyError(f"[IK] {path}: missing pose keys {missing}")
    lengths = {k: data[k].shape[0] if data[k].ndim else -1 for k in POSE_KEYS}
    T = lengths["qpos_wrist_right"]
    for k, tail in POSE_KEYS.items():
        a = data[k]
        if a.ndim != 1 + len(tail) or tuple(a.shape[1:]) != tail:
            raise ValueError(f"[IK] {k}: shape {a.shape}, expected (T, {', '.join(map(str, tail))})")
        if a.shape[0] != T or T < 1:
            raise ValueError(f"[IK] pose keys disagree on the frame count or are empty: {lengths}")
        if not np.isfinite(a).all():
            raise ValueError(f"[IK] {k}: non-finite values")
    hand_dev = max(_quat_dev(data[k]) for k in HAND_KEYS)
    if hand_dev > HAND_QUAT_ATOL:
        raise ValueError(f"[IK] hand quaternions off unit length by {hand_dev:.3g} > {HAND_QUAT_ATOL}")
    obj_dev = max(_quat_dev(data[k]) for k in OBJECT_KEYS)
    if obj_dev > OBJ_QUAT_ATOL:
        raise ValueError(f"[IK] object quaternions off unit length by {obj_dev:.3g} > {OBJ_QUAT_ATOL}")
    if window >= T:
        raise ValueError(f"[IK] {T} frames: the {window}-frame smoothing needs more than {window}")
    return T


def _extra_rows(data: dict, T: int) -> list[np.ndarray]:
    """The declared extra-object trajectories as stored, up to the first missing key (warned)."""
    extra_rows: list[np.ndarray] = []
    declared = int(data["n_extra_objects"]) if "n_extra_objects" in data else 0
    for i in range(declared):
        key = f"qpos_extra_obj_{i}"
        if key not in data:
            # stacklevel 3: the warning names load_keypoints' caller
            warnings.warn(f"[IK] n_extra_objects={declared} but {key} is missing; using the first {i}", stacklevel=3)
            break
        a = data[key]
        if a.shape != (T, 7) or not np.isfinite(a).all():
            raise ValueError(f"[IK] {key}: shape {a.shape} (expected ({T}, 7)) or non-finite values")
        extra_rows.append(a)
    return extra_rows


def _contact(data: dict, path: str, T: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-side contact flags as float64 (all ones when the archive has none) and the aggregated reference: each row
    is any(row) in every column."""
    if "contact_right" in data and "contact_left" in data:
        raw = {s: np.asarray(data[f"contact_{s}"], dtype=np.float64) for s in SIDES}
        for s, a in raw.items():
            if a.ndim != 2 or a.shape[0] != T:
                raise ValueError(f"[IK] contact_{s}: shape {a.shape}, expected ({T}, K)")
    else:
        # stacklevel 3: the warning names load_keypoints' caller
        warnings.warn(f"[IK] {path}: no contact keys; every fingertip counts as in contact", stacklevel=3)
        raw = {s: np.ones((T, 5)) for s in SIDES}
    both = np.concatenate([raw["right"], raw["left"]], axis=1)
    flag = (both != 0).any(axis=1).astype(np.float64)
    contact_ref = np.repeat(flag[:, None], both.shape[1], axis=1)
    return raw, contact_ref
