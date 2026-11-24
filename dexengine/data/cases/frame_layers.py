"""Frame layers of one hand × one object (A3): contact, hold, timeline grasp, carry and joint drive per frame, from
MANO fingertip/palm surface distances, MANO fingertip speed and the object motion (speed and in-place rotation). The
thresholds and the layer definitions are owned by hand_object_assign.py; this module applies them."""

from __future__ import annotations

import numpy as np

from dexengine.data.cases.hand_object_assign import (
    TH_ARTIC_DRIVE_RADPS,
    TH_CARRY_REL,
    TH_GRASP_MIN_TIPS,
    TH_OBJ_STILL_MPS,
    TH_PALM_M,
    TH_ROT_DPS,
    TH_SMOOTH_S,
    TH_TIMELINE_COMOVE_MPS,
    frames_of,
    smooth_velocity,
)


def angular_speed_dps(quat_wxyz, fps: float):
    """Object quaternion (wxyz, (T,4)) → smoothed angular speed (deg/s, (T,))."""
    q = np.asarray(quat_wxyz, float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if len(q) < 2:
        return np.zeros(len(q))
    dot = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0.0, 1.0)
    w = np.degrees(2.0 * np.arccos(dot)) * fps
    w = np.r_[w[:1], w]
    k = max(1, min(frames_of(TH_SMOOTH_S, fps), len(w)))
    return np.convolve(w, np.ones(k) / k, mode="same")


def hand_object_masks(
    dist, witness, palm_dist, thr: float, small: bool, tips_v, fps: float, rot_dps=None, hinge=None
) -> dict:
    """Frame layers of one hand × one object: {contact, hold, grasp_tl (timeline grasp), carry, drive (joint drive)}.
    dist (T,5) per-fingertip surface distance · witness (T,5,3) surface points near the fingertips · palm_dist (T,) ·
    tips_v (T,3) MANO fingertip centroid velocity · rot_dps (T,) object angular speed · hinge (T,) joint column."""
    near = np.asarray(dist, float) < thr
    palm = np.asarray(palm_dist, float) < TH_PALM_M
    contact = near.any(1) | palm
    hold = (near.sum(1) >= TH_GRASP_MIN_TIPS) | palm | (bool(small) & near.any(1))
    wspeed, wv = smooth_velocity(np.asarray(witness, float).mean(1), fps)
    rel = np.linalg.norm(wv - np.asarray(tips_v, float), axis=1)
    grasp_tl = hold & ((rel < np.maximum(TH_TIMELINE_COMOVE_MPS, 0.5 * wspeed)) | (wspeed < TH_OBJ_STILL_MPS))
    moving_with = (wspeed >= TH_OBJ_STILL_MPS) & (rel < TH_CARRY_REL * wspeed)
    if rot_dps is not None:
        moving_with = moving_with | (np.asarray(rot_dps, float) > TH_ROT_DPS)
    carry = hold & moving_with
    drive = np.zeros(len(near), bool)
    if hinge is not None:
        drive = near.any(1) & (np.abs(np.gradient(np.asarray(hinge, float))) * fps > TH_ARTIC_DRIVE_RADPS)
    return {"contact": contact, "hold": hold, "grasp_tl": grasp_tl | drive, "carry": carry | drive, "drive": drive}
