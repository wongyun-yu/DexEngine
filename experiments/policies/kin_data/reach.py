"""E5 kinematic baseline — the **only owner** of the reach rule (thresholds are named constants).

reach (placement) = the fraction of frames whose palm origin position error ≤ POS_TOL_M **and** orientation error ≤
  ROT_TOL_DEG (prep.py IK cache wrist_pos_error_all (Euclidean, m), wrist_rot_error_deg_all (geodesic °)). Bimanual
  clips
  = the minimum of both hands · one-hand clips = the active hand only (idle hand excluded). Pass = reach ≥ REACH_MIN,
  measured per rendered placement (source and each variant), not per clip.
Joint jumps (recorded only, not a filter): frames where the active arm's per-frame |Δq_j| exceeds the joint speed limit
×
  dt (dt = 1 / trajectory frequency — arm_vel_limit and frequency from the cache): count n_jump and max |Δq|/cap.
Standard library + numpy only (read by shells, the table builders, prep.py and the render driver).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

POS_TOL_M = 0.01
ROT_TOL_DEG = 10.0
REACH_MIN = 0.80
RULE = (
    f"reach = min over active hands of fraction of frames with palm-origin euclid error <= {POS_TOL_M} m and "
    f"geodesic rotation error <= {ROT_TOL_DEG} deg (kin_prep IK); pass = reach >= {REACH_MIN}; "
    "per rendered placement; joint jumps (frame step |dq_j| > arm velocity limit_j / trajectory frequency) are "
    "recorded, not filtered"
)
EUCLID_DEF = "euclid_palm_origin_m"


def frame_ok(pe, re_) -> np.ndarray:
    """Per frame: reached (position and orientation)?"""
    return (np.asarray(pe, float) <= POS_TOL_M) & (np.asarray(re_, float) <= ROT_TOL_DEG)


def dq_cap(z) -> np.ndarray:
    """Per-frame joint cap = speed limit / trajectory frequency (values from the cache — missing means an old cache,
    death).
    """
    if "arm_vel_limit" not in z.files or "frequency" not in z.files:
        raise ValueError("[KIN_REACH] IK cache lacks arm_vel_limit · frequency (made before the joint-continuity gate)")
    return np.asarray(z["arm_vel_limit"], float) / float(z["frequency"])


def jumps(q, cap) -> tuple[int, float]:
    """(jump frames, max |Δq|/cap) — a jump = any joint with |Δq_j| > cap_j."""
    q = np.asarray(q, float)
    if len(q) < 2:
        return 0, 0.0
    r = np.abs(np.diff(q, axis=0)) / np.asarray(cap, float)
    return int((r > 1.0).any(1).sum()), round(float(r.max()), 3)


def solver_of(z) -> str:
    """The IK cache's solver version (prep.py's solver marker) — the table and cell builders accept one version, never
    mixed.
    """
    return str(z["solver"]) if "solver" in z.files else "?"


def reach_of(ik_npz: dict[str, Path], drop_fallback: bool = False) -> dict:
    """{active hand: IK cache} → {rule, per_side{side: frac}, reach (minimum), passes, frames, solver}. No active hand,
    or a cache without the Euclidean definition, dies. drop_fallback = count frames replaced by the divergence guard
    (ik_fallback_frames) as not reached — a reference value for that repair's share (approximate, the trajectory
    changes)."""
    if not ik_npz:
        raise ValueError("[KIN_REACH] no active hand — cannot measure reach")
    per, frames, sv, jn, jr = {}, {}, set(), {}, {}
    for s, f in sorted(ik_npz.items()):
        z = np.load(f)
        if "wrist_pos_error_def" not in z.files or str(z["wrist_pos_error_def"]) != EUCLID_DEF:
            raise ValueError(f"[KIN_REACH] {f}: position error definition is not {EUCLID_DEF} (old cache)")
        ok = frame_ok(z["wrist_pos_error_all"], z["wrist_rot_error_deg_all"])
        if drop_fallback:
            if "ik_fallback_frames" not in z.files:
                raise ValueError(f"[KIN_REACH] {f}: no ik_fallback_frames (made before the divergence guard)")
            ok = ok & ~np.asarray(z["ik_fallback_frames"], bool)
        sv.add(solver_of(z))
        jn[s], jr[s] = jumps(z["arm_joint_positions_all"], dq_cap(z))
        per[s], frames[s] = round(float(ok.mean()), 4), int(len(ok))
    if len(sv) != 1:
        raise ValueError(f"[KIN_REACH] the hand caches of one clip disagree on the solver version {sorted(sv)}")
    r = min(per.values())
    return {
        "rule": RULE,
        "per_side": per,
        "reach": r,
        "passes": bool(r >= REACH_MIN),
        "frames": frames,
        "reach_min": REACH_MIN,
        "solver": sv.pop(),
        "n_jump": jn,
        "max_dq_ratio": jr,
    }
