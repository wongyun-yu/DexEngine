"""MANO <-> scene alignment gate of the TACO MANO loader (viz.mano_taco calls it once per clip in the canonical loader):
the desk anchor and the hand-object distance checks with their measured thresholds; a failure raises
ManoAlignmentError."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from dexengine.core.desk import DESK_TOP_SURFACE_Z as DESK_HEIGHT


class ManoAlignmentError(RuntimeError):
    """The MANO hands do not line up with the scene (objects, desk): a state that must not pass silently."""


# gate thresholds, all measured: on a sample of 421 clips (normal 358 = ok 300 + one hand without contact 58;
#   abnormal by diagnosis 63) the minimum distance between the MANO vertices and the assigned object surface:
#     'better hand' max of the 358 normal clips = 0.071 cm (p50 0.019 · p99 0.044)
#     'better hand' median of the 63 abnormal clips = 5.02 cm
#   -> the combination below: 0 false alarms on the 358 normal clips and 59/59 true anomalies caught
#     (the other 4 are normal clips where both hands hold one tool, false alarms of the classifier).
GATE_DESK_MIN_Z = -0.15  # G1: fails when the MANO lowest 0.5 % lies this far below the **lowest robot fingertip**
GATE_DESK_ABS_MIN_Z = -0.35  # G1 fallback (no robot fingertips): an absolute threshold relative to the desk top
GATE_BEST_M = 0.010  # G2a: fails when neither hand comes this close to its own object (normal max 0.00071 m)
GATE_FAR_M = 0.010  # G2b: farther than this from its own object
GATE_NEAR_M = 0.001  # G2b: and this close to the other object suggests a flipped assignment
GATE_FRAMES = 40  # max frames checked (even thinning)


def _desk_anchor_problems(tree, rh_verts, lh_verts, problems):
    """G1: the lowest 0.5 % of each hand's vertices against the lowest robot fingertip (or the desk top without
    robot)."""
    _ft = None
    try:
        _bp_p = Path(tree) / "0" / "body_positions.npz"
        if _bp_p.is_file():
            _bpz = np.load(_bp_p)
            _ft = {s: np.asarray(_bpz[f"{s}_fingertip_pos"])[..., 2].min() for s in ("right", "left")}
    except (OSError, KeyError, ValueError):
        _ft = None
    for tag, v in (("right", rh_verts), ("left", lh_verts)):
        if v is None or len(v) == 0:
            continue
        z = v[..., 2] - DESK_HEIGHT  # Isaac world -> scene (desk = 0)
        p05 = float(np.percentile(z, 0.5))
        if _ft is not None:  # relative check (exact): the robot is the reference
            margin = p05 - float(_ft[tag])
            if margin < GATE_DESK_MIN_Z:
                problems.append(
                    f"G1 {tag}: MANO hand vertex z lowest 0.5%={p05:+.3f}m is {-margin:.3f}m below the lowest robot "
                    f"fingertip ({float(_ft[tag]):+.3f}m) "
                    f"(allowed {-GATE_DESK_MIN_Z:.2f}m). "
                    f"Suspect the session z anchor (taco_frames.SESSION_SHIFT_Z)"
                )
        elif p05 < GATE_DESK_ABS_MIN_Z:  # fallback (no robot): a loose absolute threshold
            problems.append(
                f"G1 {tag}: hand vertex z lowest 0.5%={p05:+.3f}m is "
                f"{-p05:.3f}m below the desk top (allowed {GATE_DESK_ABS_MIN_Z:+.2f}, no robot fingertips). "
                f"Suspect the session z anchor (taco_frames.SESSION_SHIFT_Z)"
            )


def _assignment_problems(dd, problems):
    """G2: hand <-> object distances for the assignment and the swap; flags a flipped assignment or a broken frame."""
    if all(x is not None for x in dd.values()):
        ra, rs = dd[("right", "right")], dd[("right", "left")]
        la, ls = dd[("left", "left")], dd[("left", "right")]
        detail = (
            f"R assigned={ra * 100:.2f}cm/swap={rs * 100:.2f}cm · L assigned={la * 100:.2f}cm/swap={ls * 100:.2f}cm"
        )
        if min(ra, la) > GATE_BEST_M:
            problems.append(
                f"G2a: neither hand reaches its assigned object (better {min(ra, la) * 100:.2f}cm > "
                f"{GATE_BEST_M * 100:.1f}cm). {detail} — "
                + (
                    "the swap is closer -> suspect the **hand <-> object assignment (task_info/side_objects)**"
                    if min(rs, ls) < min(ra, la)
                    else "the swap is far too -> suspect the tree frame or object tracks"
                )
            )
        elif (ra > GATE_FAR_M and rs < GATE_NEAR_M and ls <= la) or (la > GATE_FAR_M and ls < GATE_NEAR_M and rs <= ra):
            problems.append(
                f"G2b: one hand is far from its own object and touches the other object, and the other hand prefers "
                f"the swap too. {detail}: suspect the hand <-> object assignment"
            )


def taco_alignment_gate(sequence, rh_verts, lh_verts, rh_obj_T, lh_obj_T, r_mesh, l_mesh, tree=None) -> None:
    """MANO <-> scene alignment gate, **on by default**.

    Why: checking that resolver, task_info and canonical values agree is circular (all three derive from task_info),
    so a flipped left/right assignment or a session z anchor off by 0.72 m passes every format check. This gate looks
    at **geometry**, not form.

    What it checks (threshold evidence = the constant comments above, 421 clips measured)
      G1 desk anchor   fails when the lowest 0.5 % of hand vertex z lies GATE_DESK_MIN_Z below the desk top (scene
                       z = 0). Measured desk heights of 24 sessions are within -0.019..+0.003, a 130 mm margin; a
                       session with a wrong world origin sits at -0.72 and is surely caught. **Only the lower side**:
                       clips with hands held high are normal.
      G2a both hands   fails when neither hand ever comes within GATE_BEST_M of its assigned object. Flipped
          off          assignments and tree frame mismatches are caught here. Clips where one hand only watches
                       (58) pass.
      G2b one hand     fails when one hand is GATE_FAR_M or more from its own object while within GATE_NEAR_M of the
          off          **other** object, and the other hand prefers the swap too. Normal clips where both hands hold
                       one tool pass, since the other hand prefers its own object (0 false alarms measured).

    Off: env TACO_MANO_ALIGN_GATE=0 (fully off) / =warn (warning only). On by default: a failure raises
    ManoAlignmentError.
    """
    mode = os.environ.get("TACO_MANO_ALIGN_GATE", "1").strip().lower()
    if mode in ("0", "off", "false", "no"):
        return
    from scipy.spatial import cKDTree

    problems: list[str] = []

    # ---- G1 desk anchor, **relative to the robot fingertips when present**. An absolute threshold alone flags clips
    # where the person really lowers a hand below the desk top (dust_roller_plate__20231019_096/106: left hand
    # -0.17/-0.28 m); there the robot fingertips also go to -0.164/-0.265, so the motion is genuine and the frame is
    # not broken. The robot fingertips are retargeted from the same MANO and live in the object frame, so "is the
    # MANO far below the robot" is the right question for a broken frame.
    _desk_anchor_problems(tree, rh_verts, lh_verts, problems)

    # ---- G2 hand <-> object: both the assignment (asg) and the swap (swp) are measured; measuring the swap too tells
    #      "the assignment is flipped" from "the frame is off".
    def _surf(mesh_path):
        if not mesh_path:
            return None
        p = Path(mesh_path).parent / "surface_points.npz"
        if not p.exists():
            return None
        z = np.load(p)
        a = np.asarray(z[z.files[0]], dtype=np.float64)[:, :3]
        return a[np.linspace(0, len(a) - 1, min(len(a), 1200)).astype(int)]

    pts = {"right": _surf(r_mesh), "left": _surf(l_mesh)}
    verts = {"right": rh_verts, "left": lh_verts}
    objT = {"right": rh_obj_T, "left": lh_obj_T}

    def _md(side_hand, side_obj):
        v, P, T_a = verts[side_hand], pts[side_obj], objT[side_obj]
        if v is None or P is None or T_a is None:
            return None
        n = min(len(v), len(T_a))
        idx = np.unique(np.linspace(0, n - 1, min(n, GATE_FRAMES)).astype(int))
        return min(float(cKDTree(P @ T_a[t][:3, :3].T + T_a[t][:3, 3]).query(v[t])[0].min()) for t in idx)

    dd = {(h, o): _md(h, o) for h in ("right", "left") for o in ("right", "left")}
    _assignment_problems(dd, problems)

    if not problems:
        return
    msg = (
        f"[MANO alignment gate] {sequence}: "
        + " | ".join(problems)
        + "\n  off (not recommended): TACO_MANO_ALIGN_GATE=0"
    )
    if mode in ("warn", "warning"):
        print("  ⚠ " + msg, flush=True)
        return
    raise ManoAlignmentError(msg)
