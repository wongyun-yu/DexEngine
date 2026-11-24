"""Rotation-aware Pinocchio arm IK and the robot facts of one (arm, hand) combo.

Frames: a wrist target (palm) in the trajectory frame becomes env-local world coordinates through Rz(-90°) and
+z = DESK_HEIGHT, and robot-base coordinates by subtracting the robot spawn. IK solves the arm joints so that
FK(palm frame) matches the target in both translation and rotation (6-DoF damped least squares).

Every consumer that needs the merged URDF, the palm frame, the arm joint names, the robot base or the palm offset
of a combo (judge, bake, reach map, gate, repair, experiments) takes them from `resolve_combo` and
`palm_frame_offset` here, so all of them solve against the same robot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import wrist_base_dir
from dexengine.core.repo import FULL_ROBOT_URDF_DIR
from dexengine.rl.tasks.cfg import constants as _SC

# The desk height and the robot base are facts of the scene, owned by the env constants (a pure constants
# module, importable without isaaclab).
DESK_HEIGHT = _SC.DESK_HEIGHT
BASE_XYZ = np.array(_SC.ROBOT_DEFAULT_POS, dtype=float)


class Rotation:
    """Minimal numpy-only stand-in for the subset of scipy.spatial.transform.Rotation used with this module
    (from_euler / from_quat / apply / * / as_matrix), for environments where scipy's compiled extension does not
    load next to pinocchio."""

    def __init__(self, mat):
        self._m = np.asarray(mat, dtype=np.float64)

    @classmethod
    def from_euler(cls, axis, angle, degrees=False):
        a = np.radians(angle) if degrees else float(angle)
        c, s = np.cos(a), np.sin(a)
        if axis == "z":
            m = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        elif axis == "y":
            m = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        elif axis == "x":
            m = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        else:
            raise ValueError(f"unsupported axis {axis!r}")
        return cls(m)

    @classmethod
    def from_quat(cls, q):
        # scipy convention: q = [x, y, z, w]
        x, y, z, w = (float(v) for v in q)
        n = (w * w + x * x + y * y + z * z) ** 0.5
        if n == 0.0:
            return cls(np.eye(3))
        w, x, y, z = w / n, x / n, y / n, z / n
        m = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        return cls(m)

    def apply(self, v):
        return self._m @ np.asarray(v, dtype=np.float64)

    def __mul__(self, other):
        return Rotation(self._m @ other._m)

    def as_matrix(self):
        return self._m


def _load_spec_module(stem):
    """The robot-spec registry by its spec stem: `arm_specs` → `dexengine.core.arms`, `core.hands` →
    `dexengine.core.hands`."""
    from dexengine.core import arms, hands  # noqa: PLC0415

    return {"arm_specs": arms, "hand_specs": hands}[stem]


def urdf_torso3_pitch(urdf, joint: str | None) -> float | None:
    """The origin rpy·y (rad) of ``joint`` (the arm's ArmSpec.generation_pitch_joint — RB-Y1 ``torso_3``) in a merged
    URDF; None for an arm without that joint (no generation axis) or a URDF without it.

    This number is the evidence of which body generation a URDF on disk belongs to: file and robot names
    can be copied unchanged between generations, this angle cannot.
    """
    import re as _re

    if joint is None:
        return None
    try:
        txt = Path(urdf).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _re.search(rf'<joint\s+name="{_re.escape(joint)}".*?<origin\s+rpy="\s*([^"]+?)\s*"', txt, _re.S)
    if not m:
        return None
    parts = m.group(1).split()
    return float(parts[1]) if len(parts) == 3 else None


def _verify_urdf_posture(arms, arm: str, urdf, expect_variant) -> None:
    """Die if the torso posture of the URDF on disk is not the one of the expected body generation.

    Guards against a generator that writes an old posture into the canonical path without changing any file
    name: every IK solved after that would be silently on another body.
    """
    joint = arms.ARM_REGISTRY[arm].generation_pitch_joint
    got = urdf_torso3_pitch(urdf, joint)
    if joint is None:
        return  # an arm with no generation pitch joint is not checked
    if got is None:
        raise SystemExit(f"[resolve_combo] {urdf}: no origin rpy on the generation pitch joint {joint!r} of arm {arm}")
    gen = arms.generation_of(arm, expect_variant)
    if gen is None:
        return  # unregistered variant: nothing to compare against
    if abs(got - gen.torso3_pitch_rad) > 1e-6:
        raise SystemExit(
            f"[resolve_combo] asset generation mismatch — the URDF posture is not the one of its variant.\n"
            f"  file             : {urdf}\n"
            f"  {joint} measured : {got:+.4f} rad\n"
            f"  variant '{expect_variant}' wants: {gen.torso3_pitch_rad:+.4f} rad "
            f"(chest {gen.chest_pitch_deg:+.2f}°)\n"
            f"  The file was overwritten with another posture; check `git diff` and rebuild or restore it."
        )


def resolve_combo(arm: str, hand: str, side: str):
    """(urdf, ee_frame, arm_joint_names, base_xyz, wrist_dir) — the robot facts of one combo and side.

    urdf      merged arm + hand URDF of the arm's canonical body variant (`<combo><variant>/<combo><variant>.urdf`
              under the robot assets); a missing canonical variant dies (no fallback to another body)
    ee_frame  the hand spec's URDF palm frame on that side (the frame the wrist targets track)
    arm_joint_names  the ArmSpec joint names of that side
    base_xyz  ArmSpec.base_pos, else the env default robot position
    wrist_dir the raw wrist-pose cache of the hand on that side (TACO layer)
    """
    arms = _load_spec_module("arm_specs")
    hands = _load_spec_module("hand_specs")
    a = arms.ARM_REGISTRY.get(arm)
    if a is None:
        raise SystemExit(f"unknown --arm {arm!r} (known: {sorted(arms.ARM_REGISTRY)})")
    h = hands.HAND_REGISTRY.get(hand)
    if h is None:
        raise SystemExit(f"unknown --hand {hand!r} (known: {sorted(hands.HAND_REGISTRY)})")
    if not h.urdf_palm_frame_right:
        raise SystemExit(
            f"HandSpec({hand!r}).urdf_palm_frame_right is empty — it names the Pinocchio EE frame and must be set."
        )
    combo = f"{arm}_{hand}"
    # The asset variant suffix comes from the same generation table as the sim side (core.arms).
    _var = arms.canon_variant(arm)
    if not (FULL_ROBOT_URDF_DIR / f"{combo}{_var}" / f"{combo}{_var}.urdf").is_file():
        raise InputMissingError(
            f"[resolve_combo] canonical URDF missing: "
            f"{FULL_ROBOT_URDF_DIR / f'{combo}{_var}' / f'{combo}{_var}.urdf'} — judging, baking and the gates need "
            f"the training body's URDF. Build it: python -m dexengine.robots.vendor build --data-root <data root>"
        )
    urdf = FULL_ROBOT_URDF_DIR / f"{combo}{_var}" / f"{combo}{_var}.urdf"
    _verify_urdf_posture(arms, arm, urdf, _var)
    ee_frame = h.urdf_palm_frame_right.replace("right_", f"{side}_", 1)
    base = (
        np.array(a.base_pos, dtype=float)
        if getattr(a, "base_pos", None) is not None
        else np.array(BASE_XYZ, dtype=float)
    )
    # An empty retarget_dirname means the hand name is the retargeting directory.
    wrist_dir = wrist_base_dir(h.retarget_dirname or h.name, side, "taco")
    return urdf, ee_frame, list(a.joint_names(side)), base, wrist_dir


def palm_frame_offset(hand: str) -> np.ndarray:
    """HandSpec.palm_frame_offset_pos — translation from the retargeting palm body to the URDF palm frame
    (palm-local).

    Every builder of a wrist target (IK, placement, reach map and its readers) uses this one function. An
    unregistered hand id (e.g. a spec directory name or a typo) raises KeyError instead of returning (0, 0, 0);
    a registered hand whose spec leaves the field at its default gets its declared (0, 0, 0)."""
    hands = _load_spec_module("hand_specs")
    h = hands.HAND_REGISTRY.get(hand)
    if h is None:
        raise KeyError(
            f"palm_frame_offset: unregistered hand {hand!r} (registered: {sorted(hands.HAND_REGISTRY)}) — "
            "pass the hand id, not the spec directory name."
        )
    return np.asarray(h.palm_frame_offset_pos, dtype=float)


def solve_ik_pose(
    model,
    data,
    target_pos_rbf,
    target_rot_rbf,
    ee_id,
    qpos_idx,
    max_iters=500,
    eps_pos=0.001,
    eps_rot_deg=0.5,
    lambda_pos=0.01,
    lambda_rot=0.01,
    q0=None,
):
    """Single-start DLS-IK for both position and orientation.

    Args:
        target_pos_rbf: (3,) target translation in robot-base frame
        target_rot_rbf: (3, 3) target rotation matrix in robot-base frame
        ee_id: pinocchio frame id of palm
        qpos_idx: tuple of arm joint indices in q vector
        q0: optional initial q (model.nq,) — defaults to neutral

    Returns:
        q (model.nq,), pos_err_m, rot_err_deg, iters_used
    """
    q = q0.copy() if q0 is not None else pin.neutral(model)
    target_se3 = pin.SE3(target_rot_rbf, target_pos_rbf)

    for it in range(max_iters):  # noqa: B007 — the iteration count is returned after the loop
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        cur_se3 = data.oMf[ee_id]
        # 6D error: log(target * cur^-1) in tangent space
        err6 = pin.log(target_se3.actInv(cur_se3)).vector  # [v(3), w(3)]
        pos_err = np.linalg.norm(err6[:3])
        rot_err = np.linalg.norm(err6[3:])
        if pos_err < eps_pos and np.degrees(rot_err) < eps_rot_deg:
            break
        # 6D Jacobian, LOCAL frame (matches log convention above)
        J6 = pin.computeFrameJacobian(model, data, q, ee_id, pin.LOCAL)
        # Weighted DLS step
        W = np.diag([lambda_pos**2] * 3 + [lambda_rot**2] * 3)
        dq_full = -J6.T @ np.linalg.solve(J6 @ J6.T + W, err6)
        # Project to arm DOFs, clip to joint limits
        for _k, qi in enumerate(qpos_idx):
            q[qi] = np.clip(q[qi] + dq_full[qi], model.lowerPositionLimit[qi], model.upperPositionLimit[qi])

    # Final error (one more FK at the latest q)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    cur_se3 = data.oMf[ee_id]
    err6 = pin.log(target_se3.actInv(cur_se3)).vector
    return q, np.linalg.norm(err6[:3]), np.degrees(np.linalg.norm(err6[3:])), it + 1


def solve_ik_multistart(model, data, target_pos, target_rot, ee_id, qpos_idx, num_starts=20, max_iters=500, seed=42):
    """Multi-start IK — pick best by 6D residual norm."""
    rng = np.random.default_rng(seed)
    best = None  # (q, pos_err, rot_err_deg, iters)
    for s in range(num_starts):
        if s == 0:
            q0 = pin.neutral(model)
        else:
            q0 = pin.neutral(model)
            for qi in qpos_idx:
                lo = model.lowerPositionLimit[qi]
                hi = model.upperPositionLimit[qi]
                if np.isfinite(lo) and np.isfinite(hi):
                    q0[qi] = rng.uniform(lo, hi)
                else:
                    q0[qi] = rng.uniform(-1.5, 1.5)
        q, pos_err, rot_err_deg, iters = solve_ik_pose(
            model, data, target_pos, target_rot, ee_id, qpos_idx, max_iters=max_iters, q0=q0
        )
        # Composite score: pos in mm + rot in deg (both small ≈ good)
        score = pos_err * 1000 + rot_err_deg
        if best is None or score < (best[1] * 1000 + best[2]):
            best = (q, pos_err, rot_err_deg, iters)
    return best
