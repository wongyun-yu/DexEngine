"""Forward kinematics of articulated objects (shared by data preparation and placement).

Articulation statement (``task_info.objects[].articulation``):

- ``parts.base`` / ``parts.moving``: mesh paths relative to ``mesh_dir`` (ARCTIC ``bottom.obj``/``top.obj``,
  OakInk2 ``visual.obj``/``../<id>/visual.obj``).
- ``axis`` / ``pos``: joint axis and anchor in the moving-part (child body) frame, as for an MJCF joint.
- ``type``: ``hinge`` (default, ``theta`` in radians) or a slide joint (``theta`` in metres).
- ``child_pose``: pose ``{pos, quat_wxyz}`` of the child body in the base frame at ``theta = 0``; absent means identity
  (ARCTIC declares the child body without pos/quat, so the base frame equals the child frame).

Quaternions are wxyz.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _Rot


def pose_points(V: np.ndarray, pose: dict | None) -> np.ndarray:
    """Move points by a link-frame pose ``{pos, quat_wxyz}`` (``None`` keeps them): ``V' = R V + p``.

    This is how a rigidly attached part is placed in its link frame (same form as ``child_pose``).
    """
    if not pose:
        return V
    R = _Rot.from_quat(np.asarray(pose["quat_wxyz"], float)[[1, 2, 3, 0]]).as_matrix()
    return np.asarray(V, float) @ R.T + np.asarray(pose["pos"], float)


def artic_moving_vertices(art: dict, Vm: np.ndarray, theta: float) -> np.ndarray:
    """Vertices ``Vm`` of the moving part, given in the child frame, expressed in the base body frame at joint value
    ``theta``: ``V' = child_pose o FK_joint(theta)(Vm)``."""
    ax = np.asarray(art["axis"], float)
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    anc = np.asarray(art.get("pos") or [0.0, 0.0, 0.0], float)
    if str(art.get("type", "hinge")) == "hinge":
        Rj = _Rot.from_rotvec(ax * float(theta)).as_matrix()
        V = (Vm - anc) @ Rj.T + anc
    else:
        V = Vm + ax * float(theta)
    cp = art.get("child_pose")
    if cp:
        Rc = _Rot.from_quat(np.asarray(cp["quat_wxyz"], float)[[1, 2, 3, 0]]).as_matrix()
        V = V @ Rc.T + np.asarray(cp["pos"], float)
    return V
