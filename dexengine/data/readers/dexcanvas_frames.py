"""DexCanvas raw pose conventions (numpy and scipy only).

The reader (``dexengine.data.readers.dexcanvas``) converts with these functions, and the raw loader
(``dexengine.data.readers.raw.dexcanvas``) checks against them which Euler reading keeps a grasped object still in the
hand.

``object_info.pose`` is (T, 6) = x y z (m) + three Euler angles (rad). The release does not name the sequence; on all
149 takes of mocap_ver0.1, intrinsic X-Y-Z holds a carried cube at 0.4 deg median drift in the hand's root frame,
while extrinsic x-y-z turns it 2.3 deg per 0.1 s. ``POSE_VERSION`` names this rule; bump it when ``OBJECT_EULER``
changes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

#: scipy Euler sequence of object_info.pose[:, 3:] — upper case = intrinsic (rotating axes)
OBJECT_EULER = "XYZ"
POSE_VERSION = "2026-10-12"


def object_rotation(euler: np.ndarray) -> Rot:
    """Rotations of object_info.pose[:, 3:] (rad) under OBJECT_EULER."""
    return Rot.from_euler(OBJECT_EULER, np.asarray(euler, dtype=np.float64))


def object_pose7(pose6: np.ndarray) -> np.ndarray:
    """(T,6) object_info.pose -> (T,7) x y z + w x y z."""
    pose6 = np.asarray(pose6, dtype=np.float64)
    pose7 = np.zeros((len(pose6), 7), dtype=np.float64)
    pose7[:, :3] = pose6[:, :3]
    pose7[:, 3:] = object_rotation(pose6[:, 3:]).as_quat()[:, [3, 0, 1, 2]]
    return pose7
