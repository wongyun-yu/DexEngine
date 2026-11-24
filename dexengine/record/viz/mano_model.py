"""MANO hand model, camera conventions and batched forward kinematics for the MANO videos."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R

# isort: split
# names and suffixes are owned by dexengine/core/paths.py (no literals)
from dexengine.core.paths import mano_model_dir as _mano_model_dir  # noqa: E402  (owner of the MANO model place)

# MANO model dir: owner = paths.mano_model_dir() (data/body_models/mano of the data root); env MANO_MODEL_DIR is the
# caller's explicit choice.
MANO_MODEL_DIR = Path(os.environ.get("MANO_MODEL_DIR") or _mano_model_dir())

# Raw -> Isaac world frame transform (the retargeting layer's convention):
#   TACO is Z-up (per converter docstring). The layer applies Rz(-90°) about Z,
#     then dexengine.rl.tasks.mdp.trajectory.manager adds z+=the desk top height (core.desk.DESK_TOP_SURFACE_Z) at
#     runtime.
#   To match the view of the c64 `_retarget.mp4` (= visualization_ik.mp4 of the hand tracker's `--save-video`)
#     use **the camera that shot that video itself**, neither inferred nor fitted: camera 'front' that the
#     scene builder puts into scene.xml (trackcom, raw TACO frame, fixed xyaxes = RETARGET_CAM_XYAXES
#     below). That video is in the raw frame before placement (yaw) and this render in the placed frame, so the camera
#     also turns **by each clip's placement yaw** (`--cam_preset retarget`). One fixed --cam_dir cannot fit 64 clips at
#     once (yaw −8°..−58°). Checked: fingertip means of the npz projected with this camera vs the green marker centres
#     of the video, median error 15-24 px (3 clips). Negative components must be written `--cam_dir=-…` for argparse
#     to read them as a value.

# camera of the retargeting diagnosis video (visualization_ik.mp4): the scene's `front` camera (
# position (0.031, 0.941, 0.844), track-COM).
# trackcom: the position follows the scene centre of mass (2-8 cm per clip) but **the direction is fixed**; only the
# direction is used. In the raw TACO frame: azimuth −91.9°, elevation −41.8°, fovy 45 (MuJoCo default).
# tests/record/test_record_units.py checks it against the scene builder's source.
RETARGET_CAM_XYAXES = (-0.999, 0.033, 0.0, -0.022, -0.667, 0.745)
RETARGET_CAM_FOVY = 45.0


def retarget_cam_dir_raw() -> np.ndarray:
    """RETARGET_CAM_XYAXES -> view direction (raw TACO frame, unit vector). MuJoCo cameras look along −z:
    fwd = −(x × y)."""
    x = np.asarray(RETARGET_CAM_XYAXES[:3], dtype=np.float64)
    y = np.asarray(RETARGET_CAM_XYAXES[3:], dtype=np.float64)
    fwd = -np.cross(x, y)
    return fwd / np.linalg.norm(fwd)


def raw_dir_to_world(dir_raw: np.ndarray, placement_yaw_deg: float) -> np.ndarray:
    """Direction vector in the raw TACO frame -> this render's world frame. Only **the same rotation chain** as the hand
    vertices (Rz(yaw) then Rz(−90°)) is applied (translation and pivot do not affect directions). It must get the same
    yaw load_taco applied to the hands."""
    rz = R.from_euler("z", float(placement_yaw_deg), degrees=True).as_matrix()
    d = RZ_MINUS_90 @ (rz @ np.asarray(dir_raw, dtype=np.float64))
    return d / np.linalg.norm(d)


RZ_MINUS_90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)  # TACO Z-up


# Isaac robot head position (anchor for the front view camera). The camera
# moves with the robot in the actual Isaac rollouts; for our MANO render we
# fix the eye at this position and aim it at the per-frame MANO scene
# centroid. Verified by comparing retargeting-layer wrist coords against
# gate2_replay logged world coords: my Rz(-90°)+z+0.76 transform for TACO
# lands within 2-5cm of Isaac's actual placement.
ISAAC_HEAD_EYE = np.array([-0.5, 0.25, 1.2])

# ---------- MANO model singletons ----------
_MANO_R: smplx.MANO | None = None
_MANO_L: smplx.MANO | None = None


def get_mano(is_right: bool, batch_size: int) -> smplx.MANO:
    """Lazy-instantiate MANO model at the requested batch size."""
    global _MANO_R, _MANO_L
    existing = _MANO_R if is_right else _MANO_L
    if existing is not None and existing.batch_size == batch_size:
        return existing
    m = smplx.MANO(
        model_path=str(MANO_MODEL_DIR),
        is_rhand=is_right,
        use_pca=False,
        flat_hand_mean=True,  # CRITICAL: matches the keypoint layer's reading of
        # axis-angle as absolute pose. With flat_hand_mean=False smplx adds
        # the MANO mean pose internally, which the layer does NOT — causes
        # distal finger joints to drift progressively (35-45mm at level 3).
        # Verified with 0.0001mm joint match on TACO brush_190 frame 0.
        batch_size=batch_size,
    )
    if is_right:
        _MANO_R = m
    else:
        _MANO_L = m
    return m


def get_rest_wrist_offset(shape: torch.Tensor, is_right: bool) -> np.ndarray:
    """Compute the MANO rest-pose wrist joint position (joint[0]) for the given
    shape, with transl=0 and zero pose. The MANO template has a non-zero rest
    wrist position (~10cm offset) which must be subtracted from the desired
    world wrist position when computing the `transl` input.
    """
    model = get_mano(is_right, batch_size=1)
    with torch.no_grad():
        out = model(
            global_orient=torch.zeros(1, 3),
            hand_pose=torch.zeros(1, 45),
            betas=shape.unsqueeze(0).float() if shape.ndim == 1 else shape[:1].float(),
            transl=torch.zeros(1, 3),
            return_verts=True,
        )
    return out.joints[0, 0].cpu().numpy()  # (3,) MANO rest wrist position


def mano_fk_batch(
    poses_aa48: torch.Tensor,  # (T, 48)
    target_wrist: torch.Tensor,  # (T, 3) — DESIRED wrist position in world (output frame)
    shape: torch.Tensor,  # (10,) or (T, 10)
    is_right: bool,
) -> np.ndarray:
    """Run MANO FK in batch with the wrist joint pinned to `target_wrist`.

    MANO's FK gives joints[0] = transl + J_rest[0] (rest wrist position is
    constant given the betas). So to put the FK output wrist AT a desired
    world point, set transl = target - J_rest[0]. This eliminates the ~10cm
    apparent hand-object offset that comes from using raw `hand_trans`
    directly (which is the MANO transl PARAMETER, not the resulting wrist
    JOINT position).
    """
    T = poses_aa48.shape[0]
    model = get_mano(is_right, batch_size=T)
    if shape.ndim == 1:
        shape = shape.unsqueeze(0).expand(T, -1)

    # Compute rest wrist offset (constant for this shape)
    J_rest_0 = get_rest_wrist_offset(shape[0] if shape.ndim == 2 else shape, is_right)
    J_rest_0_t = torch.from_numpy(J_rest_0).unsqueeze(0).expand(T, -1).float()

    corrected_transl = target_wrist.float() - J_rest_0_t

    with torch.no_grad():
        out = model(
            global_orient=poses_aa48[:, :3].float(),
            hand_pose=poses_aa48[:, 3:].float(),
            betas=shape.float(),
            transl=corrected_transl,
            return_verts=True,
        )
    return out.vertices.cpu().numpy()  # (T, 778, 3)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
