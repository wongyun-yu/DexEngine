"""MANO keypoint library: model loading, joint and fingertip forward kinematics, palm frame, resampling, and the
object-quaternion repair of a keypoint archive.

Importers: `common.py` (resampling), `hocap.py` and `hot3d.py` (model loading, the left-model statement, forward
kinematics) and the TACO reader `taco.py` (all of it).

Conventions: metres, radians, seconds; a pose 7-vector is `[x, y, z, qw, qx, qy, qz]` (scalar first). SciPy's
`Rotation` is scalar last, so every conversion here reorders explicitly.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from dexengine.data.readers import mano_conventions
from dexengine.retarget.quat_bounds import OBJ_QUAT_ATOL

#: The left-model convention rule recorded in `task_info.mano_model.rule` (owner: `mano_conventions.LEFT_FIX`).
MC_RULE = "dexengine/data/readers/mano_conventions.py:LEFT_FIX"

#: Model file of each supported model type.
MODEL_FILES = {"manor": "MANO_RIGHT.pkl", "manol": "MANO_LEFT.pkl"}
#: Entries of a MANO pickle that the library hands out (values are left exactly as unpickled).
MODEL_KEYS = ("v_template", "shapedirs", "J_regressor", "J", "kintree_table", "hands_mean", "hands_components")
#: Fingertip vertices (thumb, index, middle, ring, pinky) and the joint that carries each of them rigidly.
TIP_VERTICES = (745, 317, 445, 556, 673)
TIP_PARENT_JOINTS = (15, 3, 6, 12, 9)
#: Joints that span the palm frame: wrist, index / middle / ring MCP.
WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP = 0, 1, 4, 10
NUM_JOINTS = 16
NUM_BETAS = 10
_NORM_EPS = 1e-8


# ----------------------------------------------------------------------------------------------- model loading
def load_model(model_type: str, model_path: str, **ignored: Any) -> dict:
    """Load one MANO hand model.

    `model_type` is "manor" (right hand, `MANO_RIGHT.pkl`) or "manol" (left hand, `MANO_LEFT.pkl`); anything else
    raises `NotImplementedError`. The pickle in `model_path` is read with latin-1 decoding, and the returned dict holds
    its entries `MODEL_KEYS` unchanged. Extra keyword arguments are accepted and ignored.
    """
    if model_type not in MODEL_FILES:
        raise NotImplementedError(
            f"mano_lib: model type {model_type!r} is not supported (known: {sorted(MODEL_FILES)})"
        )
    path = Path(model_path) / MODEL_FILES[model_type]
    if not path.is_file():
        raise FileNotFoundError(f"mano_lib: MANO model file not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    missing = [k for k in MODEL_KEYS if k not in data]
    if missing:
        raise KeyError(f"mano_lib: {path} lacks the MANO entries {missing}")
    return {k: data[k] for k in MODEL_KEYS}


def left_model(dataset: str, right: dict, left: dict) -> tuple[dict, dict]:
    """The left model as `dataset` fitted its hands, and the record of that decision.

    Measures `sum |left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :]|` and asks `mano_conventions.left_negate`
    whether the x component of the ten shape directions must be negated (it raises for an unknown dataset or a
    contradiction). When it must, a new dict with a negated copy of `shapedirs` is returned and `left` is left
    untouched; otherwise `left` itself is returned.
    """
    left_sd = np.asarray(left["shapedirs"], dtype=np.float64)
    right_sd = np.asarray(right["shapedirs"], dtype=np.float64)
    total = float(np.abs(left_sd[:, 0, :] - right_sd[:, 0, :]).sum())
    flip = bool(mano_conventions.left_negate(dataset, total))
    record = {
        "rule": MC_RULE,
        "left_sd0_sum": round(total, 6),
        "threshold": mano_conventions.LEFT_SD0_DEFECT_SUM,
        "left_flipped": flip,
    }
    if not flip:
        return left, record
    fixed_sd = np.array(left_sd, dtype=np.float64, copy=True)
    fixed_sd[:, 0, :] *= -1.0
    fixed = dict(left)
    fixed["shapedirs"] = fixed_sd
    return fixed, record


# ----------------------------------------------------------------------------------------- forward kinematics
def _parents(model: dict) -> np.ndarray:
    """Parent index of every joint (-1 for the root); a parent always precedes its child."""
    raw = np.asarray(model["kintree_table"])[0].astype(np.int64)
    parents = np.where(raw > 1000, -1, raw)
    if parents.shape != (NUM_JOINTS,) or parents[0] != -1 or any(parents[j] >= j for j in range(1, NUM_JOINTS)):
        raise ValueError(f"mano_lib: unexpected kinematic tree {raw.tolist()}")
    return parents


def _dense(matrix: Any) -> np.ndarray:
    """A sparse or dense matrix as a dense NumPy array."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def mano_forward(
    model: dict,
    hand_pose,
    hand_shape,
    hand_trans,
    flat_hand_mean: bool = True,
    use_pca: bool = False,
    trans_is_wrist: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """MANO joint positions (no pose blend shapes) and true fingertip vertices of one frame.

    `hand_pose` holds 48 values: the root axis-angle, then 15 joint axis-angles (or, with `use_pca`, 45 PCA
    coefficients mapped through `hands_components`). Without `flat_hand_mean`, `hands_mean` is added to the 45 finger
    values. `hand_shape` holds at least ten betas (the first ten are used). With `trans_is_wrist`, `hand_trans` is the
    wrist (joint 0) position; otherwise it is an additive translation.

    Everything except the rotations runs in float32: the pose, shape and translation, every model array used
    (template, shape directions, the dense joint regressor, PCA components, hand mean) and the stored joint positions.
    Rotation matrices are float64 from SciPy. Returns `(joints (16, 3), tips (5, 3))`, both float32; tips are thumb,
    index, middle, ring, pinky.
    """
    pose = np.array(hand_pose, dtype=np.float32).reshape(-1)
    if pose.size != 3 * NUM_JOINTS:
        raise ValueError(f"mano_forward: hand_pose must hold {3 * NUM_JOINTS} values, got {pose.size}")
    shape = np.asarray(hand_shape, dtype=np.float32).reshape(-1)
    if shape.size < NUM_BETAS:
        raise ValueError(f"mano_forward: hand_shape must hold at least {NUM_BETAS} values, got {shape.size}")
    trans = np.asarray(hand_trans, dtype=np.float32).reshape(-1)
    if trans.size != 3:
        raise ValueError(f"mano_forward: hand_trans must hold 3 values, got {trans.size}")

    if use_pca:
        pose[3:] = pose[3:] @ np.asarray(model["hands_components"], dtype=np.float32)
    if not flat_hand_mean:
        pose[3:] = pose[3:] + np.asarray(model["hands_mean"], dtype=np.float32)

    template = np.asarray(model["v_template"], dtype=np.float32)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float32)[:, :, :NUM_BETAS]
    verts = template + np.einsum("vcb,b->vc", shapedirs, shape[:NUM_BETAS])
    rest = _dense(model["J_regressor"]).astype(np.float32) @ verts

    parents = _parents(model)
    local = Rotation.from_rotvec(pose.reshape(NUM_JOINTS, 3)).as_matrix()
    rot = np.empty((NUM_JOINTS, 3, 3))
    pos = np.empty((NUM_JOINTS, 3), dtype=np.float32)
    for j in range(NUM_JOINTS):
        p = parents[j]
        if p < 0:
            rot[j] = local[j]
            pos[j] = rest[j]
        else:
            rot[j] = rot[p] @ local[j]
            pos[j] = pos[p] + rot[p] @ (rest[j] - rest[p])

    shift = trans - pos[WRIST] if trans_is_wrist else trans
    joints = pos + shift
    tips = np.stack(
        [rot[p] @ (verts[v] - rest[p]) + pos[p] + shift for v, p in zip(TIP_VERTICES, TIP_PARENT_JOINTS, strict=True)]
    ).astype(np.float32)
    return joints.astype(np.float32), tips


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + _NORM_EPS)


def palm_matrix(joints: np.ndarray, side: str) -> np.ndarray:
    """Palm frame of one hand from its 16 MANO joints, as a 3x3 rotation matrix with columns (x, y, z).

    z runs wrist -> middle MCP, the auxiliary y runs ring MCP -> index MCP, x = y_aux x z, y = z x x; the left hand
    negates x and y. The arithmetic stays in the dtype of `joints` (float32 for `mano_forward` output); integer input
    is promoted to float64.
    """
    if side not in ("right", "left"):
        raise ValueError(f"palm_matrix: side must be 'right' or 'left', got {side!r}")
    j = np.asarray(joints)
    if not np.issubdtype(j.dtype, np.floating):
        j = j.astype(np.float64)
    z = _unit(j[MIDDLE_MCP] - j[WRIST])
    y_aux = _unit(j[INDEX_MCP] - j[RING_MCP])
    x = _unit(np.cross(y_aux, z))
    y = _unit(np.cross(z, x))
    if side == "left":
        x, y = -x, -y
    return np.stack([x, y, z], axis=1)


# ------------------------------------------------------------------------------------------------- resampling
def resample_times(num_frames: int, source_dt: float, target_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Source sample times `k * source_dt` and fixed-rate target times `k * target_dt` covering the same duration.

    The target grid ends at the last whole target step inside the source duration (the trailing fractional interval
    is dropped); a last target within 1e-10 of the duration is set to the duration exactly. One frame gives `[0.0]`.
    """
    if num_frames <= 0:
        raise ValueError(f"resample_times: num_frames must be positive, got {num_frames}")
    for name, dt in (("source_dt", source_dt), ("target_dt", target_dt)):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"resample_times: {name} must be finite and positive, got {dt}")
    source = np.arange(num_frames, dtype=np.float64) * source_dt
    if num_frames == 1:
        return source, np.zeros(1)
    duration = (num_frames - 1) * source_dt
    count = int(np.floor(duration / target_dt + 1e-9)) + 1
    target = np.arange(count, dtype=np.float64) * target_dt
    if abs(target[-1] - duration) <= 1e-10:
        target[-1] = duration
    return source, target


def _same_times(source_times: np.ndarray, target_times: np.ndarray) -> bool:
    return source_times.shape == target_times.shape and bool(np.array_equal(source_times, target_times))


def _check_times(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> None:
    if source_times.ndim != 1 or target_times.ndim != 1 or target_times.size == 0:
        raise ValueError("resample: source and target times must be non-empty 1-D arrays")
    if values.shape[0] != source_times.size:
        raise ValueError(f"resample: {values.shape[0]} frames but {source_times.size} source times")


def _interp_columns(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Linear interpolation along the first axis, column by column in float64, returned as float64."""
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    out = np.empty((target_times.size, flat.shape[1]))
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(target_times, source_times, flat[:, c])
    return out.reshape((target_times.size,) + values.shape[1:])


def resample_point_sequence(values, source_times, target_times) -> np.ndarray:
    """Resample a (T, ..., 3) point sequence (or any array along its first axis) linearly to `target_times`.

    Identical time arrays return a copy; a single source frame or a single target repeats the first frame.
    The output keeps the input dtype.
    """
    values = np.asarray(values)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    _check_times(values, source_times, target_times)
    if _same_times(source_times, target_times):
        return values.copy()
    if values.shape[0] == 1 or target_times.size == 1:
        return np.repeat(values[:1], target_times.size, axis=0)
    return _interp_columns(values, source_times, target_times).astype(values.dtype)


def _unit_quaternions(quats: np.ndarray) -> np.ndarray:
    """Scale (N, 4) quaternions to unit length; a zero quaternion raises `ValueError`."""
    norms = np.linalg.norm(quats, axis=1)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("resample_pose_sequence: zero-norm or non-finite quaternion")
    return quats / norms[:, None]


def _sign_continuous(quats: np.ndarray) -> np.ndarray:
    """Flip every quaternion whose dot product with the previous (already adjusted) one is negative."""
    out = quats.copy()
    for k in range(1, out.shape[0]):
        if np.dot(out[k - 1], out[k]) < 0.0:
            out[k] = -out[k]
    return out


def resample_pose_sequence(values, source_times, target_times) -> np.ndarray:
    """Resample a (T, ..., 7) pose sequence (`xyz + wxyz` per pose) to `target_times`.

    Per pose track: positions are interpolated linearly; quaternions are normalised, made sign-continuous, Slerped
    (SciPy) at the target times and made sign-continuous again. Identical time arrays return a copy; a single source
    frame or a single target repeats the first pose with its quaternion normalised. The output keeps the input dtype.
    """
    values = np.asarray(values)
    source_times = np.asarray(source_times, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != 7:
        raise ValueError(f"resample_pose_sequence: expected (T, ..., 7) poses, got {values.shape}")
    _check_times(values, source_times, target_times)
    if _same_times(source_times, target_times):
        return values.copy()
    tracks = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1, 7)
    n_out = target_times.size
    out = np.empty((n_out, tracks.shape[1], 7))
    for i in range(tracks.shape[1]):
        quats = _unit_quaternions(tracks[:, i, 3:])
        if tracks.shape[0] == 1 or n_out == 1:
            out[:, i, :3] = tracks[0, i, :3]
            out[:, i, 3:] = quats[0]
            continue
        out[:, i, :3] = _interp_columns(tracks[:, i, :3], source_times, target_times)
        quats = _sign_continuous(quats)
        slerp = Slerp(source_times, Rotation.from_quat(quats[:, [1, 2, 3, 0]]))
        out[:, i, 3:] = _sign_continuous(slerp(target_times).as_quat()[:, [3, 0, 1, 2]])
    return out.reshape((n_out,) + values.shape[1:]).astype(values.dtype)


# --------------------------------------------------------------------------------------------- archive repair
def repair_object_quaternions(archive: dict) -> float | None:
    """Rescale the object quaternions of a keypoint archive to unit length — every clip (quat_bounds: a reader that
    builds a rotation from an off-unit quaternion places a scaled object).

    `worst` is the largest `| ||q|| - 1 |` over `qpos_obj_right` and `qpos_obj_left` before the rescale. Both keys
    are replaced by float64 copies with unit quaternions (a zero quaternion raises). Beyond `OBJ_QUAT_ATOL` (a
    tracking glitch) `archive["quat_normalized"]` is set to `int8(1)` and `worst` is returned; within it `None` is
    returned (the source's ordinary off-unit noise, rescaled without a mark).
    """
    keys = ("qpos_obj_right", "qpos_obj_left")
    worst = 0.0
    for key in keys:
        poses = np.array(archive[key], dtype=np.float64, copy=True)
        if poses.size:
            norms = np.linalg.norm(poses[..., 3:7], axis=-1, keepdims=True)
            if np.any(norms == 0.0):
                raise ValueError(f"repair_object_quaternions: {key} holds a zero quaternion")
            worst = max(worst, float(np.abs(norms - 1.0).max()))
            poses[..., 3:7] = poses[..., 3:7] / norms
        archive[key] = poses
    if worst <= OBJ_QUAT_ATOL:
        return None
    archive["quat_normalized"] = np.int8(1)
    return worst
