"""TACO keypoint reader (phase "kinematic" of the TACO batch).

Turns one converted TACO sequence (`<dataset_dir>/raw/<raw_subdir>/<task>_<emb>.pkl`, the ManipTrans-style pickle
written by `taco_convert.py`) into the human keypoint layer:

- `processed/taco/mano/<emb>/<task>/task_info.json`
- `processed/taco/mano/<emb>/<task>/0/trajectory_keypoints.npz`
- `processed/taco/assets/objects/<id>/visual.obj` (each object's mesh in metres)

Hands come from MANO forward kinematics (`mano_lib`), objects from the converted 4x4 trajectories. One vertical shift
puts the lowest object vertex of the anchor frame on z = 0 for the whole clip, then every pose track is resampled to
the target frame rate. Called in-process by `dexengine/retarget/phases/taco.py`; every problem raises.
"""

from __future__ import annotations

import io
import json
import math
import numbers
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dexengine.data.readers import mano_lib
from dexengine.data.readers.taco_payload import HandInput, ObjectInput, extra_inputs, hand_input
from dexengine.retarget import mjutil
from dexengine.retarget.ik.prior import emb_sides

DATASET = "taco"
SIDES = ("right", "left")
EMBODIMENTS = ("right", "left", "bimanual")
#: TACO object meshes are modelled in centimetres.
MESH_SCALE = 0.01
MESH_FILE = "visual.obj"
KEYPOINT_FILE = "trajectory_keypoints.npz"
NUM_TIPS = 5
_TAG = "[TACO_READER]"


def _log(message: str) -> None:
    print(f"{_TAG} {message}", flush=True)


# ------------------------------------------------------------------------------------------------ arguments
def _check_dt(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{_TAG} {name} must be a number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{_TAG} {name} must be finite and positive, got {value}")
    return value


def _check_start(start_idx: Any) -> int:
    if isinstance(start_idx, bool) or not isinstance(start_idx, (int, np.integer)):
        raise TypeError(f"{_TAG} start_idx must be an int, got {type(start_idx).__name__}")
    if start_idx < 0:
        raise ValueError(f"{_TAG} start_idx must be >= 0, got {start_idx}")
    return int(start_idx)


# ----------------------------------------------------------------------------------------------- the pickle
def _load_storage_on_cpu(data: bytes) -> Any:
    """Rebuild a pickled torch storage on the CPU, whatever device it was saved from."""
    import torch  # noqa: PLC0415  (only pickles that hold tensors need torch)

    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


class _CpuUnpickler(pickle.Unpickler):
    """Unpickler that maps torch storages to the CPU (the converter may have run on a GPU machine)."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return _load_storage_on_cpu
        return super().find_class(module, name)


def load_sequence(path: Path) -> dict:
    """Read a converted TACO pickle; anything but a dict raises `TypeError`."""
    with open(path, "rb") as f:
        data = _CpuUnpickler(f).load()
    if not isinstance(data, dict):
        raise TypeError(f"{_TAG} {path}: expected a dict, got {type(data).__name__}")
    return data


# --------------------------------------------------------------------------------------------------- meshes
def object_id(mesh_ref: str) -> str:
    """Object id of a mesh reference: the file stem without a trailing `_cm`."""
    return Path(mesh_ref).stem.removesuffix("_cm")


def _mesh_source(dataset_dir: Path, raw_dir: Path, mesh_ref: str) -> Path | None:
    """The mesh file a reference points at: under the raw folder, else TACO's released models, else none."""
    for candidate in (
        raw_dir / mesh_ref,
        dataset_dir / "raw" / DATASET / "object_models_released" / f"{object_id(mesh_ref)}_cm.obj",
    ):
        if candidate.is_file():
            return candidate
    return None


def convert_mesh(source: Path, mesh_dir: Path) -> Path:
    """Write `source` scaled from centimetres to metres as `<mesh_dir>/visual.obj` (pymeshlab reader and writer).

    The scale is one uniform transformation matrix, frozen into the vertices. The file is written under its final name
    directly because the OBJ writer records that name in the file header.
    """
    import pymeshlab  # noqa: PLC0415  (heavy; only the TACO reader needs it)

    mesh_dir.mkdir(parents=True, exist_ok=True)
    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(str(source))
    meshes.set_matrix(transformmatrix=np.diag([MESH_SCALE, MESH_SCALE, MESH_SCALE, 1.0]))
    target = mesh_dir / MESH_FILE
    meshes.save_current_mesh(str(target))
    return target


def _mesh_vertices(mesh_file: Path) -> np.ndarray:
    """Every `v` record of a written OBJ file, in file order, as (V, 3) float64 parsed from its decimal text.

    Vertices that no face references are kept: the lowest point of the object must not depend on a loader's cleanup.
    """
    rows = []
    with open(mesh_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(("v ", "v\t")):
                parts = line.split()
                rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def _lowest_z(vertices: np.ndarray, pose: np.ndarray) -> float:
    """Lowest world z of `vertices` posed by one 4x4 transform.

    Per vertex: the three products of the rotation's third row are rounded on their own and added left to right,
    then the z translation is added — `((x*R20 + y*R21) + z*R22) + tz` in float64.
    """
    r = np.asarray(pose[2, :3], dtype=np.float64)
    tz = np.float64(pose[2, 3])
    z = vertices[:, 0] * r[0] + vertices[:, 1] * r[1]
    z = z + vertices[:, 2] * r[2]
    z = z + tz
    return float(z.min())


class _Objects:
    """Converts each referenced mesh once and remembers its folder (absolute and relative to `dataset_dir`)."""

    def __init__(self, dataset_dir: Path, raw_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.raw_dir = raw_dir
        self.files: dict[str, Path] = {}

    def stage(self, mesh_ref: str, required: bool) -> tuple[str, str | None]:
        """(object id, stored mesh dir or None). A missing source warns, or raises when `required`."""
        oid = object_id(mesh_ref)
        source = _mesh_source(self.dataset_dir, self.raw_dir, mesh_ref)
        if source is None:
            if required:
                raise FileNotFoundError(f"{_TAG} mesh of object {oid} not found ({self.raw_dir / mesh_ref})")
            _log(f"warning: mesh of object {oid} not found ({self.raw_dir / mesh_ref}); its mesh dir stays null")
            return oid, None
        folder = mjutil.object_mesh_dir(str(self.dataset_dir), DATASET, oid)
        self.files[oid] = convert_mesh(source, Path(folder))
        return oid, os.path.relpath(folder, str(self.dataset_dir))


# --------------------------------------------------------------------------------------------- pose helpers
def poses_from_matrices(mats: np.ndarray) -> np.ndarray:
    """(T, 4, 4) rigid transforms -> (T, 7) float64 translation + wxyz (SciPy, from the rotation block)."""
    mats = np.asarray(mats)
    out = np.empty((mats.shape[0], 7))
    out[:, :3] = mats[:, :3, 3]
    out[:, 3:] = Rotation.from_matrix(mats[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
    return out


def _identity_trajectory(num_frames: int) -> np.ndarray:
    return np.tile(np.eye(4), (num_frames, 1, 1))


def _is_identity(trajectory: np.ndarray) -> bool:
    return bool(np.allclose(trajectory, np.eye(4)))


def _shifted_trajectory(trajectory: np.ndarray, z_offset: float) -> np.ndarray:
    """A copy of a (T, 4, 4) trajectory with `z_offset` added to its z translation in the trajectory's own dtype.

    For the float32 TACO trajectories both operands are float32 and the sum is rounded to float32; the pose
    conversion to float64 comes after.
    """
    out = np.array(trajectory, copy=True)
    out[:, 2, 3] = out[:, 2, 3] + out.dtype.type(z_offset)
    return out


def _hand_poses(
    model: dict, hand: HandInput, side: str, flat_hand_mean: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrist poses (T, 7), fingertip positions (T, 5, 3) and wrist rotation vectors (T, 3), all float64.

    The wrist orientation is the palm frame taken through SciPy: matrix -> quaternion -> rotation (renormalised) ->
    rotation vector -> rotation -> quaternion. The round trip puts every quaternion in SciPy's canonical hemisphere;
    the rotation vectors are returned for the wrist-frame fingertips.
    """
    num_frames = hand.pose.shape[0]
    wrist = np.empty((num_frames, 7))
    tips = np.empty((num_frames, NUM_TIPS, 3))
    palms = np.empty((num_frames, 3, 3))
    for t in range(num_frames):
        joints, tip = mano_lib.mano_forward(
            model,
            hand.pose[t],
            hand.shape,
            hand.trans[t],
            flat_hand_mean=flat_hand_mean,
            use_pca=False,
            trans_is_wrist=True,
        )
        palms[t] = mano_lib.palm_matrix(joints, side)
        wrist[t, :3] = joints[mano_lib.WRIST]
        tips[t] = tip
    rotvec = Rotation.from_quat(Rotation.from_matrix(palms).as_quat()).as_rotvec()
    wrist[:, 3:] = Rotation.from_rotvec(rotvec).as_quat()[:, [3, 0, 1, 2]]
    return wrist, tips, rotvec


def _with_identity(positions: np.ndarray) -> np.ndarray:
    """(..., 3) positions -> (..., 7) poses with the identity orientation."""
    out = np.zeros(positions.shape[:-1] + (7,))
    out[..., :3] = positions
    out[..., 3] = 1.0
    return out


def _wrist_frame_tips(wrist: np.ndarray, tips: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Fingertips in the wrist frame, (T, 5, 3): each frame's five offsets `tip - wrist` (both already grounded)
    rotated by the inverse of that frame's wrist rotation (SciPy, built from the wrist rotation vector)."""
    offsets = tips - wrist[:, None, :3]
    rot = Rotation.from_rotvec(rotvec)
    out = np.empty(offsets.shape)
    for t in range(offsets.shape[0]):
        out[t] = rot[t].apply(offsets[t], inverse=True)
    return out


# --------------------------------------------------------------------------------------------------- reader
def main(
    dataset_dir: str,
    embodiment_type: str,
    task: str,
    show_viewer: bool = False,
    save_video: bool = False,
    start_idx: int = 0,
    raw_subdir: str = "taco_maniptrans",
    body_models_dir: str | None = None,
    source_ref_dt: float = 1.0 / 30.0,
    target_ref_dt: float = 0.02,
    mano_flat_hand_mean: bool = True,
) -> None:
    """Read one converted TACO sequence and write its keypoint archive, task_info.json and object meshes.

    `start_idx` drops the frames before it (it is also the grounding anchor). `body_models_dir` holds
    `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` (default `<dataset_dir>/raw/taco/body_models/mano`). Raises on any invalid
    argument or input; there is no interactive viewer, and no replay video is rendered.
    """
    if embodiment_type not in EMBODIMENTS:
        raise ValueError(f"{_TAG} embodiment_type must be one of {EMBODIMENTS}, got {embodiment_type!r}")
    source_ref_dt = _check_dt("source_ref_dt", source_ref_dt)
    target_ref_dt = _check_dt("target_ref_dt", target_ref_dt)
    start_idx = _check_start(start_idx)
    if show_viewer:
        raise NotImplementedError(f"{_TAG} the interactive viewer is not part of this reader")
    if save_video:
        _log("save_video: this reader renders no replay video; the archive is written as usual")

    root = Path(dataset_dir)
    raw_dir = root / "raw" / raw_subdir
    pkl = raw_dir / f"{task}_{embodiment_type}.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(f"{_TAG} converted sequence not found: {pkl}")
    models = _load_models(Path(body_models_dir) if body_models_dir else root / "raw" / DATASET / "body_models" / "mano")
    hands, num_frames, extras = _read_inputs(pkl, embodiment_type, start_idx)

    objects = _Objects(root, raw_dir)
    mesh_dirs, obj_ids, trajectories = _stage_hand_objects(objects, hands, num_frames)
    extra_records = []
    for extra in extras:
        oid, folder = objects.stage(extra.mesh_ref, required=True)
        extra_records.append({"obj_id": oid, "mesh_dir": folder, "convex_dir": folder})

    object_active = {side: not _is_identity(trajectories[side]) for side in SIDES}
    z_offset = _ground_offset(objects, obj_ids, mesh_dirs, trajectories, object_active, start_idx, task)
    arrays = _source_arrays(models, hands, trajectories, object_active, extras, z_offset, mano_flat_hand_mean)

    # keep frames [start_idx:], resample to the target rate
    source_times, target_times = mano_lib.resample_times(num_frames - start_idx, source_ref_dt, target_ref_dt)
    archive = {k: mano_lib.resample_pose_sequence(v[start_idx:], source_times, target_times) for k, v in arrays.items()}
    _zero_absent_hands(archive, hands, target_times.size)
    archive["timestamps"] = target_times
    _replace_grid50_objects(archive, hands, start_idx, target_times.size)

    worst = mano_lib.repair_object_quaternions(archive)
    if worst is not None:
        _log(f"[OBJ_QUAT_NORM] {task}: object quaternions rescaled to unit length (worst |norm - 1| = {worst:.6f})")
    archive["n_extra_objects"] = np.array(len(extras), dtype=np.int64)
    _write(root, embodiment_type, task, archive, mesh_dirs, extra_records, source_ref_dt, target_ref_dt)


def _load_models(mano_dir: Path) -> dict:
    """The right MANO model and the left model built from it, by side."""
    right_model = mano_lib.load_model(model_type="manor", model_path=str(mano_dir))
    left_model, _ = mano_lib.left_model(
        DATASET, right_model, mano_lib.load_model(model_type="manol", model_path=str(mano_dir))
    )
    return {"right": right_model, "left": left_model}


def _read_inputs(pkl: Path, embodiment_type: str, start_idx: int) -> tuple[dict[str, HandInput], int, list]:
    """The active hands' inputs (equal frame counts), the frame count and the extra objects of one sequence."""
    data = load_sequence(pkl)
    hands: dict[str, HandInput] = {}
    for side in emb_sides(embodiment_type):
        if side not in data:
            raise KeyError(f"{_TAG} {pkl.name}: the {side} hand payload is missing")
        hands[side] = hand_input(data[side], f"{pkl.name}:{side}")
    lengths = {side: h.pose.shape[0] for side, h in hands.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{_TAG} {pkl.name}: active hands have different frame counts {lengths}")
    num_frames = next(iter(lengths.values()))
    if start_idx >= num_frames:
        raise ValueError(f"{_TAG} start_idx {start_idx} >= frame count {num_frames}")
    extras: list[ObjectInput] = extra_inputs(data.get("extra_objects"), num_frames, pkl.name)
    return hands, num_frames, extras


def _stage_hand_objects(objects: _Objects, hands: dict[str, HandInput], num_frames: int) -> tuple[dict, dict, dict]:
    """Each side's object: mesh dir, object id and source-rate trajectory (identity for an inactive side or a left
    hand holding the right hand's object)."""
    mesh_dirs: dict[str, str | None] = dict.fromkeys(SIDES)
    obj_ids: dict[str, str | None] = dict.fromkeys(SIDES)
    trajectories: dict[str, np.ndarray] = {}
    for side in SIDES:
        trajectories[side] = _identity_trajectory(num_frames)
        if side not in hands:
            continue
        oid = object_id(hands[side].mesh_ref)
        if side == "left" and oid == obj_ids["right"]:
            _log(f"left object {oid} is the right object; the left side keeps no object")
            continue
        obj_ids[side], mesh_dirs[side] = objects.stage(hands[side].mesh_ref, required=False)
        trajectories[side] = np.asarray(hands[side].obj_trajectory)  # stored dtype (float32 for TACO)
    return mesh_dirs, obj_ids, trajectories


def _ground_offset(
    objects: _Objects,
    obj_ids: dict,
    mesh_dirs: dict,
    trajectories: dict,
    object_active: dict,
    start_idx: int,
    task: str,
) -> float:
    """One z shift putting the lowest object vertex of the anchor frame on z = 0 (0 when no object mesh exists)."""
    lows = []
    for side in SIDES:
        oid = obj_ids[side]
        if (
            object_active[side]
            and mesh_dirs[side] is not None
            and oid in objects.files
            and objects.files[oid].is_file()
        ):
            lows.append(_lowest_z(_mesh_vertices(objects.files[oid]), trajectories[side][start_idx]))
    if lows:
        return 0.0 - min(lows)
    _log(f"warning: {task}: no object mesh to ground on; z offset 0")
    return 0.0


def _source_arrays(
    models: dict,
    hands: dict[str, HandInput],
    trajectories: dict,
    object_active: dict,
    extras: list,
    z_offset: float,
    flat_hand_mean: bool,
) -> dict[str, np.ndarray]:
    """Source-rate poses by archive key, grounded by `z_offset`; object poses get the shift in their stored dtype
    before the pose conversion."""
    arrays: dict[str, np.ndarray] = {}
    rotvecs: dict[str, np.ndarray] = {}
    for side in SIDES:
        if side in hands:
            wrist, tips, rotvecs[side] = _hand_poses(models[side], hands[side], side, flat_hand_mean)
            wrist[:, 2] += z_offset
            tips[..., 2] += z_offset
            arrays[f"qpos_wrist_{side}"] = wrist
            arrays[f"qpos_finger_{side}"] = _with_identity(tips)
        traj = trajectories[side]
        arrays[f"qpos_obj_{side}"] = poses_from_matrices(
            _shifted_trajectory(traj, z_offset) if object_active[side] else traj
        )
    for side in hands:
        wrist = arrays[f"qpos_wrist_{side}"]
        arrays[f"qpos_wrist_{side}_wrist_frame"] = _with_identity(np.zeros((wrist.shape[0], 3)))
        arrays[f"qpos_finger_{side}_wrist_frame"] = _with_identity(
            _wrist_frame_tips(wrist, arrays[f"qpos_finger_{side}"][..., :3], rotvecs[side])
        )
    for k, extra in enumerate(extras):
        arrays[f"qpos_extra_obj_{k}"] = poses_from_matrices(_shifted_trajectory(extra.obj_trajectory, z_offset))
    return arrays


def _zero_absent_hands(archive: dict, hands: dict, n_out: int) -> None:
    """An absent hand is all zeros, in every key."""
    for side in SIDES:
        if side not in hands:
            archive[f"qpos_wrist_{side}"] = np.zeros((n_out, 7))
            archive[f"qpos_finger_{side}"] = np.zeros((n_out, NUM_TIPS, 7))
            archive[f"qpos_wrist_{side}_wrist_frame"] = np.zeros((n_out, 7))
            archive[f"qpos_finger_{side}_wrist_frame"] = np.zeros((n_out, NUM_TIPS, 7))


def _replace_grid50_objects(archive: dict, hands: dict, start_idx: int, n_out: int) -> None:
    """Objects already on the 50 Hz grid replace the resampled ones, for every active hand whose payload has the block
    (spec §2.6 — this includes a left hand sharing the right hand's object, whose payload carries that object's
    block). The last row repeats when the block is shorter than the output."""
    for side in SIDES:
        grid = hands[side].grid50 if side in hands else None
        if grid is None:
            continue
        if start_idx != 0:
            _log(f"warning: {side} obj_trajectory_grid50 skipped (start_idx {start_idx} != 0)")
            continue
        rows = np.asarray(grid[:n_out], dtype=np.float64)
        if rows.shape[0] < n_out:
            rows = np.concatenate([rows, np.repeat(rows[-1:], n_out - rows.shape[0], axis=0)])
        archive[f"qpos_obj_{side}"] = rows


def _write(
    root: Path,
    embodiment_type: str,
    task: str,
    archive: dict,
    mesh_dirs: dict,
    extra_records: list,
    source_ref_dt: float,
    target_ref_dt: float,
) -> None:
    """Write task_info.json and the keypoint archive in their processed folders."""
    clip_dir = Path(mjutil.processed_data_dir(str(root), DATASET, "mano", embodiment_type, task, 0))
    clip_dir.mkdir(parents=True, exist_ok=True)
    task_info = {
        "task": task,
        "dataset_name": DATASET,
        "robot_type": "mano",
        "embodiment_type": embodiment_type,
        "data_id": 0,
        "right_object_mesh_dir": mesh_dirs["right"],
        "left_object_mesh_dir": mesh_dirs["left"],
        "right_object_convex_dir": mesh_dirs["right"],
        "left_object_convex_dir": mesh_dirs["left"],
        "ref_dt": target_ref_dt,
        "source_ref_dt": source_ref_dt,
        "extra_objects": extra_records,
    }
    with open(clip_dir.parent / "task_info.json", "w", encoding="utf-8") as f:
        json.dump(task_info, f, indent=2)
    # member order is part of the file bytes: hands and objects per side, wrist-frame
    # keys, the extra-object count and poses, then the sample times; any other key (`quat_normalized`) comes last
    order = [f"qpos_{part}_{side}" for side in SIDES for part in ("wrist", "finger", "obj")]
    order += [f"qpos_{part}_{side}_wrist_frame" for side in SIDES for part in ("wrist", "finger")]
    order += ["n_extra_objects"]
    order += sorted((k for k in archive if k.startswith("qpos_extra_obj_")), key=lambda k: int(k.rsplit("_", 1)[1]))
    order += ["timestamps"]
    order += [k for k in archive if k not in order]
    np.savez(clip_dir / KEYPOINT_FILE, **{k: archive[k] for k in order})
    _log(f"wrote {clip_dir / KEYPOINT_FILE} ({archive['timestamps'].size} frames)")
