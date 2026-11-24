"""Shared plumbing for the HRDexDB / HO-Cap / DexCanvas processors.

Each of those datasets ships hand-object trajectories in its own layout, but
they all reduce to the same thing the retargeting consumes: a per-frame wrist pose,
five fingertip positions, and an object pose per hand.  This module owns the
parts that must stay identical across the three -- MANO keypoint conventions,
grounding, resampling, mesh staging, and the ``trajectory_keypoints.npz`` /
``task_info.json`` contract -- so a fix lands once instead of three times.

The output contract is taken from ``taco.py`` and is byte-compatible with it:
every pose is ``[x, y, z, qw, qx, qy, qz]``.  ``contact`` / ``contact_pos`` are
deliberately absent; the contact detector (``dexengine.retarget.contact``) adds those later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexengine.data.readers.mano_lib import resample_point_sequence, resample_pose_sequence, resample_times

# MANO's 21-keypoint layout (the one HRDexDB and DexCanvas publish directly).
# 0 wrist, then thumb/index/middle/ring/pinky as (MCP, PIP, DIP, TIP).
J21_WRIST = 0
J21_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
# The joint one step proximal of each tip: DIP for the fingers, IP for the
# thumb. IK uses these as a second target per finger so the split of the curl
# between MCP, PIP and DIP stops being the posture prior's guess -- see the
# "Distal (DIP) targets" section of README.md.
J21_DISTAL = [3, 7, 11, 15, 19]
J21_MCP_MIDDLE, J21_MCP_INDEX, J21_MCP_RING = 9, 5, 13

IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class HandTrack:
    """One hand over time.

    ``wrist`` is (T,7) wxyz-first, ``tips`` and ``distal`` are (T,5,3) in the
    finger order thumb, index, middle, ring, pinky. ``distal`` is the joint
    just proximal of each tip; leaving it None costs the distal IK targets.
    """

    wrist: np.ndarray
    tips: np.ndarray
    distal: np.ndarray | None = None


@dataclass
class SequenceData:
    """One raw sequence, already in metres and in a single world frame."""

    task: str
    source_fps: float
    right: HandTrack | None = None
    left: HandTrack | None = None
    obj_right: np.ndarray | None = None  # (T,7)
    obj_left: np.ndarray | None = None  # (T,7)
    mesh_right: Path | None = None
    mesh_left: Path | None = None
    mesh_name_right: str | None = None
    mesh_name_left: str | None = None
    # Objects present in the scene but not attached to a hand. The pipeline
    # already carries these end to end (task_info["extra_objects"] plus
    # qpos_extra_obj_<i> in the archive); a dataset that stages several objects
    # and picks them up in turn needs them, or the take is retargeted against a
    # table that looks empty.
    extra_objects: list[tuple[str, Path, np.ndarray]] = field(default_factory=list)
    # (T,) bool at the SOURCE rate, True where a reader reconstructed the frame
    # instead of reading it. Optional -- a reader that never interpolates leaves
    # it None. Written through to the archive so a consumer can weight or drop
    # those frames instead of having to trust that every frame is measured.
    interp_mask: np.ndarray | None = None
    # Object GT gaps a reader measured (hot3d `_fill_gt_gaps`) -- written through to
    # task_info["object_gt_gap"] as is. None = the reader does not measure them (key absent).
    object_gt_gap: dict | None = None
    # {side: (1,7)} first-frame object pose that anchors the ground offset, when the written poses may differ from
    # the ones the frame convention was fixed on (hot3d gap filling: an interpolated frame 0 would
    # otherwise shift the whole clip in z and silently invalidate the INTAKE ledgers measured in that frame).
    # None = ground on obj_<side>[:1] as always.
    ground_first_pose: dict | None = None
    # How a reader turned the source world into this +z-up frame (hot3d `HEADSET_UP`: device, up axis, rotation) --
    # written through to task_info["source_frame"] as is. None = the reader applies no frame rule (key absent).
    source_frame: dict | None = None
    # Which MANO hand models a reader posed the joints with when it corrects a published model (hot3d `mano_models`:
    # rule, test value, whether the left model was corrected) -- written through to task_info["mano_model"] as is.
    # None = the reader uses the model files as they are (key absent).
    mano_model: dict | None = None

    @property
    def num_frames(self) -> int:
        for track in (self.right, self.left):
            if track is not None:
                return track.wrist.shape[0]
        raise ValueError(f"{self.task}: neither hand present")


def wrist_frame_from_axes(z_vec: np.ndarray, y_aux_vec: np.ndarray, side: str) -> np.ndarray:
    """Wrist orientation (wxyz) from a palm Z axis and an auxiliary Y axis.

    Split out of ``wrist_frame_from_joints21`` so a dataset whose landmark
    layout is not MANO-21 (SHOW3D publishes the UmeTrack/HOT3D order) can reach
    the *same* geometry instead of re-deriving it. The two callers must stay on
    one implementation: a second copy of this drifts silently, and a wrist frame
    that is a few degrees off does not fail any shape check -- it just retargets
    a slightly wrong hand for the whole dataset.
    """
    z = np.asarray(z_vec, dtype=np.float64).copy()
    z /= np.linalg.norm(z) + 1e-8
    y_aux = np.asarray(y_aux_vec, dtype=np.float64).copy()
    y_aux /= np.linalg.norm(y_aux) + 1e-8
    x = np.cross(y_aux, z)
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    y /= np.linalg.norm(y) + 1e-8
    if side == "left":
        x, y = -x, -y
    xyzw = R.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
    return xyzw[[3, 0, 1, 2]]


def wrist_frame_from_joints21(joints21: np.ndarray, side: str) -> np.ndarray:
    """Wrist orientation (wxyz) from palm-rigid MCP joints.

    Mirrors ``taco.extract_hand_data``: the Z axis runs wrist -> middle MCP and
    the auxiliary Y axis spans ring MCP -> index MCP.  Those are knuckles, so
    the frame does not swing when the fingers flex; PIP joints would move it by
    tens of degrees within a single clip.
    """
    j = np.asarray(joints21, dtype=np.float64)
    return wrist_frame_from_axes(
        j[J21_MCP_MIDDLE] - j[J21_WRIST],
        j[J21_MCP_INDEX] - j[J21_MCP_RING],
        side,
    )


def hand_track_from_joints21(joints21_seq: np.ndarray, side: str) -> HandTrack:
    """Build a HandTrack from a (T,21,3) keypoint sequence."""
    joints21_seq = np.asarray(joints21_seq, dtype=np.float64)
    if joints21_seq.ndim != 3 or joints21_seq.shape[1:] != (21, 3):
        raise ValueError(f"expected (T,21,3), got {joints21_seq.shape}")
    num_frames = joints21_seq.shape[0]
    wrist = np.zeros((num_frames, 7))
    wrist[:, :3] = joints21_seq[:, J21_WRIST]
    for i in range(num_frames):
        wrist[i, 3:] = wrist_frame_from_joints21(joints21_seq[i], side)
    return HandTrack(
        wrist=wrist,
        tips=joints21_seq[:, J21_TIPS].copy(),
        distal=joints21_seq[:, J21_DISTAL].copy(),
    )


def mesh_metre_scale(mesh_path: Path) -> float:
    """Return the factor that puts ``mesh_path`` into metres.

    Decided from the geometry rather than from a units field, because DexCanvas
    mixes the two in one directory: its primitives are modelled in millimetres
    (cube1 is 80x80x80) while mug.stl is already in metres (0.117 m across).
    Its own ``object_heights.json`` assumes millimetres for every mesh, so it
    reports the mug as 8.1e-05 m tall; that file is not trustworthy here.
    A hand-held object is never more than a metre across, so the extent settles
    it unambiguously.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    extent = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
    if extent <= 0:
        raise ValueError(f"degenerate mesh: {mesh_path}")
    return 0.001 if extent > 1.0 else 1.0


def _mesh_points(mesh_path: Path, scale: float, max_points: int = 2000) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64) * scale
    if verts.shape[0] > max_points:
        stride = int(np.ceil(verts.shape[0] / max_points))
        verts = verts[::stride]
    return verts


def _pose7_to_matrices(pose7: np.ndarray) -> np.ndarray:
    """(T,7) xyz+wxyz -> (T,4,4)."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    mats = np.tile(np.eye(4), (pose7.shape[0], 1, 1))
    mats[:, :3, 3] = pose7[:, :3]
    mats[:, :3, :3] = R.from_quat(pose7[:, [4, 5, 6, 3]]).as_matrix()
    return mats


def ground_offset(
    seq: SequenceData,
    mesh_scales: dict[str, float],
    mode: str = "object_first_frame",
) -> float:
    """Height to subtract so the scene sits on the floor.

    ``mode="object_first_frame"`` (default, unchanged): the object rests on the
    floor at the FIRST frame.

    ``mode="scene_min"``: the floor goes under the LOWEST point anything reaches
    over the whole clip -- object mesh and every hand keypoint. For a capture
    whose world frame has no real floor, that is the only choice that keeps the
    hand out of the ground. See the mode note below.

    Anchored on frame 0, not on the minimum over the clip. These sequences
    start with the object sitting on a surface and then lift it, so the
    clip-wide minimum is the moment just before the lift and grounding on it
    leaves the object floating for the rest of the take -- measured on
    HRDexDB, a median of 12 to 20 cm off the floor for 96%+ of frames. Frame 0
    puts it on the floor where it starts and lets the lift be a real motion.

    Object meshes decide it: hand keypoints alone sit above the table by the
    thickness of whatever is held, and a hand grasping the side of a bowl has
    fingertips below the bowl's bottom. The minimum z over the transformed
    mesh vertices is the object's actual support surface. One offset goes to
    both hands and both objects, so the raw hand-object geometry is preserved.
    """
    if mode not in {"object_first_frame", "scene_min"}:
        raise ValueError(f"unknown ground mode: {mode!r}")

    lows: list[float] = []
    for side in ("right", "left"):
        mesh_path = getattr(seq, f"mesh_{side}")
        obj = getattr(seq, f"obj_{side}")
        if mesh_path is None or obj is None:
            continue
        verts = _mesh_points(mesh_path, mesh_scales[side])
        anchor = (seq.ground_first_pose or {}).get(side)
        poses = obj if mode == "scene_min" else (anchor if anchor is not None else obj[:1])
        lows.extend(float((verts @ mat[:3, :3].T + mat[:3, 3])[:, 2].min()) for mat in _pose7_to_matrices(poses))
    if mode == "scene_min":
        # Hands too: the object can be held high while a hand reaches below it,
        # and grounding on the object alone then buries the hand.
        for track in (seq.right, seq.left):
            if track is None:
                continue
            lows.append(float(track.wrist[:, 2].min()))
            lows.append(float(track.tips[..., 2].min()))
            if track.distal is not None:
                lows.append(float(track.distal[..., 2].min()))
    elif not lows:
        lows.extend(
            float(min(track.tips[0, :, 2].min(), track.wrist[0, 2]))
            for track in (seq.right, seq.left)
            if track is not None
        )
    return float(min(lows)) if lows else 0.0


def _stage_mesh(
    mesh_path: Path,
    object_name: str,
    dataset_dir: Path,
    dataset_name: str,
    scale: float,
) -> str:
    """Copy one object mesh under ``processed/<ds>/assets/objects/<name>``.

    Meshes are rewritten in metres so downstream MuJoCo XML generation does not
    have to carry a per-object scale.  Returns the dataset-relative directory
    recorded in ``task_info.json``.
    """
    import trimesh

    rel = Path("processed") / dataset_name / "assets" / "objects" / object_name
    out_dir = dataset_dir / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    # Named visual.obj, not <object>.obj: the decomposition, contact detector and
    # scene builder all look for that exact filename. Getting it wrong is
    # silent -- decomposition skips the object with a warning, then the scene
    # is built without it.
    out_file = out_dir / "visual.obj"
    if not out_file.exists():
        mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
        if scale != 1.0:
            mesh.apply_scale(scale)
        mesh.export(str(out_file))
    return str(rel)


def _require_hands(seq: SequenceData, embodiment_type: str) -> None:
    """Every hand the embodiment needs is present in ``seq``."""
    wanted = {"right": ["right"], "left": ["left"], "bimanual": ["right", "left"]}[embodiment_type]
    for side in wanted:
        if getattr(seq, side) is None:
            raise ValueError(f"{seq.task}: embodiment_type={embodiment_type} needs the {side} hand, which is absent")


def _hand_targets(
    out: dict[str, np.ndarray],
    track: HandTrack | None,
    side: str,
    dz: float,
    source_times: np.ndarray,
    target_times: np.ndarray,
    num_out: int,
) -> None:
    """Put one hand's wrist, fingertip and distal targets into ``out`` (identity poses when the hand is absent)."""
    if track is None:
        out[f"qpos_wrist_{side}"] = np.zeros((num_out, 7))
        out[f"qpos_wrist_{side}"][:, 3] = 1.0
        out[f"qpos_finger_{side}"] = np.zeros((num_out, 5, 7))
        out[f"qpos_finger_{side}"][:, :, 3] = 1.0
        out[f"qpos_finger_distal_{side}"] = np.zeros((num_out, 5, 7))
        out[f"qpos_finger_distal_{side}"][:, :, 3] = 1.0
    else:
        wrist = track.wrist.copy()
        wrist[:, 2] -= dz
        out[f"qpos_wrist_{side}"] = resample_pose_sequence(wrist, source_times, target_times)
        tips = track.tips.copy()
        tips[..., 2] -= dz
        tips = resample_point_sequence(tips, source_times, target_times)
        finger = np.zeros((num_out, 5, 7))
        finger[:, :, :3] = tips
        finger[:, :, 3:] = IDENTITY_WXYZ
        out[f"qpos_finger_{side}"] = finger

        # Distal joints get the same grounding shift and the same time
        # grid as the tips. A new array that skips either one is the
        # classic way to break this file: an ungrounded copy lands ~0.5 m
        # off, and an unresampled one silently indexes the source frame
        # rate against a 50 Hz trajectory.
        distal = np.zeros((num_out, 5, 7))
        distal[:, :, 3:] = IDENTITY_WXYZ
        if track.distal is not None:
            dpts = track.distal.copy()
            dpts[..., 2] -= dz
            distal[:, :, :3] = resample_point_sequence(dpts, source_times, target_times)
        else:
            # No distal joints in this source: fall back to the tips, which
            # makes every distal target a zero-length constraint that IK
            # satisfies trivially, i.e. a no-op rather than a wrong pull.
            distal[:, :, :3] = tips
        out[f"qpos_finger_distal_{side}"] = distal


def _object_targets(
    out: dict[str, np.ndarray],
    task_info: dict,
    seq: SequenceData,
    side: str,
    dz: float,
    source_times: np.ndarray,
    target_times: np.ndarray,
    num_out: int,
    dataset_dir: Path,
    dataset_name: str,
    mesh_scales: dict[str, float],
) -> None:
    """Put the pose track of one hand's object into ``out`` and stage its mesh (``task_info`` gets the mesh dirs)."""
    obj = getattr(seq, f"obj_{side}")
    if obj is None:
        out[f"qpos_obj_{side}"] = np.zeros((num_out, 7))
        out[f"qpos_obj_{side}"][:, 3] = 1.0
        task_info[f"{side}_object_mesh_dir"] = None
        task_info[f"{side}_object_convex_dir"] = None
    else:
        obj = obj.copy()
        obj[:, 2] -= dz
        out[f"qpos_obj_{side}"] = resample_pose_sequence(obj, source_times, target_times)
        rel = _stage_mesh(
            getattr(seq, f"mesh_{side}"),
            getattr(seq, f"mesh_name_{side}"),
            dataset_dir,
            dataset_name,
            mesh_scales[side],
        )
        task_info[f"{side}_object_mesh_dir"] = rel
        task_info[f"{side}_object_convex_dir"] = rel


def _extra_object_targets(
    out: dict[str, np.ndarray],
    task_info: dict,
    seq: SequenceData,
    dz: float,
    source_times: np.ndarray,
    target_times: np.ndarray,
    dataset_dir: Path,
    dataset_name: str,
) -> None:
    """Put the pose track of every extra scene object into ``out`` and stage its mesh (listed in ``task_info``)."""
    extras = list(seq.extra_objects)
    task_info["extra_objects"] = []
    out["n_extra_objects"] = np.asarray(len(extras))
    for idx, (obj_name, mesh_path, obj_pose) in enumerate(extras):
        scale = mesh_metre_scale(mesh_path)
        rel = _stage_mesh(mesh_path, obj_name, dataset_dir, dataset_name, scale)
        pose = obj_pose.copy()
        pose[:, 2] -= dz
        out[f"qpos_extra_obj_{idx}"] = resample_pose_sequence(pose, source_times, target_times)
        task_info["extra_objects"].append({"obj_id": obj_name, "mesh_dir": rel, "convex_dir": rel})


def _reader_records(
    out: dict[str, np.ndarray],
    task_info: dict,
    seq: SequenceData,
    source_times: np.ndarray,
    target_times: np.ndarray,
) -> None:
    """Carry what the reader measured (interpolated frames, GT gaps, source frame, MANO model) when it is set."""
    if seq.interp_mask is not None:
        # Nearest-in-time, then OR: an output frame is flagged if it draws on any
        # interpolated source frame. Conservative on purpose -- under-reporting
        # reconstructed frames is the failure that matters.
        mask = np.asarray(seq.interp_mask, dtype=float)
        resampled = np.interp(target_times, source_times, mask)
        out["interp_mask"] = resampled > 1e-9
    if seq.object_gt_gap is not None:
        task_info["object_gt_gap"] = seq.object_gt_gap
    if seq.source_frame is not None:
        task_info["source_frame"] = seq.source_frame
    if seq.mano_model is not None:
        task_info["mano_model"] = seq.mano_model


def write_sequence(
    seq: SequenceData,
    dataset_dir: str | Path,
    dataset_name: str,
    embodiment_type: str,
    robot_type: str = "mano",
    data_id: int = 0,
    target_ref_dt: float = 0.02,
    ground: bool = True,
    ground_mode: str = "object_first_frame",
) -> Path:
    """Resample, ground, and write the keypoint trajectory for one sequence.

    Returns the directory holding ``trajectory_keypoints.npz``.
    """
    if embodiment_type not in {"right", "left", "bimanual"}:
        raise ValueError(f"embodiment_type must be right/left/bimanual, got {embodiment_type!r}")
    dataset_dir = Path(os.path.abspath(os.path.expanduser(str(dataset_dir))))
    _require_hands(seq, embodiment_type)

    mesh_scales = {
        side: (mesh_metre_scale(getattr(seq, f"mesh_{side}")) if getattr(seq, f"mesh_{side}") else 1.0)
        for side in ("right", "left")
    }
    dz = ground_offset(seq, mesh_scales, mode=ground_mode) if ground else 0.0

    source_ref_dt = 1.0 / float(seq.source_fps)
    source_times, target_times = resample_times(seq.num_frames, source_ref_dt, target_ref_dt)
    num_out = target_times.shape[0]

    out: dict[str, np.ndarray] = {}
    task_info: dict = {
        "task": seq.task,
        "dataset_name": dataset_name,
        "robot_type": robot_type,
        "embodiment_type": embodiment_type,
        "data_id": data_id,
        "ref_dt": target_ref_dt,
        "source_ref_dt": source_ref_dt,
    }

    for side in ("right", "left"):
        track = getattr(seq, side)
        _hand_targets(out, track, side, dz, source_times, target_times, num_out)
        _object_targets(
            out, task_info, seq, side, dz, source_times, target_times, num_out, dataset_dir, dataset_name, mesh_scales
        )

    out_dir = dataset_dir / "processed" / dataset_name / robot_type / embodiment_type / seq.task / str(data_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    _extra_object_targets(out, task_info, seq, dz, source_times, target_times, dataset_dir, dataset_name)
    _reader_records(out, task_info, seq, source_times, target_times)
    # The batch runner validates a Phase 1 archive by checking that the
    # timestamps sit on an exact fixed-rate grid, so an archive without them is
    # treated as invalid and reprocessed on every --skip-existing run.
    out["timestamps"] = target_times
    np.savez(out_dir / "trajectory_keypoints.npz", **out)
    with open(out_dir.parent / "task_info.json", "w") as handle:
        json.dump(task_info, handle, indent=2)
    return out_dir
