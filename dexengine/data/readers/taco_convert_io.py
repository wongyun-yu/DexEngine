"""Raw TACO file readers of the converter (hand frames, MANO shape, side assignment, object poses), the check of an
existing converted file, and the per-sequence mesh preparation."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch


def _sorted_frame_keys(frame_data: dict, label: str) -> list:
    """Return deterministic frame keys and reject empty/non-dict payloads."""
    if not isinstance(frame_data, dict):
        raise TypeError(f"{label} hand data must be a dict, got {type(frame_data).__name__}")
    if not frame_data:
        raise ValueError(f"{label} hand data contains no frames")
    try:
        return sorted(frame_data)
    except TypeError as exc:
        raise ValueError(f"{label} frame keys must have mutually comparable types") from exc


def _validate_matching_frame_keys(rh_data: dict, lh_data: dict) -> list:
    """Require the right and left hand dictionaries to describe identical frames."""
    rh_frames = _sorted_frame_keys(rh_data, "Right")
    lh_frames = _sorted_frame_keys(lh_data, "Left")
    if rh_frames != lh_frames:
        rh_only = sorted(set(rh_frames) - set(lh_frames), key=repr)
        lh_only = sorted(set(lh_frames) - set(rh_frames), key=repr)
        raise ValueError(f"Right/left hand frame keys differ: right-only={rh_only[:5]}, left-only={lh_only[:5]}")
    return rh_frames


def _stack_hand_frames(
    frame_data: dict,
    frames: list | None = None,
    label: str = "Hand",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack and validate per-frame MANO parameters.

    Returns:
        hand_pose: (T, 48)
        hand_trans: (T, 3)
    """
    if frames is None:
        frames = _sorted_frame_keys(frame_data, label)
    hand_poses = []
    hand_trans = []
    for frame_id in frames:
        frame = frame_data[frame_id]
        if not isinstance(frame, dict):
            raise TypeError(f"{label} frame {frame_id!r} must be a dict")
        if "hand_pose" not in frame or "hand_trans" not in frame:
            raise KeyError(f"{label} frame {frame_id!r} must contain hand_pose and hand_trans")
        pose = torch.as_tensor(frame["hand_pose"])
        trans = torch.as_tensor(frame["hand_trans"])
        if tuple(pose.shape) != (48,):
            raise ValueError(f"{label} frame {frame_id!r} hand_pose must be (48,), got {tuple(pose.shape)}")
        if tuple(trans.shape) != (3,):
            raise ValueError(f"{label} frame {frame_id!r} hand_trans must be (3,), got {tuple(trans.shape)}")
        if not torch.isfinite(pose).all() or not torch.isfinite(trans).all():
            raise ValueError(f"{label} frame {frame_id!r} contains non-finite values")
        hand_poses.append(pose)
        hand_trans.append(trans)
    return torch.stack(hand_poses), torch.stack(hand_trans)


def _load_pickle(path: Path):
    """Load a pickle while ensuring the file descriptor is always closed."""
    with path.open("rb") as file:
        return pickle.load(file)


def _load_hand_shape(path: Path, label: str) -> np.ndarray:
    """Load one finite MANO shape vector from a TACO shape pickle."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} hand shape not found: {path}")
    payload = _load_pickle(path)
    if not isinstance(payload, dict) or "hand_shape" not in payload:
        raise ValueError(f"{label} hand shape file has no 'hand_shape' field: {path}")
    shape = torch.as_tensor(payload["hand_shape"]).detach().cpu().numpy()
    if shape.size != 10:
        raise ValueError(f"{label} hand shape must contain 10 MANO coefficients, got shape {shape.shape}: {path}")
    if not np.isfinite(shape).all():
        raise ValueError(f"{label} hand shape contains non-finite values: {path}")
    return shape


class ExcludedSequence(RuntimeError):
    """The measured sidecar marks the sequence excluded (the source-defect ledger owns the reason) — not converted."""


def _read_side_assign(obj_dir: Path, tool_obj_id: str, target_obj_id: str, sequence_id: str) -> dict:
    """Read the hand↔object assignment from the sidecar. Fails without it; never filled in by assumption.

    sidecar `side_assign.json` (producer dexengine/data/readers/taco_side_assign.py, measured from source MANO contact
    and sustained grasps):
        {"right": "<id>", "left": "<id>", "extra": [...], "case": "...", "method": "..."}
      hands on different objects   right ≠ left · no extra
      both hands on one object (D6-a sharing)  right == left ∈ {tool, target} · extra = [the other object] (stays in the
          scene, not duplicated)
      excluded   {"excluded": "<reason>"}, owned by the source defect ledger (configs/datasets/taco/excluded_clips.txt)
          → ExcludedSequence

    "The tool is in the right hand" is not an invariant (left-hand tool clips were measured). The assignment flows into
    the per-hand contact sensors (the contact detector's right fingers ↔ right_object) and IK contact weights, so
    filling it
    in by assumption flips the contact labels of left-hand tool clips entirely; without a measured assignment the
    sequence cannot be converted.
    """
    p = obj_dir / "side_assign.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"{sequence_id}: {p} missing — the hand↔object assignment comes only from measurement (side_assign.json). "
            "The 'tool = right hand' assumption is wrong for clips where the left hand holds the tool."
        )
    d = json.loads(p.read_text())
    if d.get("excluded"):
        raise ExcludedSequence(f"{sequence_id}: excluded — {d['excluded']}")
    ids = {tool_obj_id, target_obj_id}
    r, l_, extra = d.get("right"), d.get("left"), list(d.get("extra") or [])
    shared_ok = r == l_ and r in ids and extra == sorted(ids - {r})
    split_ok = r != l_ and {r, l_} == ids and not extra
    if not (shared_ok or split_ok):
        raise ValueError(
            f"{sequence_id}: assignment {r}/{l_} · extra {extra} in side_assign.json does not match objects "
            f"{sorted(ids)}"
        )
    print(
        f"[SIDE_ASSIGN] {sequence_id}: R={r} L={l_}{' shared · extra ' + ','.join(extra) if shared_ok else ''} "
        f"({d.get('method')})",
        flush=True,
    )
    return d


def _find_single_object_pose(obj_dir: Path, role: str) -> tuple[str, Path]:
    """Find the single tool/target pose file required by the TACO schema."""
    files = sorted(
        path for path in obj_dir.glob(f"{role}_*.npy") if path.is_file() and not path.name.endswith(".grid50.npy")
    )
    if len(files) != 1:
        names = [path.name for path in files]
        raise ValueError(f"Expected exactly one {role}_*.npy in {obj_dir}, found {len(files)}: {names}")
    path = files[0]
    obj_id = path.stem.removeprefix(f"{role}_")
    if not obj_id:
        raise ValueError(f"Could not extract object id from {path.name}")
    return obj_id, path


def _load_object_trajectory(
    path: Path,
    expected_frames: int,
    translation_scale: float,
) -> np.ndarray:
    """Load and validate one homogeneous object-pose trajectory."""
    with path.open("rb") as file:
        trajectory = np.load(file, allow_pickle=False)
    trajectory = np.asarray(trajectory, dtype=np.float32)
    expected_shape = (expected_frames, 4, 4)
    if trajectory.shape != expected_shape:
        raise ValueError(f"Object poses must have shape {expected_shape}, got {trajectory.shape}: {path}")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"Object poses contain non-finite values: {path}")
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if not np.allclose(trajectory[:, 3, :], expected_last_row, atol=1e-5):
        raise ValueError(f"Object poses must be homogeneous transforms with last row [0, 0, 0, 1]: {path}")
    rotations = trajectory[:, :3, :3].astype(np.float64)
    gram = np.swapaxes(rotations, 1, 2) @ rotations
    identity = np.broadcast_to(np.eye(3), gram.shape)
    determinants = np.linalg.det(rotations)
    if not np.allclose(gram, identity, atol=1e-3) or not np.allclose(determinants, 1.0, atol=1e-3):
        raise ValueError(f"Object pose rotations must be proper rotation matrices: {path}")
    trajectory = trajectory.copy()
    trajectory[:, :3, 3] *= translation_scale
    return trajectory


def _matches_side_assign(path: Path, side_assign_file: Path) -> bool:
    """The artifact's per-hand and extra objects equal the measured assignment; a missing or unreadable sidecar, or
    an excluded sequence, does not match."""
    if not Path(side_assign_file).is_file():
        return False
    try:
        want = json.loads(Path(side_assign_file).read_text())
        got = _load_pickle(path)
        if want.get("excluded") or any(
            str((got.get(s) or {}).get("obj_id", [None])[0]) != str(want.get(s)) for s in ("right", "left")
        ):
            return False
        if sorted(str(e["obj_id"][0]) for e in got.get("extra_objects") or []) != sorted(
            str(e) for e in want.get("extra") or []
        ):
            return False
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, pickle.UnpicklingError):
        return False
    return True


def _options_match(metadata: dict, object_poses_unit: str | None, hand_trans_offset: np.ndarray | None) -> bool:
    """The conversion options stored in the artifact equal the requested ones (an option left None is not checked,
    except the hand offset: None means the artifact must have none)."""
    if object_poses_unit is not None and metadata.get("object_poses_unit") != object_poses_unit:
        return False
    expected_offset = np.asarray(hand_trans_offset, dtype=np.float32) if hand_trans_offset is not None else None
    stored_offset = metadata.get("hand_trans_offset")
    if expected_offset is None:
        if stored_offset is not None:
            return False
    elif stored_offset is None or not np.allclose(stored_offset, expected_offset):
        return False
    return True


def _hand_is_valid(path: Path, payload: dict, side: str) -> bool:
    """One hand's entry: (T, 48) pose, (T, 3) translation and (T, 4, 4) object trajectory, all finite, and a mesh
    reference that exists next to the artifact."""
    hand = payload.get(side)
    if not isinstance(hand, dict):
        return False
    pose = np.asarray(torch.as_tensor(hand["hand_pose"]).cpu())
    trans = np.asarray(torch.as_tensor(hand["hand_trans"]).cpu())
    trajectory = np.asarray(torch.as_tensor(hand["obj_trajectory"]).cpu())
    if pose.ndim != 2 or pose.shape[1] != 48 or pose.shape[0] == 0:
        return False
    frame_count = pose.shape[0]
    if trans.shape != (frame_count, 3) or trajectory.shape != (
        frame_count,
        4,
        4,
    ):
        return False
    if not all(np.isfinite(array).all() for array in (pose, trans, trajectory)):
        return False
    mesh_paths = hand.get("obj_mesh_path")
    if not isinstance(mesh_paths, (list, tuple)) or not mesh_paths:
        return False
    mesh_ref = mesh_paths[0]
    if not isinstance(mesh_ref, str) or not mesh_ref:
        return False
    return (path.parent / mesh_ref).is_file()


def converted_output_is_valid(
    path: Path,
    object_poses_unit: str | None = None,
    hand_trans_offset: np.ndarray | None = None,
    side_assign_file: Path | None = None,
) -> bool:
    """Validate an existing Phase-0 artifact and available option metadata. With `side_assign_file` (the sequence's
    measured hand↔object assignment) the artifact is valid only when its per-hand objects equal it — a pkl converted
    under another assignment (or before one existed) is regenerated, and a missing sidecar makes it invalid."""
    if not path.is_file():
        return False
    if side_assign_file is not None and not _matches_side_assign(path, side_assign_file):
        return False
    try:
        payload = _load_pickle(path)
        if not isinstance(payload, dict) or payload.get("env_type") != "bimanual":
            return False
        metadata = payload.get("conversion_metadata")
        if isinstance(metadata, dict) and not _options_match(metadata, object_poses_unit, hand_trans_offset):
            return False
        return all(_hand_is_valid(path, payload, side) for side in ("right", "left"))
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
        pickle.UnpicklingError,
    ):
        return False


def prepare_mesh_for_taco(raw_dir: str, output_dir: str, obj_id: str) -> str:
    """Copy an object mesh into output_dir/meshes/{obj_id}/.

    Returns:
        Converter-format relative path: "meshes/{obj_id}/{obj_id}_cm.obj"
    """
    if not obj_id:
        raise ValueError("Object id must not be empty")

    src_mesh = Path(raw_dir) / "object_models_released" / f"{obj_id}_cm.obj"
    if not src_mesh.is_file():
        raise FileNotFoundError(f"Object mesh not found: {src_mesh}")

    filename = src_mesh.name
    dst_dir = Path(output_dir) / "meshes" / obj_id
    dst_path = dst_dir / filename

    if not dst_path.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_mesh, dst_path)

    return f"meshes/{obj_id}/{filename}"
