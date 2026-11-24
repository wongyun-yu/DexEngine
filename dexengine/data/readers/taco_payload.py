"""Validation of a converted TACO pickle's payloads (used by the TACO keypoint reader `taco.py`).

Each active hand's payload must hold `hand_pose (T, 48)`, `hand_trans (T, 3)`, `hand_shape` (at least ten values),
`obj_mesh_path` (a non-empty list whose first item is a non-empty path string) and `obj_trajectory (T, 4, 4)`, all
finite with T > 0; `obj_trajectory_grid50 (T50, 7)` is optional. An extra object holds `obj_mesh_path` and
`obj_trajectory (T, 4, 4)` with the hands' T. Torch tensors are converted to NumPy.

Errors: a missing field raises `KeyError`, a wrong type `TypeError`, a wrong shape, an empty value or a non-finite
number `ValueError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_TAG = "[TACO_READER]"


@dataclass(frozen=True)
class ObjectInput:
    """One object's mesh reference and its (T, 4, 4) trajectory."""

    mesh_ref: str
    obj_trajectory: np.ndarray


@dataclass(frozen=True)
class HandInput:
    """One hand's validated payload."""

    pose: np.ndarray  # (T, 48) root + 15 joint axis-angles
    trans: np.ndarray  # (T, 3) wrist positions
    shape: np.ndarray  # (>= 10,) betas
    mesh_ref: str  # first entry of obj_mesh_path
    obj_trajectory: np.ndarray  # (T, 4, 4)
    grid50: np.ndarray | None  # (T50, 7) object poses already on the 50 Hz grid, or None


def as_array(value: Any, name: str) -> np.ndarray:
    """A finite numeric NumPy array from an array-like or a (CPU) torch tensor."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{_TAG} {name}: not an array ({exc})") from exc
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{_TAG} {name}: expected numbers, got dtype {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{_TAG} {name}: holds non-finite values")
    return arr


def _shaped(value: Any, name: str, trailing: tuple[int, ...], num_frames: int | None) -> np.ndarray:
    arr = as_array(value, name)
    if arr.ndim != 1 + len(trailing) or arr.shape[1:] != trailing:
        raise ValueError(f"{_TAG} {name}: expected shape (T, {', '.join(map(str, trailing))}), got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{_TAG} {name}: no frames")
    if num_frames is not None and arr.shape[0] != num_frames:
        raise ValueError(f"{_TAG} {name}: {arr.shape[0]} frames, expected {num_frames}")
    return arr


def mesh_ref(value: Any, name: str) -> str:
    """The first path of an `obj_mesh_path` list."""
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{_TAG} {name}: obj_mesh_path must be a list, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{_TAG} {name}: obj_mesh_path is empty")
    first = value[0]
    if not isinstance(first, str):
        raise TypeError(f"{_TAG} {name}: obj_mesh_path[0] must be a path string, got {type(first).__name__}")
    if not first:
        raise ValueError(f"{_TAG} {name}: obj_mesh_path[0] is empty")
    return first


def _field(payload: dict, key: str, name: str) -> Any:
    if key not in payload:
        raise KeyError(f"{_TAG} {name}: missing field {key!r}")
    return payload[key]


def hand_input(payload: Any, name: str) -> HandInput:
    """Validate one hand's payload (`name` labels the errors)."""
    if not isinstance(payload, dict):
        raise TypeError(f"{_TAG} {name}: hand payload must be a dict, got {type(payload).__name__}")
    pose = _shaped(_field(payload, "hand_pose", name), f"{name}.hand_pose", (48,), None)
    num_frames = pose.shape[0]
    trans = _shaped(_field(payload, "hand_trans", name), f"{name}.hand_trans", (3,), num_frames)
    shape = as_array(_field(payload, "hand_shape", name), f"{name}.hand_shape").reshape(-1)
    if shape.size < 10:
        raise ValueError(f"{_TAG} {name}.hand_shape: expected at least 10 values, got {shape.size}")
    ref = mesh_ref(_field(payload, "obj_mesh_path", name), name)
    traj = _shaped(_field(payload, "obj_trajectory", name), f"{name}.obj_trajectory", (4, 4), num_frames)
    grid50 = None
    if payload.get("obj_trajectory_grid50") is not None:
        grid50 = _shaped(payload["obj_trajectory_grid50"], f"{name}.obj_trajectory_grid50", (7,), None)
    return HandInput(pose=pose, trans=trans, shape=shape, mesh_ref=ref, obj_trajectory=traj, grid50=grid50)


def extra_inputs(value: Any, num_frames: int, name: str) -> list[ObjectInput]:
    """Validate the optional `extra_objects` list (absent or None = no extra object)."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{_TAG} {name}: extra_objects must be a list, got {type(value).__name__}")
    out = []
    for k, entry in enumerate(value):
        label = f"{name}:extra_objects[{k}]"
        if not isinstance(entry, dict):
            raise TypeError(f"{_TAG} {label}: must be a dict, got {type(entry).__name__}")
        ref = mesh_ref(_field(entry, "obj_mesh_path", label), label)
        traj = _shaped(_field(entry, "obj_trajectory", label), f"{label}.obj_trajectory", (4, 4), num_frames)
        out.append(ObjectInput(mesh_ref=ref, obj_trajectory=traj))
    return out
