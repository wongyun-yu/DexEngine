"""Clip discovery, per-clip output paths and output validators of the stage-1 retargeting batch driver."""

from __future__ import annotations

import json
import re
import shutil
import threading
from pathlib import Path

import numpy as np

from dexengine.core.paths import hand_rig_dir
from dexengine.retarget.phases.common import RUN, log_info

_ASSET_COPY_IGNORE = shutil.ignore_patterns(".omc", "__pycache__")
_ENSURED_ROBOT_ASSETS: set = set()
_ROBOT_ASSET_LOCK = threading.Lock()


def normalize_phase_name(value: str) -> str:
    """Accept legacy underscore names while presenting hyphenated CLI values."""
    return value.replace("_", "-")


def canonical_filter_text(value: str) -> str:
    """Normalize raw triplets and generated task names for consistent filtering."""
    return re.sub(r"[^0-9a-z]+", "_", value.casefold()).strip("_")


def task_matches_filter(task_name: str, filter_text: str | None) -> bool:
    """Return whether a task matches a raw or normalized triplet filter."""
    if not filter_text:
        return True
    return canonical_filter_text(filter_text) in canonical_filter_text(task_name)


def mano_data_dir(dataset_dir: Path, embodiment_type: str, task: str) -> Path:
    """Return the per-clip MANO output directory used by Phases 1 and 2."""
    return dataset_dir / "processed" / RUN.dataset_name / "mano" / embodiment_type / task / "0"


def robot_task_dir(
    dataset_dir: Path,
    robot_type: str,
    embodiment_type: str,
    task: str,
) -> Path:
    """Return the task-level robot directory containing generated scene XML."""
    return dataset_dir / "processed" / RUN.dataset_name / robot_type / embodiment_type / task


def robot_data_dir(
    dataset_dir: Path,
    robot_type: str,
    embodiment_type: str,
    task: str,
) -> Path:
    """Return the per-clip robot directory containing IK trajectories."""
    return robot_task_dir(dataset_dir, robot_type, embodiment_type, task) / "0"


def xml_outputs_complete(task_dir: Path) -> bool:
    """Return whether both runtime scenes and their task metadata exist."""
    return all((task_dir / filename).is_file() for filename in ("scene.xml", "scene_eq.xml", "task_info.json"))


def has_contact_result(path: Path) -> bool:
    """Return whether contact arrays are present, shaped, and finite."""
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as data:
        required = {
            "qpos_wrist_right",
            "contact_right",
            "contact_left",
            "contact_pos_right",
            "contact_pos_left",
        }
        if not required.issubset(data.files):
            return False
        frame_count = data["qpos_wrist_right"].shape[0]
        expected_shapes = {
            "contact_right": (frame_count, 5),
            "contact_left": (frame_count, 5),
            "contact_pos_right": (5, 3),
            "contact_pos_left": (5, 3),
        }
        return all(data[key].shape == shape and np.isfinite(data[key]).all() for key, shape in expected_shapes.items())


def kinematic_result_valid(
    path: Path,
    source_ref_dt: float,
    target_ref_dt: float,
) -> bool:
    """Validate a Phase-1 archive and the frame-rate metadata beside it."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            expected_shapes = {
                "qpos_wrist_right": (7,),
                "qpos_finger_right": (5, 7),
                "qpos_obj_right": (7,),
                "qpos_wrist_left": (7,),
                "qpos_finger_left": (5, 7),
                "qpos_obj_left": (7,),
            }
            if not set(expected_shapes).issubset(data.files):
                return False
            frame_count = data["qpos_wrist_right"].shape[0]
            if frame_count == 0:
                return False
            for key, trailing_shape in expected_shapes.items():
                array = data[key]
                if array.shape != (frame_count, *trailing_shape):
                    return False
                if not np.isfinite(array).all():
                    return False
            if "timestamps" not in data.files:
                return False
            timestamps = data["timestamps"]
            if timestamps.shape != (frame_count,) or not np.isfinite(timestamps).all():
                return False
            if frame_count > 1 and not np.allclose(np.diff(timestamps), target_ref_dt, rtol=0.0, atol=1e-10):
                return False

        task_info_path = path.parent.parent / "task_info.json"
        with task_info_path.open(encoding="utf-8") as file:
            task_info = json.load(file)
        return np.isclose(task_info.get("ref_dt"), target_ref_dt) and np.isclose(
            task_info.get("source_ref_dt"), source_ref_dt
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def ik_result_valid(path: Path, ref_dt: float) -> bool:
    """Validate canonical IK arrays and their shared time axis."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"qpos", "qvel", "contact", "contact_pos", "frequency"}
            if not required.issubset(data.files):
                return False
            frame_count = data["qpos"].shape[0]
            if frame_count == 0:
                return False
            if any(data[key].shape[0] != frame_count for key in ("qvel", "contact", "contact_pos")):
                return False
            if any(not np.isfinite(data[key]).all() for key in required):
                return False
            frequency = float(np.asarray(data["frequency"]).reshape(()))
            return np.isclose(frequency, 1.0 / ref_dt)
    except (OSError, ValueError, TypeError):
        return False


def convex_parts_exist(dataset_dir: Path, task_info_path: Path, embodiment_type: str) -> bool:
    """Return whether every active hand already has convex collision parts."""
    if not task_info_path.is_file():
        return False
    try:
        with task_info_path.open(encoding="utf-8") as file:
            task_info = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False

    hands = ["right", "left"] if embodiment_type == "bimanual" else [embodiment_type]
    checked_any = False
    for hand in hands:
        if not task_info.get(f"{hand}_object_mesh_dir"):
            continue
        convex_dir = task_info.get(f"{hand}_object_convex_dir")
        if not convex_dir:
            return False
        convex_path = dataset_dir / convex_dir
        # The mesh dir itself is the pre-decomposition default; only a real
        # convex/ directory holding parts counts as done.
        if convex_path.name != "convex" or not any(convex_path.glob("*.obj")):
            return False
        checked_any = True
    return checked_any


def save_taco_text_annotation(
    dataset_dir: Path,
    robot_type: str,
    embodiment_type: str,
    triplet_id: str,
    clip_id: str,
) -> Path:
    """Write the task-description sidecar consumed by downstream code."""
    task = f"{triplet_id}__{clip_id}" if clip_id is not None else triplet_id

    dst_data_dir = robot_data_dir(dataset_dir, robot_type, embodiment_type, task)
    dst_data_dir.mkdir(parents=True, exist_ok=True)
    text_path = dst_data_dir / "text_annotations.json"

    parts = triplet_id.split("_")

    # If the triplet is in verb_tool_object format, generate a natural sentence.
    if len(parts) == 3:
        verb, tool, obj = parts
        task_desc = f"{verb.capitalize()} the {obj} using the {tool}."
    else:
        task_desc = f"Task: {triplet_id}"

    annotation_data = {
        "task": task,
        "annotations": {"task_description": task_desc},
    }

    with text_path.open("w", encoding="utf-8") as file:
        json.dump(annotation_data, file, indent=2, ensure_ascii=False)

    return text_path


def ensure_robot_assets(dataset_dir: Path, robot_type: str) -> None:
    """Mirror the packaged robot assets (MJCF, meshes) into the dataset tree."""
    src = hand_rig_dir(robot_type)
    dst = dataset_dir / "processed" / RUN.dataset_name / "assets" / "robots" / robot_type
    if not src.exists():
        raise FileNotFoundError(f"Robot assets not found: {src}")
    # With --workers several threads arrive at once: copy each (src, dst) pair once per process. The copy runs
    # even when dst exists, so an edited rig always reaches a dataset tree populated by an earlier run.
    # Plugin state and bytecode (_ASSET_COPY_IGNORE) are not assets and must not leak into the data tree.
    key = (src.resolve(), dst.resolve())
    with _ROBOT_ASSET_LOCK:
        if key in _ENSURED_ROBOT_ASSETS:
            return
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True, ignore=_ASSET_COPY_IGNORE)
        _ENSURED_ROBOT_ASSETS.add(key)


def find_processed_clips(
    dataset_dir: Path,
    embodiment_type: str,
    triplet_filter: str | None = None,
    max_clips: int | None = None,
) -> list[dict]:
    """Find clips generated after Phase 1 processing."""
    if max_clips is not None and max_clips <= 0:
        return []
    mano_root = dataset_dir / "processed" / RUN.dataset_name / "mano" / embodiment_type
    if RUN.clip_list is not None:
        present = {d.name for d in mano_root.iterdir() if d.is_dir()} if mano_root.exists() else set()
        absent = sorted(RUN.clip_list - present)
        if absent:
            raise SystemExit(
                f"[CLIP_LIST] ✗ clips in the list ({RUN.clip_list_path}) but not in the input ({mano_root}): "
                f"{len(absent)}: {absent[:5]}"
            )
        log_info(f"[CLIP_LIST] {len(RUN.clip_list)}/{len(present)} ({RUN.clip_list_path})")
    if not mano_root.exists():
        return []

    clips = []
    for task_dir in sorted(mano_root.iterdir()):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        if RUN.clip_list is not None and task_name not in RUN.clip_list:
            continue
        if not task_matches_filter(task_name, triplet_filter):
            continue

        # task_name format: "triplet__clip_id" (e.g., "bowl_soak_water__seq00"). Other datasets have task names
        # without "__" (hocap "subject_2-20231022_200657", arctic "s01-box_grab_01"); the whole name is the task.
        if "__" in task_name:
            triplet, clip_id = task_name.split("__", 1)
        else:
            triplet, clip_id = task_name, None
        clips.append(
            {
                "triplet": triplet,
                "sequence": clip_id,
                "task": task_name,
                "task_dir": task_dir / "0",
            }
        )

        if max_clips is not None and len(clips) >= max_clips:
            break

    return clips
