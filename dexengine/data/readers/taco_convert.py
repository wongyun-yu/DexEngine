#!/usr/bin/env python3
"""Convert raw TACO sequences into the ManipTrans-compatible PKL format.

This is phase 0 (convert) of ``dexengine.retarget.batch``.  It combines TACO's separate
left-hand, right-hand, tool, and target files into one bimanual record per
sequence.  The mapping used throughout the TACO pipeline is:

* right hand -> tool object
* left hand  -> target object

The converter intentionally performs strict input validation.  Both hands
must contain exactly the same frame keys, and each sequence must contain
exactly one ``tool_*.npy`` and one ``target_*.npy`` trajectory of shape
``(T, 4, 4)``.  Invalid sequences are reported and make the command exit with
a non-zero status; missing object motion is never replaced with identity poses.

TACO is already expressed in a Z-up world frame.  Coordinate conversion and
ground alignment happen after MANO forward kinematics in
the TACO reader (``dexengine/data/readers/taco.py``), not in this script.

Ref: https://github.com/leolyliu/TACO-Instructions
    (see data_visualization/visualize_data.py)

Input structure:
    datasets/raw/taco/
    ├── Hand_Poses/
    │   └── (triplet)/
    │       └── {sequence_id}/
    │           ├── left_hand.pkl
    │           ├── right_hand.pkl
    │           ├── left_hand_shape.pkl
    │           └── right_hand_shape.pkl
    ├── Object_Poses/
    │   └── (triplet)/
    │       └── {sequence_id}/
    │           ├── tool_{obj_id}.npy
    │           └── target_{obj_id}.npy
    └── object_models_released/
        └── {obj_id}_cm.obj

Output format:
    {task_name}_bimanual.pkl (ManipTrans-compatible)

Usage:
    python -m dexengine.data.readers.taco_convert \
        --raw-dir datasets/raw/taco \
        --output-dir datasets/raw/taco_maniptrans \
        --object-poses-unit m \
        --max-sequences 5 \
        --skip-existing
"""

import argparse
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from dexengine.data.readers.taco_convert_io import (  # noqa: E402,F401  (re-exported)
    ExcludedSequence,
    _find_single_object_pose,
    _load_hand_shape,
    _load_object_trajectory,
    _load_pickle,
    _read_side_assign,
    _sorted_frame_keys,
    _stack_hand_frames,
    _validate_matching_frame_keys,
    converted_output_is_valid,
    prepare_mesh_for_taco,
)


def _naming_rule():
    """The clip naming rule's owner, `dexengine.core.paths` (imported lazily)."""
    from dexengine.core import paths  # noqa: PLC0415

    return paths


def taco_task_name(triplet: str, sequence_id: str) -> str:
    """The clip id every later phase keys on, built by the naming contract's one rule (`paths.clip_id_from_raw`).

    A field with a space comes out hyphenated — `(pour in some, bowl, bowl)` → `pour-in-some_bowl_bowl__…` — so
    what the converter writes is already the name the trees, ledgers and checkpoints use.
    """
    return _naming_rule().clip_id_from_raw(triplet, sequence_id, "taco")


def _canonical_filter_text(value: str) -> str:
    """Normalize raw ``(verb, tool, target)`` and generated task spellings."""
    return re.sub(r"[^0-9a-z]+", "_", value.casefold()).strip("_")


def triplet_matches_filter(triplet: str, filter_text: str | None) -> bool:
    """Match the same filter against raw and normalized TACO task names."""
    if not filter_text:
        return True
    return _canonical_filter_text(filter_text) in _canonical_filter_text(triplet)


def converted_output_path(output_dir: str | Path, triplet: str, sequence_id: str) -> Path:
    """Return the Phase-0 output path without touching the filesystem."""
    return Path(output_dir) / f"{taco_task_name(triplet, sequence_id)}_bimanual.pkl"


def _load_hands(hand_dir: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(T, right pose, right trans, left pose, left trans) of one sequence; both hands must share their frame keys."""
    rh_pkl = hand_dir / "right_hand.pkl"
    lh_pkl = hand_dir / "left_hand.pkl"

    if not rh_pkl.is_file() or not lh_pkl.is_file():
        missing = [str(path) for path in (rh_pkl, lh_pkl) if not path.is_file()]
        raise FileNotFoundError(f"Missing hand pose file(s): {missing}")
    rh_data = _load_pickle(rh_pkl)
    lh_data = _load_pickle(lh_pkl)

    # Matching counts alone are insufficient: the actual frame identifiers must match.
    frames = _validate_matching_frame_keys(rh_data, lh_data)
    rh_pose_t, rh_trans_t = _stack_hand_frames(rh_data, frames, "Right")
    lh_pose_t, lh_trans_t = _stack_hand_frames(lh_data, frames, "Left")
    T = len(frames)

    rh_pose = rh_pose_t.detach().cpu().numpy()
    rh_trans = rh_trans_t.detach().cpu().numpy()
    lh_pose = lh_pose_t.detach().cpu().numpy()
    lh_trans = lh_trans_t.detach().cpu().numpy()
    return T, rh_pose, rh_trans, lh_pose, lh_trans


def _load_objects(obj_dir: Path, object_poses_unit: str, T: int) -> tuple[str, np.ndarray, str, np.ndarray]:
    """(tool id, tool trajectory, target id, target trajectory) of one sequence, positions in metres."""
    if object_poses_unit not in ["cm", "m"]:
        raise ValueError(f"Invalid object_poses_unit: {object_poses_unit} (use 'cm' or 'm')")
    obj_pos_scale = 0.01 if object_poses_unit == "cm" else 1.0

    if not obj_dir.is_dir():
        raise FileNotFoundError(f"Object pose directory not found: {obj_dir}")
    target_obj_id, target_file = _find_single_object_pose(obj_dir, "target")
    tool_obj_id, tool_file = _find_single_object_pose(obj_dir, "tool")
    target_obj_traj = _load_object_trajectory(target_file, T, obj_pos_scale)
    tool_obj_traj = _load_object_trajectory(tool_file, T, obj_pos_scale)
    return tool_obj_id, tool_obj_traj, target_obj_id, target_obj_traj


def _load_grid50(obj_dir: Path, role: str, obj_id: str):
    """The grid-50 sidecar ``<role>_<id>.grid50.npy`` next to the pose npy (lifted-reference fidelity; see the TACO
    reader's grid-50 passthrough for why): (T50, 7) pos + quat wxyz, carried verbatim. Required: only stage 1's lift
    writes the raw layout this converter reads. Public TACO ``Object_Poses`` placed there by hand would convert into
    object tracks that are not the released ones (dexengine.data.readers.taco_source: same-time difference p99
    12.2 mm, max 41.9 deg), so a missing sidecar dies instead of converting them."""
    path = obj_dir / f"{role}_{obj_id}.grid50.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing — the TACO converter reads only the stage-1 lift layout (scripts/stage1_retarget.py, "
            "TACO source = shipped input taco_source); the public TACO Object_Poses do not give the released tracks"
        )
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"grid50 must be (T,7), got {arr.shape}: {path}")
    return torch.from_numpy(np.asarray(arr, dtype=np.float64))


def _offset_hands(
    sequence_id: str, assign: dict, hand_trans_offset: np.ndarray | None, rh_trans: np.ndarray, lh_trans: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """(offset as float32, right trans, left trans) with the optional global offset added to both hands.

    The shim writes the hand shift **already baked** (dexengine.data.readers.taco_lift). Applying it here again would
    apply it twice, so it is refused when the sidecar says it is baked.
    """
    if hand_trans_offset is not None and assign.get("hand_shift_baked"):
        raise ValueError(
            f"{sequence_id}: input raw is a shim with the hand shift already baked"
            f"(side_assign.json.hand_shift_z={assign.get('hand_shift_z')}). "
            "--hand-trans-offset as well would apply it twice; the shift lives in one place (the shim)."
        )
    if hand_trans_offset is not None:
        hand_trans_offset = np.asarray(hand_trans_offset, dtype=np.float32)
        if hand_trans_offset.shape != (3,) or not np.isfinite(hand_trans_offset).all():
            raise ValueError(f"hand_trans_offset must contain exactly three finite values, got {hand_trans_offset}")
        rh_trans = rh_trans + hand_trans_offset[None, :]
        lh_trans = lh_trans + hand_trans_offset[None, :]
    return hand_trans_offset, rh_trans, lh_trans


def _hand_record(
    triplet: str,
    sequence_id: str,
    obj_id: str,
    mesh_ref: str,
    obj_traj: np.ndarray,
    pose: np.ndarray,
    trans: np.ndarray,
    shape: np.ndarray,
    grid50,
) -> dict:
    """One hand's entry of the ManipTrans payload (the hand and the object assigned to it)."""
    return {
        "data_path": [f"Hand_Poses/{triplet}/{sequence_id}"],
        "obj_id": [obj_id],
        "obj_mesh_path": [mesh_ref],
        "obj_urdf_path": [""],
        "obj_trajectory": torch.from_numpy(obj_traj),
        "scene_objs": [[]],
        "hand_pose": torch.from_numpy(pose),
        "hand_trans": torch.from_numpy(trans),
        "hand_shape": torch.from_numpy(shape),
        **({"obj_trajectory_grid50": grid50} if grid50 is not None else {}),
    }


def _write_atomically(output_path: Path, output_data: dict) -> None:
    """Pickle to a hidden temporary sibling, fsync, then rename: a reader never sees a half-written file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("wb") as file:
            pickle.dump(output_data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_single_taco_sequence(
    raw_dir: str,
    output_dir: str,
    triplet: str,
    sequence_id: str,
    object_poses_unit: str = "m",
    hand_trans_offset: np.ndarray | None = None,
) -> str:
    """Convert one strictly validated TACO sequence into a ManipTrans PKL."""

    logger.info(f"Processing: {triplet}/{sequence_id}")

    hand_dir = Path(raw_dir) / "Hand_Poses" / triplet / sequence_id
    T, rh_pose, rh_trans, lh_pose, lh_trans = _load_hands(hand_dir)
    rh_shape_path = hand_dir / "right_hand_shape.pkl"
    lh_shape_path = hand_dir / "left_hand_shape.pkl"
    rh_shape = _load_hand_shape(rh_shape_path, "Right")
    lh_shape = _load_hand_shape(lh_shape_path, "Left")

    obj_dir = Path(raw_dir) / "Object_Poses" / triplet / sequence_id
    tool_obj_id, tool_obj_traj, target_obj_id, target_obj_traj = _load_objects(obj_dir, object_poses_unit, T)

    # Hand↔object assignment.
    # `RH=tool, LH=target` is **not hard-coded**: "the tool is in the right hand" is **not** a TACO invariant (the
    # corpus has 51 clips where the person holds the tool in the left hand, confirmed by grasp persistence). Hard-coding
    # it would flip the contact labels of those clips entirely.
    #
    # The **measured assignment** comes from the sidecar (`side_assign.json`, written by
    # dexengine.data.readers.taco_lift from `side_objects.resolve()`). Fails without it; never filled in by assumption.
    assign = _read_side_assign(obj_dir, tool_obj_id, target_obj_id, sequence_id)
    _by_id = {tool_obj_id: (tool_obj_id, tool_obj_traj), target_obj_id: (target_obj_id, target_obj_traj)}
    rh_obj_id, rh_obj_traj = _by_id[assign["right"]]
    lh_obj_id, lh_obj_traj = _by_id[assign["left"]]
    extra_ids = list(assign.get("extra") or [])
    _role_of = {tool_obj_id: "tool", target_obj_id: "target"}

    # roles are derived from the assignment; reusing the literals "tool"/"target" would make only grid50 point at the
    # opposite object in clips with a flipped assignment.
    rh_grid50 = _load_grid50(obj_dir, _role_of[rh_obj_id], rh_obj_id)
    lh_grid50 = _load_grid50(obj_dir, _role_of[lh_obj_id], lh_obj_id)
    hand_trans_offset, rh_trans, lh_trans = _offset_hands(sequence_id, assign, hand_trans_offset, rh_trans, lh_trans)

    rh_mesh_ref = prepare_mesh_for_taco(raw_dir, output_dir, rh_obj_id)
    lh_mesh_ref = prepare_mesh_for_taco(raw_dir, output_dir, lh_obj_id)
    extra_objects = [
        {
            "obj_id": [eid],
            "obj_mesh_path": [prepare_mesh_for_taco(raw_dir, output_dir, eid)],
            "obj_trajectory": torch.from_numpy(_by_id[eid][1]),
        }
        for eid in extra_ids
    ]

    task_name = taco_task_name(triplet, sequence_id)
    output_data = {
        "data_idx": task_name,
        "env_type": "bimanual",
        "conversion_metadata": {
            "object_poses_unit": object_poses_unit,
            "hand_trans_offset": (np.asarray(hand_trans_offset).tolist() if hand_trans_offset is not None else None),
        },
        "right": _hand_record(
            triplet, sequence_id, rh_obj_id, rh_mesh_ref, rh_obj_traj, rh_pose, rh_trans, rh_shape, rh_grid50
        ),
        "left": _hand_record(
            triplet, sequence_id, lh_obj_id, lh_mesh_ref, lh_obj_traj, lh_pose, lh_trans, lh_shape, lh_grid50
        ),
        **({"extra_objects": extra_objects} if extra_objects else {}),
    }

    output_path = converted_output_path(output_dir, triplet, sequence_id)
    _write_atomically(output_path, output_data)
    logger.info(f"  ✓ Saved: {output_path.name} (T={T}, RH={rh_obj_id}, LH={lh_obj_id})")
    return str(output_path)


def _build_parser() -> argparse.ArgumentParser:
    """The converter's command line."""
    parser = argparse.ArgumentParser(description="Convert raw TACO data to ManipTrans PKL")
    parser.add_argument("--raw-dir", type=str, default="datasets/raw/taco", help="Path to raw TACO data")
    parser.add_argument(
        "--output-dir", type=str, default="datasets/raw/taco_maniptrans", help="Output path for converted PKL files"
    )
    parser.add_argument("--max-sequences", type=int, default=None, help="Maximum number of sequences to process")
    parser.add_argument("--triplet-filter", type=str, default=None, help="Triplet filter (e.g., 'bowl')")
    parser.add_argument(
        "--object-poses-unit", type=str, default="m", choices=["cm", "m"], help="Object pose translation unit (cm or m)"
    )
    parser.add_argument(
        "--hand-trans-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Global offset to add to all hand_trans frames for both hands",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave an existing *_bimanual.pkl untouched and count it as skipped",
    )
    return parser


def _discover_sequences(hand_poses_dir: Path, triplet_filter: str | None) -> list[tuple[str, str]]:
    """Sorted (triplet, sequence id) of every sequence folder whose triplet matches the filter."""
    sequences = []
    for triplet_dir in sorted(hand_poses_dir.iterdir()):
        if not triplet_dir.is_dir():
            continue
        triplet_name = triplet_dir.name

        if not triplet_matches_filter(triplet_name, triplet_filter):
            continue

        sequences.extend((triplet_name, seq_dir.name) for seq_dir in sorted(triplet_dir.iterdir()) if seq_dir.is_dir())

    sequences.sort()
    return sequences


def _clip_id_clashes(sequences: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Clip ids that more than one raw sequence maps to (one would overwrite the other)."""
    by_name: dict[str, list[str]] = {}
    for triplet_name, sequence_id in sequences:
        by_name.setdefault(taco_task_name(triplet_name, sequence_id), []).append(f"{triplet_name}/{sequence_id}")
    return {name: raws for name, raws in by_name.items() if len(raws) > 1}


def _convert_one(args, raw_dir: Path, output_dir: Path, hand_trans_offset, triplet: str, seq_id: str) -> str:
    """Convert one sequence; returns its outcome: "skipped", "success", "excluded" or "failed"."""
    output_path = converted_output_path(output_dir, triplet, seq_id)
    if args.skip_existing and converted_output_is_valid(
        output_path,
        object_poses_unit=args.object_poses_unit,
        hand_trans_offset=hand_trans_offset,
        side_assign_file=Path(raw_dir) / "Object_Poses" / triplet / seq_id / "side_assign.json",
    ):
        logger.info(f"  ⏭ Skip (valid): {output_path.name}")
        return "skipped"
    if args.skip_existing and output_path.exists():
        logger.warning(f"  Existing output is invalid; regenerating: {output_path.name}")
    try:
        process_single_taco_sequence(
            raw_dir=str(raw_dir),
            output_dir=str(output_dir),
            triplet=triplet,
            sequence_id=seq_id,
            object_poses_unit=args.object_poses_unit,
            hand_trans_offset=hand_trans_offset,
        )
    except ExcludedSequence as e:
        logger.warning(f"  ⏭ {e}")
        return "excluded"
    except Exception as e:  # noqa: BLE001  one bad sequence is logged and counted; the batch continues
        logger.error(f"  Failed ({type(e).__name__}): {e}")
        return "failed"
    return "success"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    hand_trans_offset = (
        np.asarray(args.hand_trans_offset, dtype=np.float32) if args.hand_trans_offset is not None else None
    )
    if args.max_sequences is not None and args.max_sequences < 0:
        parser.error("--max-sequences must be non-negative")
    if hand_trans_offset is not None and not np.isfinite(hand_trans_offset).all():
        parser.error("--hand-trans-offset values must be finite")

    hand_poses_dir = raw_dir / "Hand_Poses"
    if not hand_poses_dir.is_dir():
        logger.error(f"Hand_Poses directory not found: {hand_poses_dir}")
        return 2

    sequences = _discover_sequences(hand_poses_dir, args.triplet_filter)
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]

    logger.info(f"Found {len(sequences)} sequences")
    clashes = _clip_id_clashes(sequences)
    if clashes:
        logger.error(
            f"{len(clashes)} clip id(s) are shared by more than one raw sequence — one would overwrite "
            f"the other: {dict(list(clashes.items())[:3])}"
        )
        return 2
    if not sequences and args.max_sequences != 0:
        logger.error("No TACO sequences matched the requested input/filter. Check --raw-dir and --triplet-filter.")
        return 2

    outcomes = dict.fromkeys(("success", "skipped", "excluded", "failed"), 0)
    for i, (triplet, seq_id) in enumerate(sequences, 1):
        logger.info(f"\n[{i}/{len(sequences)}] {triplet}/{seq_id}")
        outcomes[_convert_one(args, raw_dir, output_dir, hand_trans_offset, triplet, seq_id)] += 1
    success, skipped, excluded, failed = (outcomes[k] for k in ("success", "skipped", "excluded", "failed"))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Conversion completed: {success} converted, {skipped} skipped, {excluded} excluded, {failed} failed")
    logger.info(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.data.readers.taco_convert")
    raise SystemExit(main())
