"""Convert one DexCanvas MANO trajectory into retarget keypoints.

DexCanvas v0.1 stores all trajectories in one nested Parquet file.  Its
``mano_model_output.joints`` field is in the MANO local frame; the documented
visualizer transforms it by the per-frame root rotation and translation before
placing it next to the object.  Applying that same transform here is essential
-- using the local joints directly puts the hand roughly a hand-length away
from the object.

Only the 21 transformed MANO joints, object pose, and a matching STL mesh are
read.  The large hand-mesh vertex column is deliberately never loaded.
"""

from __future__ import annotations

import os
from pathlib import Path

import loguru
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from dexengine.data.readers.common import SequenceData, hand_track_from_joints21, write_sequence
from dexengine.data.readers.dexcanvas_frames import OBJECT_EULER, POSE_VERSION, object_pose7
from dexengine.retarget import mjutil

DATASET_NAME = "dexcanvas"
HAND_SIDE = "right"
# Resolved under ``dataset_dir`` so a checkout carries no machine-specific
# paths, matching hocap.py and hrdexdb.py which read ``raw/<name>/data``. Point
# ``datasets/raw/dexcanvas/data`` at the download (a symlink is fine) and put
# the toolkit's object meshes in ``datasets/raw/dexcanvas/objects``; both are
# still overridable per run with --parquet-path / --objects-dir.
PARQUET_RELPATH = "raw/dexcanvas/data/mocap_ver0.1.parquet"
OBJECTS_RELPATH = "raw/dexcanvas/objects"


def _default_parquet(dataset_dir: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(dataset_dir)))) / PARQUET_RELPATH


def _default_objects(dataset_dir: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(dataset_dir)))) / OBJECTS_RELPATH


def _parquet_path(path: str | Path) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not path.is_file():
        raise FileNotFoundError(f"DexCanvas Parquet file not found: {path}")
    return path


def _object_dir(path: str | Path) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not path.is_dir():
        raise FileNotFoundError(f"DexCanvas object asset directory not found: {path}")
    return path


def _task_name(meta: dict) -> str:
    """Stable, filesystem-safe id for one row in the DexCanvas Parquet file."""
    source = meta["mocap_raw_data_source"]
    return "__".join(
        [
            source["operator"],
            source["object"],
            source["gesture"],
            source["sequence"],
        ]
    )


def _metadata(parquet_path: Path) -> list[dict]:
    """Read the compact metadata column without materializing trajectory data."""
    import pyarrow.parquet as pq

    return pq.read_table(parquet_path, columns=["trajectory_meta_data"]).column(0).to_pylist()


def list_sequences(parquet_path: str | Path) -> list[str]:
    """Return all DexCanvas ids as ``operator__object__gesture__sequence``."""
    return [_task_name(meta) for meta in _metadata(_parquet_path(parquet_path))]


def _find_row(parquet_path: Path, task: str) -> tuple[int, dict]:
    for row_idx, meta in enumerate(_metadata(parquet_path)):
        if _task_name(meta) == task:
            return row_idx, meta
    available = ", ".join(list_sequences(parquet_path)[:5])
    raise ValueError(f"DexCanvas task {task!r} not found. First available: {available}")


def _read_sequence_info(parquet_path: Path, row_idx: int) -> dict:
    """Read one row's keypoints and pose columns, skipping MANO mesh vertices."""
    import pyarrow.parquet as pq

    columns = [
        "sequence_info.mano_model_output.joints",
        "sequence_info.object_info.pose",
        "sequence_info.hand_joint.position",
        "sequence_info.hand_joint.rotation",
    ]
    parquet = pq.ParquetFile(parquet_path)
    for current_row, batch in enumerate(parquet.iter_batches(batch_size=1, columns=columns)):
        if current_row == row_idx:
            return batch.to_pydict()["sequence_info"][0]
    raise IndexError(f"row {row_idx} is outside {parquet_path}")


def _object_mesh(objects_dir: Path, object_name: str) -> Path:
    """Resolve the toolkit's inconsistent lower/upper-case STL extensions."""
    exact = objects_dir / f"{object_name}.stl"
    if exact.is_file():
        return exact
    matches = sorted(
        path
        for path in objects_dir.iterdir()
        if path.stem.lower() == object_name.lower() and path.suffix.lower() == ".stl"
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"DexCanvas mesh for {object_name!r} not found in {objects_dir}")


def _world_joints(sequence_info: dict) -> np.ndarray:
    """Apply the global MANO transform used by DexCanvas' own visualizer."""
    mano = sequence_info["mano_model_output"]
    joints = np.asarray(mano["joints"], dtype=np.float64).reshape(-1, 21, 3)
    hand = sequence_info["hand_joint"]
    translation = np.asarray(hand["position"], dtype=np.float64)
    rotation = np.asarray(hand["rotation"], dtype=np.float64)
    if translation.shape != (joints.shape[0], 3) or rotation.shape != (joints.shape[0], 3):
        raise ValueError(
            "DexCanvas hand root transform shape mismatch: "
            f"joints={joints.shape}, translation={translation.shape}, rotation={rotation.shape}"
        )
    return np.einsum("tij,tkj->tki", R.from_rotvec(rotation).as_matrix(), joints) + translation[:, None, :]


def _object_pose7(sequence_info: dict, num_frames: int) -> np.ndarray:
    """DexCanvas x y z + Euler (rad) -> keypoint x y z + w x y z under the Euler rule's owner (dexcanvas_frames)."""
    pose6 = np.asarray(sequence_info["object_info"]["pose"], dtype=np.float64)
    if pose6.shape != (num_frames, 6):
        raise ValueError(f"expected object pose ({num_frames}, 6), got {pose6.shape}")
    return object_pose7(pose6)


def load_sequence(
    parquet_path: str | Path,
    objects_dir: str | Path,
    task: str,
) -> SequenceData:
    """Load one DexCanvas row as an in-world right-hand ``SequenceData``."""
    parquet_path = _parquet_path(parquet_path)
    objects_dir = _object_dir(objects_dir)
    row_idx, meta = _find_row(parquet_path, task)
    sequence_info = _read_sequence_info(parquet_path, row_idx)
    joints = _world_joints(sequence_info)
    source = meta["mocap_raw_data_source"]
    object_name = source["object"]
    return SequenceData(
        task=task,
        source_fps=float(meta.get("data_fps", 100.0)),
        right=hand_track_from_joints21(joints, HAND_SIDE),
        obj_right=_object_pose7(sequence_info, joints.shape[0]),
        mesh_right=_object_mesh(objects_dir, object_name),
        mesh_name_right=object_name,
    )


def _convert_one(parquet, objects_dir, dataset_dir, tasks, index, sequence_task, target_ref_dt, ground) -> bool:
    """Convert one clip; a failure is logged and returns False so the batch goes on."""
    try:
        seq = load_sequence(parquet, objects_dir, sequence_task)
        out = write_sequence(
            seq,
            dataset_dir,
            DATASET_NAME,
            HAND_SIDE,
            target_ref_dt=target_ref_dt,
            ground=ground,
        )
        loguru.logger.info(f"[{index}/{len(tasks)}] {sequence_task} -> {out}")
        print(f"[DEXCANVAS_EULER] task={sequence_task} euler={OBJECT_EULER} pose_version={POSE_VERSION}", flush=True)
    except Exception as exc:  # noqa: BLE001  one bad clip is logged and counted; the batch continues
        loguru.logger.error(f"[{index}/{len(tasks)}] {sequence_task}: {type(exc).__name__}: {exc}")
        return False
    return True


def main(
    dataset_dir: str = f"{mjutil.ROOT}/datasets",
    task: str = "s02__cube1__09__003",
    parquet_path: str | None = None,
    objects_dir: str | None = None,
    target_ref_dt: float = 0.02,
    ground: bool = True,
    all_sequences: bool = False,
    list_only: bool = False,
) -> None:
    """Process one DexCanvas clip, or every clip when ``--all-sequences`` is set.

    ``parquet_path`` and ``objects_dir`` default to the DexCanvas v0.1 layout
    under ``dataset_dir/raw/dexcanvas`` and can be overridden for a different
    download location. ``--list-only`` prints every task id (one per line) and
    writes nothing -- the raw re-read driver (``python -m dexengine.data.read_raw``) lists the
    sequences through this one function. Each written clip prints
    ``[DEXCANVAS_EULER] task=<id> euler=<rule>`` -- the driver counts it.
    """
    parquet = _parquet_path(parquet_path or _default_parquet(dataset_dir))
    if list_only:
        print("\n".join(list_sequences(parquet)))
        return
    objects_dir = objects_dir or _default_objects(dataset_dir)
    tasks = list_sequences(parquet) if all_sequences else [task]
    failed = 0
    for index, sequence_task in enumerate(tasks, 1):
        if not _convert_one(parquet, objects_dir, dataset_dir, tasks, index, sequence_task, target_ref_dt, ground):
            failed += 1
    if failed:
        raise RuntimeError(f"DexCanvas conversion failed for {failed}/{len(tasks)} sequence(s)")


if __name__ == "__main__":
    tyro.cli(main)
