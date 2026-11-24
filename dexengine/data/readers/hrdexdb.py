"""Turn an HRDexDB *human* episode into a keypoint trajectory.

HRDexDB pairs human and robot-hand captures of the same grasp, but only the
``human/`` half is used here -- the robot embodiments (allegro_v5, inspire_f1,
inspire_dftp) are retargeting *targets*, not sources.

Layout consumed, relative to ``<dataset_dir>/raw/hrdexdb/data``::

    metadata/episodes.parquet            index: embodiment, fps, num_frames, mesh_path
    human/<object>/<seq>/hand/mano_params/NNNNN.json
    object_6d_pose_v2/human/<object>_<seq>.npz    frame_<i> -> 4x4  (current release layout)
    human/<object>/<seq>/object_6d_pose_v2.npz    frame_<i> -> 4x4  (older release layout)
    object_6d_pose_v1/human/<object>_<seq>.npz    frame_<i> -> 4x4  (v1 — only when no v2 exists)
    human/<object>/<seq>/object_6d/pose_NNNNNN.txt   (older v1 · marker [HRDEXDB_OBJ] says which source)
    assets/mesh_v2/<object>/<object>.obj

The per-frame JSON already carries a 21-keypoint MANO skeleton in metres, so
this reader uses those directly instead of re-running FK.  That sidesteps the
one convention this dataset does not document: whether ``transl`` is an
additive MANO translation or the wrist position outright.  ``joints`` is
unambiguous either way.

HRDexDB captures a single right hand, so ``embodiment_type`` is ``right``.

Canonical run (retarget venv, from ``retarget/``) — every human episode, the episodes without ``C2R.npy`` on the
release-wide C2R (the canonical choice, see ``source_frames.C2R_FILE``; marker ``[HRDEXDB_C2R] … source=shared``)::

    python -m dexengine.data.readers.hrdexdb --dataset-dir <stage> --all-sequences --allow-shared-c2r
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import loguru
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from dexengine.data.readers.common import SequenceData, hand_track_from_joints21, write_sequence
from dexengine.data.readers.source_frames import (
    C2R_FILE,
    FRAME_VERSION,
    hrdexdb_mesh,
    hrdexdb_pose_npz,
    load_c2r,
    release_c2r,
    robot_from_world,
)
from dexengine.retarget import mjutil

DATASET_NAME = "hrdexdb"
HAND_SIDE = "right"


def _root(dataset_dir: Path) -> Path:
    root = dataset_dir / "raw" / DATASET_NAME / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"HRDexDB not found at {root}")
    return root


def list_sequences(dataset_dir: str | Path, split: str | None = None) -> list[str]:
    """Return ``<object>/<seq>`` ids for every human episode."""
    import pandas as pd

    root = _root(Path(os.path.abspath(os.path.expanduser(str(dataset_dir)))))
    table = pd.read_parquet(root / "metadata" / "episodes.parquet")
    table = table[table["embodiment"] == "human"]
    if split is not None:
        table = table[table["split"] == split]
    return sorted(eid.split("/", 1)[1] for eid in table["episode_id"])


#: task_info ``source_frame.rule`` — where the C2R rule lives (official inverse · no-file policy); presence says it ran.
SOURCE_FRAME_RULE = "dexengine/data/readers/source_frames.py:C2R_FILE"


def _robot_from_world(episode: Path, allow_shared_c2r: bool) -> tuple[np.ndarray, str]:
    """(4x4 world -> robot-base transform, C2R source) of one episode — ``source_frames.robot_from_world`` of the
    episode's C2R, or of the release-wide one when the file is missing and the caller allowed it
    (``source_frames.C2R_FILE`` says why there is no default). Prints ``[HRDEXDB_C2R]``."""
    path = episode / C2R_FILE
    if path.is_file():
        c2r, source = load_c2r(path), "file"
    elif allow_shared_c2r:
        c2r, source = _release_c2r(str(episode.parent.parent)), "shared(allow_shared_c2r)"
    else:
        raise FileNotFoundError(
            f"{episode}: no {C2R_FILE}. The official identity default leaves this camera-rig-world episode "
            "upside down; pass --allow-shared-c2r to use the release-wide C2R (checked to be one matrix) "
            "or leave the episode out."
        )
    print(f"[HRDEXDB_C2R] episode={episode.parent.name}/{episode.name} source={source} applied=inv(C2R)", flush=True)
    return robot_from_world(c2r), source


@functools.lru_cache(maxsize=1)
def _release_c2r(human_root: str) -> np.ndarray:
    return release_c2r(Path(human_root))


def _object_poses(episode: Path, num_frames: int, rfw: np.ndarray) -> tuple[np.ndarray, str]:
    """((T,7) object pose in the robot base frame, source file) — prefers the v2 refit the dataset card recommends."""
    f = hrdexdb_pose_npz(episode)
    mats, source = None, "v1:object_6d/*.txt"
    if f is not None:
        with np.load(f) as data:
            keys = [f"frame_{i}" for i in range(num_frames)]
            if not all(k in data for k in keys):
                raise ValueError(f"{f}: object poses miss frames of {num_frames} (no silent fallback)")
            mats, source = np.stack([data[k] for k in keys]), str(f.relative_to(episode.parents[2]))
    print(f"[HRDEXDB_OBJ] task={episode.parent.name}/{episode.name} source={source}", flush=True)
    if mats is None:
        files = sorted((episode / "object_6d").glob("pose_*.txt"))
        if len(files) < num_frames:
            raise FileNotFoundError(f"{episode}: only {len(files)} v1 object poses for {num_frames} frames")
        mats = np.stack([np.loadtxt(f) for f in files[:num_frames]])
    mats = rfw[None] @ mats
    pose = np.zeros((num_frames, 7))
    pose[:, :3] = mats[:, :3, 3]
    xyzw = R.from_matrix(mats[:, :3, :3]).as_quat()
    pose[:, 3:] = xyzw[:, [3, 0, 1, 2]]
    return pose, source


def load_sequence(dataset_dir: str | Path, task: str, allow_shared_c2r: bool = False) -> SequenceData:
    """Load ``<object>/<seq>`` (e.g. ``apple/0``) into the common form (robot base frame — source_frames.C2R_FILE)."""
    dataset_dir = Path(os.path.abspath(os.path.expanduser(str(dataset_dir))))
    root = _root(dataset_dir)
    object_name, seq_id = task.split("/")
    episode = root / "human" / object_name / seq_id
    if not episode.is_dir():
        raise FileNotFoundError(f"episode not found: {episode}")

    files = sorted((episode / "hand" / "mano_params").glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{episode}: no mano_params")
    joints = np.stack([np.asarray(json.loads(Path(f).read_text())["joints"], dtype=np.float64) for f in files])
    rfw, c2r_source = _robot_from_world(episode, allow_shared_c2r)
    joints = joints @ rfw[:3, :3].T + rfw[:3, 3]

    obj_pose, obj_source = _object_poses(episode, joints.shape[0], rfw)
    mesh = hrdexdb_mesh(root, object_name)
    if mesh is None:
        raise FileNotFoundError(f"no mesh for object {object_name!r}")

    return SequenceData(
        task=f"{object_name}__{seq_id}",
        source_fps=30.0,
        right=hand_track_from_joints21(joints, HAND_SIDE),
        obj_right=obj_pose,
        mesh_right=mesh,
        mesh_name_right=object_name,
        source_frame={
            "rule": SOURCE_FRAME_RULE,
            "version": FRAME_VERSION,
            "applied": "inv(C2R)",
            "c2r_source": c2r_source,
            "object_pose": obj_source,
        },
    )


def _convert_one(dataset_dir, tasks, i, name, target_ref_dt, ground, allow_shared_c2r) -> bool:
    """Convert one episode; a failure is logged and returns False so the batch goes on."""
    try:
        seq = load_sequence(dataset_dir, name, allow_shared_c2r=allow_shared_c2r)
        out = write_sequence(seq, dataset_dir, DATASET_NAME, HAND_SIDE, target_ref_dt=target_ref_dt, ground=ground)
        loguru.logger.info(f"[{i}/{len(tasks)}] {name} -> {out}")
    except Exception as exc:  # noqa: BLE001  one bad episode is logged and counted; the batch continues
        loguru.logger.error(f"[{i}/{len(tasks)}] {name}: {type(exc).__name__}: {exc}")
        return False
    return True


def main(
    dataset_dir: str = f"{mjutil.ROOT}/datasets",
    task: str = "apple/0",
    target_ref_dt: float = 0.02,
    ground: bool = True,
    all_sequences: bool = False,
    split: str | None = None,
    allow_shared_c2r: bool = False,
) -> None:
    """Process one HRDexDB human episode, or every one with ``--all-sequences``.

    Args:
        dataset_dir: Dataset root holding ``raw/`` and receiving ``processed/``.
        task: Episode id as ``<object>/<seq>``, e.g. ``apple/0``.
        target_ref_dt: Seconds per output frame (0.02 -> 50 Hz).
        ground: Shift the scene so its lowest point sits at z = 0.
        all_sequences: Process every human episode instead of ``task``.
        split: With ``--all-sequences``, restrict to this metadata split.
        allow_shared_c2r: Process an episode without C2R.npy with the release-wide C2R
            (default: such an episode fails, see source_frames.C2R_FILE).
    """
    tasks = list_sequences(dataset_dir, split=split) if all_sequences else [task]
    failed = 0
    for i, name in enumerate(tasks, 1):
        if not _convert_one(dataset_dir, tasks, i, name, target_ref_dt, ground, allow_shared_c2r):
            failed += 1
    if failed:
        loguru.logger.warning(f"{failed}/{len(tasks)} episodes failed")


if __name__ == "__main__":
    tyro.cli(main)
