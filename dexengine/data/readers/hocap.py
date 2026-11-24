"""Turn an HO-Cap sequence into a keypoint trajectory.

Layout consumed, relative to ``<dataset_dir>/raw/hocap/data``::

    sequence_labels/<seq>.meta.yaml     mano_sides, object_ids, num_frames
    sequence_labels/<seq>.poses_m.npy   (2, T, 51)  MANO params, [right, left]
    sequence_labels/<seq>.poses_o.npy   (K, T, 7)   object poses, quat-first
    calibration/mano.<subject>.yaml     per-subject MANO betas
    models/<object_id>.cleaned_mesh_10000.obj

Three things differ from the other readers and each one is load-bearing:

*Hands are slots, not a list.* ``poses_m`` always has two rows -- right then
left -- and an absent hand is filled with -1, not zeros or NaN. ``meta.yaml``'s
``mano_sides`` says which rows are real; a row that is all -1 must be dropped
rather than fed to FK, which would otherwise produce a plausible-looking hand
at a nonsense pose.

*Poses are quaternion-first.* ``poses_o`` rows are (qx, qy, qz, qw, x, y, z),
while the keypoint layer wants (x, y, z, qw, qx, qy, qz).

*Several objects, one hand.* A sequence stages up to four object_ids and the
subject picks them up in turn. The retargeting attaches ONE object per hand -- that is
the one the fingertip constraints and contact detection work against -- so the
object nearest the hand over the take is attached and the others are carried as
``extra_objects``: present in the scene, posed per frame, but not attached.
Dropping them, as an earlier version did, retargets the take against a table
that looks empty. The attachment rule is still a guess and will pick wrong if
the subject spends most of the take on a different object than the one they
hold longest; see ``_held_object``.

MANO parameters are given, not joint positions, so this runs the same forward
kinematics ``taco.py`` uses and then reorders MANO's 16-joint output into the
21-joint layout ``_hoi_common`` expects.
"""

from __future__ import annotations

import os
from pathlib import Path

import loguru
import numpy as np
import tyro
import yaml

from dexengine.data.readers.common import SequenceData, hand_track_from_joints21, write_sequence
from dexengine.data.readers.mano_lib import left_model, load_model, mano_forward
from dexengine.retarget import mjutil

DATASET_NAME = "hocap"
SOURCE_FPS = 30.0

# MANO's 16-joint regressor order is wrist, index, middle, pinky, ring, thumb;
# the 21-joint layout _hoi_common indexes is wrist, thumb, index, middle, ring,
# pinky with the tip appended to each finger. Tips come from vertices, not the
# regressor, so they are filled separately.
_M16_TO_J21 = {
    0: 0,
    13: 1,
    14: 2,
    15: 3,
    1: 5,
    2: 6,
    3: 7,
    4: 9,
    5: 10,
    6: 11,
    10: 13,
    11: 14,
    12: 15,
    7: 17,
    8: 18,
    9: 19,
}
_J21_TIPS = [4, 8, 12, 16, 20]


def _mano_dir(dataset_dir: Path) -> Path:
    """Where MANO_LEFT.pkl / MANO_RIGHT.pkl live.

    Prefers this dataset's own ``raw/hocap/body_models/mano`` but falls back to
    the TACO location, which README.md already tells the recipient to fill in.
    MANO's licence forbids redistribution, so the package ships those folders
    empty; without the fallback a recipient who followed the TACO instructions
    would hit a bare FileNotFoundError here with no hint that the same two
    files just need to be in a second place.
    """
    own = dataset_dir / "raw" / DATASET_NAME / "body_models" / "mano"
    if (own / "MANO_LEFT.pkl").is_file() or (own / "MANO_RIGHT.pkl").is_file():
        return own
    shared = dataset_dir / "raw" / "taco" / "body_models" / "mano"
    if (shared / "MANO_LEFT.pkl").is_file() or (shared / "MANO_RIGHT.pkl").is_file():
        return shared
    raise FileNotFoundError(
        "MANO body models not found. Put MANO_LEFT.pkl and MANO_RIGHT.pkl in "
        f"{own} or {shared} (see README.md -- the licence forbids shipping them)."
    )


def _root(dataset_dir: Path) -> Path:
    root = dataset_dir / "raw" / DATASET_NAME / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"HO-Cap not found at {root}")
    return root


def list_sequences(dataset_dir: str | Path) -> list[str]:
    """Sequence ids, e.g. ``subject_1-20231025_165502``."""
    root = _root(Path(os.path.abspath(os.path.expanduser(str(dataset_dir)))))
    return sorted(p.name[: -len(".meta.yaml")] for p in (root / "sequence_labels").glob("*.meta.yaml"))


def _joints21(betas: np.ndarray, params: np.ndarray, model, flat_hand_mean: bool = False) -> np.ndarray:
    """(T, 21, 3) world joints from (T, 51) MANO parameters.

    HO-Cap's MANO convention differs from TACO's in two independent ways, and
    both are needed. Neither fails loudly on its own.

    1. ``params[3 : 48]`` are 45 MANO **PCA coefficients**, not axis-angle
       rotations, so they are expanded through ``hands_components``
       (``use_pca=True``); ``hands_mean`` is then added
       (``flat_hand_mean=False``).
    2. ``params[48 : 51]`` is smplx's **additive** translation, not the wrist
       position TACO records, so the posed mesh keeps its rest wrist anchor
       (``trans_is_wrist=False``). The two conventions differ by a constant
       |J[0]| = 9.35 cm.

    This is not inferred from hand-object geometry, which cannot separate the
    two: HO-Cap ships its own ``hand_joints_3d`` labels inside each frame's
    ``.label.npy``, and those are the reference here. Over 8 subjects, both
    hands and 364 frames, all 21 joints match to 0.0000 cm with these settings.
    Reading the pose as axis-angle and the translation as a wrist position --
    TACO's convention -- gives a median joint error of 9.54 cm instead.

    The labels are per-camera and in camera coordinates; ``calibration/
    extrinsics.*.yaml`` is cam->world (all 8 cameras then agree to 0.00 cm), and
    that world differs from the ``poses_m``/``poses_o`` frame by one rigid
    transform, recoverable from the object poses at 0.0000 cm residual.
    """
    out = np.zeros((params.shape[0], 21, 3))
    for t, p in enumerate(params):
        pose = np.concatenate([p[:3], p[3:48]])
        joints, tips = mano_forward(
            model, pose, betas, p[48:51], flat_hand_mean=flat_hand_mean, use_pca=True, trans_is_wrist=False
        )
        for src, dst in _M16_TO_J21.items():
            out[t, dst] = joints[src]
        out[t, _J21_TIPS] = tips
    return out


def _quatfirst_to_keypoint(pose: np.ndarray) -> np.ndarray:
    """(T,7) (qx,qy,qz,qw,x,y,z) -> (T,7) (x,y,z,qw,qx,qy,qz)."""
    out = np.zeros_like(pose)
    out[:, :3] = pose[:, 4:7]
    out[:, 3] = pose[:, 3]
    out[:, 4:] = pose[:, :3]
    return out


def _held_object(joints: np.ndarray, poses_o: np.ndarray) -> int:
    """Index of the object the hand actually holds, by grasp duration.

    HO-Cap takes are pick-and-place: the subject handles several objects in
    turn, so "nearest on average over the clip" picks an object the hand is
    not on for most frames. Measured on subject_1-20231025_165502, the nearest
    object changes four times across the take and the clip-mean winner is the
    held one for only part of it.

    The retargeting attaches ONE object per hand, so the rule here is the object that
    is nearest the fingertips for the largest number of frames -- the one held
    longest. That is still a single choice for the whole clip: a take that
    splits its time evenly between two objects will be wrong for half of it,
    and fixing that properly means letting the attachment change per frame,
    which the intermediate form does not currently express.
    """
    tips = joints[:, _J21_TIPS]  # (T, 5, 3)
    # (K, T): distance from each object's origin to the nearest fingertip
    dist = np.stack(
        [np.linalg.norm(tips - poses_o[k, :, None, 4:7], axis=-1).min(axis=1) for k in range(poses_o.shape[0])]
    )
    counts = np.bincount(dist.argmin(axis=0), minlength=poses_o.shape[0])
    return int(counts.argmax())


def load_sequence(dataset_dir: str | Path, task: str) -> SequenceData:
    dataset_dir = Path(os.path.abspath(os.path.expanduser(str(dataset_dir))))
    root = _root(dataset_dir)
    meta = yaml.safe_load((root / "sequence_labels" / f"{task}.meta.yaml").read_text())
    sides = list(meta["mano_sides"])
    if not sides:
        raise ValueError(f"{task}: meta.yaml lists no mano_sides")

    poses_m = np.load(root / "sequence_labels" / f"{task}.poses_m.npy").astype(np.float64)
    poses_o = np.load(root / "sequence_labels" / f"{task}.poses_o.npy").astype(np.float64)
    betas = np.asarray(
        yaml.safe_load((root / "calibration" / f"mano.{meta['subject_id']}.yaml").read_text())["betas"],
        dtype=np.float64,
    )

    seq = SequenceData(task=task, source_fps=SOURCE_FPS)
    for row, side in ((0, "right"), (1, "left")):
        if side not in sides:
            continue
        params = poses_m[row]
        if np.all(params <= -1.0 + 1e-6):
            loguru.logger.warning(f"{task}: meta lists {side} but poses_m row {row} is the -1 filler")
            continue
        body_model = load_model(model_type="manor", model_path=str(_mano_dir(dataset_dir)))
        if side == "left":  # the left model HO-Cap was fitted with — statement: mano_conventions
            body_model = left_model(
                DATASET_NAME, body_model, load_model(model_type="manol", model_path=str(_mano_dir(dataset_dir)))
            )[0]
        joints = _joints21(betas, params, body_model)
        setattr(seq, side, hand_track_from_joints21(joints, side))

        k = _held_object(joints, poses_o)
        for j, obj_id in enumerate(meta["object_ids"]):
            mesh = root / "models" / f"{obj_id}.cleaned_mesh_10000.obj"
            if not mesh.is_file():
                raise FileNotFoundError(f"{task}: no mesh for object {obj_id!r} at {mesh}")
            pose = _quatfirst_to_keypoint(poses_o[j])
            if j == k:
                setattr(seq, f"obj_{side}", pose)
                setattr(seq, f"mesh_{side}", mesh)
                setattr(seq, f"mesh_name_{side}", obj_id)
            elif not any(o[0] == obj_id for o in seq.extra_objects):
                seq.extra_objects.append((obj_id, mesh, pose))
    if seq.right is None and seq.left is None:
        raise ValueError(f"{task}: no usable hand")
    return seq


def _convert_one(dataset_dir, tasks, i, name, target_ref_dt, ground) -> bool:
    """Convert one sequence; a failure is logged and returns False so the batch goes on."""
    try:
        seq = load_sequence(dataset_dir, name)
        side = "right" if seq.right is not None else "left"
        out = write_sequence(seq, dataset_dir, DATASET_NAME, side, target_ref_dt=target_ref_dt, ground=ground)
        loguru.logger.info(f"[{i}/{len(tasks)}] {name} ({side}) -> {out}")
    except Exception as exc:  # noqa: BLE001  one bad sequence is logged and counted; the batch continues
        loguru.logger.error(f"[{i}/{len(tasks)}] {name}: {type(exc).__name__}: {exc}")
        return False
    return True


def main(
    dataset_dir: str = f"{mjutil.ROOT}/datasets",
    task: str = "subject_1-20231025_165502",
    target_ref_dt: float = 0.02,
    ground: bool = True,
    all_sequences: bool = False,
) -> None:
    """Process one HO-Cap sequence, or every one with ``--all-sequences``."""
    tasks = list_sequences(dataset_dir) if all_sequences else [task]
    failed = 0
    for i, name in enumerate(tasks, 1):
        if not _convert_one(dataset_dir, tasks, i, name, target_ref_dt, ground):
            failed += 1
    if failed:
        loguru.logger.warning(f"{failed}/{len(tasks)} sequences failed")


if __name__ == "__main__":
    tyro.cli(main)
