"""Phases 0-1 of the stage-1 retargeting batch driver — TACO only (they read raw/taco).

0. convert   — measure the hand-object assignment (dexengine.data.readers.taco_side_assign) and convert raw TACO
               into ManipTrans PKL files (dexengine.data.readers.taco_convert).
1. kinematic — MANO forward kinematics into trajectory_keypoints.npz (dexengine.data.readers.taco).
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

from dexengine.core import blas_pin
from dexengine.core.repo import REPO_ROOT
from dexengine.retarget.phases.clips import kinematic_result_valid, mano_data_dir, task_matches_filter
from dexengine.retarget.phases.common import log_error, log_info, log_subprocess_failure, log_warning

#: Prefix of the producer's lines; its one summary line is the prefixed stdout line that is not an error ("✗") line.
SIDE_ASSIGN_MARK = "[SIDE_ASSIGN_MEASURE] "


def _has_measure_summary(stdout: str | None) -> bool:
    """Whether the side-assignment producer printed its summary line (it does so only after measuring)."""
    return any(
        ln.startswith(SIDE_ASSIGN_MARK) and not ln.startswith(SIDE_ASSIGN_MARK + "✗")
        for ln in (stdout or "").splitlines()
    )


def _convert_cmd(
    hand_trans_offset, max_sequences, object_poses_unit, output_dir, raw_dir, skip_existing, triplet_filter
):
    """The taco_convert command line for this run's options."""
    cmd = [
        sys.executable,
        "-m",
        "dexengine.data.readers.taco_convert",
        "--raw-dir",
        str(raw_dir),
        "--output-dir",
        str(output_dir),
        "--object-poses-unit",
        object_poses_unit,
    ]

    if max_sequences is not None:
        cmd.extend(["--max-sequences", str(max_sequences)])

    if triplet_filter:
        cmd.extend(["--triplet-filter", triplet_filter])
    if skip_existing:
        cmd.append("--skip-existing")
    if hand_trans_offset is not None:
        cmd.append("--hand-trans-offset")
        cmd.extend(str(value) for value in hand_trans_offset)
    return cmd


def run_convert(
    dataset_dir: Path,
    max_sequences: int | None = None,
    triplet_filter: str | None = None,
    skip_existing: bool = False,
    object_poses_unit: str = "m",
    hand_trans_offset: list[float] | None = None,
) -> bool:
    """Phase 0: measure the hand-object assignment, then convert raw TACO into ManipTrans PKL files."""
    log_info("=" * 60)
    log_info("Phase 0: Convert")
    log_info("=" * 60)

    raw_dir = dataset_dir / "raw" / "taco"
    output_dir = dataset_dir / "raw" / "taco_maniptrans"

    if not raw_dir.is_dir():
        log_error(f"Raw dir not found: {raw_dir}")
        return False

    # The hand-object assignment is only ever measured: before converting, the producer measures it per sequence
    # from the raw MANO contact and sustained grasp and writes side_assign.json. A sequence it cannot measure gets
    # no file (the reason is in the report) and the converter then fails on that sequence.
    pcmd = [
        sys.executable,
        "-m",
        "dexengine.data.readers.taco_side_assign",
        "--raw",
        str(raw_dir),
        "--write",
        "--report",
        str(raw_dir / "side_assign_measure.json"),
    ]
    log_info(f"Running: {shlex.join(pcmd)}")
    presult = subprocess.run(pcmd, cwd=REPO_ROOT, capture_output=True, text=True)
    blas_pin.forward(presult, "child of taco.py")
    for line in (presult.stdout or "").splitlines():
        if line.startswith("[SIDE_ASSIGN_MEASURE]"):
            print(line, flush=True)
    if presult.returncode not in (0, 1) or not _has_measure_summary(presult.stdout):
        log_subprocess_failure("convert(side_assign)", pcmd, presult)
        return False

    cmd = _convert_cmd(
        hand_trans_offset, max_sequences, object_poses_unit, output_dir, raw_dir, skip_existing, triplet_filter
    )

    log_info(f"Running: {shlex.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    blas_pin.forward(result, "child of taco.py")
    if result.returncode != 0:
        log_subprocess_failure("convert", cmd, result)
        return False
    # Loguru writes normal converter progress to stderr; replay both streams so
    # a successful batch run remains inspectable.
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    log_info("Phase 0 completed!")
    return True


def run_kinematic(
    dataset_dir: Path,
    embodiment_type: str,
    max_clips: int | None,
    triplet_filter: str | None,
    skip_existing: bool,
    save_video: bool = False,
    body_models_dir: Path | None = None,
    source_ref_dt: float = 1.0 / 30.0,
    target_ref_dt: float = 0.02,
    mano_flat_hand_mean: bool = True,
) -> bool:
    """Phase 1: Run the TACO reader (MANO forward kinematics) to generate trajectory_keypoints.npz."""
    log_info("=" * 60)
    log_info("Phase 1: Kinematic")
    log_info("=" * 60)

    maniptrans_dir = dataset_dir / "raw" / "taco_maniptrans"
    if not maniptrans_dir.is_dir():
        log_error(f"ManipTrans dir not found: {maniptrans_dir}")
        return False

    pkl_files = sorted(maniptrans_dir.glob("*_bimanual.pkl"))
    if triplet_filter:
        pkl_files = [f for f in pkl_files if task_matches_filter(f.stem, triplet_filter)]

    if max_clips is not None:
        pkl_files = pkl_files[:max_clips]

    log_info(f"Found {len(pkl_files)} bimanual pkl files")
    if not pkl_files:
        if max_clips == 0:
            log_info("No clips selected because --max-clips=0")
            return True
        log_error("No converted TACO clips matched the requested input/filter")
        return False

    from dexengine.data.readers import taco as taco_processor

    success = 0
    for pkl_file in pkl_files:
        task_name = pkl_file.stem.removesuffix("_bimanual")
        log_info(f"Processing: {task_name}")

        marker = mano_data_dir(dataset_dir, embodiment_type, task_name) / "trajectory_keypoints.npz"
        if skip_existing and kinematic_result_valid(marker, source_ref_dt, target_ref_dt):
            log_info("  ⏭ Skip (valid trajectory_keypoints.npz exists)")
            success += 1
            continue
        if skip_existing and marker.exists():
            log_warning("  Existing kinematic output is stale/invalid; regenerating")

        try:
            taco_processor.main(
                dataset_dir=str(dataset_dir),
                embodiment_type=embodiment_type,
                task=task_name,
                show_viewer=False,
                save_video=save_video,
                raw_subdir="taco_maniptrans",
                body_models_dir=(str(body_models_dir) if body_models_dir else None),
                source_ref_dt=source_ref_dt,
                target_ref_dt=target_ref_dt,
                mano_flat_hand_mean=mano_flat_hand_mean,
            )
            if not kinematic_result_valid(marker, source_ref_dt, target_ref_dt):
                raise ValueError(f"Kinematic processor returned without a valid output: {marker}")
            log_info(f"  ✓ Processed: {task_name}")
            success += 1
        except Exception as e:  # noqa: BLE001 — per-clip phase step: any failure is logged and the clip counts as failed
            log_error(f"  ✗ Failed: {task_name} - {type(e).__name__}: {e}")

    log_info(f"Phase 1 completed! ({success}/{len(pkl_files)} successful)")
    return success == len(pkl_files)
