"""Phases 2-3 of the stage-1 retargeting batch driver (shared by every hand; run once per dataset tree).

2. decompose      — convex collision parts of each clip's objects (dexengine.retarget.decompose).
3. detect_contact — MANO fingertip-object contacts written into trajectory_keypoints.npz
                    (dexengine.retarget.contact).
Each clip runs as a child process of the retargeting unit's module, from the repository root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dexengine.core import blas_pin
from dexengine.core.repo import REPO_ROOT
from dexengine.retarget.phases.clips import (
    convex_parts_exist,
    find_processed_clips,
    has_contact_result,
    mano_data_dir,
)
from dexengine.retarget.phases.common import RUN, log_error, log_info, log_subprocess_failure, log_timeout, log_warning

#: The [DECOMP] … marker lines the decomposer printed in this run (one per clip); counted at the end of phase 2.
DECOMP_MARKS: list[str] = []


def run_single_decompose(
    dataset_dir: Path,
    triplet_id: str,
    clip_id: str,
    embodiment_type: str,
    skip_existing: bool,
) -> bool:
    """Phase 2: Build convex collision parts for one clip's objects."""
    task = f"{triplet_id}__{clip_id}" if clip_id is not None else triplet_id
    task_info_path = mano_data_dir(dataset_dir, embodiment_type, task).parent / "task_info.json"

    if skip_existing and convex_parts_exist(dataset_dir, task_info_path, embodiment_type):
        log_info("  ⏭ Skip (convex parts exist)")
        return True

    try:
        cmd = [
            sys.executable,
            "-m",
            "dexengine.retarget.decompose",
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-name",
            RUN.dataset_name,
            "--task",
            task,
            "--data-id",
            "0",
            "--embodiment-type",
            embodiment_type,
        ]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=900,
            capture_output=True,
            text=True,
        )
        blas_pin.forward(result, "child of objects.py")
        if result.returncode != 0:
            log_subprocess_failure("decompose", cmd, result)
            return False
        if not convex_parts_exist(dataset_dir, task_info_path, embodiment_type):
            log_error(f"  ✗ decompose exited successfully but wrote no convex parts: {task_info_path}")
            return False
        # Relay the decomposer's rule markers (frozen-dataset reuse "[DECOMP] frozen reuse …") into this log, so the
        # run log shows which branch each clip took.
        marks = [ln for ln in (result.stdout or "").splitlines() if ln.startswith("[DECOMP] ")]
        for ln in marks:
            log_info(f"  {ln}")
        DECOMP_MARKS.extend(marks)
        log_info("  ✓ Convex parts " + ("reused (frozen dataset)" if marks else "generated"))
        return True
    except subprocess.TimeoutExpired as exc:
        log_timeout("decompose", exc)
        return False
    except Exception as e:  # noqa: BLE001 — per-clip phase step: any failure is logged and the clip counts as failed
        log_error(f"  ✗ {e}")
        return False


def run_decompose(
    dataset_dir: Path,
    embodiment_type: str,
    max_clips: int | None,
    triplet_filter: str | None,
    skip_existing: bool,
) -> bool:
    """Phase 2: Build convex collision parts for all clips."""
    log_info("=" * 60)
    log_info("Phase 2: Decompose")
    log_info("=" * 60)

    clips = find_processed_clips(dataset_dir, embodiment_type, triplet_filter, max_clips)
    log_info(f"Found {len(clips)} processed clips")
    if not clips:
        if max_clips == 0:
            log_info("No clips selected because --max-clips=0")
            return True
        log_error("No processed TACO clips matched the requested input/filter")
        return False

    success = 0
    for i, clip in enumerate(clips, 1):
        log_info(f"[{i}/{len(clips)}] {clip['triplet']}__{clip['sequence']}")
        if run_single_decompose(
            dataset_dir,
            clip["triplet"],
            clip["sequence"],
            embodiment_type,
            skip_existing,
        ):
            success += 1

    n_frozen = sum(1 for ln in DECOMP_MARKS if ln.startswith("[DECOMP] frozen reuse "))
    if n_frozen:
        log_info(
            f"[DECOMP] frozen reuse ds={RUN.dataset_name} n={n_frozen}/{len(clips)} clips (data_layout.frozen_datasets)"
        )
    log_info(f"Phase 2 completed! ({success}/{len(clips)} successful)")
    return success == len(clips)


def run_single_detect_contact(
    dataset_dir: Path,
    triplet_id: str,
    clip_id: str,
    embodiment_type: str,
    skip_existing: bool,
    ref_dt: float,
) -> bool:
    """Phase 3: Detect contact for a single clip."""

    task = f"{triplet_id}__{clip_id}" if clip_id is not None else triplet_id
    traj_dir = mano_data_dir(dataset_dir, embodiment_type, task)
    traj_file = traj_dir / "trajectory_keypoints.npz"

    if skip_existing and traj_file.is_file():
        try:
            if has_contact_result(traj_file):
                log_info("  ⏭ Skip (contact data exists)")
                return True
        except (OSError, ValueError) as exc:
            log_warning(f"  Existing contact file is unreadable; regenerating it: {exc}")

    try:
        cmd = [
            sys.executable,
            "-m",
            "dexengine.retarget.contact",
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-name",
            RUN.dataset_name,
            "--task",
            task,
            "--data-id",
            "0",
            "--embodiment-type",
            embodiment_type,
            "--ref-dt",
            str(ref_dt),
            "--no-show-viewer",
        ]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=300,
            capture_output=True,
            text=True,
        )
        blas_pin.forward(result, "child of objects.py")
        if result.returncode != 0:
            log_subprocess_failure("detect-contact", cmd, result)
            return False
        if not has_contact_result(traj_file):
            log_error(f"  ✗ detect-contact exited successfully but did not write contact arrays: {traj_file}")
            return False
        log_info("  ✓ Contact detected")
        return True
    except subprocess.TimeoutExpired as exc:
        log_timeout("detect-contact", exc)
        return False
    except Exception as e:  # noqa: BLE001 — per-clip phase step: any failure is logged and the clip counts as failed
        log_error(f"  ✗ {e}")
        return False


def run_detect_contact(
    dataset_dir: Path,
    embodiment_type: str,
    max_clips: int | None,
    triplet_filter: str | None,
    skip_existing: bool,
    ref_dt: float,
) -> bool:
    """Phase 3: Detect contact for all clips."""
    log_info("=" * 60)
    log_info("Phase 3: Detect Contact")
    log_info("=" * 60)

    clips = find_processed_clips(dataset_dir, embodiment_type, triplet_filter, max_clips)
    log_info(f"Found {len(clips)} processed clips")
    if not clips:
        if max_clips == 0:
            log_info("No clips selected because --max-clips=0")
            return True
        log_error("No processed TACO clips matched the requested input/filter")
        return False

    success = 0
    for i, clip in enumerate(clips, 1):
        log_info(f"[{i}/{len(clips)}] {clip['triplet']}__{clip['sequence']}")
        if run_single_detect_contact(
            dataset_dir,
            clip["triplet"],
            clip["sequence"],
            embodiment_type,
            skip_existing,
            ref_dt,
        ):
            success += 1

    log_info(f"Phase 3 completed! ({success}/{len(clips)} successful)")
    return success == len(clips)
