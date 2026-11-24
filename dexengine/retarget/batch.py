#!/usr/bin/env python3
"""Stage-1 retargeting batch driver: MANO human-hand clips → floating robot-hand trajectories (MuJoCo IK).

Phases (one --phase or all):
0. convert        - measure the hand-object assignment and convert raw TACO into ManipTrans PKL files (TACO only)
1. kinematic      - process hand poses and write trajectory_keypoints.npz (TACO only)
2. decompose      - split object meshes into convex collision parts
3. detect-contact - detect contact between hands and objects
4. generate-xml   - generate MuJoCo scene XML files (per hand: --robot-type)
5. ik             - run inverse kinematics and write trajectory_kinematic.npz (per hand: --robot-type)

Phases 2-5 accept any dataset laid out as processed/<dataset>/{mano,assets,<robot>} (--dataset-name).
Paths are independent of the caller's working directory. Every phase reports per-clip failures, and the command
exits non-zero if any requested phase fails.

Usage:
    python -m dexengine.retarget.batch --dataset-dir DIR --phase all --robot-type sharpa_wave --skip-existing
    python -m dexengine.retarget.batch --dataset-dir DIR --phase convert --max-sequences 5
    python -m dexengine.retarget.batch --dataset-dir DIR --phase ik --robot-type sharpa_wave --max-clips 5 --save-video
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# MuJoCo offscreen rendering configuration (inherited by the phase subprocesses).
os.environ.setdefault("MUJOCO_GL", "egl")

from dexengine.core.repo import REPO_ROOT  # noqa: E402
from dexengine.retarget.phases.clips import (  # noqa: E402
    convex_parts_exist,
    find_processed_clips,
    has_contact_result,
    normalize_phase_name,
    xml_outputs_complete,
)
from dexengine.retarget.phases.common import RUN, log_error, log_info, log_warning  # noqa: E402
from dexengine.retarget.phases.objects import run_decompose, run_detect_contact  # noqa: E402
from dexengine.retarget.phases.robot import run_generate_xml, run_ik  # noqa: E402
from dexengine.retarget.phases.taco import run_convert, run_kinematic  # noqa: E402


def print_summary(
    dataset_dir: Path,
    embodiment_type: str,
    robot_type: str | None,
) -> None:
    """Print a summary of processing results."""
    log_info("\n" + "=" * 60)
    log_info("Summary")
    log_info("=" * 60)

    maniptrans_dir = dataset_dir / "raw" / "taco_maniptrans"
    pkl_count = len(list(maniptrans_dir.glob("*_bimanual.pkl"))) if maniptrans_dir.exists() else 0

    mano_clips = find_processed_clips(dataset_dir, embodiment_type)

    # A run of the shared phases only (no hand) does not count a robot layer.
    robot_root = (dataset_dir / "processed" / RUN.dataset_name / robot_type / embodiment_type) if robot_type else None
    robot_clips = []
    robot_task_dirs = []
    if robot_root is not None and robot_root.exists():
        for task_dir in sorted(robot_root.iterdir()):
            if task_dir.is_dir():
                robot_task_dirs.append(task_dir)
                if (task_dir / "0").exists():
                    robot_clips.append(task_dir / "0")

    contact_count = 0
    kinematic_count = 0
    decompose_count = 0
    for c in mano_clips:
        traj_file = c["task_dir"] / "trajectory_keypoints.npz"
        if traj_file.exists():
            kinematic_count += 1
            try:
                if has_contact_result(traj_file):
                    contact_count += 1
            except (OSError, ValueError):
                log_warning(f"Could not read contact result: {traj_file}")
        if convex_parts_exist(dataset_dir, c["task_dir"].parent / "task_info.json", embodiment_type):
            decompose_count += 1

    xml_count = len([d for d in robot_task_dirs if xml_outputs_complete(d)])
    ik_count = len([d for d in robot_clips if (d / "trajectory_kinematic.npz").exists()])

    video_mano = sum(1 for c in mano_clips if (c["task_dir"] / "visualization.mp4").exists())
    video_ik = len([d for d in robot_clips if (d / "visualization_ik.mp4").exists()])

    log_info(f"Phase 0 (Convert):        {pkl_count} pkl files")
    log_info(f"Phase 1 (Kinematic):      {kinematic_count} clips, {video_mano} videos")
    log_info(f"Phase 2 (Decompose):      {decompose_count} clips")
    log_info(f"Phase 3 (Detect Contact): {contact_count} clips")
    log_info(f"Phase 4 (Generate XML):   {xml_count} clips")
    log_info(f"Phase 5 (IK):             {ik_count} clips, {video_ik} videos")
    log_info("=" * 60)


def _add_scope_args(parser):
    """Dataset, robot, embodiment, phase and clip-selection arguments."""
    parser.add_argument("--dataset-dir", type=str, default=str(REPO_ROOT / "datasets"), help="Dataset root directory")
    parser.add_argument(
        "--workers", type=int, default=None, help="clips solved in parallel by phases 4-5 (default: capacity.plan)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="taco",
        help="process processed/<name>/ (default taco); only phases 2-5 accept other datasets",
    )
    # No default: the hand (its rig spec directory) is stated by the caller; the robot phases refuse to run without it.
    parser.add_argument(
        "--robot-type",
        type=str,
        default=None,
        help="robot type = the hand's rig spec directory (phases 4-5, required there)",
    )
    parser.add_argument(
        "--embodiment-type",
        type=str,
        default="bimanual",
        choices=["bimanual", "right", "left"],
        help=(
            "Embodiment type. TACO is a bimanual dataset; 'right' and 'left' process a single hand of the same clips."
        ),
    )
    parser.add_argument(
        "--phase",
        type=normalize_phase_name,
        default="all",
        choices=[
            "all",
            "convert",
            "kinematic",
            "decompose",
            "detect-contact",
            "generate-xml",
            "ik",
        ],
        help="Pipeline phase to run",
    )
    parser.add_argument("--max-clips", type=int, default=None, help="Maximum number of clips to process")
    parser.add_argument(
        "--max-sequences", type=int, default=None, help="Phase 0: maximum number of sequences to convert"
    )
    parser.add_argument(
        "--triplet-filter", type=str, default=None, help="Raw or normalized triplet substring (e.g., 'bowl')"
    )
    parser.add_argument(
        "--object-poses-unit",
        choices=["m", "cm"],
        default="m",
        help="Phase 0: translation unit stored in Object_Poses matrices",
    )


def _add_conversion_args(parser):
    """Per-phase conversion, collision, timing and output arguments."""
    parser.add_argument(
        "--hand-trans-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help=(
            "Phase 0: constant offset added to every hand translation. Only "
            "needed if a source release stores hand_trans in a different "
            "convention; leave unset for stock TACO."
        ),
    )
    parser.add_argument(
        "--collision-mode",
        choices=["decomposed", "visual-hull"],
        default="decomposed",
        help=(
            "Phase 4: 'decomposed' uses the convex parts from Phase 2 so "
            "containers stay hollow; 'visual-hull' reuses the visual mesh, "
            "which MuJoCo collides as a single convex hull (a bowl becomes "
            "solid). Use 'visual-hull' only to skip the coacd dependency."
        ),
    )
    parser.add_argument(
        "--body-models-dir",
        type=str,
        default=None,
        help=(
            "Phase 1: directory containing MANO_RIGHT.pkl and MANO_LEFT.pkl "
            "(default: DATASET_DIR/raw/taco/body_models/mano)"
        ),
    )
    parser.add_argument(
        "--source-ref-dt",
        type=float,
        default=1.0 / 30.0,
        help="Phase 1: source frame interval in seconds (default: 1/30)",
    )
    parser.add_argument(
        "--target-ref-dt",
        type=float,
        default=0.02,
        help=("Output interval used consistently by Phases 1, 3, and 5 (default: 0.02, i.e. 50 Hz)"),
    )
    parser.add_argument(
        "--no-mano-flat-hand-mean",
        dest="mano_flat_hand_mean",
        action="store_false",
        help=(
            "Phase 1: treat hand_pose[3:] as relative to MANO's mean hand pose "
            "(EasyMocap's use_flat_mean: False). The default assumes absolute "
            "poses, which matches smplx's flat_hand_mean=True exactly; see the "
            "Known limitations section before changing it."
        ),
    )
    parser.add_argument(
        "--clip-list",
        type=str,
        default=None,
        help="process only the task names in this file (one per line); an absent name fails (default: all)",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that are already processed")
    parser.add_argument("--save-video", action="store_true", help="Phases 1 and 5: render a diagnostic video")


def build_parser() -> argparse.ArgumentParser:
    """The batch CLI (one argparse table)."""
    parser = argparse.ArgumentParser(description="TACO retargeting pipeline")

    _add_scope_args(parser)
    _add_conversion_args(parser)
    return parser


def configure_run(args, parser) -> None:
    """Copy the run-wide settings into `RUN` and refuse inconsistent flags."""
    RUN.workers = None if args.workers is None else max(1, int(args.workers))
    print(f"[batch] workers option={RUN.workers} dataset={args.dataset_name}", flush=True)
    RUN.dataset_name = args.dataset_name
    if args.clip_list is not None:
        RUN.clip_list_path = str(Path(args.clip_list).expanduser().resolve())
        RUN.clip_list = {
            ln.strip() for ln in Path(RUN.clip_list_path).read_text(encoding="utf-8").splitlines() if ln.strip()
        }
        if not RUN.clip_list:
            raise SystemExit(f"[CLIP_LIST] ✗ empty list: {RUN.clip_list_path}")

    if RUN.dataset_name != "taco" and args.phase in ("convert", "kinematic", "all"):
        raise SystemExit(
            f"[batch] --dataset-name {RUN.dataset_name}: phases convert/kinematic/all are TACO-only "
            "(raw/taco); name one of phases 2-5"
        )

    if (
        not np.isfinite(args.source_ref_dt)
        or not np.isfinite(args.target_ref_dt)
        or args.source_ref_dt <= 0
        or args.target_ref_dt <= 0
    ):
        parser.error("--source-ref-dt and --target-ref-dt must be finite and positive")
    if args.max_clips is not None and args.max_clips < 0:
        parser.error("--max-clips must be non-negative")
    if args.max_sequences is not None and args.max_sequences < 0:
        parser.error("--max-sequences must be non-negative")


def select_phases(args, parser, robot_type) -> list[str]:
    """The phases to run, in order; the robot phases need --robot-type."""
    # Keep the historic underscore phase values as aliases while documenting
    # the easier-to-read hyphenated spellings.
    phase = args.phase.replace("-", "_")
    if phase == "all":
        phases_to_run = [
            "convert",
            "kinematic",
            "decompose",
            "detect_contact",
            "generate_xml",
            "ik",
        ]
        if args.collision_mode == "visual-hull":
            # Nothing downstream would read the convex parts.
            phases_to_run.remove("decompose")
    else:
        phases_to_run = [phase]
    if not robot_type and {"generate_xml", "ik"} & set(phases_to_run):
        parser.error("--robot-type is required by the robot phases (generate-xml, ik); it has no default")
    return phases_to_run


def make_phase_runners(args, dataset_dir, body_models_dir, robot_type, embodiment_type) -> dict:
    """Phase name → zero-argument runner returning True on success."""
    return {
        "convert": lambda: run_convert(
            dataset_dir=dataset_dir,
            max_sequences=(args.max_sequences if args.max_sequences is not None else args.max_clips),
            triplet_filter=args.triplet_filter,
            skip_existing=args.skip_existing,
            object_poses_unit=args.object_poses_unit,
            hand_trans_offset=args.hand_trans_offset,
        ),
        "kinematic": lambda: run_kinematic(
            dataset_dir=dataset_dir,
            embodiment_type=embodiment_type,
            max_clips=args.max_clips,
            triplet_filter=args.triplet_filter,
            skip_existing=args.skip_existing,
            save_video=args.save_video,
            body_models_dir=body_models_dir,
            source_ref_dt=args.source_ref_dt,
            target_ref_dt=args.target_ref_dt,
            mano_flat_hand_mean=args.mano_flat_hand_mean,
        ),
        "decompose": lambda: run_decompose(
            dataset_dir=dataset_dir,
            embodiment_type=embodiment_type,
            max_clips=args.max_clips,
            triplet_filter=args.triplet_filter,
            skip_existing=args.skip_existing,
        ),
        "detect_contact": lambda: run_detect_contact(
            dataset_dir=dataset_dir,
            embodiment_type=embodiment_type,
            max_clips=args.max_clips,
            triplet_filter=args.triplet_filter,
            skip_existing=args.skip_existing,
            ref_dt=args.target_ref_dt,
        ),
        "generate_xml": lambda: run_generate_xml(
            dataset_dir=dataset_dir,
            robot_type=robot_type,
            embodiment_type=embodiment_type,
            max_clips=args.max_clips,
            triplet_filter=args.triplet_filter,
            skip_existing=args.skip_existing,
            collision_mode=args.collision_mode,
        ),
        "ik": lambda: run_ik(
            dataset_dir=dataset_dir,
            robot_type=robot_type,
            embodiment_type=embodiment_type,
            max_clips=args.max_clips,
            triplet_filter=args.triplet_filter,
            skip_existing=args.skip_existing,
            save_video=args.save_video,
            ref_dt=args.target_ref_dt,
        ),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_run(args, parser)

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    body_models_dir = Path(args.body_models_dir).expanduser().resolve() if args.body_models_dir is not None else None
    robot_type = args.robot_type
    embodiment_type = args.embodiment_type
    phases_to_run = select_phases(args, parser, robot_type)

    log_info(f"Dataset: {dataset_dir}")
    log_info(f"Robot: {robot_type}")
    log_info(f"Embodiment: {embodiment_type}")
    log_info(f"Collision: {args.collision_mode}")
    log_info(f"Phases: {phases_to_run}")

    phase_runners = make_phase_runners(args, dataset_dir, body_models_dir, robot_type, embodiment_type)
    phase_results: list[bool] = []
    for phase_name in phases_to_run:
        phase_ok = phase_runners[phase_name]()
        phase_results.append(phase_ok)
        if not phase_ok:
            remaining = phases_to_run[len(phase_results) :]
            if remaining:
                log_error(f"Stopping after failed phase '{phase_name}'; dependent phases not run: {remaining}")
            break

    print_summary(dataset_dir, embodiment_type, robot_type)
    if not all(phase_results):
        log_error("One or more requested phases failed")
        return 1
    return 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.batch")
    raise SystemExit(main())
