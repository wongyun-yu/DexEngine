"""Phases 4-5 of the stage-1 retargeting batch driver (per hand: --robot-type = the hand's rig spec directory).

4. generate_xml — the MuJoCo scenes of each clip (dexengine.retarget.scene).
5. ik           — the retargeted hand trajectory trajectory_kinematic.npz (dexengine.retarget.ik).
--skip-existing reuses an output only when the solve-input fingerprint recorded beside it
(dexengine.retarget.input_fp) equals the fingerprint measured now.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from dexengine.core import blas_pin
from dexengine.core.paths import hand_rig_dir
from dexengine.core.repo import REPO_ROOT
from dexengine.retarget import input_fp as RFP
from dexengine.retarget.phases.clips import (
    ensure_robot_assets,
    find_processed_clips,
    ik_result_valid,
    robot_data_dir,
    robot_task_dir,
    save_taco_text_annotation,
    xml_outputs_complete,
)
from dexengine.retarget.phases.common import (
    RUN,
    _pmap,
    log_error,
    log_info,
    log_subprocess_failure,
    log_timeout,
    log_warning,
)

_SKIP_TALLY: dict[str, dict[str, int]] = {}
_SKIP_LOCK = threading.Lock()
#: The generate-xml and IK children never render, so MuJoCo loads no GL backend there (EGL costs ~40 ms per
#: process). This overrides the batch-wide MUJOCO_GL default, which serves the phases that do render.
_CHILD_MUJOCO_GL = "disable"


def _child_env() -> dict[str, str]:
    """Environment of a phase 4/5 child: the caller's, with the GL backend turned off."""
    return {**os.environ, "MUJOCO_GL": _CHILD_MUJOCO_GL}


def _solve_fp(dataset_dir: Path, robot_type: str, embodiment_type: str, task: str, phase: str) -> dict:
    """The clip's solve-input fingerprint for `phase`, measured now (owner: `dexengine.retarget.input_fp`)."""
    return RFP.current(
        dataset_dir, RUN.dataset_name, robot_type, embodiment_type, task, phase, hand_rig_dir(robot_type)
    )


def _skip_verdict(phase: str, task_dir: Path, cur: dict, have: bool) -> bool:
    """--skip-existing rule: reuse only a complete output whose recorded input fingerprint equals `cur`.
    Tallies reuse / nofp (output without a record) / diff (record differs) / new (no complete output)."""
    v = RFP.verdict(task_dir, phase, cur) if have else "none"
    kind = {"fresh": "reuse", "unknown": "nofp", "stale": "diff", "none": "new"}[v]
    with _SKIP_LOCK:
        t = _SKIP_TALLY.setdefault(phase, {})
        t[kind] = t.get(kind, 0) + 1
    return kind == "reuse"


def _skip_report(phase: str) -> None:
    """Marker of the applied reuse rule, one line per phase: reuse = skipped (same inputs) · stale = an existing
    output solved from other or unrecorded inputs · redo = everything solved again (stale + no complete output)."""
    t = _SKIP_TALLY.get(phase, {})
    stale = t.get("nofp", 0) + t.get("diff", 0)
    log_info(
        f"[SKIP_EXISTING] phase={phase} reuse={t.get('reuse', 0)} stale={stale} redo={stale + t.get('new', 0)} "
        f"(no fingerprint {t.get('nofp', 0)} · fingerprint differs {t.get('diff', 0)} · no output {t.get('new', 0)})"
    )


def _drop_outputs(task_dir: Path, phase: str) -> None:
    """Before a re-solve: forget the fingerprint record of this phase and the ones after it, and **unlink** the
    outputs they produced (never rewritten in place — an ingested layer may share their inodes; a failed solve must
    leave no product of the old inputs behind)."""
    RFP.forget(task_dir, phase)
    old = [q for q in (task_dir / "0").glob("*") if q.is_file() or q.is_symlink()]
    if phase == "generate-xml":
        old += list(task_dir.glob("*.xml")) + [task_dir / "task_info.json"]
    for q in old:
        q.unlink(missing_ok=True)


def run_single_generate_xml(
    dataset_dir: Path,
    triplet_id: str,
    clip_id: str,
    robot_type: str,
    embodiment_type: str,
    skip_existing: bool,
    collision_mode: str,
) -> bool:
    """Phase 4: Generate MuJoCo XML for a single clip."""
    task = f"{triplet_id}__{clip_id}" if clip_id is not None else triplet_id
    # generate_xml writes scene files one level above the per-data-id directory.
    task_dir = robot_task_dir(dataset_dir, robot_type, embodiment_type, task)
    xml_file = task_dir / "scene_eq.xml"

    try:
        fp = _solve_fp(dataset_dir, robot_type, embodiment_type, task, "generate-xml")
        if skip_existing and _skip_verdict("generate-xml", task_dir, fp, xml_outputs_complete(task_dir)):
            log_info("  ⏭ Skip (scene files exist and were built from the same inputs)")
            return True
        _drop_outputs(task_dir, "generate-xml")
        ensure_robot_assets(dataset_dir, robot_type)

        cmd = [
            sys.executable,
            "-m",
            "dexengine.retarget.scene",
            "--dataset-dir",
            str(dataset_dir),
            "--dataset-name",
            RUN.dataset_name,
            "--task",
            task,
            "--data-id",
            "0",
            "--robot-type",
            robot_type,
            "--embodiment-type",
            embodiment_type,
            "--no-show-viewer",
        ]
        if collision_mode == "visual-hull":
            # MuJoCo convexifies every mesh geom, so this makes each object
            # collide as its convex hull: a bowl becomes a solid blob.
            cmd.append("--use-visual-mesh-as-collision")
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=300,
            capture_output=True,
            text=True,
            env=_child_env(),
        )
        blas_pin.forward(result, "child of robot.py")
        if result.returncode != 0:
            log_subprocess_failure("generate-xml", cmd, result)
            return False
        if not xml_outputs_complete(task_dir):
            log_error(f"  ✗ generate-xml exited successfully but did not create all required outputs in: {task_dir}")
            return False
        RFP.stamp(task_dir, "generate-xml", fp)
        log_info(f"  ✓ XML generated: {xml_file}")
        return True
    except subprocess.TimeoutExpired as exc:
        log_timeout("generate-xml", exc)
        return False
    except Exception as e:  # noqa: BLE001 — per-clip phase step: any failure is logged and the clip counts as failed
        log_error(f"  ✗ {e}")
        return False


def run_generate_xml(
    dataset_dir: Path,
    robot_type: str,
    embodiment_type: str,
    max_clips: int | None,
    triplet_filter: str | None,
    skip_existing: bool,
    collision_mode: str,
) -> bool:
    """Phase 4: Generate MuJoCo XML for all clips."""
    log_info("=" * 60)
    log_info(f"Phase 4: Generate XML (collision: {collision_mode})")
    log_info("=" * 60)

    clips = find_processed_clips(dataset_dir, embodiment_type, triplet_filter, max_clips)
    log_info(f"Found {len(clips)} processed clips")
    if not clips:
        if max_clips == 0:
            log_info("No clips selected because --max-clips=0")
            return True
        log_error("No processed TACO clips matched the requested input/filter")
        return False

    def _one(i_clip):
        i, clip = i_clip
        log_info(f"[{i}/{len(clips)}] {clip['task']}")
        return run_single_generate_xml(
            dataset_dir, clip["triplet"], clip["sequence"], robot_type, embodiment_type, skip_existing, collision_mode
        )

    success = sum(1 for ok in _pmap(_one, list(enumerate(clips, 1))) if ok)
    if skip_existing:
        _skip_report("generate-xml")

    log_info(f"Phase 4 completed! ({success}/{len(clips)} successful)")
    return success == len(clips)


_IK_MODULE = "dexengine.retarget.ik"
_IK_TIMEOUT_S = 600  # per clip
_IK_INPUT = "y\n"  # answer to the module's rollout prompt


def _ik_cmd(dataset_dir, robot_type, task, embodiment_type, ref_dt, save_video):
    """The IK module command line for one clip."""
    cmd = [
        sys.executable,
        "-m",
        _IK_MODULE,
        "--dataset-dir",
        str(dataset_dir),
        "--dataset-name",
        RUN.dataset_name,
        "--robot-type",
        robot_type,
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

    if save_video:
        cmd.append("--save-video")
    return cmd


def run_single_ik(
    dataset_dir: Path,
    triplet_id: str,
    clip_id: str,
    robot_type: str,
    embodiment_type: str,
    skip_existing: bool,
    save_video: bool,
    ref_dt: float,
) -> bool:
    """Phase 5: Run inverse kinematics for a single clip."""
    task = f"{triplet_id}__{clip_id}" if clip_id is not None else triplet_id
    ik_dir = robot_data_dir(dataset_dir, robot_type, embodiment_type, task)
    # trajectory_kinematic.npz is the deterministic retargeting product.
    # trajectory_ikrollout.npz contains a noisy diagnostic rollout and is not a
    # suitable completion marker.
    ik_file = ik_dir / "trajectory_kinematic.npz"

    try:
        fp = _solve_fp(dataset_dir, robot_type, embodiment_type, task, "ik")
    except (OSError, ValueError) as e:
        log_error(f"  ✗ solve-input fingerprint: {e}")
        return False
    if skip_existing and _skip_verdict("ik", ik_dir.parent, fp, ik_result_valid(ik_file, ref_dt)):
        log_info("  ⏭ Skip (valid trajectory_kinematic.npz exists and was solved from the same inputs)")
        annotation_path = ik_dir / "text_annotations.json"
        if not annotation_path.is_file():
            annotation_path = save_taco_text_annotation(
                dataset_dir=dataset_dir,
                robot_type=robot_type,
                embodiment_type=embodiment_type,
                triplet_id=triplet_id,
                clip_id=clip_id,
            )
            log_info(f"  ✓ Missing text annotation created: {annotation_path}")
        return True
    if skip_existing and ik_file.exists():
        log_warning("  Existing IK output is stale/invalid; regenerating")

    try:
        _drop_outputs(ik_dir.parent, "ik")
        env = _child_env()

        cmd = _ik_cmd(dataset_dir, robot_type, task, embodiment_type, ref_dt, save_video)

        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            input=_IK_INPUT,
            timeout=_IK_TIMEOUT_S,
            capture_output=True,
            text=True,
            env=env,
        )
        blas_pin.forward(result, "child of robot.py")
        if result.returncode != 0:
            log_subprocess_failure("ik", cmd, result)
            return False
        if not ik_result_valid(ik_file, ref_dt):
            log_error(f"  ✗ ik exited successfully but output is invalid: {ik_file}")
            return False
        RFP.stamp(ik_dir.parent, "ik", fp)

        annot_path = save_taco_text_annotation(
            dataset_dir=dataset_dir,
            robot_type=robot_type,
            embodiment_type=embodiment_type,
            triplet_id=triplet_id,
            clip_id=clip_id,
        )
        log_info(f"  ✓ Text annotation saved: {annot_path}")
        log_info(f"  ✓ IK computed: {ik_file}")
        return True
    except subprocess.TimeoutExpired as exc:
        log_timeout("ik", exc)
        return False
    except Exception as e:  # noqa: BLE001 — per-clip phase step: any failure is logged and the clip counts as failed
        log_error(f"  ✗ {e}")
        return False


def run_ik(
    dataset_dir: Path,
    robot_type: str,
    embodiment_type: str,
    max_clips: int | None,
    triplet_filter: str | None,
    skip_existing: bool,
    save_video: bool,
    ref_dt: float,
) -> bool:
    """Phase 5: Run inverse kinematics for all clips."""
    log_info("=" * 60)
    log_info("Phase 5: Inverse Kinematics")
    log_info("=" * 60)

    clips = find_processed_clips(dataset_dir, embodiment_type, triplet_filter, max_clips)
    log_info(f"Found {len(clips)} processed clips")
    if not clips:
        if max_clips == 0:
            log_info("No clips selected because --max-clips=0")
            return True
        log_error("No processed TACO clips matched the requested input/filter")
        return False

    def _one(i_clip):
        i, clip = i_clip
        log_info(f"[{i}/{len(clips)}] {clip['task']}")
        return run_single_ik(
            dataset_dir,
            clip["triplet"],
            clip["sequence"],
            robot_type,
            embodiment_type,
            skip_existing,
            save_video,
            ref_dt,
        )

    success = sum(1 for ok in _pmap(_one, list(enumerate(clips, 1))) if ok)
    if skip_existing:
        _skip_report("ik")

    log_info(f"Phase 5 completed! ({success}/{len(clips)} successful)")
    return success == len(clips)
