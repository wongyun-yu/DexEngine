#!/usr/bin/env python3
"""Stage-1 TACO retargeting batch: one clip (base) list × robot hands, through the retargeting phases.

Steps (each step's output goes to ``<log>/<step>.log`` and its exit code to ``<log>/<step>.rc``):
  shared (skipped with --robot-phases-only)
    lift            dexengine/data/readers/taco_lift.py — TACO object trajectories from the TACO source tree
                    (``layout.taco_source_root`` under the data root) into the stage's ``raw/taco``
    convert · kinematic · decompose · detect-contact
                    ``python -m dexengine.retarget.batch --phase <p> --skip-existing``
  per robot (in parallel)
    <robot>_generate-xml · <robot>_ik   (ik reuses valid outputs unless --ik-force)
    <robot>.count   clips of the list whose ``processed/taco/<robot>/bimanual/<clip>/0/trajectory_kinematic.npz``
                    exists (names compared through ``paths.canonical_clip``)
Done = count match: every robot's count equals the number of clips in the list. ``<log>/STATUS`` holds RUNNING, then
DONE, FAIL_COUNT, FAIL_lift or FAIL_<phase>.

The interpreter is the retarget venv (``paths.retarget_py``, env RETARGET_PY wins). Phases run from the repository
root with PYTHONPATH = the repository root and DEXENGINE_MAIN_ROOT = the data root; the MuJoCo phases with
MUJOCO_GL=egl.

Usage:
    python -m dexengine.retarget.run --bases <clips.txt> --robots dg5f_s [--stage-dir DIR] [--log-dir DIR]
        [--data-root DIR] [--robot-phases-only] [--ik-force]
rc 0 = the batch ran to the end (the result is STATUS) · 1 = a step failed · 2 = missing input.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dexengine.core import config, layout
from dexengine.core import paths as P
from dexengine.core.repo import REPO_ROOT

SHARED_PHASES = ("convert", "kinematic", "decompose", "detect-contact")
DEFAULT_LOG_DIR = Path("/tmp/f32_batch")


def _selected_hands() -> list[str]:
    """Every registered hand whose vendor source is set (vendor.unavailable_hands names the others), with a [HANDS]
    line when one is left out."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.robots import vendor  # noqa: PLC0415

    un = vendor.unavailable_hands()
    if un:
        print(f"[HANDS] selected {[h for h in P.HAND_IDS if h not in un]} · not selected {un}", flush=True)
    return [h for h in P.HAND_IDS if h not in un]


def list_clips(bases: Path) -> list[str]:
    """Non-empty lines of the clip list (names may contain spaces)."""
    return [ln for ln in bases.read_text(encoding="utf-8").splitlines() if ln]


def count_done(stage_dir: Path, robot: str, clips: list[str], dataset: str = "taco") -> int:
    """Clips of ``clips`` with a retargeted ``0/trajectory_kinematic.npz`` for ``robot`` (bimanual). A name the name
    rule refuses is reported and not counted (the count then fails)."""
    root = stage_dir / "processed" / dataset / robot / "bimanual"
    n = 0
    for c in clips:
        try:
            n += (root / P.canonical_clip(c, dataset) / "0" / "trajectory_kinematic.npz").is_file()
        except ValueError as e:  # noqa: PERF203 — a refused name is reported per clip and the count goes on
            print(f"[f32] ✗ {e}", file=sys.stderr)
    return n


def _step(log_dir: Path, name: str, cmd: list[str], env: dict | None = None) -> int:
    """Run one step with its log and rc file; returns the rc."""
    with open(log_dir / f"{name}.log", "w") as fh:
        rc = subprocess.run(
            cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO_ROOT, env={**os.environ, **(env or {})}
        ).returncode
    (log_dir / f"{name}.rc").write_text(f"{rc}\n")
    return rc


def _status(log_dir: Path, word: str) -> str:
    (log_dir / "STATUS").write_text(word + "\n")
    return word


def taco_batch(
    bases: Path,
    robots: list[str],
    *,
    data_root: Path,
    stage_dir: Path,
    log_dir: Path = DEFAULT_LOG_DIR,
    robot_phases_only: bool = False,
    ik_force: bool = False,
    python: str | None = None,
) -> tuple[str, int]:
    """Run the batch; returns (STATUS word, rc) — see the module docstring."""
    py = python or P.retarget_py()
    if not (py and os.access(py, os.X_OK)):
        print(f"[f32] retargeting interpreter not found: {py or '<empty>'}", file=sys.stderr)
        return "", 2
    if not bases.is_file() or not bases.stat().st_size:
        print(f"[f32] the base list is empty or missing: {bases}", file=sys.stderr)
        return "", 2
    clips = list_clips(bases)
    log_dir.mkdir(parents=True, exist_ok=True)
    _status(log_dir, "RUNNING")
    env = {"PYTHONPATH": str(REPO_ROOT), "DEXENGINE_MAIN_ROOT": str(data_root)}
    gl = {**env, "MUJOCO_GL": "egl"}
    mano = str(data_root / "data" / "body_models" / "mano")
    batch = [py, "-m", "dexengine.retarget.batch", "--dataset-dir", str(stage_dir)]

    if not robot_phases_only:
        lift_root = data_root / layout.taco_source_root()
        if not lift_root.is_dir():
            print(f"[f32] lift input (TACO source, shipped input taco_source) missing: {lift_root}", file=sys.stderr)
            return _status(log_dir, "FAIL_lift"), 2
        print(f"[f32] lift input = {lift_root}", flush=True)
        rc = _step(
            log_dir,
            "lift",
            [
                py,
                str(REPO_ROOT / "dexengine/data/readers/taco_lift.py"),
                "--tasks-file",
                str(bases),
                "--trajectories-root",
                str(lift_root),
                "--meshes-root",
                str(data_root / "data/taco/meshes"),
                "--real-raw",
                str(data_root / "data/taco/raw_unzipped"),
                "--shim-raw-dir",
                str(stage_dir / "raw/taco"),
            ],
            env,
        )
        if rc:
            return _status(log_dir, "FAIL_lift"), 1
        for ph in SHARED_PHASES:
            if _step(log_dir, ph, batch + ["--phase", ph, "--skip-existing", "--body-models-dir", mano], gl):
                return _status(log_dir, f"FAIL_{ph}"), 1

    def one_robot(r: str) -> int:
        for ph, skip in (("generate-xml", True), ("ik", not ik_force)):
            cmd = (
                batch
                + ["--phase", ph, "--robot-type", r]
                + (["--skip-existing"] if skip else [])
                + ["--body-models-dir", mano]
            )
            if _step(log_dir, f"{r}_{ph}", cmd, gl):
                return -1
        n = count_done(stage_dir, r, clips)
        (log_dir / f"{r}.count").write_text(f"{n}\n")
        return n

    from dexengine.core import capacity  # noqa: PLC0415

    step = "dexengine.retarget.run"  # capacity.NEEDS key: one retarget subprocess chain per robot
    with capacity.sized(step, len(robots)) as cap, ThreadPoolExecutor(cap.workers) as ex:
        counts = list(ex.map(one_robot, robots))
    ok = all(n == len(clips) for n in counts)
    return _status(log_dir, "DONE" if ok else "FAIL_COUNT"), 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bases", type=Path, required=True, help="clip (base) names, one per line")
    ap.add_argument(
        "--robots",
        nargs="+",
        default=None,
        help="hand spec directories (default: every registered hand's, paths.hand_spec_dir)",
    )
    config.add_data_root_arg(ap)
    ap.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="retargeting dataset root (raw/ and processed/); default <data root>/data/taco_retarget",
    )
    ap.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    ap.add_argument(
        "--robot-phases-only",
        action="store_true",
        help="skip the shared steps (lift … detect-contact) and run only the per-robot steps",
    )
    ap.add_argument("--ik-force", action="store_true", help="re-run ik on every clip")
    a = ap.parse_args(argv)
    config.apply_data_root(a)
    data_root = P.main_root()
    robots = a.robots or [P.hand_spec_dir(h) for h in _selected_hands()]
    status, rc = taco_batch(
        a.bases,
        robots,
        data_root=data_root,
        stage_dir=a.stage_dir or data_root / "data" / "taco_retarget",
        log_dir=a.log_dir,
        robot_phases_only=a.robot_phases_only,
        ik_force=a.ik_force,
    )
    if status:
        print(f"[f32] STATUS={status}", flush=True)
    return rc


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.run")
    sys.exit(main())
