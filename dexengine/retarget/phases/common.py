"""Shared state, logging and subprocess helpers of the stage-1 retargeting batch driver (`dexengine.retarget.batch`)."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class BatchRun:
    """Settings of one batch invocation, filled once by `batch.main`.

    dataset_name: the dataset processed under ``processed/<name>/``. Phases 2-5 follow it; phases 0-1 read
        ``raw/taco`` and ``raw/taco_maniptrans`` and are TACO-only. Other datasets reuse the same driver with
        ``--dataset-name <ds>`` (the ``processed/<ds>/{mano,assets,<robot>}`` layout of
        ``dexengine.retarget.mjutil.processed_data_dir``).
    workers: clips solved in parallel by phases 4-5 (each worker runs one subprocess) — an explicit --workers; None =
        capacity.plan decides.
    clip_list: task directory names to process (``--clip-list``); None = every clip. A listed name missing
        from the input (``mano/<emb>``) is an error.
    """

    dataset_name: str = "taco"
    workers: int | None = None
    clip_list: set | None = None
    clip_list_path: str = ""


#: The settings of the running invocation.
RUN = BatchRun()
STEP = "dexengine.retarget.phases.common"  # capacity.NEEDS key (the phase 4-5 clip pool)


def log_info(msg):
    print(f"[INFO] {msg}", flush=True)


def log_warning(msg):
    print(f"[WARNING] {msg}", flush=True)


def log_error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


def _write_captured_output(label: str, content: str | bytes | None) -> None:
    """Print captured subprocess output with a label, preserving useful context."""
    if isinstance(content, bytes):
        content = content.decode(errors="replace")
    if not content or not content.strip():
        return
    log_error(f"  {label}:")
    for line in content.rstrip().splitlines():
        print(f"    {line}", file=sys.stderr, flush=True)


def log_subprocess_failure(
    phase: str,
    cmd: list[str],
    result: subprocess.CompletedProcess,
) -> None:
    """Report a failed command together with its captured stdout and stderr."""
    log_error(f"  ✗ {phase} failed (code {result.returncode})")
    log_error(f"  Command: {shlex.join(cmd)}")
    _write_captured_output("stdout", result.stdout)
    _write_captured_output("stderr", result.stderr)


def log_timeout(phase: str, timeout: subprocess.TimeoutExpired) -> None:
    """Report a timeout and any partial subprocess output."""
    log_error(f"  ✗ {phase} timed out after {timeout.timeout}s")
    _write_captured_output("stdout before timeout", timeout.stdout)
    _write_captured_output("stderr before timeout", timeout.stderr)


def _pmap(fn, items):
    """Map `fn` over `items` on a thread pool sized by capacity (`RUN.workers` = an explicit option; each call runs one
    subprocess, so the GIL does not matter).
    Order is preserved."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from dexengine.core import capacity  # noqa: PLC0415

    with capacity.sized(STEP, len(items), RUN.workers) as cap, ThreadPoolExecutor(cap.workers) as ex:
        return list(ex.map(fn, items))
