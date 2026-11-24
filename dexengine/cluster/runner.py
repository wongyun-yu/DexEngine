"""Run a job list on the local machine's GPUs.

Jobs start in list order as soon as enough GPUs are free; each job sees only its GPUs (CUDA_VISIBLE_DEVICES) and
runs in its own session (so a stop reaches its whole process tree). A job's prep (Job.prep) runs first with no GPU
(CUDA_VISIBLE_DEVICES empty) while other jobs hold the GPUs; at most one prepared or preparing job waits per GPU, and
prepared jobs take the GPUs in list order. Output (prep and command) goes to `<log_dir>/<name>.log` and the exit
code to `<log_dir>/<name>.rc`. A job whose `.rc` file already says 0 is skipped, so a rerun resumes.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

from dexengine.cluster.jobs import Job, check_unique


def _start(job: Job, gpus: list[str], log_dir: Path, argv: tuple[str, ...] | None = None):
    env = {**os.environ, **job.env, "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
    log = open(log_dir / f"{job.name}.log", "ab")  # noqa: SIM115 — closed when the process ends (run_local)
    proc = subprocess.Popen(
        list(argv or job.cmd),
        env=env,
        cwd=job.workdir or None,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, log


def _done(log_dir: Path, name: str) -> bool:
    f = log_dir / f"{name}.rc"
    return f.is_file() and f.read_text().strip() == "0"


def run_local(jobs: list[Job], gpus: list[str], log_dir: str | os.PathLike, poll_s: float = 2.0) -> dict[str, int]:
    """Run `jobs` on the GPU ids `gpus` (CPU-only jobs need no GPU) → {job name: exit code}.

    A job asking for more GPUs than `gpus` holds raises before anything starts. SIGTERM/SIGINT stop the running jobs
    (their sessions get SIGTERM) and return the codes collected so far."""
    jobs = check_unique(list(jobs))
    if bad := [j.name for j in jobs if j.gpus > len(gpus)]:
        raise ValueError(f"jobs {bad} need more GPUs than the {len(gpus)} given")
    log = Path(log_dir)
    log.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not _done(log, j.name)]
    rcs = {j.name: 0 for j in jobs if j not in todo}
    free = list(gpus)
    ahead = max(1, len(gpus))  # prepared + preparing jobs that wait for GPUs
    prepping: dict[str, tuple] = {}
    ready: list[Job] = []  # prep done (or none): waits for GPUs in list order
    running: dict[str, tuple] = {}
    stop = []
    old = {s: signal.signal(s, lambda *_: stop.append(1)) for s in (signal.SIGTERM, signal.SIGINT)}

    def end(name: str, rc: int) -> None:
        (log / f"{name}.rc").write_text(f"{rc}\n")
        rcs[name] = rc

    try:
        while (todo or prepping or ready or running) and not stop:
            while todo and (not todo[0].prep or len(prepping) + sum(bool(j.prep) for j in ready) < ahead):
                job = todo.pop(0)
                if job.prep:
                    prepping[job.name] = (*_start(job, [], log, job.prep), job)
                else:
                    ready.append(job)
            for name, (proc, fh, job) in list(prepping.items()):
                rc = proc.poll()
                if rc is not None:
                    fh.close()
                    del prepping[name]
                    if rc:
                        end(name, rc)  # a failing prep is the job's code; cmd does not run
                    else:
                        ready.append(job)
                        ready.sort(key=jobs.index)  # list order among the prepared jobs
            while ready and ready[0].gpus <= len(free):
                job = ready.pop(0)
                mine, free = free[: job.gpus], free[job.gpus :]
                running[job.name] = (*_start(job, mine, log), mine)
            for name, (proc, fh, mine) in list(running.items()):
                rc = proc.poll()
                if rc is not None:
                    fh.close()
                    end(name, rc)
                    free += mine
                    del running[name]
            time.sleep(poll_s if running or prepping else 0)
    finally:
        for proc, fh, _ in (*running.values(), *prepping.values()):
            fh.close()
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        for s, h in old.items():
            signal.signal(s, h)
    return rcs
