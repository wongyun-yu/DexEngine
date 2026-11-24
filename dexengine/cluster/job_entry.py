"""Run one entry of a job list inside a scheduled job: `python -m dexengine.cluster.job_entry --jobs F --index I`.

The job's prep (if any) runs first as a child; then the command replaces this process (same pid, so the scheduler's
signals reach it) with the job's environment added. Exit code = the command's (a failing prep's when it fails); 2 = a
bad list or index.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from dexengine.cluster.jobs import load_jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, help="job list (JSON Lines, jobs.save_jobs)")
    ap.add_argument("--index", required=True, type=int, help="0-based entry (Slurm: $SLURM_ARRAY_TASK_ID)")
    a = ap.parse_args(argv)
    try:
        jobs = load_jobs(a.jobs)
        job = jobs[a.index]
    except (OSError, ValueError, IndexError) as e:
        print(f"[job_entry] {a.jobs} index {a.index}: {e}", file=sys.stderr)
        return 2
    if job.workdir:
        os.chdir(job.workdir)
    env = {**os.environ, **job.env}
    if job.prep:
        print(f"[job_entry] {job.name} prep: {' '.join(job.prep)}", flush=True)
        if rc := subprocess.run(list(job.prep), env=env).returncode:
            print(f"[job_entry] {job.name} prep exited {rc}: the command does not run", flush=True)
            return rc
    print(f"[job_entry] {job.name}: {' '.join(job.cmd)}", flush=True)
    os.execvpe(job.cmd[0], list(job.cmd), env)
    return 0  # not reached


if __name__ == "__main__":
    raise SystemExit(main())
