"""Job list → one Slurm array script (`sbatch <file>`).

Every array task runs `python -m dexengine.cluster.job_entry --jobs <list> --index $SLURM_ARRAY_TASK_ID`, so the job
list file must be readable from the compute nodes. The resource request of the array is the largest request of its
jobs; give jobs with very different needs separate lists. Site settings (partition, account, extra `#SBATCH` lines,
setup commands) come from `configs/cluster/slurm.yaml` (`SlurmConfig`).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from dexengine.cluster.jobs import Job, check_unique


@dataclass(frozen=True)
class SlurmConfig:
    """`configs/cluster/slurm.yaml`."""

    partition: str = ""
    account: str = ""
    qos: str = ""
    python: str = "python"  # interpreter on the compute nodes
    max_parallel: int = 0  # array throttle (`%N`), 0 = none
    extra_sbatch: tuple[str, ...] = ()  # extra `#SBATCH` options, e.g. "--exclude=<node>"
    setup: tuple[str, ...] = ()  # shell lines run before the job (module load, conda activate, ...)


def render(jobs: list[Job], jobs_file: str, name: str, log_dir: str, cfg: SlurmConfig | None = None) -> str:
    """The sbatch script text for `jobs` stored in `jobs_file` (as written by jobs.save_jobs); cfg None = the default
    SlurmConfig."""
    cfg = SlurmConfig() if cfg is None else cfg
    jobs = check_unique(list(jobs))
    if not jobs:
        raise ValueError("empty job list")
    n = len(jobs)
    gpus, cpus, mem = max(j.gpus for j in jobs), max(j.cpus for j in jobs), max(j.mem_gb for j in jobs)
    tmin = max(j.time_min for j in jobs)
    array = f"0-{n - 1}" + (f"%{cfg.max_parallel}" if cfg.max_parallel else "")
    opts = [
        f"--job-name={name}",
        f"--array={array}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}G",
        f"--output={Path(log_dir) / (name + '_%a.log')}",
    ]
    if gpus:
        opts.append(f"--gres=gpu:{gpus}")
    if tmin:
        opts.append(f"--time={tmin // 60}:{tmin % 60:02d}:00")
    opts += [f"--{k}={v}" for k, v in (("partition", cfg.partition), ("account", cfg.account), ("qos", cfg.qos)) if v]
    opts += list(cfg.extra_sbatch)
    lines = [
        "#!/bin/bash",
        *(f"#SBATCH {o}" for o in opts),
        "set -euo pipefail",
        *cfg.setup,
        f"exec {shlex.quote(cfg.python)} -m dexengine.cluster.job_entry --jobs {shlex.quote(str(jobs_file))} "
        '--index "$SLURM_ARRAY_TASK_ID"',
    ]
    return "\n".join(lines) + "\n"
