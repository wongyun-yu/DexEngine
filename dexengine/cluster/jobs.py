"""Generic job lists: what to run, with which resources — the input of every execution backend.

A job is one command with its environment and resource request. Stages build a job list (for example one job per
RL bundle or per record piece); a backend runs it on local GPUs (`dexengine.cluster.runner`) or turns it into
scheduler files (`dexengine.cluster.slurm`, `dexengine.cluster.k8s`). Inside a scheduled job,
`python -m dexengine.cluster.job_entry --jobs <file> --index <i>` runs job `i` of the list.

Job lists are stored as JSON Lines (one job object per line), so a list can be appended to and read back in order.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


@dataclass(frozen=True)
class Job:
    """One command and its resource request.

    name      unique within a list; used for log files and scheduler job names (letters, digits, `.`, `_`, `-`)
    cmd       argv of the command (no shell)
    env       extra environment variables of the command
    gpus      GPUs the command needs (0 = CPU only)
    cpus      CPU cores
    mem_gb    host memory
    time_min  wall-time limit in minutes (0 = no limit)
    workdir   working directory ("" = the caller's)
    prep      argv of a CPU step run before `cmd`, with the same env and workdir (() = none; it must be safe to rerun).
              The local runner runs it before the job's GPUs are bound, so staging inputs does not hold a GPU; a
              scheduled job (job_entry, Kubernetes) runs it inside its allocation. A failing prep is the job's exit
              code and `cmd` does not run.
    """

    name: str
    cmd: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    gpus: int = 1
    cpus: int = 8
    mem_gb: int = 64
    time_min: int = 0
    workdir: str = ""
    prep: tuple[str, ...] = ()

    def __post_init__(self):
        if not _NAME_RE.match(self.name):
            raise ValueError(f"job name {self.name!r}: 1-63 characters of letters, digits, '.', '_', '-'")
        if not self.cmd or not all(isinstance(a, str) for a in self.cmd):
            raise ValueError(f"job {self.name}: cmd must be a non-empty list of strings")
        if not all(isinstance(a, str) for a in self.prep):
            raise ValueError(f"job {self.name}: prep must be a list of strings")
        if min(self.gpus, self.cpus, self.mem_gb, self.time_min) < 0:
            raise ValueError(f"job {self.name}: negative resource request")
        object.__setattr__(self, "cmd", tuple(self.cmd))
        object.__setattr__(self, "prep", tuple(self.prep))
        object.__setattr__(self, "env", {str(k): str(v) for k, v in self.env.items()})


def job_from_dict(d: dict) -> Job:
    """A Job from its stored dict (unknown keys raise)."""
    known = {f.name for f in dataclasses.fields(Job)}
    if extra := sorted(set(d) - known):
        raise ValueError(f"job {d.get('name')!r}: unknown keys {extra}")
    return Job(**{**d, "cmd": tuple(d["cmd"]), "prep": tuple(d.get("prep", ()))})


def job_to_dict(job: Job) -> dict:
    """The stored dict of a Job (an empty prep is left out, so lists without one keep their stored form)."""
    d = {**dataclasses.asdict(job), "cmd": list(job.cmd), "prep": list(job.prep)}
    if not job.prep:
        del d["prep"]
    return d


def check_unique(jobs: list[Job]) -> list[Job]:
    """`jobs` unchanged, or ValueError naming a duplicated job name."""
    seen: set[str] = set()
    for j in jobs:
        if j.name in seen:
            raise ValueError(f"duplicated job name {j.name!r}")
        seen.add(j.name)
    return jobs


def save_jobs(path: str | os.PathLike, jobs: list[Job]) -> Path:
    """Write a job list as JSON Lines (atomically) and return its path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    tmp.write_text(
        "".join(json.dumps(job_to_dict(j), sort_keys=True) + "\n" for j in check_unique(list(jobs))), encoding="utf-8"
    )
    os.replace(tmp, p)
    return p


def load_jobs(path: str | os.PathLike) -> list[Job]:
    """Read a JSON Lines job list (blank lines are skipped)."""
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(job_from_dict(json.loads(line)))
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    return check_unique(out)
