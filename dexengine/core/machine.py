"""The machine a process runs on, as the pool owner (dexengine.core.capacity) reads it: usable CPUs and memory.

CPUs   = the process's affinity, capped by the cgroup CPU quota of its chain (v2 cpu.max, v1 cpu.cfs_quota_us /
         cpu.cfs_period_us); a fractional quota rounds up (cpu_limit doc). Never the host core count.
memory = the smallest of the host's MemAvailable, the cgroup memory limit of its chain (v2 memory.max, v1
         memory.limit_in_bytes), the Slurm job's request and DEXENGINE_MEM_GIB; ``hidden`` when the limit cannot be read
         (mem_limit doc).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

TAG = "[CPU]"
CGROUP = Path("/sys/fs/cgroup")
MEM_ENV = "DEXENGINE_MEM_GIB"  # an explicit memory budget [GiB]: the only way past a cgroup this process cannot read


def cgroup_dirs(proc_cgroup: str = "/proc/self/cgroup", root: Path = CGROUP) -> list[Path]:
    """This process's cgroup v2 directory and every ancestor up to the mount root (empty without cgroup v2)."""
    try:
        rel = next(ln.split(":", 2)[2] for ln in Path(proc_cgroup).read_text().splitlines() if ln.startswith("0::"))
    except (OSError, StopIteration):
        return []
    d = root / rel.strip().lstrip("/")
    out = []
    while True:
        out.append(d)
        if d == root:
            return out
        d = d.parent


def _v1_cpu_quota(proc_cgroup: str, root: Path) -> float | None:
    """The smallest cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us of the cpu chain (-1 = none), or None."""
    try:
        rel = next(
            ln.split(":", 2)[2]
            for ln in Path(proc_cgroup).read_text().splitlines()
            if "cpu" in ln.split(":", 2)[1].split(",")
        )
    except (OSError, StopIteration, IndexError):
        return None
    top = next((root / m for m in ("cpu", "cpu,cpuacct", "cpuacct,cpu") if (root / m).is_dir()), None)
    if top is None:
        return None
    d, best = top / rel.strip().lstrip("/"), None
    while True:
        try:
            q, per = (int((d / f"cpu.cfs_{k}_us").read_text()) for k in ("quota", "period"))
            if q > 0 and per > 0:
                best = q / per if best is None else min(best, q / per)
        except (OSError, ValueError):
            pass
        if d == top or d == d.parent:
            return best
        d = d.parent


def cpu_limit(
    dirs: list[Path] | None = None, *, proc_cgroup: str = "/proc/self/cgroup", root: Path = CGROUP
) -> tuple[int, str]:
    """(CPUs, source): the affinity (never the host core count: a pod or a pinned session has fewer), capped by the
    smallest cgroup CPU quota of the process's chain — v2 cpu.max 'quota period', v1 cpu.cfs_quota_us /
    cpu.cfs_period_us ('max' / -1 = none). A fractional quota rounds UP (7.5 -> 8): the choice for batch throughput
    (all of the quota is used; rounding down leaves 0.5 CPU idle); a 7.5-CPU measurement could not tell them apart
    (noise +-1.2 s larger than the difference). At least 1."""
    n, src = len(os.sched_getaffinity(0)), "affinity"
    quotas = []
    for d in cgroup_dirs(proc_cgroup, root) if dirs is None else dirs:
        try:
            q, per = (d / "cpu.max").read_text().split()
        except (OSError, ValueError):
            continue
        if q != "max" and int(per) > 0:
            quotas.append((int(q) / int(per), "cpu.max"))
    if dirs is None and (v1 := _v1_cpu_quota(proc_cgroup, root)) is not None:
        quotas.append((v1, "cpu.v1"))
    for q, qsrc in quotas:
        if max(1, math.ceil(q)) < n:
            n, src = max(1, math.ceil(q)), qsrc
    return n, src


def _host_available(meminfo: str = "/proc/meminfo") -> int:
    for ln in Path(meminfo).read_text().splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) * 1024
    raise SystemExit(f"{TAG} ✗ {meminfo} has no MemAvailable: cannot size a pool")


def _v1_memory(proc_cgroup: str, root: Path) -> int | None:
    """The smallest cgroup v1 memory.limit_in_bytes of the memory controller's chain (a v1 or hybrid host), or None."""
    try:
        rel = next(
            ln.split(":", 2)[2]
            for ln in Path(proc_cgroup).read_text().splitlines()
            if "memory" in ln.split(":", 2)[1].split(",")
        )
    except (OSError, StopIteration, IndexError):
        return None
    top = root / "memory"
    d, best = top / rel.strip().lstrip("/"), None
    while True:
        try:
            v = int((d / "memory.limit_in_bytes").read_text())
            best = v if best is None else min(best, v)
        except (OSError, ValueError):
            pass
        if d == top or d == d.parent:
            return best
        d = d.parent


def _slurm_memory(environ) -> int | None:
    """The job's memory [bytes] from Slurm's environment (MB values): SLURM_MEM_PER_NODE, or SLURM_MEM_PER_CPU x the
    task's CPUs (SLURM_CPUS_PER_TASK, else SLURM_CPUS_ON_NODE); None outside a Slurm job or without a memory request."""
    if not environ.get("SLURM_JOB_ID"):
        return None
    if (v := environ.get("SLURM_MEM_PER_NODE", "")).isdigit():
        return int(v) * 2**20
    per, cpus = (
        environ.get("SLURM_MEM_PER_CPU", ""),
        environ.get("SLURM_CPUS_PER_TASK") or environ.get("SLURM_CPUS_ON_NODE", ""),
    )
    return int(per) * int(cpus) * 2**20 if per.isdigit() and cpus.isdigit() else None


def mem_limit(
    dirs: list[Path] | None = None,
    meminfo: str = "/proc/meminfo",
    *,
    environ=None,
    proc_cgroup: str = "/proc/self/cgroup",
    root: Path = CGROUP,
) -> tuple[int, str]:
    """(bytes, source): the smallest of the host's MemAvailable (host), the cgroup v2 memory.max of the process's
    chain (memory.max) or v1 memory.limit_in_bytes (memory.v1), the Slurm job's request (slurm) and an explicit budget
    MEM_ENV (env). A process whose cgroup path is set but not readable here (a container without the cgroup
    filesystem, e.g. singularity), or that runs in a Slurm job's cgroup with no finite memory.max, and that has no
    Slurm or env value gets source ``hidden``: its limit is unknown, and a process pool refuses to plan on the host's
    memory (``plan``)."""
    if environ is None:
        from dexengine.core import config  # noqa: PLC0415 — the registry of process-boundary variables

        environ = {**os.environ, MEM_ENV: config.env(MEM_ENV)}
    dirs = cgroup_dirs(proc_cgroup, root) if dirs is None else dirs
    b, src = _host_available(meminfo), "host"
    seen = not dirs or dirs[0].is_dir()  # no v2 path (no limit to read) or the v2 directory is visible
    # A Slurm job's cgroup (slurmstepd's job_<id>): its budget is the job's request, which a site may keep out of
    # memory.max (measured on one cluster: memory.max = max at every level of a --mem=6G job) — known only from
    # Slurm's variables, a finite memory.max or MEM_ENV.
    slurm_job = any(d.name.startswith("job_") and d.name[4:].isdigit() and "slurmstepd" in d.parent.name for d in dirs)
    seen = seen and not slurm_job
    for d in dirs:
        try:
            raw = (d / "memory.max").read_text().strip()
        except OSError:
            continue
        if raw != "max":
            seen = True
            if int(raw) < b:
                b, src = int(raw), "memory.max"
    if not seen and (v1 := _v1_memory(proc_cgroup, root)) is not None:
        seen = True
        if v1 < b:
            b, src = v1, "memory.v1"
    if (sl := _slurm_memory(environ)) is not None:
        seen = True
        if sl < b:
            b, src = sl, "slurm"
    if (budget := _env_budget(environ)) is not None:
        seen = True
        if budget < b:
            b, src = budget, "env"
    return b, (src if seen else "hidden")


def _env_budget(environ) -> int | None:
    """MEM_ENV in bytes (None when unset); not a positive number of GiB dies."""
    raw = environ.get(MEM_ENV, "")
    if not raw:
        return None
    try:
        gib = float(raw)
    except ValueError:
        gib = 0.0
    if gib <= 0:
        raise SystemExit(f"{TAG} ✗ {MEM_ENV}={raw!r} is not a positive number of GiB")
    return int(gib * 2**30)
