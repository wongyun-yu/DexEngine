"""How many workers and threads a step runs with — the one owner (every process or thread pool in dexengine sizes
itself here; tests/core/test_capacity.py checks that structurally).

Machine (dexengine.core.machine; read when a step sizes its pool, never cached across machines):
  CPUs   = the process's affinity, capped by cgroup v2 ``cpu.max`` (quota / period) of its cgroup and every ancestor.
  memory = the smallest of the host's MemAvailable, cgroup v2 ``memory.max`` of its cgroup and every ancestor (v1:
           memory.limit_in_bytes), the Slurm job's request (SLURM_MEM_PER_NODE / _PER_CPU) and DEXENGINE_MEM_GIB.
  Without a limit the host value stands (a pod, a systemd scope and a plain host read the same way). A cgroup this
  process cannot read (a container without the cgroup filesystem) with no Slurm or env value is ``hidden``: a
  process pool refuses to plan then (it would size itself from the whole host and be killed by the hidden limit).
Statement (NEEDS, one per step): the calling process's own peak (``parent_gib``), one worker's peak (``per_job_gib``)
and the most workers that help (``cap`` — the step's old fixed count, so a large machine runs as before). Measured
statements name their records (capacity_peaks.tsv, checked by the test against the statement); a step without a
measurement says so (``measured=""``) and is listed in the marker as unmeasured.
Plan: workers = max(1, min(CPUs, (memory x SAFETY - parent) // per_job, jobs, cap)); an explicit option (--workers /
--jobs) replaces that value here and only here (the marker shows both). Threads per worker = 1: a pool already
spreads over the CPUs, and BLAS / OpenMP reductions may change their bytes with the thread count — so outputs do not
depend on the machine. A caller who sets a thread variable keeps it (source=env).
For a process pool the thread variables are set while the pool is open (THREAD_ENV — PXR_WORK_THREAD_LIMIT caps the
USD TBB pool, which otherwise starts one thread per CPU and ignores OMP) and, for libraries a forked worker inherits
already loaded, applied in every worker (``worker_init`` — threadpoolctl, torch, pxr Work); afterwards the calling
process has its own values back. A thread or io pool shares the process's library pools and changes no thread
setting (threads=-(process-wide)); it only gets its size and, when its jobs import native modules, the preload.
Marker: ``[CPU] step=… usable=… mem=… parent=… per_job=… workers=… threads=… source=…`` when the pool starts and the
same line with ``peak_parent=… peak_job=…`` (resource.getrusage — RSS of this process and of its largest reaped
child) when it ends; a run without the marker did not size its pool here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import resource
import sys
from pathlib import Path

from dexengine.core.machine import (  # noqa: F401 — the readers of the machine, used and re-exported here
    CGROUP,
    MEM_ENV,
    TAG,
    cgroup_dirs,
    cpu_limit,
    mem_limit,
)

SAFETY = 0.85  # share of the memory limit a pool may plan for (page cache, the parent's growth, other processes)
#: a measured statement is at least MARGIN x the largest value its records hold (parent and per job): the samples are
#: a few datasets, a heavier input passes them (a 2.8 GiB statement from one-tree samples met 4.74 GiB per worker)
MARGIN = 1.15
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PXR_WORK_THREAD_LIMIT",
)
THREADS = 1
_SAID: set = set()  # steps whose marker this process printed


@dataclasses.dataclass(frozen=True)
class Need:
    """One step's memory statement [GiB] and its worker cap; `measured` names the records (capacity_peaks.tsv)."""

    parent_gib: float
    per_job_gib: float
    cap: int
    measured: str = ""
    kind: str = "process"  # process (or threads that each run a subprocess) | thread (shares this process's memory)
    # | io (network or disk waits: neither CPUs nor memory bound it)
    preload: tuple[str, ...] = ()  # modules a thread pool's jobs import: loaded here, in the main thread, first


def unmeasured(cap: int, kind: str = "process", preload: tuple[str, ...] = ()) -> Need:
    """The statement of a step without records (1 GiB parent, 2 GiB a worker — named unmeasured in the marker)."""
    return Need(1.0, 2.0, cap, "", kind, preload)


#: step (the module that owns the pool, ":part" for a second pool in it) -> statement. Measured values name their
#: records (capacity_peaks.tsv); `cap` is the step's old fixed count. A pool whose step is missing here dies.
NEEDS: dict[str, Need] = {
    "dexengine.data.base_adapter": Need(0.5, 4.8, 8, measured="stage2"),
    "dexengine.placement.repair.wrist_desk_stage": Need(2.0, 3.6, 16, measured="stage2e"),
    "dexengine.data.contact.sidecar": unmeasured(8),
    "dexengine.data.cases.taxonomy": unmeasured(8),
    "dexengine.data.support.gap": unmeasured(8),
    "dexengine.data.support.oakink2": unmeasured(6),
    "dexengine.data.oakink2.required": unmeasured(16),
    "dexengine.data.stage1.clip_scope": unmeasured(8),
    "dexengine.data.handoff.checks": unmeasured(8),
    "dexengine.data.artic.spec_collect": unmeasured(4),
    "dexengine.data.readers.raw.hocap": unmeasured(8),
    "dexengine.data.readers.raw.dexcanvas": unmeasured(2),
    "dexengine.data.readers.taco_side_assign": unmeasured(20),
    "dexengine.data.contact.fingerprint": unmeasured(8),
    "dexengine.data.objects.build": unmeasured(1),
    "dexengine.data.objects.decompose": unmeasured(1),
    "dexengine.data.read_raw.stage:read": unmeasured(8),
    "dexengine.data.read_raw.stage:decomp": unmeasured(8),
    "dexengine.data.read_raw.stage:contact": unmeasured(8),
    "dexengine.retarget.phases.common": unmeasured(8),
    "dexengine.rl.launch.train_scene": unmeasured(4),
    "dexengine.placement.gate.core": unmeasured(6),
    "dexengine.placement.train_ready.trees": unmeasured(8),
    "dexengine.placement.train_ready.judge_fp": unmeasured(8),
    "dexengine.placement.judge.guard_pool": unmeasured(4),
    "dexengine.record.augment.coll_gate": Need(2.7, 1.9, 16, measured="coll_gate_post_sim"),
    "dexengine.record.augment.place_gen": Need(2.0, 2.3, 6, measured="generation_sim"),
    "dexengine.record.augment.prefetch": unmeasured(2, "thread"),
    "dexengine.record.text.sources": unmeasured(8, "process"),
    "dexengine.record.text.api": unmeasured(8, "io"),
    "dexengine.record.looks.augvis.assets": unmeasured(8, "io"),
    "dexengine.record.store.merge_check": unmeasured(4, "process"),
    "dexengine.record.store.lerobot:prewarm": unmeasured(6, "process"),
    "dexengine.record.store.lerobot:stats": unmeasured(16, "process"),
    "dexengine.placement.bundle.rl_bundle": unmeasured(8, "thread"),
    "dexengine.rl.render.replay:close": unmeasured(16, "thread"),
    "dexengine.rl.render.replay:probe": unmeasured(8, "process"),
    "dexengine.retarget.run": unmeasured(8),
    "dexengine.record.augment.bake_run": unmeasured(8),
    "dexengine.record.augment.recv_clearance": unmeasured(4),
    "dexengine.record.scene.render_only_build": unmeasured(1),
    "dexengine.retarget.f0.shadow_gpu": unmeasured(16, "thread"),
}


@dataclasses.dataclass(frozen=True)
class Plan:
    step: str
    workers: int
    threads: int
    cpus: int
    cpu_src: str
    mem_gib: float
    mem_src: str
    need: Need
    source: str  # owner | option
    owner_workers: int
    threads_src: str  # owner | env | process-wide (a thread or io pool: its jobs use the process's library pools)

    @property
    def initargs(self) -> tuple:
        return (self.threads,)

    def line(self, end: bool = False) -> str:
        n = self.need
        meas = n.measured or "unmeasured"
        s = (
            f"{TAG} step={self.step} usable={self.cpus}({self.cpu_src}) mem={self.mem_gib:.1f}GiB({self.mem_src}) "
            f"parent={n.parent_gib}GiB per_job={n.per_job_gib}GiB({meas}) cap={n.cap} workers={self.workers} "
            f"threads={self.threads or '-'}({self.threads_src}) source={self.source}"
        )
        if self.source != "owner":
            s += f" owner_workers={self.owner_workers}"
        if (u := usd_threads()) is not None:  # USD's TBB pool reads its limit once, when pxr is first imported
            s += f" usd_threads={u}" + ("(loaded before the cap)" if self.threads and u > self.threads else "")
        if end:
            me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            kid = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
            s += f" peak_parent={me:.2f}GiB peak_job={kid:.2f}GiB"
        return s


def plan(
    step: str, jobs: int, option: int | None = None, *, share: int | None = None, dirs: list[Path] | None = None
) -> Plan:
    """The pool size of `step` for `jobs` work items (module doc). `option` = the caller's explicit --workers/--jobs
    (replaces the owner's value); `share` = a CPU budget a caller hands down (a lane's share of the machine: it caps
    the CPUs, the memory statement still applies)."""
    need = NEEDS.get(step)
    if need is None:
        raise SystemExit(f"{TAG} ✗ step {step!r} has no memory statement (capacity.NEEDS): cannot size its pool")
    cpus, csrc = cpu_limit(dirs)
    if share is not None and 1 <= int(share) < cpus:
        cpus, csrc = int(share), "share"
    mem, msrc = mem_limit(dirs)
    if msrc == "hidden" and need.kind == "process":
        raise SystemExit(
            f"{TAG} ✗ step {step}: the memory limit of this process is unknown — its cgroup "
            f"({cgroup_dirs()[:1]}) is not readable here (a container without the cgroup filesystem) or is a Slurm "
            f"job's without a finite memory.max — so a process pool cannot be sized from the host's memory. Pass the "
            f"Slurm job variables (singularity --cleanenv drops them: SINGULARITYENV_SLURM_JOB_ID, "
            f"SINGULARITYENV_SLURM_MEM_PER_NODE) or state the budget: {MEM_ENV}=<GiB>."
        )
    gib = mem / 2**30
    by_mem = int((gib * SAFETY - need.parent_gib) // need.per_job_gib) if need.kind == "process" else need.cap
    by_cpu = need.cap if need.kind == "io" else cpus
    own = max(1, min(by_cpu, by_mem, max(1, jobs), need.cap))
    if option is not None and int(option) < 1:
        raise SystemExit(f"{TAG} ✗ step {step}: workers option {option} < 1")
    workers, source = (own, "owner") if option is None else (int(option), "option")
    n, tsrc = threads() if need.kind == "process" else (0, "process-wide")  # threads share the process's pools
    return Plan(step, workers, n, cpus, csrc, gib, msrc, need, source, own, tsrc)


def threads(environ=None) -> tuple[int, str]:
    """(threads per worker, source): the caller's OMP_NUM_THREADS when set (env), else THREADS (owner)."""
    v = (os.environ if environ is None else environ).get("OMP_NUM_THREADS", "")
    return (int(v), "env") if v.isdigit() and int(v) >= 1 else (THREADS, "owner")


def child_env(n: int, environ=None) -> dict:
    """The thread variables a child process gets (a variable the caller set is kept)."""
    env = dict(os.environ if environ is None else environ)
    for k in THREAD_ENV:
        env.setdefault(k, str(n))
    return env


def limit_here(n: int) -> list[str]:
    """Cap the thread pools of libraries already loaded in this process at `n` -> what was capped."""
    done = []
    with contextlib.suppress(ImportError):
        from threadpoolctl import threadpool_limits  # noqa: PLC0415

        threadpool_limits(limits=n)
        done.append("threadpoolctl")
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(n)
        done.append("torch")
    if "pxr" in sys.modules:
        with contextlib.suppress(ImportError):
            from pxr import Work  # noqa: PLC0415

            Work.SetConcurrencyLimit(n)
            done.append("pxr")
    return done


def _module_name(m: str) -> str:
    if m.startswith("registry:"):
        from dexengine.core.locations.modules import _REGISTRY_MODULES  # noqa: PLC0415

        return _REGISTRY_MODULES[m.split(":", 1)[1]]
    return m


def preload(mods: tuple[str, ...]) -> None:
    """Import `mods` in this (the main) thread, before a pool whose jobs import them starts."""
    import importlib  # noqa: PLC0415

    for m in mods:
        importlib.import_module(_module_name(m))


def check_preloaded(mods: tuple[str, ...]) -> None:
    """Dies unless every module of `mods` is loaded (sys.modules) — the pool must not be the first to import one."""
    miss = [m for m in mods if _module_name(m) not in sys.modules]
    if miss:
        raise SystemExit(f"{TAG} ✗ pool opened before its jobs' native modules were loaded in the main thread: {miss}")


def usd_threads() -> int | None:
    """USD's work concurrency in this process (None = pxr not loaded). It follows PXR_WORK_THREAD_LIMIT read at the
    first import of pxr; a later SetConcurrencyLimit does not lower a pool already running (measured, usd-core here)."""
    if "pxr" not in sys.modules:
        return None
    from pxr import Work  # noqa: PLC0415

    return int(Work.GetConcurrencyLimit())


_WORKER: int | None = None  # this process's threads when it is a process-pool worker (set by worker_init)


def worker_init(n: int) -> None:
    """Pool initializer: the same caps in every worker (a forked worker inherits loaded libraries, not new env)."""
    global _WORKER  # noqa: PLW0603 — one value per process, set once when the worker starts
    _WORKER = n
    limit_here(n)


def apply(p: Plan) -> dict:
    """Process pool: the thread variables for its workers and subprocesses (kept when the caller set them), set while
    the pool is open -> the previous values (``restore``). A thread or io pool changes nothing: its jobs share this
    process's library pools, as before this owner."""
    if p.need.kind != "process":
        return {}
    old = {k: os.environ.get(k) for k in THREAD_ENV}
    for k, v in child_env(p.threads).items():
        if k in THREAD_ENV:
            os.environ[k] = v
    return old


def restore(old: dict) -> None:
    """Put back the thread variables ``apply`` replaced (the process after its pool runs as it did before)."""
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@contextlib.contextmanager
def sized(step: str, jobs: int, option: int | None = None, *, share: int | None = None):
    """``with capacity.sized(STEP, len(jobs), a.workers) as cap, Pool(cap.workers, worker_init, cap.initargs):`` —
    plans, applies, prints the start marker, and the end marker with the peaks once the pool is gone."""
    p = plan(step, jobs, option, share=share)
    old = apply(p)
    preload(p.need.preload)
    check_preloaded(p.need.preload)
    first = step not in _SAID  # a pool opened again and again (per flush, per episode) marks its first opening only
    _SAID.add(step)
    if first:
        print(p.line(), file=sys.stderr, flush=True)  # stderr: a step's stdout may be a table or JSON
    try:
        yield p
    finally:
        restore(old)
        if first:
            print(p.line(end=True), file=sys.stderr, flush=True)
        if p.step not in _OVER and (bad := oracle(p)):
            _OVER.add(p.step)
            print(bad, file=sys.stderr, flush=True)


_OVER: set[str] = set()


def oracle(p: Plan) -> str | None:
    """Statement against measurement (an allowed duplicate: the statement lives in NEEDS, this reads it): the line to
    print when a process pool's largest reaped worker used more than its per-job statement, else None.

    A warning, not a stop: when the pool closes the step's outputs are complete and correct, and RUSAGE_CHILDREN's
    ru_maxrss is an upper bound (pages shared with the parent count; any child this process reaped before counts) — a
    stop would discard a good run on a noisy measure. The fix is the statement and its record (capacity_peaks.tsv)."""
    if p.need.kind != "process":
        return None
    kid = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    if kid <= p.need.per_job_gib:
        return None
    return (
        f"{TAG} ✗ peak_job={kid:.2f}GiB > per_job={p.need.per_job_gib}GiB step={p.step} "
        f"({p.need.measured or 'unmeasured'}) — a worker used more than its statement; a machine sized by it would "
        f"kill the pool. Raise capacity.NEEDS[{p.step!r}] and record the measurement in capacity_peaks.tsv."
    )


def threads_here() -> int:
    """Threads a library call may start in this process (scipy cKDTree ``workers=`` and the like; its result does not
    depend on it): a process-pool worker's threads inside a worker, else the usable CPUs (cgroup cpu.max, where
    ``workers=-1`` took every host CPU). OMP_NUM_THREADS is a BLAS pin (configs/placement/canon_env.sh sets 1 for the
    placement drivers) and is not read here."""
    return _WORKER if _WORKER is not None else cpu_limit()[0]


def main(argv: list[str] | None = None) -> int:
    """``python -m dexengine.core.capacity cpus`` — the usable CPUs (for shell callers; the one CPU reader)."""
    args = sys.argv[1:] if argv is None else argv
    if args != ["cpus"]:
        print("usage: python -m dexengine.core.capacity cpus", file=sys.stderr)
        return 2
    print(cpu_limit()[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
