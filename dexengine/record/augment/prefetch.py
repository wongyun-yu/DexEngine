"""Runs the augment placement generation (gen: augment.place_gen and the bake, CPU heavy) ahead for every clip at job
start, overlapped with the previous clip's Isaac rollout session, which takes the generation share (measured 29-40
s/clip on the v2 jobs) off the critical path. Outputs and order stay: the generation records (variants_<clip>.json)
are written by the same command in the place augment.stage gen always uses, and one_clip waits in clip order for that
record to finish before reading it (an existing record = gen's reuse path).

Canonical (no switch; A/B passed): concurrency = min(WIDTH, CPU ÷ processes per generation). One MARK line
(concurrency · clips).
Failure: the exception of a generation run ahead is raised again at that clip's turn (no silent pass) · when the job
dies the remaining generations are cancelled.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor

from dexengine.core import capacity

# the run-ahead pool (capacity.NEEDS key; cap 2 = one generation is the gen processes of pool_sizes, the share that
# overlaps the previous rollout is the next one or two clips)
STEP = "dexengine.record.augment.prefetch"
MARK = "[AUG_PREFETCH]"
#: the job's lane count env (written by every record job) and the
#: post gate / position generation steps whose caps and memory statements capacity.NEEDS holds (records:
#: capacity_peaks.tsv). More workers than the job's CPU share are not faster and only add RAM. Outputs do not depend
#: on the pool size (post gate post.json and generation variants are equal except sec values, measured on the same
#: clip).
LANES_ENV = "COLLECT_JOB_LANES"
POST_STEP, GEN_STEP = "dexengine.record.augment.coll_gate", "dexengine.record.augment.place_gen"  # caps: NEEDS
POOL_MARK = "[POOL_SIZE]"
_POOL: ThreadPoolExecutor | None = None
_FUT: dict[str, Future] = {}


def cpus() -> int:
    """CPUs this process may use (capacity.cpu_limit: the affinity, capped by cpu.max of its cgroup chain)."""
    return capacity.cpu_limit()[0]


@functools.cache
def gpus_seen() -> int:
    """GPUs this job sees (nvidia-smi; a cluster allocation shows only its own GPU, a host all). Unreadable dies."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, timeout=60
    )
    n = len(r.stdout.split())
    if r.returncode != 0 or n < 1:
        raise SystemExit(f"{POOL_MARK} ✗ nvidia-smi rc {r.returncode} · GPU {n}: cannot count the share")
    return n


def sharers(lanes: int) -> int:
    """Record jobs sharing the CPU set this job sees: a job that sees one GPU (cluster allocation) = that job's lane
    count · a job that sees several (hosts · nodes) = GPUs x data_layout.COLLECT_SESSIONS_PER_GPU (home of the
    host seat count)."""
    g = gpus_seen()
    from dexengine.core import paths as P

    return lanes if g == 1 else g * P.data_layout.COLLECT_SESSIONS_PER_GPU


def pool_sizes(environ=None) -> dict:
    """{cpu, lanes, sharers, share, post, gen}: share = cpus() ÷ sharers · post gate = share ÷ 2 · generation = share
    (values above the cap · floor 1). A lane count that is not an integer >= 1 dies."""
    raw = (os.environ if environ is None else environ).get(LANES_ENV, "1")
    if not raw.isdigit() or int(raw) < 1:
        raise SystemExit(f"{POOL_MARK} ✗ {LANES_ENV}={raw!r}: an integer >= 1")
    c, lanes = cpus(), int(raw)
    n = sharers(lanes)
    share = max(1, c // n)
    return {
        "cpu": c,
        "lanes": lanes,
        "sharers": n,
        "share": share,
        "post": max(1, min(capacity.NEEDS[POST_STEP].cap, share // 2)),
        "gen": max(1, min(capacity.NEEDS[GEN_STEP].cap, share)),
    }


def pool_line(environ=None) -> str:
    z = pool_sizes(environ)
    return " ".join([POOL_MARK, *(f"{k}={z[k]}" for k in ("cpu", "lanes", "sharers", "share", "post", "gen"))])


def start(jobs: list[tuple], fn, per_job: int) -> int:
    """jobs = [(key, argument tuple)]: runs fn(*arguments) ahead in order -> concurrency (no work = 0). One generation
    already uses per_job processes, so concurrency = min(WIDTH, lane share CPU // per_job), written in the marker (no
    CPU contention with rollouts)."""
    global _POOL
    if not jobs:
        return 0
    c = pool_sizes()["share"]
    p = capacity.plan(STEP, len(jobs), share=max(1, c // max(1, per_job)))  # cap WIDTH (NEEDS)
    n = p.workers
    print(p.line(), file=sys.stderr, flush=True)
    _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="aug_gen")
    for key, args in jobs:
        _FUT[key] = _POOL.submit(fn, *args)
    print(
        f"{MARK} concurrent {n} (the [CPU] line · CPU {c} ÷ per generation {per_job}) · clips {len(jobs)}: position "
        "generation overlaps the rollouts",
        flush=True,
    )
    return n


def take(key: str):
    """The run-ahead result of that key (waits until done; exceptions are raised again) · None when not run ahead."""
    f = _FUT.pop(key, None)
    return None if f is None else f.result()


def stop() -> None:
    """Cancels the remaining run-ahead work and closes (when the job ends or dies). Generations already running (at
    most the concurrency) are waited for; one generation's cap is augment.subrun.GenBound (work size, bundle deadline;
    past it the child's whole process group is reaped)."""
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True, cancel_futures=True)
    _POOL = None
    _FUT.clear()
