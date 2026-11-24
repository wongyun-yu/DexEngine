#!/usr/bin/env python3
"""Guard prefetch — the judge's mesh guard (judge.search._guard_eval → collision.core.pen_cross_guard) of a flush's
FULL passes evaluated ahead in forked worker processes, then consumed by the judge in candidate order.

Why: the deciding guard is the exact mesh ruler (numpy + cKDTree on the CPU, ~3 s per side guard), and a flush
(judge.exhaust._flush_bq · the anchor batch) guards its FULL passes one after another — one core busy, the machine's
other cores idle. The guard is a pure function of its inputs (z, the q arrays, the clip's fingers · supports · desk box
· idle sides) and of the judge's static rulers, so a fork of the judge at the flush computes the same value; the caller
still takes the values one by one in its own order, so every decision of the sequential loop stays: which candidate is
adopted, each rejection's tally, the stop at the n-th adoption, the memo.

Contract:
  * jobs = [(key, thunk)] in the order the caller will consume them; key = the caller's memo key, compared with the
    caller's `same`. Nothing but the job index and the result crosses the process boundary — the thunks and every
    ruler are the parent's memory at the fork (copy-on-write).
  * take(key) → (True, value): the worker's value, its recorded stdout/stderr written first, in the order the worker
    wrote it (the guard's markers [GUARD_BAND] · [TORSO_GUARD] … land where the sequential run prints them).
    (False, None): not prefetched (no pool, unknown key) or the worker failed — the caller evaluates inline, which
    reproduces the guard's own exception in its own place, or the value. A failure is never silent: marker
    `[GUARD_POOL] ✗` (worker exception · dead or hung pool → the rest of this pool is inline).
  * close() always (judge.guard_prefetch is a with block): results nobody took are dropped unread (a stop at the n-th
    adoption), running workers are terminated, marker `[GUARD_POOL] candidates … taken … dropped …` per pool.
Workers are forked (start method fork): the guard path is numpy · scipy only — a forked child that touched CUDA would
die (torch marks the child), and that is a worker failure above, not a wrong value. Thread counts stay what the parent
was started with (configs/placement/canon_env.sh: OMP · MKL · OPENBLAS = 1).
Size: capacity.plan (step STEP: CPUs, memory, cap 4); env DEXENGINE_GUARD_PROCS (≥ 1 — 1 = no pool, the guard runs
inline) is an explicit option; markers `[GUARD_POOL] workers` and capacity's `[CPU]` once per process.
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from dexengine.core import capacity, stdio
from dexengine.core.errors import InputMissingError

ENV = "DEXENGINE_GUARD_PROCS"
#: capacity.NEEDS key: workers per pool — a machine runs many judge lanes at once; a pool lives for one flush's burst.
STEP = "dexengine.placement.judge.guard_pool"
#: seconds to wait for one worker result before the pool counts as hung (a side guard is ~3 s)
WAIT_S = 600.0
_JOBS: list = []  # the thunks of the pool being forked — inherited by its workers, never pickled
_SAID: dict = {}


def plan(jobs: int | None = None) -> capacity.Plan:
    """The pool plan: capacity.plan for `jobs` guards (None = as many as the cap); env DEXENGINE_GUARD_PROCS (an
    integer ≥ 1, anything else dies) is an explicit option. Markers once per process."""
    raw = os.environ.get(ENV)
    opt = None
    if raw is not None:
        try:
            opt = int(raw)
        except ValueError:
            opt = 0
        if opt < 1:
            raise InputMissingError(f"[GUARD_POOL] {ENV}={raw!r} — must be an integer ≥ 1 (1 = no pool · sequential)")
    p = capacity.plan(STEP, capacity.NEEDS[STEP].cap if jobs is None else jobs, opt)
    if "procs" not in _SAID:
        _SAID["procs"] = n = p.workers
        print(p.line(), file=sys.stderr, flush=True)
        src = "env" if ENV in os.environ else "capacity"
        print(
            f"[GUARD_POOL] workers {n} ({src}) — the guards of a flush's FULL passes are evaluated ahead and "
            "consumed in candidate order" + (" · 1 = no pool (sequential)" if n == 1 else ""),
            flush=True,
        )
    return p


def procs(jobs: int | None = None) -> int:
    """Worker count of a pool (plan)."""
    return plan(jobs).workers


class _Rec:
    """A text stream recording (stream name, text) — a worker's stdout and stderr as one ordered sequence."""

    encoding = "utf-8"
    errors = "strict"

    def __init__(self, name: str, log: list):
        self.name, self.log = name, log

    def write(self, s: str) -> int:
        self.log.append((self.name, s))
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _work(i: int):
    """Worker body: job i with its output recorded. Never raises — (True, value, log, s) | (False, reason, log, s)."""
    log: list = []
    so = stdio.process_stream("stdout", _Rec("stdout", log))  # a forked worker process: process level
    se = stdio.process_stream("stderr", _Rec("stderr", log))
    t = time.perf_counter()
    try:
        return True, _JOBS[i](), log, time.perf_counter() - t
    except BaseException as e:  # noqa: BLE001 — SystemExit too: reported, the parent re-evaluates inline (same failure)
        return False, f"{type(e).__name__}: {e}", log, time.perf_counter() - t
    finally:
        stdio.process_stream("stdout", so)
        stdio.process_stream("stderr", se)


class Prefetch:
    """The guards of `jobs` [(key, thunk)] evaluated ahead by min(procs(), len(jobs)) forked workers (none for fewer
    than two jobs or procs() == 1). same(k1, k2) = the caller's key equality."""

    def __init__(self, jobs: list, same):
        self.keys = [k for k, _ in jobs]
        self.same = same
        self.fut: list = [None] * len(jobs)
        self.ex = None
        self.n = 0
        self.taken = 0
        self.wait_s = 0.0
        self.work_s = 0.0
        self.t0 = time.perf_counter()
        p = plan(len(jobs))
        n = min(p.workers, len(jobs))
        if n < 2:
            return
        if _JOBS:
            raise RuntimeError("[GUARD_POOL] pools overlap — a new pool was opened before the previous one closed")
        _JOBS[:] = [t for _, t in jobs]
        sys.stdout.flush()
        sys.stderr.flush()
        self.n = n
        self.ex = ProcessPoolExecutor(
            n, mp_context=_mp.get_context("fork"), initializer=capacity.worker_init, initargs=p.initargs
        )
        self.fut = [self.ex.submit(_work, i) for i in range(len(jobs))]

    def take(self, key):
        """(True, value) after replaying the worker's output · (False, None) = evaluate inline (see the module doc)."""
        if self.ex is None:
            return False, None
        i = next((j for j, k in enumerate(self.keys) if self.fut[j] is not None and self.same(k, key)), None)
        if i is None:
            return False, None
        f, self.fut[i] = self.fut[i], None
        t = time.perf_counter()
        try:
            ok, val, log, dt = f.result(timeout=WAIT_S)
        except BaseException as e:  # noqa: BLE001 — dead worker · hung pool: the rest of this pool goes inline
            self.wait_s += time.perf_counter() - t
            print(
                f"[GUARD_POOL] ✗ pool broken ({type(e).__name__}: {e}) — the rest of this pool is evaluated "
                "sequentially in the parent",
                flush=True,
            )
            self._stop()
            return False, None
        self.wait_s += time.perf_counter() - t
        if not ok:
            print(
                f"[GUARD_POOL] ✗ worker exception ({val}) — this candidate is re-evaluated in the parent (same "
                "inputs · same place)",
                flush=True,
            )
            return False, None
        for name, s in log:
            getattr(sys, name).write(s)
        sys.stdout.flush()
        sys.stderr.flush()
        self.taken += 1
        self.work_s += dt
        return True, val

    def _stop(self) -> int:
        """Cancel what nobody took, terminate the workers, join. Returns the number of jobs dropped unread."""
        left = sum(f is not None for f in self.fut)
        for f in self.fut:
            if f is not None:
                f.cancel()
        self.fut = [None] * len(self.fut)
        if self.ex is not None:
            # workers still computing guards nobody will read — end them instead of waiting a guard out (private
            # attribute of concurrent.futures; without it shutdown waits for them, same result, later)
            for p in list((getattr(self.ex, "_processes", None) or {}).values()):
                p.terminate()
            self.ex.shutdown(wait=True, cancel_futures=True)
            self.ex = None
        _JOBS.clear()
        return left

    def close(self) -> None:
        if self.ex is None and not self.n:
            return
        left = self._stop()
        print(
            f"[GUARD_POOL] candidates {len(self.keys)} · workers {self.n} · taken {self.taken} · dropped {left} · "
            f"wait {self.wait_s:.1f}s · worker sum {self.work_s:.1f}s · wall {time.perf_counter() - self.t0:.1f}s",
            flush=True,
        )
        self.n = 0
