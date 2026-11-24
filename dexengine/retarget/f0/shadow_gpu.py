"""f0 shadow GPU pass: the worker environment and the driver that feeds prepared clips to `gpu.py --worker` processes
and collects their results (run_gpu)."""

from __future__ import annotations

import json
import os
import queue as _queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dexengine.core import blas_pin, capacity
from dexengine.core.repo import REPO_ROOT as _REPO
from dexengine.retarget.f0.shadow_tree import _guard_prep, finish

REPO = _REPO


def worker_env() -> dict:
    """Environment of the GPU worker processes (`gpu.py --worker`); this function is its only owner. One thread, pinned
    BLAS kernel.

    These two pins affect the output even with the GPU engine: workers run the initial n_pen check, the final
    feasibility verdict (`geom["n_pen"]` = repair._n_pen_strict, the FINAL rows of gpu `GpuEngine.round`), strict
    verification (strict_overlap), contact links, wrist rotation matmuls and blend writes in host numpy, and those
    matmuls use BLAS."""
    # ONE THREAD: multithreaded BLAS may change the reduction order with the thread count, and several workers waking
    # threads oversubscribe the machine (the output stays identical with one thread per worker).
    # GPU workers do the same host arithmetic, so the pin stays (the cross-host identity was measured under this pin).
    env = dict(os.environ)
    env.update(dict.fromkeys(capacity.THREAD_ENV, "1"))  # forced, not the caller's: bytes depend on it
    # SAME BLAS KERNEL ON EVERY MACHINE. OpenBLAS DYNAMIC_ARCH picks a different kernel per CPU (different CPU
    # models), which splits trajectory_kinematic.npz by 1 ULP: repair decisions and n_evals agree, but the
    # byte-identity gate fails. Pinning the core type gives 50/50 identical output on both (measured) at no speed cost.
    # Owned by this worker env alone (next to the thread pin).
    blas_pin.pin_env(env)  # the kernel's owner is blas_pin; the worker's apply proves what loaded
    return env


# GPU engine (the only engine)
# GPU workers (`gpu.py --worker`) run many clips in one process from the argv built by prep (repair.main_steps
# generators, which own the control flow and the output text). Only the objective and NM are batched on the GPU (see the
# gpu module docstring). Classification (finish), markers and the ledger are functions here.
# Markers: one `[F0_ENGINE] engine=gpu …` line plus `[F0_GPU_WORKER] …` per worker; without them the run is invalid.
STEP = "dexengine.retarget.f0.shadow_gpu"  # capacity.NEEDS key (the prep thread pool)
GPU_PROTO = "@@F0GPU@@ "  # worker stdout protocol prefix (must equal gpu.PROTO; the worker checks)


def _prep_all(a, clips, dst_tree, dst_wrist_dirs, stage_src, wrist_dirs):
    """Run prep for every clip in threads (capacity sizes the pool and first loads, in this thread, the native modules
    prep imports — torch, pxr; two threads importing them for the first time deadlocked); returns the GPU jobs by clip
    and the rows of clips that finished in prep (cached or failed)."""
    jobs, results = {}, []
    with capacity.sized(STEP, len(clips), a.prep_threads) as cap, ThreadPoolExecutor(cap.workers) as ex:
        for c, jb in zip(
            clips,
            ex.map(lambda c: _guard_prep(c, a, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs), clips),
            strict=True,
        ):
            if jb.get("job"):
                jobs[c] = jb
            else:
                results.append(jb)
    return jobs, results


def _check_worker_python(a, env):
    """Fail within seconds when the worker interpreter cannot import warp (otherwise every clip would end as
    FAIL_GPU_WORKER with the cause hidden)."""
    # if the worker interpreter cannot import warp, every clip would just become FAIL_GPU_WORKER with the cause hidden;
    # fail loudly within ~10 s instead
    pre = subprocess.run(
        [a.python, "-c", "import warp, numpy; print(warp.__version__)"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if pre.returncode != 0:
        raise SystemExit(
            f"[f0_prejudge] GPU engine: worker interpreter {a.python} cannot import warp — "
            f"{(pre.stderr or pre.stdout).strip().splitlines()[-1:]}"
        )


def _start_workers(a, devs, env, nw):
    """Start the GPU worker processes (one per device slot), each with its stdin writer thread."""
    workers = []
    for wi in range(nw):
        e = dict(env)
        e["CUDA_VISIBLE_DEVICES"] = devs[wi % len(devs)]
        p = subprocess.Popen(
            [
                a.python,
                str(REPO / "dexengine/retarget/f0/gpu.py"),
                "--worker",
                "--max-inst",
                str(a.gpu_max_inst),
                "--proto",
                GPU_PROTO.strip(),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=e,
            text=True,
            bufsize=1,
        )
        # Each worker's stdin is written by a dedicated thread. If the main thread wrote the first feed (ahead, 600
        # lines) with blocking writes while not reading the worker's stdout, then once the worker's output filled the 64
        # KB pipe the worker would block on stdout and the parent on stdin (deadlock). The main thread only reads and
        # hands writes to a queue, so reads continue while a write blocks and no circular wait can occur for any output
        # or feed size.
        w = {"p": p, "out": set(), "dev": e["CUDA_VISIBLE_DEVICES"], "buf": "", "q": _queue.Queue()}
        w["wt"] = threading.Thread(target=_stdin_writer, args=(w,), daemon=True)
        w["wt"].start()
        workers.append(w)
    return workers


def _send(w, c, jobs):
    """Queue one clip's job line for a worker and count it as outstanding there."""
    w["out"].add(c)
    w["q"].put(json.dumps({"clip": c, "argv": jobs[c]["argv"]}, ensure_ascii=False) + "\n")


def _feed(w, queue, jobs, a):
    """Top a worker up to --gpu-ahead outstanding clips; once the queue is empty, close its stdin (worker EOF)."""
    while queue and len(w["out"]) < int(a.gpu_ahead):
        _send(w, queue.pop(0), jobs)
    if not queue and not w.get("closed"):
        w["q"].put(None)  # the writer thread closes stdin (worker EOF)
        w["closed"] = True


def run_gpu(clips, a, stage_src, wrist_dirs, dst_tree, dst_wrist_dirs):
    """prep (mirroring) runs here in threads, execution in GPU worker processes, finish here. Rows come out in
    completion order."""
    import selectors

    devs = [x.strip() for x in str(a.gpus).split(",") if x.strip()]
    if not devs:
        raise SystemExit("[f0_prejudge] --gpus (DEXENGINE_F0_GPUS) is empty — f0 has only the GPU engine")
    nw = len(devs) * int(a.gpu_workers_per_device)
    print(
        f"[F0_ENGINE] engine=gpu gpus={','.join(devs)} workers={nw} max_inst={a.gpu_max_inst} "
        f"ahead={a.gpu_ahead} (no process per clip · control flow=repair.main_steps)",
        flush=True,
    )
    jobs, results = _prep_all(a, clips, dst_tree, dst_wrist_dirs, stage_src, wrist_dirs)
    for r in results:  # noqa: UP028  (kept as a loop: a moved body stays byte-identical)
        yield r
    if not jobs:
        return
    env = worker_env()
    _check_worker_python(a, env)
    workers = _start_workers(a, devs, env, nw)
    queue = list(jobs)
    sel = selectors.DefaultSelector()
    for w in workers:
        sel.register(w["p"].stdout, selectors.EVENT_READ, w)

    # The first feed goes round-robin one clip at a time (if one worker took its whole `ahead` first, the other GPUs
    # would idle on small batches); afterwards each worker is refilled whenever a result arrives.
    ahead = int(a.gpu_ahead)
    while queue and any(len(w["out"]) < ahead for w in workers):
        for w in workers:
            if queue and len(w["out"]) < ahead:
                _send(w, queue.pop(0), jobs)
    for w in workers:
        _feed(w, queue, jobs, a)
    alive = len(workers)
    while alive:
        for key, _ in sel.select():
            w = key.data
            line = w["p"].stdout.readline()
            if not line:  # worker exited
                sel.unregister(w["p"].stdout)
                rc = w["p"].wait()
                alive -= 1
                for c in sorted(w["out"]):  # unfinished clips = loud failure
                    jb = jobs[c]
                    yield {
                        "clip": c,
                        "status": "FAIL_GPU_WORKER",
                        "P": 0,
                        "t": time.time() - jb["t0"],
                        "log": f"GPU worker (dev {w['dev']}) exited rc={rc} — this clip is unfinished",
                        "stale": jb["stale"],
                    }
                w["out"].clear()
                continue
            if not line.startswith(GPU_PROTO):
                print(line.rstrip("\n"), flush=True)  # the worker's human-readable lines (markers, progress)
                continue
            msg = json.loads(line[len(GPU_PROTO) :])
            c = msg["clip"]
            w["out"].discard(c)
            try:
                yield finish(jobs[c], int(msg["rc"]), str(msg["out"]).strip(), a, stage_src, wrist_dirs)
            except Exception as e:  # noqa: BLE001
                yield {
                    "clip": c,
                    "status": "FAIL_EXC",
                    "P": 0,
                    "t": 0.0,
                    "log": f"{type(e).__name__}: {e}"[:400],
                    "stale": jobs[c]["stale"],
                }
            _feed(w, queue, jobs, a)


def _stdin_writer(w: dict) -> None:
    """Dedicated stdin writer thread of one worker: writes queued lines and closes stdin on None. Ends quietly when the
    pipe breaks because the worker died (the main loop reports that worker's unfinished clips as FAIL_GPU_WORKER on
    stdout EOF)."""
    try:
        while True:
            item = w["q"].get()
            if item is None:
                w["p"].stdin.close()
                return
            w["p"].stdin.write(item)
            w["p"].stdin.flush()
    except (BrokenPipeError, OSError, ValueError):
        return
