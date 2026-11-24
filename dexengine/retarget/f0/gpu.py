#!/usr/bin/env python3
"""GPU engine of the f0 shadow: the `repair` objective and Nelder-Mead, run as a lock-step batch across clips.

Each side's NM evaluates `n_pen` (FK + desk + object KD + reverse Delaunay, 0.5–20 ms) 16k–45k times; this module moves
both layers to the GPU and is the only f0 engine. "CPU" below means the host (numpy) objective of `repair`, which still
owns the initial checks and the final feasibility verdict:

  1. Objective (`GpuEvaluator`): Warp kernels. Count verdicts equal the fp64 version: fp32 rejects or accepts only when
     a value is more than the guard g (per problem, FP32_GUARD × (1 + coordinate scale)) away from the boundary; inside
     it, the CPU comparison is redone in fp64.
     FK, hand point transforms, desk (fp64), object (sample AABB ± clearance in fp64 -> 26-DOP fp32 rejection only ->
     nearest sample -> convex hull containment -> normal sign), reverse check (object surface points inside link hulls,
     bounding-sphere fp32 rejection -> half-planes fp32/fp64).
     The nearest sample is found by a per-object sample kd-tree (fp32 boxes, home leaf gives the bound, then pre-order
     traversal with pruning) and an fp64 argmin (ties -> lowest index). The pruning bound is √best32 + guard, so the
     node holding the true nearest sample is never dropped: this is exact, not an approximation. Depth sums are
     reduced in a fixed (candidate, link) order, so the same input gives the same bits.
  2. Optimiser (`NMPool`): a numpy vectorisation of scipy `_minimize_neldermead` (adaptive, no bounds) that keeps the
     arithmetic order. K instances each take their own step (reflect/expand/contract/shrink); each round collects only
     the "next evaluation points" and evaluates them on the GPU at once. Given the same f values it follows the same
     trajectory as scipy (self-test `--selftest-nm`).

Where this can differ from the host objective (the only source of output differences): fp64 reduction order
(pinocchio/BLAS vs kernels, numpy pairwise sums vs fixed-order sums), at the 1-ULP level. A count verdict flips only for
a point within ~1e-16 m of a boundary, and a depth-sum ULP flips an NM comparison only when two candidates' f agree to
~1e-13 relative. Most sides therefore give the CPU's solution and evaluation count; the others are counted by name in
the report.

This module does not own the control flow (class order, budget, strict verification, acceptance), the selection rule,
clipping or cost: the `repair` generator (`main_steps` -> `_side_steps`) owns them and passes a `SolveRequest`. The
vectorised clip and cost are checked against each request's own functions before use (`_oracle_check`; a mismatch is
fatal).

Entry points: `prejudge.py` (the only caller that repairs; starts `--worker` processes). Run directly: `--selftest-nm`
(self-test). The command line is described by the parser in `__main__` (`--help`).
"""

import os
import sys
import time

import numpy as np

from dexengine.retarget.f0.gpu_engine import run_jobs
from dexengine.retarget.f0.gpu_eval import GpuEvaluator
from dexengine.retarget.f0.gpu_nm import PH_DONE, NMPool
from dexengine.retarget.f0.gpu_specs import ENGINE_TAG, GPU_FUSE_FP

# sequence driver (continuous batching)
PROTO = "@@F0GPU@@ "  # result-line prefix worker -> prejudge; must equal shadow_gpu.GPU_PROTO (the worker checks)


class _StdinJobs:
    """JSON lines (clip, argv) sent by the prejudge on stdin -> repair.main_steps generators. Reads without blocking
    (does not wait while jobs run; the parent sends more only after it receives results)."""

    def __init__(self, F):
        import selectors

        self.F = F
        self.fd = sys.stdin.fileno()
        os.set_blocking(self.fd, False)
        self.sel = selectors.DefaultSelector()
        self.sel.register(self.fd, selectors.EVENT_READ)
        self.buf = b""
        self.eof = False
        self.lines = []

    def _pull(self, block):
        while not self.lines and not self.eof:
            ev = self.sel.select(None if block else 0)
            if not ev:
                return
            try:
                chunk = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                if not block:
                    return
                continue
            if not chunk:
                self.eof = True
                break
            self.buf += chunk
            *done, self.buf = self.buf.split(b"\n")
            self.lines.extend(x for x in done if x.strip())
            if not block:
                return

    def __call__(self, block=False):
        import json

        self._pull(block)
        if self.lines:
            j = json.loads(self.lines.pop(0).decode())
            return ("job", j["clip"], self.F.main_steps(list(j["argv"])))
        return ("eof",) if self.eof else ("none",)


def worker_main(a):
    """`gpu.py --worker`: a worker process of the prejudge (one device = CUDA_VISIBLE_DEVICES). a = the parsed
    `__main__` arguments (proto, max_inst; the concurrent-instance cap is owned by the parent's --gpu-max-inst)."""
    import json

    if a.proto + " " != PROTO:
        raise SystemExit(f"[f0_gpu] protocol prefix mismatch: parent {a.proto!r} ≠ worker {PROTO.strip()!r}")
    from dexengine.retarget.f0 import repair as F

    ev = GpuEvaluator("cuda:0")
    wp = ev.wp
    dname = wp.get_device("cuda:0").name
    print(
        f"[F0_GPU_WORKER] pid={os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"device={dname} warp={wp.__version__} max_inst={a.max_inst} tree={F.REPO} engine={ENGINE_TAG} "
        f"fuse_fp={GPU_FUSE_FP}",
        file=sys.stderr,
        flush=True,
    )
    out = sys.stdout

    def on_done(key, rc, text, t):
        out.write(PROTO + json.dumps({"clip": key, "rc": rc, "out": text, "t": round(t, 3)}, ensure_ascii=False) + "\n")
        out.flush()

    t0 = time.time()
    eng = run_jobs(
        _StdinJobs(F), ev, max_inst=a.max_inst, on_done=on_done, log=lambda m: print(m, file=sys.stderr, flush=True)
    )
    s = ev.stats
    print(
        f"[F0_GPU_WORKER_DONE] pid={os.getpid()} evals={s['n_eval']} calls={s['n_call']} "
        f"q/eval={s['n_query'] / max(s['n_eval'], 1):.1f} t_gpu={s['t_gpu']:.1f}s t_host_pose={s['t_host']:.1f}s "
        f"t_nm_host={eng.stats['t_round_host']:.1f}s rounds={eng.stats['rounds']} "
        f"oracle={eng.stats['oracle_checked']}/{eng.stats['oracle_mismatch']}mis wall={time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )


# self-test
def _selftest_nm():
    """NMPool ≡ scipy: bit-identical trajectory, evaluation count and final x on the same function (incl. step functions
    with many ties)."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(1)
    ok = bad = 0
    for N in (6, 20, 26):
        W = rng.choice([1.0, 2.0], N)
        C = rng.normal(size=N) * 0.2

        def f1(z, W=W, C=C):
            # move cost + steps (cause ties) + continuous depth: the shape of the f0 objective
            pen = np.floor(np.maximum(0, 0.3 - z[0]) * 40)
            return float(np.sum((W * z) ** 2)) + 1000.0 * pen + 5e4 * max(0.0, 0.05 - abs(z[1] - C[1])) * 0.01

        inits = [np.full(N, v) for v in (-0.15, -0.35, 0.0, 0.2)] + [rng.normal(size=N) * 0.1 for _ in range(4)]
        pool = NMPool(N, 64, 2500, 1e-4, 1e-4)
        slots = [pool.add(x0, i) for i, x0 in enumerate(inits)]
        res = {}
        while len(res) < len(inits):
            s, X, kd = pool.points()
            f = np.array([f1(x) for x in X])
            pool.feed(s, kd, f)
            for sl in slots:
                if pool.phase[sl] == PH_DONE and sl not in res:
                    res[sl] = (pool.sim[sl, 0].copy(), int(pool.nfev[sl]))
        for i, x0 in enumerate(inits):
            r = minimize(
                f1, x0, method="Nelder-Mead", options={"maxiter": 2500, "xatol": 1e-4, "fatol": 1e-4, "adaptive": True}
            )
            x, nf = res[slots[i]]
            if np.array_equal(x, r.x) and nf == r.nfev:
                ok += 1
            else:
                bad += 1
                print(f"  NM MISMATCH N={N} init={i} nfev {nf} vs {r.nfev} maxdx {np.abs(x - r.x).max():.3e}")
    print(f"[F0_GPU_SELFTEST_NM] identical={ok} mismatch={bad}")
    return bad == 0


if __name__ == "__main__":
    from dexengine.core import blas_pin  # stage-1 entry: pinned OpenBLAS kernel, proven

    blas_pin.apply("dexengine.retarget.f0.gpu")
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--worker", action="store_true", help="worker process of the prejudge (stdin jobs -> stdout results)"
    )
    mode.add_argument("--selftest-nm", action="store_true", help="NMPool vs scipy Nelder-Mead bit check (self-test)")
    ap.add_argument("--proto", help="(--worker) the parent's protocol prefix; must equal the worker's PROTO")
    ap.add_argument(
        "--max-inst",
        type=int,
        help="(--worker) concurrent NM instances per worker; value from the parent's --gpu-max-inst",
    )
    a = ap.parse_args()
    if a.worker != (a.proto is not None) or a.worker != (a.max_inst is not None):
        ap.error("--proto and --max-inst go only with --worker, and --worker needs both")
    if a.worker:
        worker_main(a)
        sys.exit(0)
    if a.selftest_nm:  # the required group's other mode
        sys.exit(0 if _selftest_nm() else 1)
    ap.error("one of --worker / --selftest-nm is required")
