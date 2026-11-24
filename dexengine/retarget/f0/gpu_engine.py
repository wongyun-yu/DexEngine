"""f0 GPU engine: the sequence driver (continuous batching of clip solves over one evaluator and one NM pool)."""

import io
import time
import traceback

import numpy as np

from dexengine.core import stdio
from dexengine.retarget.f0.gpu_eval import GpuEvaluator
from dexengine.retarget.f0.gpu_nm import NMPool


# engine: requests -> instances -> rounds
class _ReqState:
    __slots__ = ("req", "slot", "idx", "n_init", "insts", "dq", "ok", "left", "evals", "clip_v", "key")


class GpuEngine:
    """Solve SolveRequests in lock-step; each request gets one NM instance per initial value."""

    def __init__(self, evaluator: GpuEvaluator, cap_per_pool: int = 4096):
        from dexengine.retarget.f0 import repair_cfg as F

        self.F = F
        self.ev = evaluator
        self.opts = dict(F.NM_OPTIONS)
        if not self.opts.get("adaptive", False):
            raise SystemExit("[f0_gpu] NM_OPTIONS.adaptive=False is not implemented by NMPool")
        self.cap = cap_per_pool
        self.pools = {}  # key(mask bytes) -> NMPool
        self.reqs = {}  # rid -> _ReqState
        self.inst_owner = {}  # (poolkey, slot) -> (rid, init index)
        self._rid = 0
        self.stats = {"rounds": 0, "t_round_host": 0.0, "oracle_checked": 0, "oracle_mismatch": 0}

    def n_instances(self):
        return sum(p.n_active() for p in self.pools.values())

    def submit(self, req) -> int:
        F = self.F
        g = req.geom
        idx = np.flatnonzero(req.mask)
        key = bytes(np.asarray(req.mask, bool))
        N = len(idx)
        if key not in self.pools:
            self.pools[key] = NMPool(N, self.cap, self.opts["maxiter"], self.opts["xatol"], self.opts["fatol"])
        pool = self.pools[key]
        rid = self._rid
        self._rid += 1
        st = _ReqState()
        st.req, st.idx, st.key = req, idx, key
        st.slot = self.ev.add_problem(g)
        st.n_init = len(req.inits)
        st.dq = [None] * st.n_init
        st.ok = [None] * st.n_init
        st.left = st.n_init
        st.evals = 0
        st.insts = []
        # check the vectorised clip and cost against this request's functions (independent oracle; mismatch is fatal)
        lo, hi, indep0 = np.asarray(g["lo"]), np.asarray(g["hi"]), np.asarray(g["indep0"])
        st.clip_v = (lo, hi, indep0)
        self._oracle_check(req, st)
        for i, x0 in enumerate(req.inits):
            s = pool.add(np.asarray(x0, float)[idx], rid)
            st.insts.append(s)
            self.inst_owner[(key, s)] = (rid, i)
        self.reqs[rid] = st
        del F
        return rid

    def _clip_rows(self, st, D):
        lo, hi, indep0 = st.clip_v
        D = D.copy()
        D[:, 6:] = np.clip(indep0 + D[:, 6:], lo, hi) - indep0
        return D

    def _oracle_check(self, req, st):
        rng = np.random.default_rng(0)
        nd = int(req.geom["nd"])
        Z = rng.normal(size=(8, nd)) * 0.5
        C = self._clip_rows(st, Z)
        for i in range(len(Z)):
            ref = req.clip(Z[i])
            if not np.array_equal(ref, C[i]):
                raise SystemExit("[f0_gpu] vectorised clip ≠ the request's _clip — the contracts diverged")
            if float(np.sum((req.geom["w_dq"] * C[i : i + 1]) ** 2, axis=1)[0]) != req.cost(C[i]):
                raise SystemExit("[f0_gpu] vectorised cost ≠ the request's _cost — the contracts diverged")

    def round(self):
        """Collect the next evaluation points of every pool, evaluate them in one GPU call and advance. Returns the
        request ids finished this round.

        One pool = one mask (shared idx and nd). Clip bounds and target widths differ per request, so they are applied
        per row through the owning-request array."""
        F = self.F
        th = time.perf_counter()
        batch, rows_slot, rows_dq = [], [], []
        for key, pool in self.pools.items():
            s, X, kd = pool.points()
            if s.size == 0:
                continue
            owners = pool.owner[s]
            u, inv = np.unique(owners, return_inverse=True)
            sts = [self.reqs[int(o)] for o in u]
            st0 = sts[0]
            nd = int(st0.req.geom["nd"])
            D = np.zeros((s.size, nd))
            D[:, st0.idx] = X
            lo = np.stack([t.clip_v[0] for t in sts])[inv]
            hi = np.stack([t.clip_v[1] for t in sts])[inv]
            i0 = np.stack([t.clip_v[2] for t in sts])[inv]
            D[:, 6:] = np.clip(i0 + D[:, 6:], lo, hi) - i0  # = the request's _clip (checked at submit)
            W = np.stack([t.req.geom["w_dq"] for t in sts])[inv]
            ps = np.array([t.slot for t in sts], np.int32)[inv]
            batch.append((key, pool, s, kd, D, owners, W))
            rows_slot.append(ps)
            rows_dq.append(D)
        if not batch:
            return []
        S = np.concatenate(rows_slot)
        DQ = np.concatenate(rows_dq)
        self.stats["t_round_host"] += time.perf_counter() - th
        cnt, dep = self.ev.evaluate(S, DQ)
        th = time.perf_counter()
        finished = []
        off = 0
        for key, pool, s, kd, D, _owners, W in batch:
            m = s.size
            c, dp = cnt[off : off + m], dep[off : off + m]
            off += m
            cost = np.sum((W * D) ** 2, axis=1)
            f = F.objective_value(cost, c, dp)
            pool.feed(s, kd, f)
            # FINAL rows: the final-point feasibility check `_n_pen_strict(dq)[0] == 0` runs on the host function
            # (geom["n_pen"]): the host version owns feasibility (and its meter counts it); GPU values are used only for
            # the oracle check.
            fm = np.flatnonzero(kd == 3)
            for j in fm:
                sl = int(s[j])
                rid, i = self.inst_owner.pop((key, sl))
                st = self.reqs[rid]
                dqrow = D[j].copy()
                cc = int(st.req.geom["n_pen"](dqrow)[0])
                if (cc == 0) != (int(c[j]) == 0) or cc != int(c[j]):
                    self.stats["oracle_mismatch"] += 1
                self.stats["oracle_checked"] += 1
                st.dq[i] = dqrow
                st.ok[i] = cc == 0
                st.evals += int(pool.nfev[sl])
                st.left -= 1
                pool.release(sl)
                if st.left == 0:
                    finished.append(rid)
        self.stats["rounds"] += 1
        self.stats["t_round_host"] += time.perf_counter() - th
        return finished

    def result(self, rid):
        """Finished request -> (x, c). Selection = repair_cfg.pick_solution (the request's _cost); GPU evaluations are
        added via add_evals (the FINAL checks were counted by the host n_pen itself; the sum is the side's evaluation
        count)."""
        st = self.reqs.pop(rid)
        st.req.add_evals(st.evals)
        self.ev.free_problem(st.slot)
        return self.F.pick_solution(st.dq, st.ok, st.req.cost)


ADMIT_PER_ROUND = 8  # cap on new clips admitted between rounds (no cap when nothing is running)


class _Job:
    __slots__ = ("key", "gen", "buf", "rc", "t0", "done")


def _step(job, value, first=False):
    """Advance the generator to its next solve request (CPU work: checks, strict verification, writing); output goes to
    the job's buffer. Returns a SolveRequest, or None when finished (job.rc set). rc follows the CPU process convention
    (SystemExit code; exception = 1)."""
    with stdio.capture(job.buf, stderr=True):
        try:
            return next(job.gen) if first else job.gen.send(value)
        except StopIteration:
            job.rc = 0
        except SystemExit as e:
            if isinstance(e.code, int):
                job.rc = e.code
            elif e.code is None:
                job.rc = 0
            else:
                print(e.code)
                job.rc = 1
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            job.rc = 1
    job.done = True
    return None


def run_jobs(next_job, evaluator, max_inst: int, on_done=None, log=None):
    """next_job(block) -> ("job", key, gen) | ("none",) | ("eof",). Runs everything to completion with continuous
    batching. on_done(key, rc, out, t) is called whenever a job finishes."""
    eng = GpuEngine(evaluator, cap_per_pool=max_inst + 64)
    active = {}  # rid -> job
    eof = False
    n_done = 0
    t_start = last = time.time()

    def handle(job, req):
        nonlocal n_done
        if req is not None:
            with stdio.capture(job.buf, stderr=True):
                rid = eng.submit(req)
            active[rid] = job
            return
        if on_done:
            on_done(job.key, job.rc, job.buf.getvalue(), time.time() - job.t0)
        n_done += 1

    def admit():
        # Limit the clips admitted between two rounds to ADMIT_PER_ROUND: admission (object and hand geometry, initial
        # checks) is CPU work, and admitting hundreds at once leaves the GPU idle meanwhile (measured: minutes at 0 % at
        # a full taco start).
        nonlocal eof
        k = 0
        while not eof and eng.n_instances() < max_inst and (not active or k < ADMIT_PER_ROUND):
            k += 1
            got = next_job(block=not active)
            if got[0] == "eof":
                eof = True
                return
            if got[0] == "none":
                return
            _, key, gen = got
            job = _Job()
            job.key, job.gen, job.buf, job.rc, job.t0, job.done = key, gen, io.StringIO(), None, time.time(), False
            handle(job, _step(job, None, first=True))

    while True:
        admit()
        if not active:
            if eof:
                break
            continue
        for rid in eng.round():
            job = active.pop(rid)
            x, c = eng.result(rid)
            handle(job, _step(job, (x, c)))
        if log and time.time() - last > 60:
            last = time.time()
            s = evaluator.stats
            log(
                f"[F0_GPU_PROGRESS] done={n_done} active_req={len(active)} inst={eng.n_instances()} "
                f"rounds={eng.stats['rounds']} evals={s['n_eval']} q/eval={s['n_query'] / max(s['n_eval'], 1):.0f} "
                f"t_gpu={s['t_gpu']:.0f}s t_host={s['t_host'] + eng.stats['t_round_host']:.0f}s "
                f"wall={time.time() - t_start:.0f}s"
            )
    return eng
