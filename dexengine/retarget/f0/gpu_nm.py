"""f0 GPU engine: batched Nelder-Mead with scipy's arithmetic (NMPool and its phase codes).

NMPool follows SciPy's `_minimize_neldermead` (BSD-3-Clause; Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy
Developers). The full notice is in THIRD_PARTY.md."""

import numpy as np

# batched Nelder-Mead (scipy arithmetic)
PH_FREE, PH_INIT, PH_R, PH_E, PH_C, PH_CC, PH_S, PH_FINAL, PH_DONE = range(9)


class NMPool:
    """NM instances of one dimension N (= one class mask). The operations of scipy `_minimize_neldermead(adaptive=True,
    bounds=None, maxfev=None)` vectorised by row with the same formulas in the same order (bit-identical,
    `--selftest-nm`).

    One round = `points()` (each instance's next evaluation points: N+1 for INIT, N for a shrink, else 1) -> external
    evaluation -> `feed(f)`. A finished instance goes through FINAL (one feasibility check of the final point, host
    `geom["n_pen"]`) to DONE."""

    def __init__(self, N: int, cap: int, maxiter: int, xatol: float, fatol: float):
        self.N = N
        self.cap = cap
        self.maxiter, self.xatol, self.fatol = maxiter, xatol, fatol
        dim = float(N)  # same formula as scipy (adaptive)
        self.rho = 1
        self.chi = 1 + 2 / dim
        self.psi = 0.75 - 1 / (2 * dim)
        self.sigma = 1 - 1 / dim
        self.sim = np.zeros((cap, N + 1, N))
        self.fsim = np.full((cap, N + 1), np.inf)
        self.xbar = np.zeros((cap, N))
        self.xr = np.zeros((cap, N))
        self.xe = np.zeros((cap, N))
        self.xc = np.zeros((cap, N))
        self.fxr = np.zeros(cap)
        self.phase = np.zeros(cap, np.int8)
        self.iters = np.zeros(cap, np.int64)
        self.nfev = np.zeros(cap, np.int64)
        self.owner = np.full(cap, -1, np.int64)  # instance -> (external) id
        self._free = list(range(cap - 1, -1, -1))

    def n_active(self):
        return int(np.count_nonzero(self.phase != PH_FREE))

    def add(self, x0, owner: int) -> int:
        """scipy initial simplex: sim[0] = x0, sim[k+1][k] = (1+0.05)·x0[k] or 0.00025."""
        s = self._free.pop()
        N = self.N
        x0 = np.asarray(x0, float).flatten()
        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + 0.05) * y[k]
            else:
                y[k] = 0.00025
            sim[k + 1] = y
        self.sim[s] = sim
        self.fsim[s] = np.inf
        self.phase[s] = PH_INIT
        self.iters[s] = 0
        self.nfev[s] = 0
        self.owner[s] = owner
        return s

    def release(self, s):
        self.phase[s] = PH_FREE
        self.owner[s] = -1
        self._free.append(s)

    def points(self):
        """(slot array, point array (M, N), kind array); kind: 0 = all simplex vertices (INIT), 1 = single point, 2 = N
        shrink points, 3 = FINAL."""
        ph = self.phase
        N = self.N
        out_s, out_x, out_k = [], [], []
        si = np.flatnonzero(ph == PH_INIT)
        if si.size:
            out_s.append(np.repeat(si, N + 1))
            out_x.append(self.sim[si].reshape(-1, N))
            out_k.append(np.zeros(si.size * (N + 1), np.int8))
        for p, arr in ((PH_R, self.xr), (PH_E, self.xe), (PH_C, self.xc), (PH_CC, self.xc)):
            sp = np.flatnonzero(ph == p)
            if sp.size:
                out_s.append(sp)
                out_x.append(arr[sp])
                out_k.append(np.ones(sp.size, np.int8))
        ss = np.flatnonzero(ph == PH_S)
        if ss.size:
            out_s.append(np.repeat(ss, N))
            out_x.append(self.sim[ss, 1:].reshape(-1, N))
            out_k.append(np.full(ss.size * N, 2, np.int8))
        sf = np.flatnonzero(ph == PH_FINAL)
        if sf.size:
            out_s.append(sf)
            out_x.append(self.sim[sf, 0])
            out_k.append(np.full(sf.size, 3, np.int8))
        if not out_s:
            return np.zeros(0, np.int64), np.zeros((0, N)), np.zeros(0, np.int8)
        return np.concatenate(out_s), np.concatenate(out_x), np.concatenate(out_k)

    def _sort(self, s):
        ind = np.argsort(self.fsim[s], axis=1)
        self.fsim[s] = np.take_along_axis(self.fsim[s], ind, 1)
        self.sim[s] = np.take_along_axis(self.sim[s], ind[:, :, None], 1)

    def _iter_start(self, s):
        """Loop head: iterations < maxiter -> convergence check -> xbar, xr -> R; otherwise FINAL."""
        if s.size == 0:
            return
        N = self.N
        go = self.iters[s] < self.maxiter
        fin = s[~go]
        s = s[go]
        if s.size:
            sim, fsim = self.sim[s], self.fsim[s]
            conv = (np.abs(sim[:, 1:] - sim[:, :1]).reshape(s.size, -1).max(1) <= self.xatol) & (
                np.abs(fsim[:, :1] - fsim[:, 1:]).max(1) <= self.fatol
            )
            fin = np.concatenate([fin, s[conv]])
            s = s[~conv]
        if fin.size:
            self.phase[fin] = PH_FINAL
        if s.size:
            xbar = np.add.reduce(self.sim[s, :-1], 1) / N
            self.xbar[s] = xbar
            self.xr[s] = (1 + self.rho) * xbar - self.rho * self.sim[s, -1]
            self.phase[s] = PH_R

    def _end_iter(self, s):
        if s.size == 0:
            return
        self.iters[s] += 1
        self._sort(s)
        self._iter_start(s)

    def _feed_init(self, slots, kinds, f, N):
        """INIT rows: the N+1 simplex values of new instances, sorted twice (as scipy does), then the first
        iteration."""
        # INIT
        mi = kinds == 0
        if np.any(mi):
            si = slots[mi][:: N + 1]
            self.fsim[si] = f[mi].reshape(-1, N + 1)
            self.nfev[si] += N + 1
            self._sort(si)  # scipy: sort in the finally block
            self._sort(si)  # scipy: and once more after it
            self.iters[si] = 1
            self._iter_start(si)

    def _feed_reflect(self, s1, f1, p1, end):
        """R rows: accept the reflection, or set up expansion / outside or inside contraction."""
        # R
        r = p1 == PH_R
        if np.any(r):
            sr, fr = s1[r], f1[r]
            self.fxr[sr] = fr
            f0 = self.fsim[sr, 0]
            fm2 = self.fsim[sr, -2]
            fm1 = self.fsim[sr, -1]
            e = fr < f0
            acc = ~e & (fr < fm2)
            c = ~e & ~acc & (fr < fm1)
            cc = ~e & ~acc & ~(fr < fm1)
            se = sr[e]
            if se.size:
                self.xe[se] = (1 + self.rho * self.chi) * self.xbar[se] - self.rho * self.chi * self.sim[se, -1]
                self.phase[se] = PH_E
            sa = sr[acc]
            if sa.size:
                self.sim[sa, -1] = self.xr[sa]
                self.fsim[sa, -1] = fr[acc]
                end.append(sa)
            sc = sr[c]
            if sc.size:
                self.xc[sc] = (1 + self.psi * self.rho) * self.xbar[sc] - self.psi * self.rho * self.sim[sc, -1]
                self.phase[sc] = PH_C
            scc = sr[cc]
            if scc.size:
                self.xc[scc] = (1 - self.psi) * self.xbar[scc] + self.psi * self.sim[scc, -1]
                self.phase[scc] = PH_CC

    def _feed_expand(self, s1, f1, p1, end):
        """E rows: keep the better of the expansion and the reflection."""
        # E
        e_ = p1 == PH_E
        if np.any(e_):
            se, fe = s1[e_], f1[e_]
            take = fe < self.fxr[se]
            a_ = se[take]
            self.sim[a_, -1] = self.xe[a_]
            self.fsim[a_, -1] = fe[take]
            b_ = se[~take]
            self.sim[b_, -1] = self.xr[b_]
            self.fsim[b_, -1] = self.fxr[b_]
            end.append(se)

    def _feed_contract(self, s1, f1, p1, end, shrink):
        """C and CC rows: accept the contraction or mark the instance for a shrink."""
        # C (outside contraction): accept if fxc <= fxr, else shrink
        c_ = p1 == PH_C
        if np.any(c_):
            sc, fc = s1[c_], f1[c_]
            take = fc <= self.fxr[sc]
            a_ = sc[take]
            self.sim[a_, -1] = self.xc[a_]
            self.fsim[a_, -1] = fc[take]
            end.append(a_)
            shrink.append(sc[~take])
        # CC (inside contraction): accept if fxcc < fsim[-1], else shrink
        cc_ = p1 == PH_CC
        if np.any(cc_):
            sc, fc = s1[cc_], f1[cc_]
            take = fc < self.fsim[sc, -1]
            a_ = sc[take]
            self.sim[a_, -1] = self.xc[a_]
            self.fsim[a_, -1] = fc[take]
            end.append(a_)
            shrink.append(sc[~take])

    def feed(self, slots, kinds, f, final_ok=None):
        """Take f in the order of points() and advance one step. FINAL rows take final_ok (bool). Returns finished
        slots."""
        N = self.N
        ph = self.phase.copy()  # phase at the start of this round
        self._feed_init(slots, kinds, f, N)
        # single point (R/E/C/CC)
        m1 = kinds == 1
        s1, f1 = slots[m1], f[m1]
        self.nfev[s1] += 1
        p1 = ph[s1]
        end = []
        shrink = []
        self._feed_reflect(s1, f1, p1, end)
        self._feed_expand(s1, f1, p1, end)
        self._feed_contract(s1, f1, p1, end, shrink)
        # shrink result (S)
        ms = kinds == 2
        if np.any(ms):
            ss = slots[ms][::N]
            self.fsim[ss, 1:] = f[ms].reshape(-1, N)
            self.nfev[ss] += N
            end.append(ss)
        # shrink start (decided this round): build the points now, evaluate them next round
        if shrink:
            sh = np.concatenate(shrink)
            if sh.size:
                s0 = self.sim[sh, :1]
                self.sim[sh, 1:] = s0 + self.sigma * (self.sim[sh, 1:] - s0)
                self.phase[sh] = PH_S
        if end:
            self._end_iter(np.concatenate(end))
        # FINAL
        mf = kinds == 3
        done = slots[mf]
        if done.size:
            self.phase[done] = PH_DONE
        return done
