"""The greedy half of the cross-repair batch path (xr_gpu.py): RowSet (near pairs of a pose batch), the
quaternion helper and BatchSolver (the lock-step greedy — the only implementation).

xr_gpu.py re-exports every name. Nothing here holds module state. BatchSolver names PoseBatch (xr_gpu.py) only in an
annotation, which `from __future__ import annotations` never evaluates.
"""

from __future__ import annotations

import numpy as np


class RowSet:
    """The near pairs (distance < distmax) of B poses as arrays; the CPU-form rows (list of tuples) are built only for
    the poses asked for. worst_v[b] = the smallest distance among the hand pairs of worst_sets (the cross repair's
    SET_REPAIR), distmax if none — the worst-hand-pair rule is owned here."""

    def __init__(self, G, B, qb, qk, d, w1, w2, distmax, worst_sets):
        order = np.lexsort((qk, qb))  # per pose, in pair order (the CPU loop order)
        self.G, self.B = G, B
        self.qb, self.qk, self.d = qb[order], qk[order], d[order]
        self.ft = np.concatenate([w1[order], w2[order]], axis=1)
        self.start = np.searchsorted(self.qb, np.arange(B + 1))
        self.worst_v = np.full(B, distmax)
        self.worst_i = np.full(B, -1)
        if worst_sets is not None:
            ok = np.isin(G.pset[self.qk], [G.sets.index(x) for x in worst_sets]) & G.phand[self.qk]
            ix = np.where(ok)[0]
            if len(ix):
                o = ix[
                    np.lexsort((self.qk[ix], self.d[ix], self.qb[ix]))
                ]  # pose -> distance -> pair order (a tie takes the first pair = CPU min())
                first = np.r_[True, self.qb[o][1:] != self.qb[o][:-1]]
                self.worst_v[self.qb[o][first]] = self.d[o][first]
                self.worst_i[self.qb[o][first]] = o[first]

    def row(self, i):
        """Row of global index i — the same form as a CPU rows element (set, class, link A, link B, distance,
        fromto6)."""
        P = self.G.M.pairs
        gi = self.G.M.ginfo
        k = self.qk[i]
        return (P[k][2], P[k][3], gi[P[k][0]][1], gi[P[k][1]][1], float(self.d[i]), self.ft[i].copy())

    def worst(self, b):
        """Row of pose b's worst hand pair (the first minimum over worst_sets) — None if there is none. The rule is the
        worst_i computation in __init__."""
        return self.row(int(self.worst_i[b])) if self.worst_i[b] >= 0 else None

    def rows(self, b):
        P = self.G.M.pairs
        gi = self.G.M.ginfo
        out = []
        for i in range(self.start[b], self.start[b + 1]):
            k = self.qk[i]
            out.append((P[k][2], P[k][3], gi[P[k][0]][1], gi[P[k][1]][1], float(self.d[i]), self.ft[i].copy()))
        return out


def _quat2mat(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz -> (N,3,3) — geometry for the filter only (exactness comes from the CPU re-query)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


# ── batch form of the cross_repair greedy step ─────────────────────────────────────────────────────
class _Inst:
    __slots__ = ("t", "fq0", "fq", "shift", "grasp", "fc", "rec", "done", "rs", "arm6")


class BatchSolver:
    """The only implementation of the cross repair's per-frame greedy.

    Many frames are solved at once (lock-step): each iteration evaluates the candidates of every active instance
    (finger +-, wrist direction x +-) in one batch, then applies the selection rule (order, ties, 1e-4 improvement
    threshold, break) per instance. In an iteration where a finger candidate is adopted the wrist candidates' results
    are discarded (speculative evaluation — the selection equals evaluating one candidate at a time).
    One iteration:
      no worst hand pair (RowSet.worst) or it is at or above the clearance -> done.
      1) finger abduction: of the pair's two links, one that has a finger, no contact of that finger in this frame, an
         abduction joint and is not on a frozen side is moved by +-step (range +-fmax, joint limits) — the wider side is
         adopted if it beats dist + 1e-4, then the next iteration.
      2) wrist shift: the movable end (hand link or held object) is shifted along the fromto normal, the 2nd/3rd pair
         normals, their mean and the wrist-wrist line x +- by need = clearance - dist (a holding hand moves with its
         object, never pushed down, capped at wmax; 10 mm for the holding hand when the partner is empty-handed; split
         in proportion to the remaining slack, partial steps beyond the cap) — done when the best does not improve.
    ctx is the bundle of cross_repair main()'s locals (cross_repair owns the values)."""

    def __init__(self, PB: PoseBatch, ctx: dict):  # noqa: F821 — annotation only (never evaluated): PoseBatch is in xr_gpu.py
        self.PB, self.c = PB, ctx
        m = PB.M.m
        self.b6 = {s: m.body(PB.M.arm_spec.ee_body_name(s)).id for s in ("right", "left")}
        self.nbatch = 0
        self.nposes = 0

    def _eval(self, items):
        """items: [(t, fq{s:arr}, shift{s:arr}, grasp{s:bool})] → [(rows, arm6{s:xyz})]"""
        if not items:
            return []
        ts = np.array([x[0] for x in items], int)
        fq = {s: np.stack([x[1][s] for x in items]) for s in ("right", "left")}
        sh = {s: np.stack([x[2][s] for x in items]) for s in ("right", "left")}
        gr = {s: np.array([bool(x[3][s]) for x in items]) for s in ("right", "left")}
        _, _, _, _, RS, (bR, bp) = self.PB.eval(ts, fq, sh, gr, want_pairs=True, worst_sets=self.c["SET_REPAIR"])
        self.nbatch += 1
        self.nposes += len(items)
        return [(RS, i, {s: bp[i, self.b6[s]] for s in ("right", "left")}) for i in range(len(items))]

    def new(self, t, warm=None):
        c = self.c
        L = c["L"]
        ins = _Inst()
        fq0, _ = L.hand_blocks(c["qpos"][t])
        ins.t = int(t)
        ins.fq0 = fq0
        ins.fq = {"right": fq0["right"].copy(), "left": fq0["left"].copy()}
        ins.shift = {"right": np.zeros(3), "left": np.zeros(3)}
        ins.grasp = {k: bool(c["grasp_all"][k][t]) for k in c["grasp_all"]}
        ins.fc = {"right": c["contact"][t, :5] > 0, "left": c["contact"][t, 5:10] > 0}
        ins.rec = {"t": int(t), "steps": []}
        if warm is not None:
            ins.shift = {k: warm[1][k].copy() for k in ins.shift}
            for k in ins.fq:
                for aa in L.aa[k].values():
                    if aa is None:
                        continue
                    j, lo, hi = aa
                    cand = fq0[k][j] + warm[2][k][j]
                    ins.fq[k][j] = float(np.clip(cand, max(lo, fq0[k][j] - c["fmax"]), min(hi, fq0[k][j] + c["fmax"])))
            ins.rec["warm"] = True
        ins.done = False
        ins.rs = None
        ins.arm6 = None
        return ins

    def _wrist_cands(self, ins, wp):
        """Candidates of step 2 (wrist shift), in order. Returns (cands [list of sh2], ends_empty)."""
        c = self.c
        CC = c["CC"]
        sname, cls, A, B, dist, ft = wp
        n = ft[3:] - ft[:3]
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])
        need_m = c["clearance"] - dist
        ends = []
        for link, dirn in ((A, -1.0), (B, +1.0)):
            sd = c["side_of_link"](link)
            if c["is_hand_link"](link) and sd and sd not in c["FREEZE"] and sd not in [e[0] for e in ends]:
                if link.endswith("_object") and not (c["carry"] and ins.grasp[sd]):
                    continue
                ends.append((sd, dirn))
        if not ends:
            return [], True
        wmax = c["wmax"]
        shift = ins.shift
        grasp = ins.grasp
        cap = dict.fromkeys(shift, wmax)
        movers = self._movers(ends, grasp, wmax, cap, shift)
        dirs = self._shift_dirs(n, ins, c, CC)
        out = self._shift_cands(dirs, shift, movers, need_m, c, grasp, cap)
        return out, False

    def _movers(self, ends, grasp, wmax, cap, shift):
        """Which ends move and by what fraction: a grasping end is capped at 10 mm when the other end is free, and the
        shift splits by the wrists' remaining headroom (the cap dict is updated in place)."""
        if len(ends) == 2:
            g0, g1 = grasp[ends[0][0]], grasp[ends[1][0]]
            if g0 and not g1:
                cap[ends[0][0]] = min(wmax, 0.010)
            if g1 and not g0:
                cap[ends[1][0]] = min(wmax, 0.010)
            head = [max(cap[e[0]] - float(np.linalg.norm(shift[e[0]])), 0.0) for e in ends]
            hs = head[0] + head[1]
            if hs > 1e-9:
                fr = (head[0] / hs, head[1] / hs)
            else:
                fr = (0.0, 1.0) if (g0 and not g1) else (1.0, 0.0) if (g1 and not g0) else (0.5, 0.5)
            movers = [(ends[0][0], ends[0][1], fr[0]), (ends[1][0], ends[1][1], fr[1])]
        else:
            movers = [(ends[0][0], ends[0][1], 1.0)]
        return movers

    def _shift_dirs(self, n, ins, c, CC):
        """Shift directions: the closest-pair normal, the other hand-pair normals, and the wrist-to-wrist line."""
        dirs = [n]
        rows = ins.rs[0].rows(ins.rs[1])
        hp = sorted([r for r in rows if r[0] in c["SET_REPAIR"] and CC.is_hand_pair(r[1])], key=lambda r: r[4])[1:3]
        for r in hp:
            m_ = r[5][3:] - r[5][:3]
            mm_ = np.linalg.norm(m_)
            if mm_ > 1e-9:
                m_ = m_ / mm_
                dirs.append(m_ if float(m_ @ n) >= 0 else -m_)
        if len(dirs) > 1:
            mean_ = np.sum(dirs, axis=0)
            mn_ = np.linalg.norm(mean_)
            if mn_ > 1e-9:
                dirs.append(mean_ / mn_)
        pr_, pl_ = ins.arm6["right"], ins.arm6["left"]
        wl_ = pl_ - pr_
        wn_ = np.linalg.norm(wl_)
        if wn_ > 1e-9:
            dirs.append(wl_ / wn_)
        return dirs

    def _shift_cands(self, dirs, shift, movers, need_m, c, grasp, cap):
        """Shifted wrist sets along each direction and sign, capped per side; unchanged or over-cap sets are skipped."""
        out = []
        for n_dir, sgn in [(d_, s_) for d_ in dirs for s_ in (+1.0, -1.0)]:
            sh2 = {k: v.copy() for k, v in shift.items()}
            for sd, dirn, frac in movers:
                if frac <= 0.0:
                    continue
                dv = sgn * dirn * frac * need_m * n_dir
                if c["carry"] and grasp[sd] and dv[2] < 0.0:
                    dv = dv.copy()
                    dv[2] = 0.0
                tot = sh2[sd] + dv
                if np.linalg.norm(tot) > cap[sd] + 1e-12:
                    s0, dn = sh2[sd], dv
                    aa_, bb_, cc_ = float(dn @ dn), 2.0 * float(s0 @ dn), float(s0 @ s0) - cap[sd] ** 2
                    lam = 0.0
                    if aa_ > 1e-18:
                        disc = bb_ * bb_ - 4 * aa_ * cc_
                        if disc >= 0.0:
                            lam = max(0.0, min(1.0, (-bb_ + np.sqrt(disc)) / (2 * aa_)))
                    dv = dv * lam
                sh2[sd] = sh2[sd] + dv
            if any(np.linalg.norm(sh2[k]) > cap[k] + 1e-9 for k in sh2):
                continue
            if all(np.linalg.norm(sh2[k] - shift[k]) < 1e-7 for k in sh2):
                continue
            out.append(sh2)
        return out

    def solve(self, insts):
        """Lock-step greedy. When it returns, every instance holds its final (shift, fq, final, rec)."""
        c = self.c
        L = c["L"]
        DM = c["distmax"]
        clr = c["clearance"]
        pend = [ins for ins in insts if ins.rs is None]
        for ins, (RS, i, a6) in zip(
            pend, self._eval([(ins.t, ins.fq, ins.shift, ins.grasp) for ins in pend]), strict=True
        ):
            ins.rs, ins.arm6 = (RS, i), a6
        for _ in range(c["iters"] * 3):
            act = [ins for ins in insts if not ins.done]
            if not act:
                break
            items = []
            plan = []
            self._plan_round(act, clr, c, L, items, plan)
            res = self._eval(items)
            self._apply_round(plan, res, L)
        out = []
        for ins in insts:
            wp = ins.rs[0].worst(ins.rs[1])
            out.append((ins.shift, ins.fq, ins.fq0, (wp[4] if wp else DM), ins.rec, ins.grasp))
        return out

    def _plan_round(self, act, clr, c, L, items, plan):
        """Per active instance: the worst pair; finger abduction candidates (± one step) at each end, then the wrist
        shift candidates. Fills items (to evaluate) and plan."""
        for ins in act:
            wp = ins.rs[0].worst(ins.rs[1])
            if wp is None or wp[4] >= clr:
                ins.done = True
                continue
            sname, cls, A, B, dist, ft = wp
            fin = []  # [(link, sd, f, j, [(cand, item_idx)])]
            for link in (A, B):
                sd, f = c["side_of_link"](link), L.finger_of(link)
                if sd is None or f is None or ins.fc[sd][f] or L.aa[sd][f] is None or sd in c["FREEZE"]:
                    continue
                j, aa_lo, aa_hi = L.aa[sd][f]
                base_v = ins.fq0[sd][j]
                cl = []
                for sgn in (+1.0, -1.0):
                    cand = ins.fq[sd].copy()
                    cand[j] = float(
                        np.clip(
                            ins.fq[sd][j] + sgn * c["step"],
                            max(aa_lo, base_v - c["fmax"]),
                            min(aa_hi, base_v + c["fmax"]),
                        )
                    )
                    if abs(cand[j] - ins.fq[sd][j]) < 1e-6:
                        continue
                    fq2 = dict(ins.fq)
                    fq2[sd] = cand
                    cl.append((cand, len(items)))
                    items.append((ins.t, fq2, ins.shift, ins.grasp))
                fin.append((link, sd, f, j, cl))
            wc, noends = self._wrist_cands(ins, wp)
            wl = []
            for sh2 in wc:
                wl.append((sh2, len(items)))
                items.append((ins.t, ins.fq, sh2, ins.grasp))
            plan.append((ins, wp, fin, wl, noends))

    def _apply_round(self, plan, res, L):
        """Per instance: adopt the best finger candidate that improves the worst-pair distance by > 0.1 mm, else the
        best wrist shift; the instance is done when nothing is left or the wrist shift did not improve."""
        for ins, wp, fin, wl, noends in plan:
            dist = wp[4]
            took = False
            for _link, sd, f, j, cl in fin:
                best = None
                for cand, ix in cl:
                    v = float(res[ix][0].worst_v[res[ix][1]])
                    if best is None or v > best[0]:
                        best = (v, sd, cand, ix)
                if best is not None and best[0] > dist + 1e-4:
                    ins.fq[best[1]] = best[2]
                    ins.rec["steps"].append(
                        {
                            "kind": "finger",
                            "side": best[1],
                            "finger": L.tokens[f],
                            "deg": round(float(np.rad2deg(best[2][j] - ins.fq0[best[1]][j])), 1),
                            "dist_mm": round(best[0] * 1000, 2),
                        }
                    )
                    RS, i_, a6 = res[best[3]]
                    ins.rs, ins.arm6 = (RS, i_), a6
                    took = True
                    break
            if took:
                continue
            if noends:
                ins.done = True
                continue
            best = None
            for sh2, ix in wl:
                v = float(res[ix][0].worst_v[res[ix][1]])
                if best is None or v > best[0]:
                    best = (v, sh2, ix)
            if best is None:
                ins.done = True
                continue
            improved = best[0] > dist + 1e-4
            ins.shift = best[1]
            sname, cls, A, B = wp[0], wp[1], wp[2], wp[3]
            ins.rec["steps"].append(
                {
                    "kind": "wrist",
                    "pair": f"{A}~{B}",
                    "set": sname,
                    "cls": cls,
                    "dist_mm": round(best[0] * 1000, 2),
                    "shift_mm": {k: [round(x * 1000, 1) for x in v] for k, v in ins.shift.items()},
                }
            )
            RS, i_, a6 = res[best[2]]
            ins.rs, ins.arm6 = (RS, i_), a6
            if not improved:
                ins.done = True

    def backoff(self, jobs):
        """alpha back-off (0.25, 0.5, 0.75 in order, first pass wins).
        jobs: [(t, shift, fq, fq0, grasp)] → [(shift, fq, final, alpha)|None]"""
        c = self.c
        clr = c["clearance"]
        items = []
        items.extend(
            (t, {k: fq0[k] + (fq[k] - fq0[k]) * alpha for k in fq}, {k: shift[k] * alpha for k in shift}, grasp)
            for t, shift, fq, fq0, grasp in jobs
            for alpha in (0.25, 0.5, 0.75)
        )
        res = self._eval(items)
        out = []
        for n, (_t, _shift, _fq, _fq0, _grasp) in enumerate(jobs):
            got = None
            for a_i, alpha in enumerate((0.25, 0.5, 0.75)):
                it = items[3 * n + a_i]
                v_a = float(res[3 * n + a_i][0].worst_v[res[3 * n + a_i][1]])
                if v_a >= clr - 1e-6:
                    got = (it[2], it[1], v_a, alpha)
                    break
            out.append(got)
        return out
