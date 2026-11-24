"""Exhaustive search (certificate) of one judged clip: when every anchor fails, judge the whole volume in coverage
order. Every judge call goes through the Judge instance it is given (`self.J`)."""

from __future__ import annotations

import json
import time

import numpy as np

from dexengine.placement.judge import clip_time as _CT  # [CLIP_T] per-clip stage seconds

#: Head of the reject-reason distribution appended to every EXHAUSTED / TRUNCATED row (reject_tally). Parsed by the
#: failure classifiers (e.g. experiments posthoc_counts) — data, keep the value.
REJECT_TALLY = " · reject_tally "


def reject_tally(rej: dict) -> str:
    """Every judge reject tally (`stage/key` → count, Judge._tally), most frequent first — reasons outside the rank
    table (_MISS_RANK) stay too. Spaces in a key become `_` (so the `key=count` tokens stay whole). No tally = `-`."""
    toks = [f"{'_'.join(str(k).split())}={v}" for k, v in sorted(rej.items(), key=lambda x: -x[1])]
    return REJECT_TALLY + (" ".join(toks) or "-")


class _Exhaustion:
    """Exhaustive search of one clip: when every anchor fails, judge the whole volume in coverage order.

    The state shared by adoption and the queue flushes (_adopt_ex · _flush_bq · _drain_gq) lives in this object
    (n_ex · n_found · _nfl · pruned · _bq · _gq …). J = the Judge, ck = this clip's context (ck.got_any is set here).
    """

    def __init__(self, J, ck):
        self.J = J
        self.ck = ck

    @_CT.timed("exhaust")
    def run(self) -> None:
        """One exhaustive search of the clip."""
        a = self.J.a
        self.cands = self._candidates()
        # D37 resume (--exhaust-resume): the cells the earlier judge covered (checks + bound exclusions) are not
        # seen again — 0 without a table. n_ex counts **list positions** (earlier cells included), so adoption rank,
        # coverage and the [JUDGE_BUDGET] check count keep the scale of a from-scratch judge.
        self._start = self.J.exh_resume_start(self.ck.clip, len(self.cands))
        self.n_ex = self._start
        self._nfl = 0  # number of FULL flushes
        self.pruned = set()  # cells **proved** excluded by branch-and-bound
        self.n_pruned = 0
        self.n_found = 0  # number of placement candidates collected (for bake retries)
        self._cut_wall = False  # the per-clip time budget stopped the loop (budget record, _budget_cut)
        self._cl = self.cands[self._start : a.exhaust_max]
        self._t_ex = time.time()
        self._ncex = min(a.n_candidates, a.exhaust_cands)  # cap on the number of exhaustion candidates
        self._bq = []  # batch queue: gate-passing cells (FULL as one batch at flush)
        self._gq = []  # gate waiting queue
        self._loop()
        self._finish()

    def _candidates(self):
        """Every volume cell as a candidate — (strict coverage, optimistic coverage, cell, dz, margin sum,
        natural-pose distance), sorted by the search keys."""
        J, ck = self.J, self.ck
        a = J.a
        V, COV, COVS, MGV, cent_cache = ck.V, ck.COV, ck.COVS, ck.MGV, ck.cent_cache
        cands = []
        for dzi in J.DZ:
            ky, kx, kyy = np.nonzero(V[dzi])
            cv = COV[dzi]
            cvs = COVS[dzi]
            mgv = MGV[dzi]
            for u in range(len(ky)):
                _off = np.array([(kx[u] - J.Wx) * J.step, (kyy[u] - J.Wy) * J.step, -dzi * J.step])
                _nat = max(float(np.linalg.norm(cent_cache[s3][ky[u]] + _off - J.W_NAT[s3])) for s3 in cent_cache)
                cands.append(
                    (
                        int(cvs[ky[u], kx[u], kyy[u]]),
                        int(cv[ky[u], kx[u], kyy[u]]),
                        int(ky[u]),
                        int(kx[u]),
                        int(kyy[u]),
                        dzi,
                        float(mgv[ky[u], kx[u], kyy[u]]),
                        _nat,
                    )
                )
        # Ordering: **strict coverage first**. Optimistic coverage saturates at 100% over thousands
        # of cells thanks to the dilation and the loose threshold, carrying no ordering
        # information; with a descending-dz tie-break, once the cap (200~400 cells) is hit it
        # **never even reaches the dz=-2 cells that actually work** (where, by the strict criterion, both arms
        # are at 100%).
        # Ordering: strict coverage → **ascending |dz|** → optimistic coverage.
        # The pool's clearance (VCA) is a link **origin** z, so it cannot see the capsule body and
        # is a different physical quantity from the real penetration check — "100% coverage" dz=-4
        # cells would fill the top and all die of early penetration (measured 39~135mm).
        # Penetration worsens the lower the robot goes, so at equal coverage **the less-lowered
        # placement is tried first** (the original design).
        # 2nd key = sum of the two-arm minimum capsule margins — breaks the saturation
        # of key 1 (strict coverage).
        # With --order-nat 1, the 2nd key = **the distance to the natural-pose (elbow 90°)
        # wrist** (ascending) — a search spreading outward from the centre.
        if a.order_nat == 1:
            cands.sort(key=lambda t: (-t[0], t[7], -t[6], abs(t[5])))
        else:
            cands.sort(key=lambda t: (-t[0], -t[6], abs(t[5]), -t[1]))
        return cands

    def _adopt_ex(self, yaw_, dcap_, dzi_, q_both, met_=None, nex_=None):
        """One exhaustion adoption — recorded after the final desk check. True = adopted."""
        J, ck = self.J, self.ck
        a = J.a
        clip, cpiv = ck.clip, ck.cpiv
        p2_ = J._adopted_pivot(clip, cpiv)
        th_ = np.deg2rad(yaw_)
        R2_ = np.array([[np.cos(th_), -np.sin(th_)], [np.sin(th_), np.cos(th_)]])
        d_ = np.array([dcap_[0], dcap_[1]]) + (np.eye(2) - R2_) @ (cpiv[:2] - p2_)
        if not J.desk_ok_final(ck, d_, p2_, yaw_):
            J._tally("confirm", {"desk_xy": "desk_xy(footprint)"})
            return False
        # **One single function**, shared with the other 4 adoption sites (with two copies the
        # thresholds, the order and the reasons drift apart). q_both is passed through q_override
        # so LASTQ is not polluted on failure.
        if q_both is not None:
            _gok, _gwhy = J._pen_cross_guard(dzi_ * J.step, q_override=q_both, place=(d_[0], d_[1], yaw_, p2_))
            if not _gok:
                J._tally("confirm", {"guard": "guard:" + str(_gwhy)})
                return False
            J.LASTQ.clear()  # batch path — inject LASTQ directly
            J.LASTQ.update(q_both)
        J.fpl.write(
            json.dumps(
                {
                    "clip": clip,
                    "anchor": self.n_found,
                    "dx": round(float(d_[0]), 4),
                    "dy": round(float(d_[1]), 4),
                    "yaw": round(yaw_, 2),
                    "pivot": [float(p2_[0]), float(p2_[1]), 0.0],
                    "robot_z_off": round(dzi_ * J.step, 4),
                    "sat": 0.0,
                    "checks": self.n_ex,
                    "exhaustive": True,
                    "rank_true": nex_,  # not the flush boundary but
                    # the winner's check rank
                    # observability: the adopted seed's stretch/error/penetration/
                    # swing — gate and audit can see the basis of the verdict.
                    "met": {k: str(v) for k, v in (met_ or {}).items()},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        J.save_ik(clip, self.n_found)
        ck.got_any = True
        self.n_found += 1
        print(
            f"[judge] {clip[:34]} exhaustive adoption {self.n_found}/{a.n_candidates} ({self.n_ex}/{len(self.cands)})",
            flush=True,
        )
        return True

    def _flush_bq(self):
        """Flush the queue: confirm FULL as a batch, adopt the first pass in candidate order.
        True = nothing more to look at (n_candidates satisfied)."""
        J = self.J
        if not self._bq:
            return False
        res = J.track_full_multi(self._bq)
        _pairs_ = list(zip(self._bq, res, strict=True))
        stop = False
        # the guards _adopt_ex will ask for (FULL passes with IK, in this order) are evaluated ahead in workers and
        # taken one by one in the loop below — the loop, its order and its stop are unchanged (judge.guard_prefetch)
        # placement = _adopt_ex's (the clip pivot; the swapped pivot is the same rigid motion — judge._place_key)
        _gj = [
            (it_["cell"][3] * J.step, q_, (it_["dcap"][0], it_["dcap"][1], it_["yaw"], self.ck.cpiv))
            for it_, (ok_, _m, q_, _e) in _pairs_
            if ok_ and q_ is not None
        ]
        with J.guard_prefetch(_gj):
            for _pi, (it_, (ok_, met_, q_, err_)) in enumerate(_pairs_):
                stop = self._flush_one(_pi, it_, ok_, met_, q_, err_, stop)
        self._bq.clear()
        return stop

    def _flush_one(self, _pi, it_, ok_, met_, q_, err_, stop) -> bool:
        """One FULL result of a flush, in candidate order: a pass is adopted (until n_candidates), a rejection
        tallied (branch-and-bound pruning). Returns the flush's stop."""
        J = self.J
        a = J.a
        (ky2, kx2, kyy2, dz2) = it_["cell"]
        if ok_ and not stop:
            if self._adopt_ex(it_["yaw"], it_["dcap"], dz2, q_, met_, it_.get("nex")):
                for _dy in range(-2, 3):
                    for _dx in range(-2, 3):
                        self.pruned.add((ky2, kx2 + _dx, kyy2 + _dy, dz2))
                if self.n_found >= self._ncex:
                    stop = True
            return stop
        if not ok_:
            J._tally("confirm", met_)
            _e2 = max(err_.values()) if err_ else 0.0
            _sl2 = _e2 - a.dp_err_mm - a.prune_radius_mm
            if _sl2 > 0:
                _k2 = int(_sl2 / (J.step * 1000.0))
                if _k2 > 0:
                    for _dy in range(-_k2, _k2 + 1):
                        for _dx in range(-_k2, _k2 + 1):
                            if _dy * _dy + _dx * _dx <= _k2 * _k2:
                                self.pruned.add((ky2, kx2 + _dx, kyy2 + _dy, dz2))
        return stop

    def _drain_gq(self):
        """Judge the gate queue in one go → passes go to the FULL queue (same flow as the
        serial path).
        True = exhaustion ends (enough adoptions)."""
        J, ck = self.J, self.ck
        a = J.a
        traj = ck.traj
        if not self._gq:
            return False
        _gres = J.gate_multi_g(self._gq, traj)
        for it_, r_ in zip(list(self._gq), _gres, strict=True):
            if not r_["ok"]:
                J._tally("sparse", r_["met"])
                continue
            self._bq.append(
                {
                    "piv": it_["piv"],
                    "yaw": it_["yaw"],
                    "dcap": it_["dcap"],
                    "dzm": it_["dzm"],
                    "cell": it_["cell"],
                    "nex": it_["nex"],
                    "gateq": {s2: r_["gateq"].get(s2) for s2 in J.sides},
                }
            )
            if len(self._bq) >= a.batch_cells:
                self._nfl += 1
                if self._flush_bq():
                    self._gq.clear()
                    return True
        self._gq.clear()
        return False

    def _loop(self):
        """Main loop: per cell → gate queue / FULL queue / serial verdict, with branch-and-bound pruning."""
        J, ck = self.J, self.ck
        a = J.a
        clip, traj, cpiv, d0cap = ck.clip, ck.traj, ck.cpiv, ck.d0cap
        for _cvs, _cv, ky_, kx_, kyy_, dzi, _mgs6, _nat7 in self._cl:
            # Per-clip time budget: makes a hard clip finish in **the same time** as an easy
            # one. Exhaustion is an open-ended search by nature, so a
            # cell cap alone leaves the time swinging clip to clip (trajectory length T2 differs).
            # Over budget, the remaining cells go unchecked and it ends as "budget exhausted" — the
            # marker that this is not a certificate is the same as 'cap reached'.
            if a.clip_budget_s > 0 and time.time() - self._t_ex > a.clip_budget_s:
                print(
                    f"[judge] {clip[:30]} exhaustive budget {a.clip_budget_s:.0f}s reached — stopped after "
                    f"{self.n_ex}/{self._start + len(self._cl)} checks",
                    flush=True,
                )
                self._cut_wall = True
                break
            if (ky_, kx_, kyy_, dzi) in self.pruned:
                self.n_pruned += 1
                continue
            yaw = float(J.yaws[ky_])
            bxy = np.array([(kx_ - J.Wx) * J.step, (kyy_ - J.Wy) * J.step, 0.0])
            dcap = J.rz.inv().apply(bxy) + d0cap  # origin displacement
            self.n_ex += 1
            J.n_chk += 1
            if a.batch_cells > 1 and a.gate_batch > 0:
                # Queue the gate too, so the refinement _dls becomes one merged batch.
                self._gq.append(
                    {
                        "piv": cpiv,
                        "yaw": yaw,
                        "dcap": dcap.copy(),
                        "dzm": dzi * J.step,
                        "cell": (ky_, kx_, kyy_, dzi),
                        "nex": self.n_ex,
                    }
                )
                if len(self._gq) >= a.gate_batch and self._drain_gq():
                    break
                continue
            if a.batch_cells > 1:
                # The gate runs on the spot (the barrier that keeps judge = bake consistent), FULL is
                # queued and run through track_full_multi as **one GPU batch** — track quality, B× speed.
                okg, metg = J.dp_confirm(traj, cpiv, yaw, dcap, dzi * J.step, step_mult=float(a.stride * a.dp_stride))
                if not okg:
                    J._tally("sparse", metg)
                    okd, met = False, metg  # shares the exclusion logic below
                else:
                    self._bq.append(
                        {
                            "piv": cpiv,
                            "yaw": yaw,
                            "dcap": dcap.copy(),
                            "dzm": dzi * J.step,
                            "cell": (ky_, kx_, kyy_, dzi),
                            "nex": self.n_ex,  # the true check rank
                            "gateq": {s2: (J.GATEQ[s2].copy() if s2 in J.GATEQ else None) for s2 in J.sides},
                        }
                    )
                    if len(self._bq) >= a.batch_cells:
                        self._nfl += 1
                        if self._flush_bq():
                            break
                    continue
            else:
                okd, met = J.dp_accept(traj, cpiv, yaw, dcap, dzi * J.step)
            if not okd:
                self._reject_cell(met, yaw, dcap, dzi, ky_, kx_, kyy_)
                continue
            if self._serial_adopt(yaw, dcap, dzi, ky_, kx_, kyy_):
                break
            continue

    def _reject_cell(self, met, yaw, dcap, dzi, ky_, kx_, kyy_):
        """A cell rejected on the serial path: branch-and-bound exclusion."""
        J = self.J
        a = J.a
        # Branch-and-bound: the tracking error is 1-Lipschitz in placement movement (moving
        # the placement by 1cm changes the best error by at most 1cm). If the cell-centre error
        # exceeds (threshold + cell radius), **the whole** cell is proved to have no solution,
        # and with more slack the neighbouring cells are proved too. It is an exclusion in
        # **continuous space**, not an enumeration over the grid.
        _e = max(J.LASTERR.values()) if J.LASTERR else 0.0
        _slack = _e - a.dp_err_mm - a.prune_radius_mm
        if _slack > 0:
            _k = int(_slack / (J.step * 1000.0))
            if _k > 0:
                for _dy in range(-_k, _k + 1):
                    for _dx in range(-_k, _k + 1):
                        if _dy * _dy + _dx * _dx > _k * _k:
                            continue
                        self.pruned.add((ky_, kx_ + _dx, kyy_ + _dy, dzi))

    def _serial_adopt(self, yaw, dcap, dzi, ky_, kx_, kyy_) -> bool:
        """Adopt a passing cell on the serial path: pivot swap → final desk check → guard (_pen_cross_guard) → plan
        line · IK save → exclude the neighbours. True = enough candidates (the search ends); False = next cell."""
        J, ck = self.J, self.ck
        a = J.a
        clip, cpiv = ck.clip, ck.cpiv
        p2 = J._adopted_pivot(clip, cpiv)
        th = np.deg2rad(yaw)
        R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        d = np.array([dcap[0], dcap[1]]) + (np.eye(2) - R2) @ (cpiv[:2] - p2)
        if not J.desk_ok_final(ck, d, p2, yaw):
            return False  # off the desk after the pivot swap — next cell
        _gok, _gwhy = J._pen_cross_guard(dzi * J.step, place=(d[0], d[1], yaw, p2))
        if not _gok:
            return False  # the non-batched exhaustion path uses the same guard
        J.fpl.write(
            json.dumps(
                {
                    "clip": clip,
                    "anchor": self.n_found,
                    "dx": round(float(d[0]), 4),
                    "dy": round(float(d[1]), 4),
                    "yaw": round(yaw, 2),
                    "pivot": [float(p2[0]), float(p2[1]), 0.0],
                    "robot_z_off": round(dzi * J.step, 4),
                    "sat": 0.0,
                    "checks": self.n_ex,
                    "exhaustive": True,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        J.save_ik(clip, self.n_found)
        ck.got_any = True
        self.n_found += 1
        print(
            f"[judge] {clip[:34]} exhaustive adoption {self.n_found}/{a.n_candidates} ({self.n_ex}/{len(self.cands)})",
            flush=True,
        )
        if self.n_found >= self._ncex:
            return True
        # The next candidate is sought far enough from this placement — a neighbouring cell is
        # likely to inherit the same collision with f0, which makes the retry meaningless.
        for _dy in range(-2, 3):
            for _dx in range(-2, 3):
                self.pruned.add((ky_, kx_ + _dx, kyy_ + _dy, dzi))
        return False

    def _finish(self):
        """Drain the tail queues, then write the EXHAUSTED / TRUNCATED row and the markers."""
        J, ck = self.J, self.ck
        clip = ck.clip
        # Drain **both** tail queues: a clip with fewer candidates than gate_batch(256) never
        # fills `_bq`, so a condition on `_bq` alone would leave the gate queue `_gq` undrained,
        # run **confirm 0 times** and record "exhaustive failure = certificate".
        if (self._bq or self._gq) and self.n_found < self._ncex:
            self._drain_gq()  # judge the leftover gate queue
            self._flush_bq()  # confirm the tail queue as well
        if not ck.got_any:
            # If the cap was hit it is not 'exhaustive' — it must not be called a certificate.
            # Cells excluded by branch-and-bound (n_pruned) are a **proved
            # exclusion**, so they count towards the coverage.
            _cov = self.n_ex + self.n_pruned
            full = _cov >= len(self.cands)
            tag = "EXHAUSTED" if full else "TRUNCATED"
            _pr = f" (checked {self.n_ex} + bound-pruned {self.n_pruned})" if self.n_pruned else ""  # row data
            _miss_txt2 = f" · best_candidate {J._BEST['txt']}" if J._BEST.get("txt") else ""
            J.fno.write(
                f"{clip}\t{tag}\tcoverage {_cov}/{len(self.cands)} candidates{_pr}"
                f"{' (exhaustive = proof)' if full else ' (cap reached — not a proof)'}"
                f"{_miss_txt2}{reject_tally(J.REJ)}\n"
            )
            print(
                f"[judge] {clip[:34]} exhaustive {_cov}/{len(self.cands)} (checked {self.n_ex} + pruned "
                f"{self.n_pruned}) {'all failed = certificate' if full else 'cap reached (no certificate)'}",
                flush=True,
            )
        # Budget record (owner judge.budget — the verdict is unchanged): hand the cut kind to the judge, which
        # writes the per-clip cell and the marker. Read after the tail drain — the tail can still fill the candidates.
        J.note_budget(
            clip,
            self._budget_cut(),
            f"checked {self.n_ex + self.n_pruned}/{len(self.cands)} · adopted {self.n_found}/{self._ncex} · "
            f"elapsed {time.time() - self._t_ex:.0f}s" + (f" · resumed from {self._start}" if self._start else ""),
        )

    def _budget_cut(self):
        """Was this search cut by a budget? None (no) or a name of budget.KINDS; enough candidates is never a cut.
        wall = the time budget stopped the loop · flush = the loop stopped before the end of the list for another
        reason · cell = the whole list was checked but the ranked list was cut at --exhaust-max."""
        if self.n_found >= self._ncex:
            return None
        if self._cut_wall:
            return "wall"
        if self.n_ex + self.n_pruned < self._start + len(self._cl):
            return "flush"
        if len(self.cands) > self.J.a.exhaust_max:
            return "cell"
        return None
