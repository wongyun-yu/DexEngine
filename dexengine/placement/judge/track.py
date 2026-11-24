"""Batched all-frame confirm (track_full_multi): targets, random starts, tracking, post checks, selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexengine.placement.collision import core as SCL
from dexengine.placement.collision import core_torch as SCT
from dexengine.placement.judge import clip_time as _CT  # [CLIP_T] per-clip stage seconds
from dexengine.placement.reach_map import batch_ik as OPG
from dexengine.placement.retime import stretch as VST

DEV, DT = OPG.DEV, OPG.DT


class TrackMixin:
    _ABL = frozenset()  # --track-ablate parts switched off (set at boot; empty = canonical)
    _TFM_STAT = None  # graft / ball counters, only while --track-ablate is given

    @_CT.timed("confirm")
    def track_full_multi(self, plc):
        """Confirm the FULL (track) of several placement candidates as **one
        GPU batch**.

        The only reason track is slow is the sequential per-frame GPU round trips (T2 of
        them), and the number of round trips is independent of batch size — stacking the seeds
        of B candidates into (B×M) rows advances B of them together in one round trip (same
        clip, so T2 is identical). The meaning of each item is the same as the FULL (track)
        branch of a single dp_confirm: same seed count, same reseed rule, same early abort,
        same scoring, same post-checks (error p99 → stretch → desk penetration). Only the
        randomness differs. The gate (sparse) is the
        barrier that keeps judge = bake consistent, so it is **not batched** — the caller puts
        only gate-passing candidates in here.

        plc: [dict(piv, yaw, dcap, dzm, gateq={side: (T2g,n)|None})]
        Returns: [(ok, met, q_both, err)] — q_both is {side: (T2,n)} on a pass, err is the
        per-side ep99 (for the branch-and-bound exclusion radius of a failing cell).

        Per side: _tfm_targets → _tfm_seeds → _tfm_track → _tfm_post_pen → _tfm_select; tk = the side's shared
        values (side · IK · length · budget · candidate indices · result lists).
        """
        a = self.a
        B = len(plc)
        r_ok = [True] * B
        r_met = [{} for _ in range(B)]
        r_q = [{} for _ in range(B)]
        r_err = [{} for _ in range(B)]
        act = list(range(B))
        _M = max(1, a.track_seeds)
        NS = max(a.random_start_mult * _M, a.random_start_min)  # random IK starts (paper: max(4 M, 32))
        DDl = np.deg2rad(12)
        for side, S in self.sides.items():
            if not act:
                break
            _ik = S["ik"]
            nj = _ik.n
            P0t, R0t = (self.FULL if self.FULL else {side: None})[side]
            T2 = len(P0t)
            budF = self.BUDGET[side]
            Ba = len(act)
            tk = SimpleNamespace(
                side=side,
                _ik=_ik,
                nj=nj,
                T2=T2,
                budF=budF,
                Ba=Ba,
                act=act,
                plc=plc,
                _M=_M,
                NS=NS,
                r_ok=r_ok,
                r_met=r_met,
                r_q=r_q,
                r_err=r_err,
            )
            Tt_all, qdp_all = self._tfm_targets(tk, S, P0t, R0t)
            if a.arm_flip:
                tk.Tt_all = Tt_all  # the arm-flip rule in _tfm_select reads the candidates' all-frame targets
            seeds = self._tfm_seeds(tk, Tt_all, qdp_all, DDl)
            E, QQ, alive = self._tfm_track(tk, Tt_all, qdp_all, seeds)
            _vs_pre, _pen_pre = self._tfm_post_pen(tk, E, QQ, alive)
            act = self._tfm_select(tk, QQ, alive, _vs_pre, _pen_pre)
        return [(r_ok[b] and len(r_q[b]) == len(self.sides), r_met[b], r_q[b], r_err[b]) for b in range(B)]

    def _tfm_targets(self, tk, S, P0t, R0t):
        """Per candidate: all-frame targets Tt_all and the gate-path seed qdp_all (interpolated to the confirm
        spacing)."""
        side, act, plc, T2, nj, Ba = tk.side, tk.act, tk.plc, tk.T2, tk.nj, tk.Ba
        Tt_all = torch.zeros((Ba, T2, 4, 4), dtype=DT, device=DEV)
        qdp_all = np.zeros((Ba, T2, nj), np.float64)
        for bi, b in enumerate(act):
            p = plc[b]
            Ry = Rotation.from_euler("z", p["yaw"], degrees=True)
            Pc = Ry.apply(P0t - p["piv"]) + p["piv"] + np.array([p["dcap"][0], p["dcap"][1], 0.0])
            Rc = Ry * R0t
            pb = self.rz.apply(Pc + Rc.apply(self.palm_off)) + np.array([0, 0, self.Z_OFF]) - S["base"]
            pb[:, 2] -= p["dzm"]
            Tt_all[bi, :, 3, 3] = 1
            Tt_all[bi, :, :3, :3] = torch.tensor((self.rz * Rc).as_matrix(), dtype=DT, device=DEV)
            Tt_all[bi, :, :3, 3] = torch.tensor(pb, dtype=DT, device=DEV)
            _gq = (p.get("gateq") or {}).get(side)
            if _gq is not None and len(_gq) > 1 and "coarse" not in self._ABL:
                _ksg = np.arange(len(_gq)) * self.a.stride
                for j in range(nj):
                    qdp_all[bi, :, j] = np.interp(np.arange(T2), _ksg, _gq[:, j])
        return Tt_all, qdp_all

    def _tfm_seeds(self, tk, Tt_all, qdp_all, DDl):
        """Seeds: solve (Ba x NS) random starts at the first frame and keep M per candidate after the 12° dedup."""
        _ik, nj, Ba, NS, _M = tk._ik, tk.nj, tk.Ba, tk.NS, tk._M
        # seed init: (Ba×NS) multistart → M per item via 12° dedup
        q0 = _ik.lo + torch.rand(Ba * NS, nj, dtype=DT, device=DEV, generator=self._JGEN) * _ik.rng_
        _nocoarse = "coarse" in self._ABL  # ablation: every start stays random, the coarse path seeds nothing
        if not _nocoarse:
            q0.view(Ba, NS, nj)[:, 0] = torch.tensor(qdp_all[:, 0], dtype=DT, device=DEV)
        T0 = Tt_all[:, 0].repeat_interleave(NS, 0)
        qs0, pe0, re0, _ = _ik._dls(T0, 60, 1e-2, 0.0, q0)
        ok0 = ((pe0 < 0.010) & (re0 < 8.0)).view(Ba, NS).cpu().numpy()
        qn0 = qs0.view(Ba, NS, nj).cpu().numpy()
        seeds = np.empty((Ba, _M, nj))
        for bi in range(Ba):
            sd = []
            for i0 in np.flatnonzero(ok0[bi]):
                qc = qn0[bi, i0]
                if not sd or min(np.abs(np.asarray(sd) - qc).max(1)) > DDl:
                    sd.append(qc)
                if len(sd) >= _M:
                    break
            if not sd:  # no start reached frame 0: the coarse-path pose — ablation: the closest random start
                sd = [qn0[bi, int(pe0.view(Ba, NS)[bi].argmin())]] if _nocoarse else [qdp_all[bi, 0]]
            while len(sd) < _M:
                sd.append(sd[len(sd) % len(sd)])
            seeds[bi] = np.stack(sd[:_M])
        return seeds

    def _tfm_track(self, tk, Tt_all, qdp_all, seeds):
        """Frame-sequential tracking (B candidates x M seeds per round trip): reseed, branch switch (ball samples),
        early abort. Returns (E per-seed error, QQ per-seed joints, alive candidates); dead candidates are written to
        tk.r_ok / tk.r_met."""
        a = self.a
        side, _ik, nj, T2, Ba, act, _M, NS = tk.side, tk._ik, tk.nj, tk.T2, tk.Ba, tk.act, tk._M, tk.NS
        r_ok, r_met = tk.r_ok, tk.r_met
        Q = torch.tensor(seeds, dtype=DT, device=DEV)  # (Ba,M,n)
        E = np.zeros((Ba, _M, T2), np.float32)
        QQ = np.zeros((Ba, _M, T2, nj), np.float32)
        hope = np.zeros(Ba, int)
        hmax = max(3, int(0.01 * T2))
        rbud = np.full(Ba, int(a.track_reseeds))
        alive = np.ones(Ba, bool)
        qdp_t = torch.tensor(qdp_all, dtype=DT, device=DEV)
        for f_ in range(T2):
            ai = np.flatnonzero(alive)
            if not len(ai):
                break
            idx = torch.tensor(ai, dtype=torch.long, device=DEV)
            # The previous frame's solution — the continuity reference for a rescue (DP graft or
            # random). The branch flips that the jump guard catches are produced by this rescue:
            # switching without a reference only manufactures solutions adoption will reject wholesale.
            Qp = Q[idx].clone() if (a.jump_max > 0 and f_ > 0) else None
            Tf = Tt_all[idx, f_]  # (Na,4,4)
            Qa, pe, re_, _ = _ik._dls(Tf.repeat_interleave(_M, 0), a.track_iters, 1e-2, 0.0, Q[idx].reshape(-1, nj))
            Qb = Qa.view(len(ai), _M, nj)
            pe_v = pe.view(len(ai), _M)
            bad = pe_v > 0.010
            need = bad.any(1).cpu().numpy() & (rbud[ai] > 0)
            if need.any():
                self._reseed_round(need, rbud, ai, Qb, bad, qdp_t, idx, f_, _M, _ik, Tf, nj, pe_v, Qp, a, NS)
            Q[idx] = Qb
            QQ[ai, :, f_] = Qb.cpu().numpy()
            E[ai, :, f_] = pe_v.cpu().numpy()
            pmins = pe_v.min(1).values.cpu().numpy() * 1000.0
            worse = pmins > a.dp_err_mm
            hope[ai[worse]] += 1
            died = hope[ai] > hmax
            for k3 in np.flatnonzero(died):
                b = act[int(ai[k3])]
                r_ok[b] = False
                r_met[b] = {side: f"early_stop(unreachable {int(hope[ai[k3]])} frames)"}
            alive[ai[died]] = False
        return E, QQ, alive

    def _reseed_round(self, need, rbud, ai, Qb, bad, qdp_t, idx, f_, _M, _ik, Tf, nj, pe_v, Qp, a, NS):
        """Reseed the seeds that broke at this frame: re-solve from the coarse path (graft), then ball samples around
        the previous pose."""
        nb = np.flatnonzero(need)
        rbud[ai[nb]] -= 1
        sel = torch.tensor(nb, dtype=torch.long, device=DEV)
        if not (self._ABL & {"coarse", "graft"}):  # ablation: no re-solve from the coarse path
            Q2 = Qb[sel].clone()
            badb = bad[sel]
            prop = qdp_t[idx[sel], f_].unsqueeze(1).expand(-1, _M, -1)
            Q2 = torch.where(badb.unsqueeze(-1), prop, Q2)
            Q2f, pe2, _, _ = _ik._dls(Tf[sel].repeat_interleave(_M, 0), 8, 1e-2, 0.0, Q2.reshape(-1, nj))
            pe2v = pe2.view(len(nb), _M)
            imp = pe2v < pe_v[sel]
            if Qp is not None:  # a DP graft that breaks the branch must not be adopted
                imp = imp & ((Q2f.view(len(nb), _M, nj) - Qp[sel]).norm(dim=-1) <= a.jump_max)
            Qb[sel] = torch.where(imp.unsqueeze(-1), Q2f.view(len(nb), _M, nj), Qb[sel])
            pe_v[sel] = torch.where(imp, pe2v, pe_v[sel])
            if self._TFM_STAT is not None:
                self._TFM_STAT["graft_tried"] += len(nb)
                self._TFM_STAT["graft_taken"] += int(imp.any(1).sum())
        bad2 = pe_v[sel] > 0.010
        self._ball_rescue(bad2, Qp, sel, NS, _M, nj, a, _ik, Tf, Qb, pe_v)

    def _ball_rescue(self, bad2, Qp, sel, NS, _M, nj, a, _ik, Tf, Qb, pe_v):
        """Seeds still broken after the graft: multistart IK samples (inside the jump ball around the previous pose when
        set), the closest valid one per seed."""
        if bool(bad2.any()) and "ball" not in self._ABL:  # ablation: no ball-sample rescue
            if self._TFM_STAT is not None:
                self._TFM_STAT["ball_tried"] += int(bad2.any(1).sum())
            nb2 = torch.nonzero(bad2.any(1)).squeeze(-1)
            Ns2 = int(nb2.numel())
            if Qp is not None:
                # Ball sample around the previous solution instead of the whole
                # joint range — a discontinuous rescue is discarded by the jump guard anyway
                _ctr = Qp[sel][nb2].repeat_interleave((NS + _M - 1) // _M, 1)[:, :NS]
                q3 = (
                    _ctr
                    + (torch.rand(Ns2, NS, nj, dtype=DT, device=DEV, generator=self._JGEN) * 2 - 1) * (a.jump_max * 0.5)
                ).reshape(Ns2 * NS, nj)
                q3 = torch.max(torch.min(q3, _ik.lo + _ik.rng_), _ik.lo)
            else:
                q3 = _ik.lo + torch.rand(Ns2 * NS, nj, dtype=DT, device=DEV, generator=self._JGEN) * _ik.rng_
            q3, pe3, re3, _ = _ik._dls(Tf[sel][nb2].repeat_interleave(NS, 0), 40, 1e-2, 0.0, q3)
            v3 = ((pe3 < 0.010) & (re3 < 8.0)).view(Ns2, NS)
            q3v = q3.view(Ns2, NS, nj)
            pe3v = pe3.view(Ns2, NS)
            for k2 in range(Ns2):
                if not bool(v3[k2].any()):
                    continue
                bs2 = sel[nb2[k2]]
                _ref = Qb[bs2] if Qp is None else Qp[bs2]
                d3 = (q3v[k2][None, :, :] - _ref[:, None, :]).abs().amax(-1)
                if Qp is not None:  # solutions outside the ball are void
                    d3 = torch.where(
                        (q3v[k2][None, :, :] - _ref[:, None, :]).norm(dim=-1) <= a.jump_max,
                        d3,
                        torch.full_like(d3, 1e9),
                    )
                d3 = torch.where(v3[k2][None, :], d3, torch.full_like(d3, 1e9))
                pick = d3.argmin(1)
                take = (pe_v[bs2] > 0.010) & (pe3v[k2][pick] < pe_v[bs2])
                if Qp is not None:
                    take = take & (d3.gather(1, pick[:, None]).squeeze(1) < 1e8)
                Qb[bs2] = torch.where(take[:, None], q3v[k2][pick], Qb[bs2])
                pe_v[bs2] = torch.where(take, pe3v[k2][pick], pe_v[bs2])
                if self._TFM_STAT is not None:
                    self._TFM_STAT["ball_taken"] += int(bool(take.any()))

    def _tfm_post_pen(self, tk, E, QQ, alive):
        """Post-check penetration (FULL) of every surviving (candidate, tried seed) pair in one tensor call.
        Returns (_vs_pre, _pen_pre)."""
        a = self.a
        side, T2, budF, act, plc, _M = tk.side, tk.T2, tk.budF, tk.act, tk.plc, tk._M
        # The FULL post-check penetration (chords ×8, 13 samples, self merge) for every
        # (candidate, tried seed) pair in ONE tensor call. vs/order/tries are computed here once
        # per alive candidate and read below; a tried seed only reaches the penetration check when
        # vs ≤ dp_err_mm, so every pair the loop below asks for is in _pen_pre by construction.
        _vs_pre, _pen_pre = {}, {}
        for bi, _b in enumerate(act):
            if not alive[bi]:
                continue
            vs = np.array(
                [
                    (
                        VST.weighted_p99(
                            E[bi, m_].astype(np.float64) * 1000.0,
                            QQ[bi, m_].astype(np.float64),
                            budF,
                            a.dp_safety,
                            floor=self._ff(1, T2),
                        )
                        if T2 > 1
                        else float(np.percentile(E[bi, m_] * 1000.0, 99))
                    )
                    for m_ in range(_M)
                ]
            )
            order_m = np.argsort(vs)
            tries = [int(m_) for m_ in order_m if vs[m_] <= a.dp_err_mm][:4] or [int(order_m[0])]
            _vs_pre[bi] = (vs, order_m, tries)
        _items = [(bi, m_) for bi in _vs_pre for m_ in _vs_pre[bi][2] if _vs_pre[bi][0][m_] <= a.dp_err_mm]
        if _items:
            _qb = torch.tensor(
                np.stack([QQ[bi, m_].astype(np.float64) for bi, m_ in _items]), dtype=torch.float64, device=DEV
            )
            _zb = torch.tensor([float(plc[act[bi]]["dzm"]) for bi, m_ in _items], dtype=torch.float64, device=DEV)
            _CHf = max(1, self.POST_CHECK_POSES // max(1, T2 * 8))
            _res = []
            for c0 in range(0, len(_items), _CHf):
                with SCL.AP.mesh_ruler():  # the decision — torch twin of DESK_CTX (the original mesh)
                    _res += SCT.desk_penetration_t(
                        self._TCD[side], _qb[c0 : c0 + _CHf], z_off=_zb[c0 : c0 + _CHf], desk_xy=self.DESK_XY
                    )
            _pen_pre = dict(zip(_items, _res))
        return _vs_pre, _pen_pre

    def _tfm_select(self, tk, QQ, alive, _vs_pre, _pen_pre):
        """Check each candidate's tried seeds (error → stretch → jump → penetration → swivel) and adopt the passer with
        the smallest swivel. Results go to tk.r_ok / r_met / r_q / r_err; returns the candidate indices for the next
        side (new_act)."""
        a = self.a
        side, budF, act, plc = tk.side, tk.budF, tk.act, tk.plc
        r_ok, r_met, r_q, r_err = tk.r_ok, tk.r_met, tk.r_q, tk.r_err
        new_act = []
        for bi, b in enumerate(act):
            if not alive[bi]:
                continue
            # ── Design: among passing seeds adopt the **minimum swivel** one.
            # Checking only the seed with the smallest weighted p99 would (1) throw the placement away
            # if that seed died on stretch or penetration, even when another passing seed exists, and
            # (2) ship seeds whose elbow sweeps a large arc even when they pass (measured p90
            # 194°/0.5s after the bake). So the top ≤4 by ascending p99 (within the threshold) go
            # through every check and the minimum-swivel passer is adopted — the passing set only
            # grows versus checking the p99-best seed alone, and only the adopted trajectory improves.
            vs, order_m, tries = _vs_pre[bi]
            fail_met = None
            passing = []
            for m_ in tries:
                qf2 = QQ[bi, m_].astype(np.float64)
                ep99 = float(vs[m_])
                if ep99 > a.dp_err_mm:
                    fail_met = fail_met or {side: f"error_p99={ep99:.0f}mm"}
                    continue
                stretch = VST.total_stretch(qf2, budF, a.dp_safety, floor=self._ff(1, len(qf2)))
                if stretch > a.dp_max_stretch:
                    mx = VST.raw_ratio_max(qf2, budF)
                    fail_met = fail_met or {side: f"stretch_x{stretch:.1f}(max_speed_ratio{mx:.0f})"}
                    continue
                jm_ = float(np.linalg.norm(np.diff(qf2, axis=0), axis=1).max()) if len(qf2) > 1 else 0.0  # jump guard
                if a.jump_max > 0 and jm_ > a.jump_max:
                    fail_met = fail_met or {side: f"jump{jm_:.1f}rad"}
                    continue
                _afp = None
                if a.arm_flip:
                    qf2, _afe, _afw = self._arm_flip_check(side, tk._ik, tk.Tt_all[bi], qf2, budF)
                    if _afw:
                        fail_met = fail_met or {side: _afw}
                        continue
                    if _afe is not None:  # replaced by a continuous solution: score and penetration on it
                        ep99 = _afe
                        stretch = VST.total_stretch(qf2, budF, a.dp_safety, floor=self._ff(1, len(qf2)))
                        jm_ = float(np.linalg.norm(np.diff(qf2, axis=0), axis=1).max())
                        with SCL.AP.mesh_ruler():
                            _afp = SCT.desk_penetration_t(
                                self._TCD[side],
                                torch.tensor(qf2[None], dtype=torch.float64, device=DEV),
                                z_off=torch.tensor([float(plc[b]["dzm"])], dtype=torch.float64, device=DEV),
                                desk_xy=self.DESK_XY,
                            )[0]
                # KeyError here = the ep99 guard above changed
                pen_, _pf, _plk = _pen_pre[(bi, m_)] if _afp is None else _afp
                if pen_ * 1000.0 > SCL.DESK_SURF_PEN_MM:
                    fail_met = fail_met or {
                        side: f"{'torso_pen' if _plk == SCL.TORSO_LINK else 'desk_pen'}{pen_ * 1000:.0f}mm"
                    }
                    continue
                sw_seed = self._swivel_win(side, qf2)
                # If the elbow sweeps the arc by more than --swivel-max-deg within a 0.5 s window it is a motion no
                # human made: reject it, even when it is the minimum-swivel passer.
                if sw_seed > a.swivel_max_deg:
                    fail_met = fail_met or {side: f"swivel{sw_seed:.0f}°>{a.swivel_max_deg:.0f}°"}
                    continue
                passing.append((sw_seed, qf2, ep99, stretch, pen_, jm_))
            r_err[b][side] = float(vs[order_m[0]])
            if not passing:
                r_ok[b] = False
                r_met[b] = fail_met or {side: f"error_p99={float(vs[order_m[0]]):.0f}mm"}
                continue
            sw_, qf2, ep99, stretch, pen_, jm2_ = min(passing, key=lambda t: t[0])
            r_err[b][side] = ep99
            r_met[b][side] = (
                f"stretch{stretch:.2f}x/{ep99:.0f}mm/pen{pen_ * 1000:+.0f}mm/swivel{sw_:.0f}°/jump{jm2_:.1f}"
            )
            r_q[b][side] = qf2
            new_act.append(b)
        return new_act
