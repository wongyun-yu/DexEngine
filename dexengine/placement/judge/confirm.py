"""All-frame confirm of one placement: sequential multi-seed tracking, arm-flip check, scoring and the post checks."""

from __future__ import annotations

import numpy as np
import torch

from dexengine.placement import arm_flip as _AF  # arm-flip gate + re-solve (shared with the bake)
from dexengine.placement.collision import core as SCL
from dexengine.placement.collision import hand_surface as HSF  # finger stretch floor of the clip
from dexengine.placement.judge.args import TRACK_ABLATE_TAG
from dexengine.placement.reach_map import batch_ik as OPG
from dexengine.placement.retime import stretch as VST

DEV, DT = OPG.DEV, OPG.DT


class ConfirmMixin:
    def _dpc_track(self, ws, step_mult):
        """Confirm (FULL): multi-seed continuous tracking (warm-start refinement, branch switch, ball samples, early
        abort)."""
        a = self.a
        if self._ABL:
            raise SystemExit(
                f"{TRACK_ABLATE_TAG} ✗ the serial confirm (_dpc_track) was reached — --track-ablate "
                "is implemented in the batched confirm (track_full_multi) only"
            )
        side, S, T2, Tt, q_dp, DD = ws.side, ws.S, ws.T2, ws.Tt, ws.q_dp, ws.DD
        # ── Multi-seed continuous tracking ───────────────────────────
        # Seeds = the distinct pool candidates of the first valid frame (representatives of the
        # homotopy classes) + the DP solution. Following each seed frame by frame with warm-start
        # refinement moves it **continuously** along the nullspace circle — solutions between the
        # 4 valley points come along naturally.
        # ── The confirm stage **does not use the pool**.
        # Rationale: pool candidates are solutions at a neighbouring cell and an approximate
        # orientation, so they must be refined to the exact target anyway, and the result is
        # "several IK solutions of one wrist pose" — in that case, solving multistart IK straight
        # at the exact target and **sampling the nullspace arc directly** has better coverage and
        # wastes nothing (324 scraped in and refined → most of them discarded).
        # The pool's worth lies in the **pre-screening** that sweeps 145 × 10⁴ placements; for
        # trajectory generation its role as a seed is enough.
        _M = max(1, a.track_seeds)
        _ik = S["ik"]
        _NS = max(a.random_start_mult * _M, a.random_start_min)  # random IK starts (paper: max(4 M, 32))
        _sd = self._track_seed_starts(_NS, _ik, q_dp, Tt, DD, _M)
        _Q = torch.tensor(np.stack(_sd[:_M]), dtype=DT, device=DEV)
        _E = torch.empty((_M, T2), dtype=DT, device=DEV)
        _QQ = torch.empty((_M, T2, q_dp.shape[1]), dtype=DT, device=DEV)
        _hope = 0  # number of frames no seed could bring within the threshold
        _hope_max = max(3, int(0.01 * T2))  # minimum count at which p99 comes to exceed the threshold
        _reseed_budget = int(a.track_reseeds)
        for f_ in range(T2):
            _Qp = _Q.clone() if (a.jump_max > 0 and f_ > 0) else None  # centre of the jump ball (jump_max)
            _Tf = Tt[f_ : f_ + 1].expand(_M, -1, -1).contiguous()
            _Q, _pe, _re, _ = S["ik"]._dls(_Tf, a.track_iters, 1e-2, 0.0, _Q)
            _bad = _pe > 0.010
            if bool(_bad.any()) and _reseed_budget > 0:
                _reseed_budget -= 1
                _Q, _pe = self._switch_branch(_Q, q_dp, f_, _bad, S, _Tf, _pe, _Qp, a, _NS, _ik, _M, Tt)
            _QQ[:, f_] = _Q
            _E[:, f_] = _pe
            # Early abort: once the frames where **every seed** exceeds the threshold pass 1%
            # of T2, the weighted p99 must exceed the threshold — no reason to track to the end
            # (tracking doomed candidates to the end measured 62% of the time, outside the metering).
            if float(_pe.min()) * 1000.0 > a.dp_err_mm:
                _hope += 1
                if _hope > _hope_max:
                    return False, {side: f"early_stop(unreachable {_hope} frames)"}
        _Qn, _En, _bs = self._best_seed(_QQ, _E, _M, T2, side, step_mult, a)
        qf2 = _Qn[_bs]
        pef2 = torch.tensor(_En[_bs] / 1000.0)
        ws.qf2, ws.pef2, ws._En, ws._bs = qf2, pef2, _En, _bs
        return None

    def _track_seed_starts(self, _NS, _ik, q_dp, Tt, DD, _M):
        """The track seeds: valid multistart IK solutions at frame 0 (the DP solution first), padded to M by
        repetition."""
        _rq = torch.rand(_NS, _ik.n, dtype=DT, device=DEV, generator=self._JGEN)
        _q0 = _ik.lo + _rq * _ik.rng_
        _q0[0] = torch.tensor(q_dp[0], dtype=DT, device=DEV)  # the DP solution is a candidate too
        _T0 = Tt[0:1].expand(_NS, -1, -1).contiguous()
        _qs0, _pe0, _re0, _ = _ik._dls(_T0, 60, 1e-2, 0.0, _q0)
        _ok0 = ((_pe0 < 0.010) & (_re0 < 8.0)).cpu().numpy()
        _qn0 = _qs0.cpu().numpy()
        _sd = []
        for _i0 in np.flatnonzero(_ok0):
            _qc = _qn0[_i0]
            if not _sd or min(np.abs(np.asarray(_sd) - _qc).max(1)) > DD:
                _sd.append(_qc)
            if len(_sd) >= _M:
                break
        if not _sd:
            _sd = [q_dp[0]]
        while len(_sd) < _M:
            _sd.append(_sd[len(_sd) % len(_sd)] if len(_sd) else q_dp[0])
        return _sd

    def _switch_branch(self, _Q, q_dp, f_, _bad, S, _Tf, _pe, _Qp, a, _NS, _ik, _M, Tt):
        """Branch switch at a broken frame: the DP proposal, then a multistart IK (inside the jump ball), whichever is
        better per seed."""
        # Branch switch: try both the DP proposal and **a multistart IK at that frame**,
        # and take whichever is better. Where tracking breaks is usually where branches split,
        # so a freshly solved solution at the exact target is what is needed.
        _Q2 = _Q.clone()
        _Q2[_bad] = torch.tensor(q_dp[f_], dtype=DT, device=DEV)
        _Q2, _pe2, _, _ = S["ik"]._dls(_Tf, 8, 1e-2, 0.0, _Q2)
        _imp = _pe2 < _pe
        if _Qp is not None:  # jump ball
            _imp = _imp & ((_Q2 - _Qp).norm(dim=-1) <= a.jump_max)
        if bool(_imp.any()):
            _Q = torch.where(_imp[:, None], _Q2, _Q)
            _pe = torch.where(_imp, _pe2, _pe)
        _bad2 = _pe > 0.010
        if bool(_bad2.any()):
            _rq3 = torch.rand(_NS, _ik.n, dtype=DT, device=DEV, generator=self._JGEN)
            if _Qp is not None:  # sample a ball around the previous solution
                _c3 = _Qp.repeat_interleave((_NS + _M - 1) // _M, 0)[:_NS]
                _q3 = _c3 + (_rq3 * 2 - 1) * (a.jump_max * 0.5)
                _q3 = torch.max(torch.min(_q3, _ik.lo + _ik.rng_), _ik.lo)
            else:
                _q3 = _ik.lo + _rq3 * _ik.rng_
            _Tf3 = Tt[f_ : f_ + 1].expand(_NS, -1, -1).contiguous()
            _q3, _pe3, _re3, _ = _ik._dls(_Tf3, 40, 1e-2, 0.0, _q3)
            _v3 = (_pe3 < 0.010) & (_re3 < 8.0)
            if bool(_v3.any()):
                # switch to the valid solution closest to the previous pose (continuity first)
                _rf = _Q if _Qp is None else _Qp
                _d3 = (_q3[None, :, :] - _rf[:, None, :]).abs().amax(-1)
                if _Qp is not None:  # solutions outside the ball are void
                    _d3 = torch.where(
                        (_q3[None, :, :] - _rf[:, None, :]).norm(dim=-1) <= a.jump_max,
                        _d3,
                        torch.full_like(_d3, 1e9),
                    )
                _d3 = torch.where(_v3[None, :], _d3, torch.full_like(_d3, 1e9))
                _pick = _d3.argmin(1)
                _cand3 = _q3[_pick]
                _pec3 = _pe3[_pick]
                _take = (
                    _bad2
                    & (_pec3 < _pe)
                    & (_d3.gather(1, _pick[:, None]).squeeze(1) < 1e8 if _Qp is not None else torch.ones_like(_bad2))
                )
                if bool(_take.any()):
                    _Q = torch.where(_take[:, None], _cand3, _Q)
                    _pe = torch.where(_take, _pec3, _pe)
        return _Q, _pe

    def _best_seed(self, _QQ, _E, _M, T2, side, step_mult, a):
        """Score each seed's trajectory by the bake's criterion (weighted p99); returns the joints, errors (mm) and the
        best seed."""
        # score each seed's trajectory by the same criterion as the bake (weighted p99) → take the best
        _Qn = _QQ.cpu().numpy()
        _En = _E.cpu().numpy() * 1000.0
        _bs, _bv = 0, float("inf")
        for m_ in range(_M):
            _v = (
                VST.weighted_p99(
                    _En[m_], _Qn[m_], self.BUDGET[side] * step_mult, a.dp_safety, floor=self._ff(step_mult, T2)
                )
                if T2 > 1
                else float(np.percentile(_En[m_], 99))
            )
            if _v < _bv:
                _bs, _bv = m_, _v
        return _Qn, _En, _bs

    def _ff(self, step_mult, T2) -> np.ndarray:
        """This clip's finger stretch floor for a trajectory of T2 frames `step_mult` original frames apart —
        hand_surface.finger_floor_for, the function retime.py and stage_dp call (judge = bake). The finger columns
        come from the f0 shadow tree (--traj-root) and the arms from the wrist cache (--wrist-root): both must sit on
        one time axis, and the trajectory must be what step_mult says (else it dies — no silent re-fit)."""
        step = int(round(step_mult))
        if step not in self._FFC:
            n_w = len(next(iter(self.FULL.values()))[0]) if self.FULL else None
            n_f = {s: len(f) for s, f in self._FFQ.items() if len(f) != n_w}  # every block the tree has (one-hand too)
            if n_f:
                raise RuntimeError(
                    f"[judge] {self.clip}: finger columns {n_f} (tree) ≠ wrist cache {n_w}f — "
                    "different time axes (--traj-root / --wrist-root generation)"
                )
            self._FFC[step] = HSF.finger_floor_for(None, self.a.hand, self.a.dp_safety, step=step, fq=self._FFQ)
        ff = self._FFC[step]
        if len(ff) != max(int(T2) - 1, 1):
            raise RuntimeError(
                f"[judge] {self.clip}: finger floor {len(ff)} segments (spacing {step}) ≠ joint "
                f"trajectory {T2}f — this trajectory's frame spacing is not step_mult"
            )
        return ff

    def _arm_flip_check(self, side, ik, Tt, qf2, bud):
        """Apply the arm-flip rule to one all-frame trajectory about to be confirmed: arm_flip.fix (the bake's
        function and thresholds) with the judge's wrist-error and stretch thresholds. Returns (q, weighted p99 error
        mm when q was replaced | None, reject reason | None):
          off (--arm-flip 0)                       → (q unchanged, None, None)
          gate not hit · rotation follows the hand → (q unchanged, None, None)
          replaced by a continuous solution         → (new q, its error, None) — later checks use the new q
          no continuous candidate in the thresholds → (q, None, "arm_flip<deg>°") — the caller drops this seed"""
        a = self.a
        if not a.arm_flip:
            return qf2, None, None
        _fl = self._ff(1, len(qf2))
        q2, _pe, _re, info = _AF.fix(
            ik,
            Tt,
            qf2,
            err_mm_fn=lambda e, q: VST.weighted_p99(e, q, bud, a.dp_safety, floor=_fl),
            err_mm_max=a.dp_err_mm,
            stretch_fn=lambda q: VST.total_stretch(q, bud, a.dp_safety, floor=_fl),
            stretch_max=a.dp_max_stretch,
            polish_iters=8,
            tag=f"{self.clip}/{side}",
            torch=torch,
        )
        if info is None:
            return qf2, None, None
        if q2 is not qf2:
            _jm = float(np.linalg.norm(np.diff(q2, axis=0), axis=1).max())
            if not (a.jump_max > 0 and _jm > a.jump_max):
                return q2, float(info["err_mm"]), None
            print(
                f"[ARM_FLIP] {self.clip}/{side}: the replaced trajectory exceeds the jump threshold "
                f"({_jm:.2f} > {a.jump_max:g} rad) → replacement undone · candidate dropped",
                flush=True,
            )
            info = dict(info, verdict=_AF.VERDICT_KEPT, feasible=0)  # the replacement jumps one step: not continuous
        if _AF.unresolved(info):
            return qf2, None, f"arm_flip{np.rad2deg(info['travel']):.0f}°"
        return qf2, None, None

    def _dpc_score_and_check(self, ws, met, yaw, dcap, dzm, step_mult):
        """Score (weighted p99 after retiming) and the post checks of the all-frame confirm: error → stretch → jump
        → arm flip → desk / torso penetration. A pass writes met[side]."""
        a = self.a
        side, T2, qf2, pef2, _bud_s = ws.side, ws.T2, ws.qf2, ws.pef2, ws._bud_s
        self.TMPQ[side] = qf2
        _e_mm = pef2.cpu().numpy() * 1000.0
        # same computation as the bake: judged on the post-retiming distribution (v5_stretch.weighted_p99)
        _ffl = self._ff(step_mult, T2)  # the fingers' stretch floor — the same as retime.py's (judge = bake)
        ep99 = (
            VST.weighted_p99(_e_mm, qf2, _bud_s, a.dp_safety, floor=_ffl) if T2 > 1 else float(np.percentile(_e_mm, 99))
        )
        self.LASTERR[side] = ep99
        # Error and stretch are judged here, on all frames, only: the sparse gate assembles from candidates 20
        # frames apart and its error is larger at the same placement, so it never rejects on them.
        if ep99 > a.dp_err_mm:
            return False, {side: f"error_p99={ep99:.0f}mm"}
        # Retiming comes afterwards, so the question is not "is it slow now" but "by what factor
        # does it stretch". A stretch running k times the limit fits exactly when lengthened k times.
        # stride makes it sparse, so the budget grows by the frame spacing (× stride) — reflected here.
        stretch = VST.total_stretch(qf2, _bud_s, a.dp_safety, floor=_ffl)
        mx = VST.raw_ratio_max(qf2, _bud_s)  # arm only: a placement's branch-flip spike (fingers: clip constant)
        if stretch > a.dp_max_stretch:
            return False, {side: f"stretch_x{stretch:.1f}(max_speed_ratio{mx:.0f})"}
        # Jump (branch flip) guard — a defect wrist error, stretch and penetration all miss.
        # When two branches leave the wrist identical the error stays continuous, stretch is a mean
        # and is insensitive, and penetration is sampled only at judged frames — the sweep appears
        # only when retiming interpolates between the jumps. The absolute amount (rad/frame) is
        # the only quantity that separates the layers.
        jm_ = float(np.linalg.norm(np.diff(qf2, axis=0), axis=1).max()) if len(qf2) > 1 else 0.0
        if a.jump_max > 0 and jm_ > a.jump_max:
            return False, {side: f"jump{jm_:.1f}rad"}
        if a.arm_flip:
            qf2, _afe, _afw = self._arm_flip_check(side, ws.S["ik"], ws.Tt, qf2, _bud_s)
            if _afw:
                return False, {side: _afw}
            if _afe is not None:  # replaced by a continuous solution: confirm and score the new trajectory
                ws.qf2 = qf2
                self.TMPQ[side] = qf2
                ep99 = _afe
                self.LASTERR[side] = ep99
                stretch = VST.total_stretch(qf2, _bud_s, a.dp_safety, floor=_ffl)
                jm_ = float(np.linalg.norm(np.diff(qf2, axis=0), axis=1).max())
        # Arm-desk penetration: checked only on candidates that passed error and stretch
        # (≈70ms per side). Judge = bake = gate use the same function and threshold (SCL.DESK_SURF_PEN_MM).
        # Penetration is checked on the all-frame trajectory only: retiming interpolates the chord at the original
        # frame spacing, which is what this sweep models (a sparse chord overestimates without bound).
        with SCL.AP.mesh_ruler():  # the decision — the original mesh (DESK_CTX), not the search proxy
            _pen, _pf, _plk, _pdk, _ptr = SCL.desk_penetration(
                self.DESK_CTX[side], qf2, z_off=dzm, want_components=True, desk_xy=self.DESK_XY
            )
        if _pen * 1000.0 > SCL.DESK_SURF_PEN_MM:
            return False, {side: f"{'torso_pen' if _plk == SCL.TORSO_LINK else 'desk_pen'}{_pen * 1000:.0f}mm"}
        # met fields (data, parsed downstream): pen = desk penetration · torso = torso overlap (own ruler)
        met[side] = (
            f"stretch{stretch:.2f}x/{ep99:.0f}mm/pen{_pdk * 1000:+.0f}mm/torso{_ptr * 1000:+.0f}mm/jump{jm_:.1f}"
        )
        return None
