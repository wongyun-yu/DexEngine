"""Coarse gate of one placement (dp_confirm): per-side targets, pool gate on the sparse tier, gate-path seed for
the all-frame confirm."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexengine.placement.collision import core as SCL
from dexengine.placement.judge.common import _NEXT_SIDE
from dexengine.placement.reach_map import batch_ik as OPG

DEV, DT = OPG.DEV, OPG.DT


class GateDPMixin:
    def dp_confirm(self, traj, piv, yaw, dcap, dzm, step_mult=1.0):
        # step_mult = **how many original frames apart the consecutive frames of this trajectory are**.
        # Sparse = a.stride(5) × a.dp_stride(4) = 20, confirm (FULL) = 1.
        # The speed budget needs both factors: without the --stride 5 factor the sparse budget is
        # 5× too small → stretch 5× too large, the DP jump penalty (50×) misfires, and the
        # gatekeeper becomes 5× stricter than the judge (clips with large motion die before FULL).
        """Confirm by actually running the strided DP extraction — passing here means the bake
        passes too (same algorithm).

        Per side: _dpc_targets (targets), then on the sparse tier (step_mult > 1) _dpc_gate_pool (pool candidates and
        DP), otherwise _dpc_gate_seed (gate-path seed) → _dpc_track → _dpc_score_and_check. Stage return values: see
        _NEXT_SIDE in judge.common."""
        a = self.a
        Ry = Rotation.from_euler("z", yaw, degrees=True)
        st = a.dp_stride
        met = {}
        self.TMPQ.clear()
        if step_mult > 1.0:
            self.GATEQ.clear()  # new placement's gatekeeper starts — the previous placement's path is void
        self.LASTERR.clear()
        for side, S in self.sides.items():
            ws = self._dpc_targets(side, S, traj, piv, Ry, st, dcap, dzm)  # this side's working state
            # Confirm (FULL) **uses neither the pool nor DP.** The seed is a multistart IK solved at
            # the exact target, and a branch switch is also solved directly at that frame — candidate
            # collection and DP here would only produce one fallback proposal q_dp (30% of the
            # profile). Only the sparse tier (the gatekeeper) filters cheaply with the pool and DP.
            if step_mult > 1.0:
                verdict = self._dpc_gate_pool(ws, met, step_mult, dzm)
            else:
                self._dpc_gate_seed(ws)
                verdict = self._dpc_track(ws, step_mult)
                if verdict is None:
                    ws._bud_s = self.BUDGET[side] * step_mult  # measured on the original frame spacing
                    verdict = self._dpc_score_and_check(ws, met, yaw, dcap, dzm, step_mult)
            if verdict == _NEXT_SIDE:
                continue
            if verdict is not None:
                return verdict
        return True, met

    def _dpc_targets(self, side, S, traj, piv, Ry, st, dcap, dzm):
        """Targets: the wrist trajectory (spacing st) moved by the placement (yaw · dcap · dzm) into robot-base poses
        Tt, plus the values the stages share. Working state (ws): side · S · T2 · Tt · pb · qb2 · still · P0 · clr_thr
        · aiN · gi2 · DD."""
        a = self.a
        P0t, R0t = traj[side]
        Pc = Ry.apply(P0t[::st] - piv) + piv + np.array([dcap[0], dcap[1], 0.0])
        Rc = Ry * R0t[::st]
        pb = self.rz.apply(Pc + Rc.apply(self.palm_off)) + np.array([0, 0, self.Z_OFF]) - S["base"]
        pb[:, 2] -= dzm
        Rb = (self.rz * Rc).as_matrix()
        T2 = len(pb)
        Tt = torch.zeros((T2, 4, 4), dtype=DT, device=DEV)
        Tt[:, 3, 3] = 1
        Tt[:, :3, :3] = torch.tensor(Rb, dtype=DT, device=DEV)
        Tt[:, :3, 3] = torch.tensor(pb, dtype=DT, device=DEV)
        P0 = S["P"]
        # Same formula as the volume: dr = desk_rel − dz (lowering the robot gives dz<0 → dr grows)
        clr_thr = (S["desk_rel"] - dzm) - 0.05
        qb2 = (self.rz * Rc).as_quat()
        # Reference-still signal (anti-drift): on frames where the target is effectively
        # standing still (Δposition<2mm ∧ Δrotation<1°), joint motion costs dp_still_boost times more
        # — this directly suppresses, over the still stretches, the self-motion in which the robot
        # wrist unwound 170° while the reference turned only 32°.
        _dpb = np.linalg.norm(np.diff(pb, axis=0), axis=1)
        _dot = np.abs(np.sum(qb2[:-1] * qb2[1:], axis=1)).clip(0, 1)
        still = np.concatenate([[False], (_dpb < 0.002) & (2 * np.degrees(np.arccos(_dot)) < 1.0)])
        # Orientation uses the **top N atoms**. With only 96 atoms, looking at the single nearest
        # one can be off the true orientation by up to 31°, and failing at that one atom alone makes
        # the whole frame "infeasible" (a fake certificate).
        simA = np.abs(qb2 @ P0["atoms"].T)
        aiN = np.argsort(-simA, axis=1)[:, : a.dp_atoms]
        gi2 = np.round((pb - P0["lo"]) / self.step).astype(int)
        DD = np.deg2rad(12)  # candidate dedup threshold — shared by sparse and confirm
        return SimpleNamespace(
            side=side, S=S, T2=T2, Tt=Tt, pb=pb, qb2=qb2, still=still, P0=P0, clr_thr=clr_thr, aiN=aiN, gi2=gi2, DD=DD
        )

    def _dpc_gate_pool(self, ws, met, step_mult, dzm):
        """Sparse gate (step_mult > 1): pool candidates with 12° dedup → refinement → penetration removal → budget
        DP → path. Ends this side (_NEXT_SIDE) or rejects; the path is kept in GATEQ as the confirm seed."""
        a = self.a
        side, S, T2, Tt, P0, clr_thr = ws.side, ws.S, ws.T2, ws.Tt, ws.P0, ws.clr_thr
        aiN, gi2, still = ws.aiN, ws.gi2, ws.still
        cand, off, clrs = self._pool_candidates(gi2, P0, aiN, S, clr_thr, T2)
        if not cand:
            return False, {side: "no_candidate"}
        # clearance deficit (m) for node cost: how far the pool solution's lowest link sinks below the top
        clr_def = np.maximum(0.0, (S["desk_rel"] - dzm) - np.asarray(clrs))
        q0 = torch.tensor(np.stack(cand), dtype=DT, device=DEV)
        idxf = torch.tensor(np.repeat(np.arange(T2), [c for _, c in off]), dtype=torch.long, device=DEV)
        qp, pe, red, mg = S["ik"]._dls(Tt[idxf], 8, 1e-2, 0.0, q0)
        okc = ((pe < 0.01) & (red < 8.0)).cpu().numpy()
        # **Remove penetrating poses from the DP graph** — the final check is "max penetration
        # along the path ≤ threshold", so a pose exceeding threshold + POLICY_MOTION_ALLOWANCE_MM (the
        # policy's motion allowance) cannot enter any passing path. A weight (clr_w) is not enough: a 5cm deficit =
        # cost 0.05 always loses to geometric proximity. A probe confirmed a clean refinement-passing
        # pose exists on every failing clip and every frame (pen_bestok all negative), so this
        # filter removes no solution — it forces the choice.
        # It is a per-pose check, so it is unrelated to the chord problem — valid at the sparse or
        # the confirm resolution alike.
        # Penetration FK runs **only on refinement survivors** — 30~70% of all candidates are
        # already dead at okc, and the 13-point capsule FK is not cheap.
        _qp_np = qp.cpu().numpy()
        _srv = np.flatnonzero(okc)
        if len(_srv):
            _pen_srv = SCL.desk_penetration_each(self.PEN_CTX[side], _qp_np[_srv], z_off=dzm, desk_xy=self._filt_xy())
            lim = SCL.DESK_SURF_PEN_MM + SCL.POLICY_MOTION_ALLOWANCE_MM + self.FILT_SLACK_MM
            okc[_srv[_pen_srv * 1000.0 > lim]] = False
        qp2 = _qp_np
        mgc = mg.cpu().numpy()
        prev_cost, choice, last, order = self._frame_viterbi(T2, off, okc, qp2, mgc, a, clr_def, side, step_mult, still)
        if last < 0:
            return False, {side: "none_after_polish"}
        cov = len(choice) / T2
        if cov < a.dp_cov:
            return False, {side: f"coverage{cov * 100:.0f}%"}
        path = self._backtrack_path(prev_cost, order, choice)
        ks = np.array(sorted(path))
        qsel = np.stack([qp2[path[f]] for f in ks])
        # From here on this is **the same computation** as the bake (stage_dp): frames that had no
        # candidate are filled in by interpolating neighbour choices, the final policy is run once,
        # then error and speed are measured on **every frame**. Measuring only the selected frames
        # hides the missing stretches (a judge 30mm clip can bake at 317mm).
        q_dp = np.empty((T2, qsel.shape[1]), np.float64)
        for j in range(qsel.shape[1]):
            q_dp[:, j] = np.interp(np.arange(T2), ks, qsel[:, j])
        self.GATEQ[side] = q_dp.copy()  # confirm uses this path as its seed
        # The gate runs no final policy: every gate rejection (no candidates, none after the policy, coverage) is
        # decided before it, and no gate threshold reads its outputs (error and stretch are judged on the FULL
        # tier only), so the rejection set is the same without it.
        met[side] = f"coverage{cov * 100:.0f}%"
        return _NEXT_SIDE

    def _pool_candidates(self, gi2, P0, aiN, S, clr_thr, T2):
        """Per frame: valid reach-map candidates of the 27 neighbour cells × N atoms, 12° deduplicated; returns
        (candidates, per-frame (start, count) offsets, clearances)."""
        cand, off = [], []
        DD = np.deg2rad(12)
        # Candidate collection was **80%** of the judging time (27 neighbours × 3 atoms × 4 valleys
        # per frame, run as a 6-deep Python loop). The indexing is folded into one numpy call.
        # The traversal order (dx3 → dy3 → dz3 → atom → valley) is preserved exactly — dedup is
        # order-dependent, so a changed order leaves a different set of candidates.
        g = gi2[:, None, :] + self.OFFS27[None, :, :]  # (T2,27,3)
        inb = ((g >= 0) & (g < [P0["nx"], P0["ny"], P0["nz"]])).all(-1)
        gc = np.clip(g, 0, [P0["nx"] - 1, P0["ny"] - 1, P0["nz"] - 1])
        xi = gc[:, :, 0][:, :, None]
        yi = gc[:, :, 1][:, :, None]
        zi = gc[:, :, 2][:, :, None]
        atN = aiN[:, None, :]  # (T2,1,N)
        okv_all = (
            S["VOK"][xi, yi, zi, atN] & (S["VCLR"][xi, yi, zi, atN] > clr_thr) & inb[:, :, None, None]
        )  # (T2,27,N,K)
        qv_all = S["VQ"][xi, yi, zi, atN]  # (T2,27,N,K,n)
        # keep per-candidate desk clearance (VCLR) in the same gather — for DP node cost (desk steering)
        clr_all = S["VCLR"][xi, yi, zi, atN]  # (T2,27,N,K)
        nj = qv_all.shape[-1]
        okf = okv_all.reshape(T2, -1)
        # Boolean indexing happens **once only** (per frame it would just become the bottleneck again).
        # Valid candidates of all frames are gathered into one row; frame boundaries are cut by offsets.
        _cnt = okf.sum(1)
        _bnd = np.concatenate([[0], np.cumsum(_cnt)])
        _allq = qv_all.reshape(-1, nj)[okf.reshape(-1)].astype(np.float64)
        _allclr = clr_all.reshape(-1)[okf.reshape(-1)].astype(np.float64)
        clrs = []
        for f in range(T2):
            qq = _allq[_bnd[f] : _bnd[f + 1]]
            cc = _allclr[_bnd[f] : _bnd[f + 1]]
            # The 12° sequential dedup, recast as **suppression by the accepted** — a per-candidate
            # loop iterates in Python once per candidate N (hundreds per frame); this one
            # iterates only once per acceptance K (tens). "j is dropped if within DD of an earlier
            # acceptance" ≡ "acceptance i suppresses every candidate within DD" — provably the same set.
            nqq = len(qq)
            ki = []
            if nqq:
                keepm = np.ones(nqq, bool)
                while True:
                    rem = np.flatnonzero(keepm)
                    if not len(rem):
                        break
                    i = int(rem[0])
                    ki.append(i)
                    keepm &= np.abs(qq - qq[i]).max(1) > DD
            off.append((len(cand), len(ki)))
            cand.extend(qq[ki].copy())
            clrs.extend(cc[ki].copy())
        return cand, off, clrs

    def _frame_viterbi(self, T2, off, okc, qp2, mgc, a, clr_def, side, step_mult, still):
        """Frame DP over the surviving candidates (viterbi with the travel/clearance cost)."""
        prev_ids = None
        prev_cost = None
        choice = {}
        last = -1
        order = []  # valid frames in order — so backtracking need not sweep everything each time
        for f in range(T2):
            s0, c0 = off[f]
            ids = (np.flatnonzero(okc[s0 : s0 + c0]) + s0).tolist()
            if not ids:
                continue
            order.append(f)
            qs2 = qp2[ids]
            if prev_ids is None:
                cost = -1e-3 * mgc[ids] + a.dp_clr_w * clr_def[ids]
                back = [-1] * len(ids)
            else:
                dd = np.abs(qs2[:, None, :] - qp2[prev_ids][None, :, :]) / (
                    self.BUDGET[side] * step_mult
                )  # edges use the spacing basis too
                dq = dd.max(-1)  # bottleneck (jump prevention) unchanged
                # Total-travel normalisation (anti-drift): with only the bottleneck, self-motion where many
                # joints each turn a little is free — if the reference stands still, the robot must too
                _tw = a.dp_travel * (1.0 + a.dp_still_boost * float(still[f]))
                pen = np.where(dq > a.dp_budget_mult, dq * 50.0, dq) + _tw * dd.mean(-1)
                tot = pen + prev_cost[None, :]
                barg = tot.argmin(1)
                # the clearance bonus (-margin) and the deficit cost (+clr_def) together pick the valley
                cost = tot[np.arange(len(ids)), barg] - 1e-3 * mgc[ids] + a.dp_clr_w * clr_def[ids]
                back = barg.tolist()
            choice[f] = (ids, back)
            prev_ids, prev_cost = ids, cost
            last = f
        return prev_cost, choice, last, order

    def _backtrack_path(self, prev_cost, order, choice):
        """The best path frame → candidate index, read back from the last frame."""
        j = int(np.argmin(prev_cost))
        path = {}
        # Backtracking walks straight back down the ordered list of valid frames — what back[j]
        # points at is exactly the candidate index of the immediately preceding valid frame (a
        # per-frame `[g for g in choice if g < f]` sweep would be O(T²): 250 × 10⁴ operations at T=1588).
        for oi in range(len(order) - 1, -1, -1):
            f = order[oi]
            ids, back = choice[f]
            path[f] = ids[j]
            if back[j] < 0 or oi == 0:
                break
            j = back[j]
        return path

    def _dpc_gate_seed(self, ws):
        """Confirm (FULL) seed: the path the gate just built (GATEQ), interpolated to the confirm spacing (zeros if
        there is none)."""
        side, T2 = ws.side, ws.T2
        # ── Confirm's seed = **the gatekeeper's successful path**. Confirm uses neither the pool
        # nor DP, but q_dp is seed 0 and the branch-switch proposal — a 0 vector would throw away
        # the path the gatekeeper just assembled out of pool candidates and passed, and restart
        # from random. Random 8 seeds miss an arm with a narrow solution basin (cells holding a full
        # existence proof then end in FULL 'early abort'); interpolating the gatekeeper path into
        # the seed makes tracking start on that trajectory.
        # Grid correspondence: gatekeeper frame i ↔ confirm frame i×a.stride (both share [::dp_stride] and differ
        # only in the input load interval).
        _gq = self.GATEQ.get(side)
        if _gq is not None and len(_gq) > 1:
            _ksg = np.arange(len(_gq)) * self.a.stride
            q_dp = np.empty((T2, _gq.shape[1]), np.float64)
            for j in range(_gq.shape[1]):
                q_dp[:, j] = np.interp(np.arange(T2), _ksg, _gq[:, j])
        else:
            q_dp = np.zeros((T2, len(self.BUDGET[side])), np.float64)
        ws.q_dp = q_dp
