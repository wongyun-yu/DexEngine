"""Acceptance of a placement (sparse gate, then the all-frame confirm), rejection tally, swivel window, and the
batched gate over many cells (gate_multi_g)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexengine.core.paths import ikc_name
from dexengine.placement.collision import core as SCL
from dexengine.placement.collision import core_torch as SCT
from dexengine.placement.collision import geom as SCG
from dexengine.placement.judge import clip_time as _CT  # [CLIP_T] per-clip stage seconds
from dexengine.placement.judge.common import DD12, _dedup_side_batch
from dexengine.placement.reach_map import batch_ik as OPG

DEV, DT = OPG.DEV, OPG.DT


class AcceptMixin:
    def _tally(self, stage, met):
        """Tally of rejection reasons — you can only fix what you know is dropping placements."""
        for side, why in (met or {}).items():
            for key in (
                "no_candidate",
                "none_after_polish",
                "coverage",
                "stretch",
                "error",
                "desk_pen",
                "torso_pen",
                "early_pen",
                "early_stop",
                "frame_infeasible",
                "desk_pen",
                "speed_ratio",
                "guard",
                "jump",
                "desk_xy",
                "swivel",  # last stage of check_batch (swivel)
                "arm_flip",
            ):
                if why.startswith(key):
                    self.REJ[f"{stage}/{key}"] = self.REJ.get(f"{stage}/{key}", 0) + 1
                    _rank = self._MISS_RANK.get(key)
                    if _rank is not None and _rank >= self._BEST.get("rank", -1):
                        self._BEST["rank"] = _rank
                        self._BEST["txt"] = f"{stage}/{side}:{why}"
                    return
            # Reasons absent from the list are not discarded either (a guard failure that does not
            # arrive under the guard key would otherwise leave the failure with no reason at all)
            self.REJ[f"{stage}/{why[:16]}"] = self.REJ.get(f"{stage}/{why[:16]}", 0) + 1
            self.REJ[f"{stage}/other"] = self.REJ.get(f"{stage}/other", 0) + 1
            return

    def save_ik(self, clip, anchor):
        """Save the all-frame joint trajectory of the adopted placement — the bake reuses it as-is."""
        a = self.a
        if not a.save_ik or not self.LASTQ:
            return
        d = Path(a.save_ik)
        d.mkdir(parents=True, exist_ok=True)
        for side, q in self.LASTQ.items():
            np.savez(
                d / ikc_name(clip, anchor, side),  # canonical name owner = paths.ikc_name
                arm_joint_positions_all=q.astype(np.float32),
                num_frames=np.int32(len(q)),
            )

    @_CT.timed("confirm")
    def dp_accept(self, traj, piv, yaw, dcap, dzm):
        """Adoption verdict — filter fast on the sparse tier (stride N), and confirm only what
        passed at **stride 1, the same as the bake**.

        Why: stride 4 looks at only 1 frame in 4. If the valley changes
        or a candidate is missing in the other 3, the judge passes it without knowing, and the
        bake (stride 1) fails only then. The evidence for adoption must always be stride 1 for
        judge = bake to hold. The 1st sparse check exists only to throw candidates away
        cheaply; it is not evidence for adoption.
        """
        a = self.a
        ok, met = self.dp_confirm(traj, piv, yaw, dcap, dzm, step_mult=float(a.stride * a.dp_stride))
        if not ok:
            self._tally("sparse", met)
            return ok, met
        # Confirm runs on **all frames** (FULL). traj is already thinned by --stride 5, so even
        # with dp_stride=1 it sees only 20% of the original — bake failures are reachability
        # failures on 2~3% of the frames mid-clip that are not in the judge's sample.
        _st = a.dp_stride
        a.dp_stride = 1
        try:
            ok1, met1 = self.dp_confirm(self.FULL if self.FULL else traj, piv, yaw, dcap, dzm, step_mult=1.0)
        finally:
            a.dp_stride = _st
        if ok1:
            self.LASTQ.clear()
            self.LASTQ.update(self.TMPQ)  # all-frame solution — the bake reuses it
        else:
            self._tally("confirm", met1)
        return ok1, met1

    def _swivel_win(self, side, q):
        """Largest change of the swivel over a 0.5s window (degrees) — the speed at which the elbow
        travels the arc around the shoulder-wrist axis (the swing metric, same maths
        as drift_audit). Undefined near the singularity (<3cm)."""
        P = SCG.arm_link_transforms(self.PEN_CTX[side], q)[:, :, :3, 3]
        sh, el, wr = P[:, 2], P[:, 3], P[:, 4]
        ax = wr - sh
        ax = ax / (np.linalg.norm(ax, axis=1, keepdims=True) + 1e-9)
        v = el - sh
        vp = v - (v * ax).sum(1, keepdims=True) * ax
        r = np.linalg.norm(vp, axis=1)
        ref = np.tile([0.0, 0.0, 1.0], (len(P), 1))
        ref = ref - (ref * ax).sum(1, keepdims=True) * ax
        ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
        sw = np.arctan2((vp * np.cross(ax, ref)).sum(1), (vp * ref).sum(1))
        d = np.abs(np.diff(np.degrees(np.unwrap(sw))))
        if len(d) == 0:
            return 0.0
        d[np.minimum(r[:-1], r[1:]) < 0.03] = 0.0
        W = min(25, len(d))
        return float(np.convolve(d, np.ones(W), "valid").max())

    # ══ Gate cell batching ══════════════════════
    # In the gate path of dp_confirm [collect → refine → assemble], the refinement _dls is merged across N cells.
    # stage1 / stage2 below compute what dp_confirm's gate branch (_dpc_gate_pool) computes per cell; the two must
    # stay in sync.
    def _gate_stage1(self, side, S, traj, piv, yaw, dcap, dzm):
        a = self.a
        Ry = Rotation.from_euler("z", yaw, degrees=True)
        st = a.dp_stride
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
        clr_thr = (S["desk_rel"] - dzm) - 0.05
        qb2 = (self.rz * Rc).as_quat()
        _dpb = np.linalg.norm(np.diff(pb, axis=0), axis=1)
        _dot = np.abs(np.sum(qb2[:-1] * qb2[1:], axis=1)).clip(0, 1)
        still = np.concatenate([[False], (_dpb < 0.002) & (2 * np.degrees(np.arccos(_dot)) < 1.0)])
        simA = np.abs(qb2 @ P0["atoms"].T)
        aiN = np.argsort(-simA, axis=1)[:, : a.dp_atoms]
        gi2 = np.round((pb - P0["lo"]) / self.step).astype(int)
        g = gi2[:, None, :] + self.OFFS27[None, :, :]
        inb = ((g >= 0) & (g < [P0["nx"], P0["ny"], P0["nz"]])).all(-1)
        gc = np.clip(g, 0, [P0["nx"] - 1, P0["ny"] - 1, P0["nz"] - 1])
        xi = gc[:, :, 0][:, :, None]
        yi = gc[:, :, 1][:, :, None]
        zi = gc[:, :, 2][:, :, None]
        atN = aiN[:, None, :]
        okv_all = S["VOK"][xi, yi, zi, atN] & (S["VCLR"][xi, yi, zi, atN] > clr_thr) & inb[:, :, None, None]
        qv_all = S["VQ"][xi, yi, zi, atN]
        clr_all = S["VCLR"][xi, yi, zi, atN]
        nj = qv_all.shape[-1]
        okf = okv_all.reshape(T2, -1)
        _cnt = okf.sum(1)
        _bnd = np.concatenate([[0], np.cumsum(_cnt)])
        _allq = qv_all.reshape(-1, nj)[okf.reshape(-1)].astype(np.float64)
        _allclr = clr_all.reshape(-1)[okf.reshape(-1)].astype(np.float64)
        if not len(_allq):  # greedy dedup keeps ≥ 1 row per non-empty frame → "none" is decided before it
            return None, "no_candidate"
        # The 12° dedup runs once per side over all cells (gate_multi_g → _dedup_side_batch →
        # _gate_stage1_finish); this cell's raw rows wait in `pre`.
        return {"pre": (_allq, _allclr, _bnd, T2), "Tt": Tt, "T2": T2, "still": still, "dzm": dzm}, None

    def _gate_stage1_finish(self, side, S, st_, cand, off, clrs):
        """Second half of _gate_stage1 after the side-batched dedup: q0 / idxf / off / clr_def of the cell
        (same arrays the per-cell version built from the list form)."""
        T2 = st_["T2"]
        st_["q0"] = np.asarray(cand)
        st_["idxf"] = np.repeat(np.arange(T2), [c for _, c in off])
        st_["off"] = off
        st_["clr_def"] = np.maximum(0.0, (S["desk_rel"] - st_["dzm"]) - np.asarray(clrs))
        del st_["pre"]

    def _okc_of(self, side, qp2, pe2, red2, pen_pre, dzm):
        """Stage-2 row survival (position error < 1 cm · rotation < 8° · desk penetration ≤ DESK_SURF_PEN_MM +
        POLICY_MOTION_ALLOWANCE_MM + the filter slack)."""
        okc = (pe2 < 0.01) & (red2 < 8.0)
        _srv = np.flatnonzero(okc)
        if len(_srv):
            _pen_srv = pen_pre[_srv]  # computed for the whole gate group in one tensor call (gate_multi_g)
            lim = SCL.DESK_SURF_PEN_MM + SCL.POLICY_MOTION_ALLOWANCE_MM + self.FILT_SLACK_MM
            okc[_srv[_pen_srv * 1000.0 > lim]] = False
        return okc

    def _gate_stage2_batch(self, side, S, cells):
        """The gate's frame DP (viterbi over the surviving candidates of each frame) for **every cell of a
        side at once**. cells = [(st_, qp2, pe2, red2, mgc, pen_pre)] (the cell's row
        slices of the group). Returns [(ok, met, q_dp)] = what the per-cell frame DP of _dpc_gate_pool gives:
        the transition cost is the same arithmetic in the same order (float32 sub/abs → float64 /budget →
        max · sequential 7-term mean · where · +prev · argmin first index), prev carries over empty frames
        exactly as the per-cell loop does, and the backtracking / interp tail is the per-cell one, in numpy."""
        a = self.a
        C = len(cells)
        if not C:
            return []
        step_mult = float(a.stride * a.dp_stride)
        T2 = int(cells[0][0]["T2"])
        # per cell: ids_tab[f, k] = cell-local row of the k-th surviving candidate of frame f (ascending; -1 pad)
        ncnt = np.zeros((C, T2), np.int64)
        tabs = []
        for c, (st_, qp2, pe2, red2, _mgc, pen_pre) in enumerate(cells):
            ok_rows = np.flatnonzero(self._okc_of(side, qp2, pe2, red2, pen_pre, st_["dzm"]))
            fr = st_["idxf"][ok_rows]
            cnt = np.bincount(fr, minlength=T2)
            ncnt[c] = cnt
            pos = np.arange(len(ok_rows)) - np.concatenate([[0], np.cumsum(cnt)[:-1]])[fr]
            tabs.append((fr, pos, ok_rows))
        Nmax = int(ncnt.max())
        out = [None] * C
        if Nmax == 0:
            return [(False, "none_after_polish", None)] * C
        ids_tab = np.full((C, T2, Nmax), -1, np.int64)
        for c, (fr, pos, ok_rows) in enumerate(tabs):
            ids_tab[c, fr, pos] = ok_rows
        nrow = np.array([len(cl[1]) for cl in cells])
        offs = np.concatenate([[0], np.cumsum(nrow)[:-1]])
        nj = cells[0][1].shape[1]
        qp_t = torch.tensor(np.concatenate([cl[1] for cl in cells]), dtype=torch.float32, device=DEV)
        mg_t = torch.tensor(np.concatenate([cl[4] for cl in cells]), dtype=torch.float32, device=DEV)
        clr_t = torch.tensor(np.concatenate([cl[0]["clr_def"] for cl in cells]), dtype=torch.float64, device=DEV)
        gid_np = np.where(ids_tab >= 0, ids_tab + offs[:, None, None], -1)
        still_np = np.stack([np.asarray(cl[0]["still"], bool) for cl in cells])  # (C, T2)
        Bs = torch.tensor(self.BUDGET[side] * step_mult, dtype=torch.float64, device=DEV)  # (nj,)
        tw0 = torch.tensor(a.dp_travel * (1.0 + a.dp_still_boost * 0.0), dtype=torch.float64, device=DEV)
        tw1 = torch.tensor(a.dp_travel * (1.0 + a.dp_still_boost * 1.0), dtype=torch.float64, device=DEV)
        inf = float("inf")
        back_np = np.full((C, T2, Nmax), -1, np.int64)
        pc_np = np.full((C, Nmax), inf, np.float64)
        # cell chunk: dd (CH,Nmax,Nmax,nj) float64 ≤ 256 MiB → transient peak ≈ 384 MiB (float32 diff + float64 dd, both
        # in place). A 1 GiB chunk with out-of-place abs/double/div peaks ≈ 2.5 GiB and can OOM a shared GPU.
        CH = max(1, int(2**28 // max(1, Nmax * Nmax * nj * 8)))
        self._dp_forward(C, CH, gid_np, still_np, Nmax, nj, inf, T2, qp_t, mg_t, clr_t, Bs, tw1, tw0, a, back_np, pc_np)
        self._dp_backtrack(cells, ncnt, out, T2, a, pc_np, ids_tab, back_np)
        return out

    def _dp_forward(
        self, C, CH, gid_np, still_np, Nmax, nj, inf, T2, qp_t, mg_t, clr_t, Bs, tw1, tw0, a, back_np, pc_np
    ):
        """Chunked frame DP forward pass over the cells (cell chunks sized to the memory cap); fills back_np and
        pc_np."""
        for c0 in range(0, C, CH):
            c1 = min(C, c0 + CH)
            gid = torch.tensor(gid_np[c0:c1], dtype=torch.long, device=DEV)  # (Cc,T2,Nmax)
            valid_all = gid >= 0
            gsafe = gid.clamp(min=0)
            still_t = torch.tensor(still_np[c0:c1], dtype=torch.bool, device=DEV)
            Cc = c1 - c0
            prev_q = torch.zeros((Cc, Nmax, nj), dtype=torch.float32, device=DEV)
            prev_cost = torch.full((Cc, Nmax), inf, dtype=torch.float64, device=DEV)
            has_prev = torch.zeros(Cc, dtype=torch.bool, device=DEV)
            back_all = torch.full((Cc, T2, Nmax), -1, dtype=torch.long, device=DEV)
            for f in range(T2):
                vf = valid_all[:, f]  # (Cc,Nmax)
                has = vf.any(1)  # (Cc,)
                g = gsafe[:, f]
                q_f = qp_t[g]  # (Cc,Nmax,nj) float32
                mg_f = mg_t[g]  # (Cc,Nmax)   float32
                clr_f = clr_t[g]  # (Cc,Nmax)   float64
                dd = (q_f[:, :, None, :] - prev_q[:, None, :, :]).abs_().double().div_(Bs)  # (Cc,Nmax,Nmax,nj)
                dq = dd.amax(-1)
                s = dd[..., 0]
                for j in range(1, nj):  # numpy's n<8 add.reduce order
                    s = s + dd[..., j]
                m = s / nj
                tw = torch.where(still_t[:, f], tw1, tw0)  # (Cc,)
                pen = torch.where(dq > a.dp_budget_mult, dq * 50.0, dq) + tw[:, None, None] * m
                tot = pen + prev_cost[:, None, :]
                barg = tot.argmin(-1)  # (Cc,Nmax) first minimal index
                best = tot.gather(-1, barg[..., None]).squeeze(-1)
                cost_new = best - 1e-3 * mg_f + a.dp_clr_w * clr_f
                cost_first = -1e-3 * mg_f + a.dp_clr_w * clr_f
                cost_f = torch.where(has_prev[:, None], cost_new, cost_first)
                cost_f = torch.where(vf, cost_f, inf)
                back_all[:, f] = torch.where(has_prev[:, None], barg, -1)
                prev_q = torch.where(has[:, None, None], q_f, prev_q)
                prev_cost = torch.where(has[:, None], cost_f, prev_cost)
                has_prev = has_prev | has
            back_np[c0:c1] = back_all.cpu().numpy()
            pc_np[c0:c1] = prev_cost.cpu().numpy()

    def _dp_backtrack(self, cells, ncnt, out, T2, a, pc_np, ids_tab, back_np):
        """Backtrack each cell's best path and expand it to every frame; fills out."""
        for c, (_st, qp2, _pe2, _red2, _mgc, _pen_pre) in enumerate(cells):
            cnt = ncnt[c]
            order = np.flatnonzero(cnt > 0)
            if not len(order):
                out[c] = (False, "none_after_polish", None)
                continue
            cov = len(order) / T2
            if cov < a.dp_cov:
                out[c] = (False, f"coverage{cov * 100:.0f}%", None)
                continue
            j = int(np.argmin(pc_np[c, : cnt[order[-1]]]))
            path = {}
            for oi in range(len(order) - 1, -1, -1):
                f = int(order[oi])
                path[f] = int(ids_tab[c, f, j])
                bj = int(back_np[c, f, j])
                if bj < 0 or oi == 0:
                    break
                j = bj
            ks = np.array(sorted(path))
            qsel = np.stack([qp2[path[f]] for f in ks])
            q_dp = np.empty((T2, qsel.shape[1]), np.float64)
            for jj in range(qsel.shape[1]):
                q_dp[:, jj] = np.interp(np.arange(T2), ks, qsel[:, jj])
            out[c] = (True, f"coverage{cov * 100:.0f}%", q_dp)

    def gate_multi_g(self, items, traj):
        """N-cell batched gate — refinement _dls merged per side. Returns: (ok, met, gateq) per item."""
        res = [{"ok": True, "met": {}, "gateq": {}} for _ in items]
        group = {sd: [] for sd in self.sides}
        for i, it in enumerate(items):
            for side, S in self.sides.items():
                st_, fail = self._gate_stage1(side, S, traj, it["piv"], it["yaw"], it["dcap"], it["dzm"])
                if fail:
                    res[i]["ok"] = False
                    res[i]["met"] = {side: fail}
                    break
                group[side].append((i, st_))
        # The 12° dedup of every surviving cell of a side in ONE tensor call
        # (rows are independent → the same picks as one call per cell).
        for side, S in self.sides.items():
            _g = [(i, st_) for (i, st_) in group[side] if res[i]["ok"]]
            for (_i, st_), (cand, off, clrs) in zip(
                _g, _dedup_side_batch([st_["pre"] for _, st_ in _g], DD12), strict=True
            ):
                self._gate_stage1_finish(side, S, st_, cand, off, clrs)
        for side, S in self.sides.items():
            lst = [(i, st_) for (i, st_) in group[side] if res[i]["ok"]]
            if not lst:
                continue
            rows = torch.cat([st_["Tt"][torch.tensor(st_["idxf"], dtype=torch.long, device=DEV)] for _, st_ in lst])
            q0 = torch.tensor(np.concatenate([st_["q0"] for _, st_ in lst]), dtype=DT, device=DEV)
            # The intermediate tensors inside _dls (Jacobians and such) are tens of
            # times the row count, so a large batch blows up the GPU (measured as a simultaneous spike
            # across 16 shards) — it is called in row chunks. The op is row-independent, so the result
            # is identical.
            CH = 49152
            _qs, _ps, _rs, _ms = [], [], [], []
            for c0 in range(0, len(q0), CH):
                qp_, pe_, red_, mg_ = S["ik"]._dls(rows[c0 : c0 + CH], 8, 1e-2, 0.0, q0[c0 : c0 + CH])
                _qs.append(qp_.cpu().numpy())
                _ps.append(pe_.cpu().numpy())
                _rs.append(red_.cpu().numpy())
                _ms.append(mg_.cpu().numpy())
            qp = np.concatenate(_qs)
            pe = np.concatenate(_ps)
            red = np.concatenate(_rs)
            mg = np.concatenate(_ms)
            # The stage2 per-row penetration for every cell of the group in one tensor call
            # (rows × caps × samples), per-row z_off = the cell's dzm; _gate_stage2 reads its slice.
            _okc_all = (pe < 0.01) & (red < 8.0)
            _srv_all = np.flatnonzero(_okc_all)
            pen_all = np.full(len(qp), np.nan)
            if len(_srv_all):
                _zrow = np.concatenate([np.full(len(st_["q0"]), float(st_["dzm"])) for _, st_ in lst])
                _qs64 = torch.tensor(qp[_srv_all].astype(np.float64), dtype=torch.float64, device=DEV)
                _zs64 = torch.tensor(_zrow[_srv_all], dtype=torch.float64, device=DEV)
                _CHp = self.PEN_EACH_CHUNK
                _fxy = self._filt_xy()
                _pp = [
                    SCT.desk_penetration_each_t(
                        self._TC[side], _qs64[c0 : c0 + _CHp], z_off=_zs64[c0 : c0 + _CHp], desk_xy=_fxy
                    )
                    for c0 in range(0, len(_srv_all), _CHp)
                ]
                pen_all[_srv_all] = torch.cat(_pp).cpu().numpy()
            o = 0
            _cells = []
            for i, st_ in lst:
                n = len(st_["q0"])
                _cells.append(
                    (i, (st_, qp[o : o + n], pe[o : o + n], red[o : o + n], mg[o : o + n], pen_all[o : o + n]))
                )
                o += n
            # the frame DP of every cell of the side in one device loop (was one Python DP per cell)
            for (i, _cl), (ok_, met_s, q_dp) in zip(
                _cells, self._gate_stage2_batch(side, S, [cl for _, cl in _cells]), strict=True
            ):
                if not ok_:
                    res[i]["ok"] = False
                    res[i]["met"] = {side: met_s}
                else:
                    res[i]["met"][side] = met_s
                    res[i]["gateq"][side] = q_dp
        return res
