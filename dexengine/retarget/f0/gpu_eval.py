"""f0 GPU engine: GpuEvaluator registers hands, objects and problems on the device and evaluates the repair objective
for a batch of candidates."""

import time

import numpy as np

from dexengine.core import runtime_env as RE
from dexengine.retarget.f0.gpu_specs import (
    FP32_GUARD,
    HULL_TOL,
    MAX_BATCH,
    GpuHandSpec,
    GpuObjectSpec,
    _require_warp,
    wrist_R_rows,
)
from dexengine.retarget.f0.gpu_upload import _DeviceUploads


class GpuEvaluator(_DeviceUploads):
    """Registers hands, objects and problems (one solve step of a side) and maps a candidate batch (problem, dq) to
    (count, depth): the batched `n_pen(dq)` (strict clearance version). Batches may mix problems (cross-clip batching).

    Uploads are incremental: object samples and hulls are appended to device arrays whose capacity doubles, and problems
    (sides) are fixed-size slots (REV_CAP reverse points) of which only new slots are copied; re-uploading everything
    for each new object or side would be O(n²) transfer."""

    REV_CAP = 400  # reverse query-point cap = ObjectGeom.local_surface cap (400, owned by repair)

    def __init__(self, device: str = "cuda:0"):
        self.wp = _require_warp()
        self.dev = device
        from dexengine.retarget.f0 import gpu_kernels  # noqa: PLC0415 — imports warp (GPU workers only)

        self.K = gpu_kernels.KERNELS
        self.point_pass_structs = gpu_kernels.point_pass_structs
        self.pts_args = ()  # (HandPoints, ProblemFrames, ObjectBounds), set by _sync_static
        self.hands, self.hand_key = [], {}
        self.objs, self.obj_key = [], {}
        self.g = {}
        self._hands_dirty = False
        self._objs_new = []  # object ids not yet on the device
        self._o_cap = {"s": 0, "h": 0, "k": 0, "p": 0}
        self._o_fill = {"s": 0, "h": 0, "k": 0, "p": 0}
        self.P = {}  # slot -> problem dict (live ones only)
        self._free = []
        self._nslot = 0
        self._p_cap = 0
        self._p_dirty = set()
        self._p_small_dirty = False
        self._cap_c = 0
        self._cap_q = 0
        self._expand_identity = {}
        self._hm_w, self._hm_R = np.zeros((0, 3)), np.zeros((0, 3, 3))
        self._hm_i, self._hm_h = np.zeros((0, 0)), np.zeros(0, np.int32)
        self.stats = {"n_eval": 0, "n_query": 0, "t_gpu": 0.0, "t_host": 0.0, "n_call": 0, "t_upload": 0.0}

    # registration
    def hand_id(self, key, hand_model):
        if key not in self.hand_key:
            self.hand_key[key] = len(self.hands)
            self.hands.append(GpuHandSpec(hand_model))
            self._hands_dirty = True
        return self.hand_key[key]

    def obj_id(self, key, obj_geom):
        if key not in self.obj_key:
            self.obj_key[key] = len(self.objs)
            self.objs.append(GpuObjectSpec(obj_geom))
            self._objs_new.append(len(self.objs) - 1)
        return self.obj_key[key]

    def add_problem(self, geom) -> int:
        hid = self.hand_id(geom["hand_key"], geom["hand"])
        oid = self.obj_id(geom["obj_key"], geom["obj"])
        o = self.objs[oid]
        oR = np.asarray(geom["oR"], float)
        opos = np.asarray(geom["opos"], float)
        # reverse query points = the CPU's formula (ObjectGeom.surface_world), host numpy, bit-identical
        rev = o.local_surface @ oR.T + opos
        if len(rev) > self.REV_CAP:
            raise SystemExit(
                f"[f0_gpu] reverse query points {len(rev)} > REV_CAP {self.REV_CAP} — the cap in repair changed"
            )
        wpos0 = np.asarray(geom["wpos0"], float)
        sc = max(float(np.abs(opos).max()), float(np.abs(wpos0).max()), float(np.abs(rev).max()))
        slot = self._free.pop() if self._free else self._alloc()
        self.P[slot] = {
            "h": hid,
            "o": oid,
            "opos": opos,
            "oR": oR,
            "floor": float(geom["floor"]),
            "clr": float(geom["clr"]),
            "pm": float(geom["pm"]),
            "rev": rev,
            "wpos0": wpos0,
            "g": FP32_GUARD * (1.0 + sc),
            "wR0": np.asarray(geom["wR0"], float),
            "indep0": np.asarray(geom["indep0"], float),
            "expand": geom["expand"],
        }
        self._p_dirty.add(slot)
        # host mirror (slot indices): gathers the base poses of thousands of candidates per round without Python loops
        ni = len(self.P[slot]["indep0"])
        if slot >= len(self._hm_w) or ni > self._hm_i.shape[1]:
            cap = max(2 * len(self._hm_w), slot + 1, 256)
            w, R, I_buf, H = (
                np.zeros((cap, 3)),
                np.tile(np.eye(3), (cap, 1, 1)),
                np.zeros((cap, max(ni, self._hm_i.shape[1]))),
                np.zeros(cap, np.int32),
            )
            k = len(self._hm_w)
            w[:k], R[:k], I_buf[:k, : self._hm_i.shape[1]], H[:k] = self._hm_w, self._hm_R, self._hm_i, self._hm_h
            self._hm_w, self._hm_R, self._hm_i, self._hm_h = w, R, I_buf, H
        self._hm_w[slot] = self.P[slot]["wpos0"]
        self._hm_R[slot] = self.P[slot]["wR0"]
        self._hm_i[slot, :ni] = self.P[slot]["indep0"]
        self._hm_h[slot] = hid
        return slot

    def _alloc(self):
        s = self._nslot
        self._nslot += 1
        return s

    def free_problem(self, slot):
        self.P.pop(slot, None)
        self._free.append(slot)

    def evaluate(self, slots, dq):
        """slots (n,), dq (n, nd, already clipped) -> (cnt (n,) int64, depth (n,) float64), = n_pen_strict(dq)."""
        slots = np.asarray(slots, np.int32)
        dq = np.asarray(dq, float)
        n = len(slots)
        cnt = np.empty(n, np.int64)
        dep = np.empty(n)
        for a in range(0, n, MAX_BATCH):
            b = min(n, a + MAX_BATCH)
            cnt[a:b], dep[a:b] = self._evaluate(slots[a:b], dq[a:b])
        return cnt, dep

    def _host_poses(self, slots, dq):
        th0 = time.perf_counter()
        n = len(slots)
        P = self.P
        wpos0 = self._hm_w[slots]
        wR0 = self._hm_R[slots]
        hands = self._hm_h[slots]
        wpv = wpos0 + dq[:, :3]
        wR = wrist_R_rows(dq[:, 3:6], wR0)
        fs = np.zeros((n, self.NFmax))
        # expand = the request's function (HandModel.fingers12). Checked once per hand for identity: identity means one
        # call for all rows (fully actuated hand), otherwise (coupled-joint hand) the function runs per row; the formula
        # is not restated.
        ind = self._hm_i[slots, : dq.shape[1] - 6] + dq[:, 6:]
        for h in np.unique(hands):
            m = hands == h
            if h not in self._expand_identity:
                i0 = int(np.flatnonzero(m)[0])
                probe = P[int(slots[i0])]["expand"](ind[i0])
                self._expand_identity[h] = bool(probe.shape == ind[i0].shape and np.array_equal(probe, ind[i0]))
            if self._expand_identity[h]:
                fs[m, : ind.shape[1]] = ind[m]
            else:
                for i in np.flatnonzero(m):
                    f = P[int(slots[i])]["expand"](ind[i])
                    fs[i, : len(f)] = f
        self.stats["t_host"] += time.perf_counter() - th0
        return hands, wpv, wR, fs

    def _launch_fk(self, wp, d, K, g, n, c_hand, c_wp, c_wR, c_f):
        """Forward kinematics of every candidate: link rotations and translations (fp64 and fp32 copies)."""
        RE.launch(
            K["fk"],
            dim=n,
            device=d,
            inputs=[
                c_hand,
                c_wp,
                c_wR,
                c_f,
                g["h_j_off"],
                g["h_nj"],
                g["j_parent"],
                g["j_R"],
                g["j_t"],
                g["j_axis"],
                g["j_kind"],
                g["j_q"],
                g["h_l_off"],
                g["h_nl"],
                g["l_joint"],
                g["l_R"],
                g["l_t"],
                g["h_pinvR"],
                g["h_pinvt"],
                self.b_sR,
                self.b_st,
                self.b_LR,
                self.b_Lt,
                self.b_LR32,
                self.b_Lt32,
            ],
        )

    def _launch_pts(self, wp, d, K, n, L, c_prob, c_hand):
        """Point pass, twice: count the nearest-sample queries per (candidate, link), scan them to offsets, then
        write the queries and the desk and box depths. Returns the query count."""
        mid = [
            c_prob,
            c_hand,
            self.b_LR,
            self.b_Lt,
            self.b_LR32,
            self.b_Lt32,
            *self.pts_args,
            self.b_dsum,
            self.b_dcnt,
            self.b_qcnt,
            self.b_qoff,
        ]
        self._ensure_q(1)
        RE.launch(K["pts"], dim=(n, L), device=d, inputs=[0, L] + mid + [self.b_qk, self.b_qloc])
        wp.utils.array_scan(self.b_qcnt, self.b_qoff, inclusive=False)
        nq = int(self.b_qoff[n * L - 1 : n * L].numpy()[0]) + int(self.b_qcnt[n * L - 1 : n * L].numpy()[0])
        self._ensure_q(nq)
        RE.launch(K["pts"], dim=(n, L), device=d, inputs=[1, L] + mid + [self.b_qk, self.b_qloc])
        return nq

    def _launch_nn(self, wp, d, K, g, nq, c_prob):
        """Nearest-sample and hull test of every query point (object penetration count and depth)."""
        if nq > 0:
            RE.launch(
                K["nn"],
                dim=nq,
                device=d,
                inputs=[
                    self.b_qk,
                    self.b_qloc,
                    c_prob,
                    g["p_obj"],
                    g["p_clr"],
                    g["p_pm"],
                    g["p_g"],
                    g["o_s_off"],
                    g["o_s_n"],
                    g["o_kd_off"],
                    g["o_kp_off"],
                    g["kd_lo"],
                    g["kd_hi"],
                    g["kd_left"],
                    g["kd_right"],
                    g["kd_miss"],
                    g["kd_a"],
                    g["kd_b"],
                    g["kd_perm"],
                    g["s32"],
                    g["s64"],
                    g["n64"],
                    g["o_h_off"],
                    g["o_h_n"],
                    g["oh_n"],
                    g["oh_d"],
                    g["oh_n32"],
                    g["oh_d32"],
                    HULL_TOL,
                    self.b_rc,
                    self.b_rd,
                    self.b_rcl,
                ],
            )

    def _launch_rev(self, wp, d, K, g, n, L, c_prob, c_hand):
        """Reverse test: object surface points inside the shrunk link hulls."""
        self.b_rev.zero_()
        RE.launch(
            K["rev"],
            dim=(n, L),
            device=d,
            inputs=[
                c_prob,
                c_hand,
                self.b_LR,
                self.b_Lt,
                self.b_LR32,
                self.b_Lt32,
                g["h_l_off"],
                g["h_nl"],
                g["l_hull"],
                g["f_a"],
                g["f_b"],
                g["f_n"],
                g["f_d"],
                g["f_n32"],
                g["f_d32"],
                g["sph_c32"],
                g["sph_r"],
                g["p_rev_off"],
                g["p_rev_n"],
                g["rev_pts"],
                g["rev32"],
                g["p_g"],
                HULL_TOL,
                self.b_rev,
            ],
        )

    def _launch_red(self, wp, d, K, n, L, nq):
        """Reduce the per-link counts and depths to one (count, depth) per candidate."""
        RE.launch(
            K["red"],
            dim=n,
            device=d,
            inputs=[
                L,
                n,
                nq,
                self.b_dsum,
                self.b_dcnt,
                self.b_qoff,
                self.b_rc,
                self.b_rd,
                self.b_rcl,
                self.b_rev,
                self.b_ocnt,
                self.b_odep,
            ],
        )

    def _evaluate(self, slots, dq):
        wp, d, K = self.wp, self.dev, self.K
        self._sync_static()
        g = self.g
        n = len(slots)
        hands, wpv, wR, fs = self._host_poses(slots, dq)
        t0 = time.perf_counter()
        self._ensure_cap(n)
        L = self.Lmax
        A = self._A
        c_prob = A(slots, wp.int32)
        c_hand = A(hands, wp.int32)
        c_wp = A(wpv, wp.vec3d)
        c_wR = A(wR, wp.mat33d)
        c_f = A(fs, wp.float64)
        self._launch_fk(wp, d, K, g, n, c_hand, c_wp, c_wR, c_f)
        nq = self._launch_pts(wp, d, K, n, L, c_prob, c_hand)
        self._launch_nn(wp, d, K, g, nq, c_prob)
        self._launch_rev(wp, d, K, g, n, L, c_prob, c_hand)
        self._launch_red(wp, d, K, n, L, nq)
        cnt = self.b_ocnt.numpy()[:n].astype(np.int64)
        dep = self.b_odep.numpy()[:n].copy()
        self.stats["t_gpu"] += time.perf_counter() - t0
        self.stats["n_eval"] += n
        self.stats["n_query"] += nq
        self.stats["n_call"] += 1
        return cnt, dep
