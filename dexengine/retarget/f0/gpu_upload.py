"""f0 GPU engine: device storage of GpuEvaluator — upload of hands, objects and problems, buffer growth and the
capacity checks (a base class; GpuEvaluator keeps registration and evaluation)."""

import time

import numpy as np


class _DeviceUploads:
    """Device-side storage of GpuEvaluator (base class; every method reads the evaluator's own fields)."""

    # upload
    def _A(self, x, dt):
        return self.wp.array(np.ascontiguousarray(x), dtype=dt, device=self.dev)

    def _upload_hands(self):
        wp, A, H, g = self.wp, self._A, self.hands, self.g
        self.Lmax = max(h.n_links for h in H)
        self.NFmax = max(h.n_f for h in H)
        self.NJmax = max(h.nj for h in H)
        g["h_j_off"] = A(np.cumsum([0] + [h.nj for h in H])[:-1], wp.int32)
        g["h_nj"] = A([h.nj for h in H], wp.int32)
        g["j_parent"] = A(np.concatenate([h.j_parent for h in H]), wp.int32)
        g["j_R"] = A(np.concatenate([h.j_R for h in H]), wp.mat33d)
        g["j_t"] = A(np.concatenate([h.j_t for h in H]), wp.vec3d)
        g["j_axis"] = A(np.concatenate([h.j_axis for h in H]), wp.vec3d)
        g["j_kind"] = A(np.concatenate([h.j_kind for h in H]), wp.int32)
        g["j_q"] = A(np.concatenate([h.j_q for h in H]), wp.int32)
        g["h_l_off"] = A(np.cumsum([0] + [h.n_links for h in H])[:-1], wp.int32)
        g["h_nl"] = A([h.n_links for h in H], wp.int32)
        g["l_joint"] = A(np.concatenate([h.l_joint for h in H]), wp.int32)
        g["l_R"] = A(np.concatenate([h.l_R for h in H]), wp.mat33d)
        g["l_t"] = A(np.concatenate([h.l_t for h in H]), wp.vec3d)
        g["h_pinvR"] = A(np.stack([h.palm_inv_R for h in H]), wp.mat33d)
        g["h_pinvt"] = A(np.stack([h.palm_inv_t for h in H]), wp.vec3d)
        g["h_p_off"] = A(np.cumsum([0] + [h.n_pts for h in H])[:-1], wp.int32)
        g["pt_a"] = A(np.concatenate([h.pt_a for h in H]), wp.int32)
        g["pt_b"] = A(np.concatenate([h.pt_b for h in H]), wp.int32)
        g["pts"] = A(np.concatenate([h.pts for h in H]), wp.vec3d)
        g["pts32"] = A(np.concatenate([h.pts for h in H]).astype(np.float32), wp.vec3f)
        g["l_hull"] = A(np.concatenate([h.l_hull for h in H]), wp.int32)
        fbase = np.cumsum([0] + [len(h.facets) for h in H])[:-1]
        g["f_a"] = A(np.concatenate([h.f_a + fb for h, fb in zip(H, fbase, strict=True)]), wp.int32)
        g["f_b"] = A(np.concatenate([h.f_b + fb for h, fb in zip(H, fbase, strict=True)]), wp.int32)
        FF = np.concatenate([h.facets for h in H])
        g["f_n"] = A(FF[:, :3], wp.vec3d)
        g["f_d"] = A(FF[:, 3], wp.float64)
        g["f_n32"] = A(FF[:, :3].astype(np.float32), wp.vec3f)
        g["f_d32"] = A(FF[:, 3].astype(np.float32), wp.float32)
        g["sph_c32"] = A(np.concatenate([h.sph_c for h in H]).astype(np.float32), wp.vec3f)
        g["sph_r"] = A(np.concatenate([h.sph_r for h in H]), wp.float64)
        self._hands_dirty = False
        self._cap_c = 0  # Lmax/NJmax may have changed: reallocate candidate buffers

    def _grow(self, key, fields, need):
        """Ensure capacity of the large object arrays (samples s, hulls h); when short, allocate double and copy the
        used part."""
        wp, d, g = self.wp, self.dev, self.g
        if need <= self._o_cap[key]:
            return
        cap = max(need, 2 * self._o_cap[key], 1 << 16)
        for name, dt in fields:
            new = wp.zeros(cap, dtype=dt, device=d)
            if name in g and self._o_fill[key] > 0:
                wp.copy(new, g[name], count=self._o_fill[key])
            g[name] = new
        self._o_cap[key] = cap

    def _upload_objs(self):
        wp, A, g, objects = self.wp, self._A, self.g, self.objs
        new = self._objs_new
        need_s = self._o_fill["s"] + sum(len(objects[i].s_pts) for i in new)
        need_h = self._o_fill["h"] + sum(len(objects[i].hull_eq) for i in new)
        self._grow("s", (("s64", wp.vec3d), ("s32", wp.vec3f), ("n64", wp.vec3d)), need_s)
        self._grow(
            "h", (("oh_n", wp.vec3d), ("oh_d", wp.float64), ("oh_n32", wp.vec3f), ("oh_d32", wp.float32)), need_h
        )
        need_k = self._o_fill["k"] + sum(len(objects[i].kd_lo) for i in new)
        need_p = self._o_fill["p"] + sum(len(objects[i].kd_perm) for i in new)
        self._grow(
            "k",
            (
                ("kd_lo", wp.vec3f),
                ("kd_hi", wp.vec3f),
                ("kd_left", wp.int32),
                ("kd_right", wp.int32),
                ("kd_miss", wp.int32),
                ("kd_a", wp.int32),
                ("kd_b", wp.int32),
            ),
            need_k,
        )
        self._grow("p", (("kd_perm", wp.int32),), need_p)
        for i in new:
            o = objects[i]
            o.s_off, o.h_off = self._o_fill["s"], self._o_fill["h"]
            ns, nh = len(o.s_pts), len(o.hull_eq)
            for name, arr, dt in (
                ("s64", o.s_pts, wp.vec3d),
                ("s32", o.s_pts.astype(np.float32), wp.vec3f),
                ("n64", o.s_nrm, wp.vec3d),
            ):
                wp.copy(g[name], A(arr, dt), dest_offset=o.s_off, count=ns)
            HE = o.hull_eq
            for name, arr, dt in (
                ("oh_n", HE[:, :3], wp.vec3d),
                ("oh_d", HE[:, 3], wp.float64),
                ("oh_n32", HE[:, :3].astype(np.float32), wp.vec3f),
                ("oh_d32", HE[:, 3].astype(np.float32), wp.float32),
            ):
                wp.copy(g[name], A(arr, dt), dest_offset=o.h_off, count=nh)
            o.k_off, o.p_off = self._o_fill["k"], self._o_fill["p"]
            nk, npm = len(o.kd_lo), len(o.kd_perm)
            for name, arr, dt in (
                ("kd_lo", o.kd_lo, wp.vec3f),
                ("kd_hi", o.kd_hi, wp.vec3f),
                ("kd_left", o.kd_left, wp.int32),
                ("kd_right", o.kd_right, wp.int32),
                ("kd_miss", o.kd_miss, wp.int32),
                ("kd_a", o.kd_a, wp.int32),
                ("kd_b", o.kd_b, wp.int32),
            ):
                wp.copy(g[name], A(arr, dt), dest_offset=o.k_off, count=nk)
            wp.copy(g["kd_perm"], A(o.kd_perm, wp.int32), dest_offset=o.p_off, count=npm)
            self._o_fill["s"] += ns
            self._o_fill["h"] += nh
            self._o_fill["k"] += nk
            self._o_fill["p"] += npm
        self._objs_new = []
        # small per-object arrays are uploaded whole (objects × tens of bytes)
        g["o_s_off"] = A([o.s_off for o in objects], wp.int32)
        g["o_s_n"] = A([len(o.s_pts) for o in objects], wp.int32)
        g["o_h_off"] = A([o.h_off for o in objects], wp.int32)
        g["o_h_n"] = A([len(o.hull_eq) for o in objects], wp.int32)
        g["o_lo"] = A(np.stack([o.lo for o in objects]), wp.vec3d)
        g["o_hi"] = A(np.stack([o.hi for o in objects]), wp.vec3d)
        g["o_lo32"] = A(np.stack([o.lo for o in objects]).astype(np.float32), wp.vec3f)
        g["o_hi32"] = A(np.stack([o.hi for o in objects]).astype(np.float32), wp.vec3f)
        g["dop_D32"] = A(np.concatenate([o.dop_D for o in objects]).astype(np.float32), wp.vec3f)
        g["dop_hi32"] = A(np.concatenate([o.dop_hi for o in objects]).astype(np.float32), wp.float32)
        g["o_kd_off"] = A([o.k_off for o in objects], wp.int32)
        g["o_kp_off"] = A([o.p_off for o in objects], wp.int32)

    def _upload_probs(self):
        """Problem slots: small arrays (slots × tens of bytes) whole, reverse points (slots × REV_CAP) only for changed
        slots."""
        wp, A, g = self.wp, self._A, self.g
        n = max(self._nslot, 1)
        if n > self._p_cap:
            cap = max(n, 2 * self._p_cap, 256)
            for name, dt in (("rev_pts", wp.vec3d), ("rev32", wp.vec3f)):
                new = wp.zeros(cap * self.REV_CAP, dtype=dt, device=self.dev)
                if name in g and self._p_cap > 0:
                    wp.copy(new, g[name], count=self._p_cap * self.REV_CAP)
                g[name] = new
            self._p_cap = cap
        obj = np.zeros(n, np.int32)
        opos = np.zeros((n, 3))
        oR = np.tile(np.eye(3), (n, 1, 1))
        fl, clr, pm, gg = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, np.float32)
        rn = np.zeros(n, np.int32)
        for s_, p in self.P.items():
            obj[s_], opos[s_], oR[s_] = p["o"], p["opos"], p["oR"]
            fl[s_], clr[s_], pm[s_], gg[s_], rn[s_] = p["floor"], p["clr"], p["pm"], p["g"], len(p["rev"])
        g["p_obj"], g["p_opos"], g["p_oR"] = A(obj, wp.int32), A(opos, wp.vec3d), A(oR, wp.mat33d)
        g["p_opos32"], g["p_oR32"] = A(opos.astype(np.float32), wp.vec3f), A(oR.astype(np.float32), wp.mat33f)
        g["p_floor"], g["p_clr"], g["p_pm"], g["p_g"] = (
            A(fl, wp.float64),
            A(clr, wp.float64),
            A(pm, wp.float64),
            A(gg, wp.float32),
        )
        g["p_rev_off"] = A(np.arange(n, dtype=np.int32) * self.REV_CAP, wp.int32)
        g["p_rev_n"] = A(rn, wp.int32)
        for s_ in self._p_dirty:
            p = self.P.get(s_)
            if p is None:
                continue
            r = p["rev"]
            wp.copy(g["rev_pts"], A(r, wp.vec3d), dest_offset=s_ * self.REV_CAP, count=len(r))
            wp.copy(g["rev32"], A(r.astype(np.float32), wp.vec3f), dest_offset=s_ * self.REV_CAP, count=len(r))
        self._p_dirty = set()

    def _sync_static(self):
        t = time.perf_counter()
        changed = self._hands_dirty or bool(self._objs_new) or bool(self._p_dirty)
        if self._hands_dirty:
            self._upload_hands()
        if self._objs_new:
            self._upload_objs()
        if self._p_dirty:
            self._upload_probs()
        if changed:  # an upload replaces device arrays: the point pass's struct arguments point at the new ones
            self.pts_args = self.point_pass_structs(self.g)
        self.stats["t_upload"] += time.perf_counter() - t

    def _ensure_cap(self, n):
        wp, d, L = self.wp, self.dev, self.Lmax
        if n <= self._cap_c:
            return
        cap = max(n, 256)
        self._cap_c = cap
        self.b_sR = wp.zeros((cap, self.NJmax), dtype=wp.mat33d, device=d)
        self.b_st = wp.zeros((cap, self.NJmax), dtype=wp.vec3d, device=d)
        self.b_LR = wp.zeros((cap, L), dtype=wp.mat33d, device=d)
        self.b_Lt = wp.zeros((cap, L), dtype=wp.vec3d, device=d)
        self.b_LR32 = wp.zeros((cap, L), dtype=wp.mat33f, device=d)
        self.b_Lt32 = wp.zeros((cap, L), dtype=wp.vec3f, device=d)
        self.b_dsum = wp.zeros(cap * L, dtype=wp.float64, device=d)
        self.b_dcnt = wp.zeros(cap * L, dtype=wp.int32, device=d)
        self.b_qcnt = wp.zeros(cap * L, dtype=wp.int32, device=d)
        self.b_qoff = wp.zeros(cap * L, dtype=wp.int32, device=d)
        self.b_rev = wp.zeros(cap, dtype=wp.int32, device=d)
        self.b_ocnt = wp.zeros(cap, dtype=wp.int32, device=d)
        self.b_odep = wp.zeros(cap, dtype=wp.float64, device=d)

    def _ensure_q(self, nq):
        wp, d = self.wp, self.dev
        if nq <= self._cap_q:
            return
        cap = max(nq, int(self._cap_q * 1.5), 4096)
        self._cap_q = cap
        self.b_qk = wp.zeros(cap, dtype=wp.int32, device=d)
        self.b_qloc = wp.zeros(cap, dtype=wp.vec3d, device=d)
        self.b_rc = wp.zeros(cap, dtype=wp.int32, device=d)
        self.b_rd = wp.zeros(cap, dtype=wp.float64, device=d)
        self.b_rcl = wp.zeros(cap, dtype=wp.float64, device=d)
