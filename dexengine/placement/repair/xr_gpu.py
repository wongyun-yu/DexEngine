#!/usr/bin/env python3
"""Batch evaluation and greedy of the cross repair (cross_repair) — the bake's canonical path.

The cost of the cross repair is one "pose -> distances" evaluation repeated thousands to tens of thousands of times;
this module evaluates B poses at once. The distance values are the CPU mj_geomDistance (float64) values: the GPU
(mujoco_warp GJK/EPA, float32) only filters out pairs that are too far to matter, so the output equals a CPU-only run.

Parts (the pair sets, bounding-sphere radii, 0.0 re-query and pose formulas are owned by collision.cross_sets):
  GpuDist    — bounding-sphere prefilter -> (with a GPU) keep pairs whose GPU distance < distmax + EXACT_MARGIN -> CPU
               re-query (0.0 re-query included). With CUDA_VISIBLE_DEVICES set to the empty string (a CPU bake,
               DEXENGINE_BAKE_CPU=1) every prefiltered pair is measured on the CPU — same values, only slower. A visible
               device without CUDA dies.
  PoseBatch  — B instances (t, fingers, wrist shift, grasp) -> arm joints (2 damped-least-squares steps), achieved
               shift, mocap poses, per-set minimum distances, RowSet (CrossSetModel.{build_q, hand_frame_pos, arm_dq,
               mocap_pose}).
  BatchSolver— the only implementation of the cross-repair greedy (finger abduction -> multi-direction wrist shift ->
               1e-4 improvement threshold), in xr_solver.py; the warm chain (t-1) is ordered by cross_repair.
The kernel launch runs inside runtime_env.nvrtc_scope() (NVRTC rejects a long TMPDIR; owner dexengine.core.runtime_env).
"""

from __future__ import annotations

import os

import numpy as np

from dexengine.core import runtime_env as RE

# isort: split
from dexengine.placement.repair.xr_kernels import _kernel, _mjw_commit, _warp, struct_of  # noqa: F401 — re-exported
from dexengine.placement.repair.xr_solver import BatchSolver, RowSet, _Inst, _quat2mat  # noqa: F401 — re-exported

MESH, CAPSULE = 7, 3  # mjtGeom
EXACT_MARGIN = (
    0.005  # margin of the GPU (float32) filter: only pairs with GPU distance < distmax + this are re-measured
)

#: head of the failure when the GPU path finds no CUDA device; a bake log holding it is a GPU-path failure
NO_CUDA = "[XR_GPU] no CUDA device"


class GpuDist:
    """Batch distances over the pair set of one CrossSetModel (one clip) — the GPU only filters, the values are CPU."""

    def __init__(self, M, tol: float | None = None, iters: int | None = None):
        self.M = M
        m = M.m
        if (m.mesh_graphadr < 0).any():
            raise RuntimeError("[XR_GPU] a mesh has no hull graph — CrossSetModel did not build the hulls")
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.use_gpu = vis != ""  # empty string = the caller asked for a CPU bake (DEXENGINE_BAKE_CPU=1)
        self.tol = float(m.opt.ccd_tolerance if tol is None else tol)
        self.iters = int(m.opt.ccd_iterations if iters is None else iters)
        self.gtype = np.asarray(m.geom_type, int)
        bad = sorted(set(self.gtype.tolist()) - {MESH, CAPSULE})
        if bad:
            raise ValueError(f"[XR_GPU] geom types {bad} are not supported (mesh and capsule only)")
        if self.use_gpu:
            W = _warp()
            wp = W["wp"]
            if wp.get_cuda_device_count() < 1:
                raise RuntimeError(
                    f"{NO_CUDA} (CUDA_VISIBLE_DEVICES={vis!r}) — a CPU bake is asked for "
                    "explicitly with stage2_ground.sh --bake-cpu"
                )
            self.dev = "cuda:0"
            self.mjw = W["commit"]
            f32, i32 = np.float32, np.int32
            dev = self.dev
            self.model_args = struct_of(
                "model",
                {
                    "geom_type": wp.array(m.geom_type.astype(i32), dtype=int, device=dev),
                    "geom_dataid": wp.array(m.geom_dataid.astype(i32), dtype=int, device=dev),
                    "geom_size": wp.array(m.geom_size.astype(f32), dtype=wp.vec3, device=dev),
                    "mesh_vertadr": wp.array(m.mesh_vertadr.astype(i32), dtype=int, device=dev),
                    "mesh_vertnum": wp.array(m.mesh_vertnum.astype(i32), dtype=int, device=dev),
                    "mesh_graphadr": wp.array(m.mesh_graphadr.astype(i32), dtype=int, device=dev),
                    "mesh_vert": wp.array(m.mesh_vert.astype(f32), dtype=wp.vec3, device=dev),
                    "mesh_graph": wp.array(m.mesh_graph.astype(i32), dtype=int, device=dev),
                },
            )
        else:
            self.dev = "cpu"
            self.mjw = "-"
        # pair table (owners: CrossSetModel.pairs · SETS order · is_hand_pair)
        from dexengine.placement.collision import cross_sets as CC

        self.CC = CC
        self.sets = list(CC.SETS)
        P = M.pairs
        self.pi = np.array([q[0] for q in P], int)
        self.pj = np.array([q[1] for q in P], int)
        self.pset = np.array([self.sets.index(q[2]) if q[2] in self.sets else -1 for q in P], int)
        self.phand = np.array([CC.is_hand_pair(q[3]) for q in P], bool)
        self.rs = np.asarray(m.geom_rbound, float)[self.pi] + np.asarray(m.geom_rbound, float)[self.pj]
        self.cap = 0
        self.nq_total = 0
        self.ncall = 0
        self.nexact = [0, 0]

    def _ensure(self, n: int):
        if n <= self.cap:
            return
        W = _warp()
        wp = W["wp"]
        cap = max(n, int(self.cap * 1.5) + 1024)
        it = self.iters
        d = self.dev
        nv, nf = 5 + it, 6 + W["NF"] * it
        self.buf = {
            "ev": wp.empty((cap, nv), dtype=wp.vec3, device=d),
            "ev1": wp.empty((cap, nv), dtype=wp.vec3, device=d),
            "ev2": wp.empty((cap, nv), dtype=wp.vec3, device=d),
            "evi1": wp.empty((cap, nv), dtype=int, device=d),
            "evi2": wp.empty((cap, nv), dtype=int, device=d),
            "ef": wp.empty((cap, nf), dtype=wp.vec3i, device=d),
            "epr": wp.empty((cap, nf), dtype=wp.vec3, device=d),
            "en2": wp.empty((cap, nf), dtype=float, device=d),
            "eix": wp.empty((cap, nf), dtype=int, device=d),
            "emap": wp.empty((cap, nf), dtype=int, device=d),
            "ehor": wp.empty((cap, 2 * W["NH"]), dtype=int, device=d),
            "out_d": wp.empty(cap, dtype=float, device=d),
            "out_w1": wp.empty(cap, dtype=wp.vec3, device=d),
            "out_w2": wp.empty(cap, dtype=wp.vec3, device=d),
        }
        self.epa_args, self.out_args = struct_of("epa", self.buf), struct_of("out", self.buf)
        self.cap = cap

    CHUNK = 65536

    def _cpu(self, qb, g1, g2, sel, distmax, robust_zero, mocap):
        """Measure the sel queries on the CPU (float64): per pose b, put the mocap (from CrossSetModel.mocap_pose) into
        the CrossSetModel's MjData, run mj_kinematics, then mj_geomDistance + the 0.0 re-query (same _fromto buffer).
        Queries outside sel are "far" (= distmax)."""
        import mujoco

        M = self.M
        m, dd = M.m, M.d
        mpos, mquat = mocap
        n = len(qb)
        out_d = np.full(n, float(distmax))
        ow1 = np.zeros((n, 3))
        ow2 = np.zeros((n, 3))
        ft = M._fromto
        self.nexact[0] += len(sel)
        self.nexact[1] += n - len(sel)
        if len(sel):
            order = sel[np.argsort(qb[sel], kind="stable")]
            cur = -1
            for i in order:
                b = int(qb[i])
                if b != cur:
                    dd.mocap_pos[:] = mpos[b]
                    dd.mocap_quat[:] = mquat[b]
                    mujoco.mj_kinematics(m, dd)
                    cur = b
                gi, gj = int(g1[i]), int(g2[i])
                dist = mujoco.mj_geomDistance(m, dd, gi, gj, distmax, ft)
                if dist == 0.0 and robust_zero:
                    dist = M._requery_zero(gi, gj, distmax)
                out_d[i] = dist
                ow1[i] = ft[:3]
                ow2[i] = ft[3:]
        return out_d, ow1, ow2

    def _launch(self, qg1, qg2, qdx, p2, r1, r2, cutoff):
        n = len(qg1)
        if n > self.CHUNK:  # the EPA buffers grow with the query count — large batches run in chunks
            parts = [
                self._launch(
                    qg1[i : i + self.CHUNK],
                    qg2[i : i + self.CHUNK],
                    qdx[i : i + self.CHUNK],
                    p2[i : i + self.CHUNK],
                    r1[i : i + self.CHUNK],
                    r2[i : i + self.CHUNK],
                    cutoff,
                )
                for i in range(0, n, self.CHUNK)
            ]
            return tuple(np.concatenate([x[j] for x in parts]) for j in range(3))
        W = _warp()
        wp = W["wp"]
        if n == 0:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
        t1s, t2s = self.gtype[qg1], self.gtype[qg2]
        kinds = sorted(set(zip(t1s.tolist(), t2s.tolist(), strict=True)))
        if len(kinds) > 1:  # one specialised kernel per type pair — mixed batches are split
            d = np.empty(n)
            w1 = np.empty((n, 3))
            w2 = np.empty((n, 3))
            for k1, k2 in kinds:
                s_ = np.where((t1s == k1) & (t2s == k2))[0]
                d[s_], w1[s_], w2[s_] = self._launch(qg1[s_], qg2[s_], qdx[s_], p2[s_], r1[s_], r2[s_], cutoff)
            return d, w1, w2
        kern = _kernel(*kinds[0])
        self._ensure(n)
        bf = self.buf

        def a(x, dt):
            return wp.array(x, dtype=dt, device=self.dev)

        RE.launch(  # inside runtime_env.nvrtc_scope: the first launch compiles the kernel with NVRTC
            kern,
            dim=n,
            device=self.dev,
            inputs=[
                self.model_args,
                struct_of(
                    "query",
                    {
                        "q_g1": a(qg1.astype(np.int32), int),
                        "q_g2": a(qg2.astype(np.int32), int),
                        "q_dx": a(qdx.astype(np.float32), float),
                        "q_p2": a(p2.astype(np.float32), wp.vec3),
                        "q_r1": a(r1.astype(np.float32), wp.mat33),
                        "q_r2": a(r2.astype(np.float32), wp.mat33),
                    },
                ),
                float(self.tol),
                float(cutoff),
                int(self.iters),
                self.epa_args,
                self.out_args,
            ],
        )
        d = bf["out_d"].numpy()[:n].astype(float)
        w1 = bf["out_w1"].numpy()[:n].astype(float)
        w2 = bf["out_w2"].numpy()[:n].astype(float)
        return d, w1, w2

    def query(
        self,
        gx: np.ndarray,
        gR: np.ndarray,
        distmax: float,
        mocap,
        sets=None,
        want_pairs: bool = True,
        robust_zero: bool = True,
        worst_sets=None,
    ):
        """gx (B, ngeom, 3) · gR (B, ngeom, 3, 3) · mocap (pos, quat) -> (mins (B,S), hand_mins (B,S), RowSet | None).
        Same meaning as CrossSetModel.distances() (set filter ``sets``; pairs beyond distmax are not in the rows; the
        minima start at distmax)."""
        B = gx.shape[0]
        S = len(self.sets)
        use = np.ones(len(self.pi), bool) if sets is None else np.isin(self.pset, [self.sets.index(s) for s in sets])
        # bounding-sphere prefilter — the same formula as CrossSetModel.distances()
        d2 = np.sum((gx[:, self.pi] - gx[:, self.pj]) ** 2, axis=2)  # (B, npair)
        keep = (d2 <= (self.rs[None] + distmax) ** 2) & use[None]
        qb, qk = np.nonzero(keep)
        g1, g2 = self.pi[qk], self.pj[qk]
        if self.use_gpu:
            o1 = gx[qb, g1]  # per-query origin (float64)
            p2 = gx[qb, g2] - o1
            r1 = gR[qb, g1]
            r2 = gR[qb, g2]
            dg, _, _ = self._launch(g1, g2, np.zeros(len(qb)), p2, r1, r2, distmax)
            sel = np.where(dg < distmax + EXACT_MARGIN)[0]
        else:
            sel = np.arange(len(qb))
        d, w1, w2 = self._cpu(qb, g1, g2, sel, distmax, robust_zero, mocap)
        self.ncall += 1
        self.nq_total += len(qb)
        near = d < distmax
        mins = np.full((B, S), distmax)
        hmins = np.full((B, S), distmax)
        qs = self.pset[qk]
        nb, ns, nd = qb[near], qs[near], d[near]
        np.minimum.at(mins, (nb, ns), nd)
        hn = near & self.phand[qk]
        np.minimum.at(hmins, (qb[hn], qs[hn]), d[hn])
        rows = None
        if want_pairs:
            rows = RowSet(self, B, qb[near], qk[near], d[near], w1[near], w2[near], distmax, worst_sets)
        return mins, hmins, rows


class PoseBatch:
    """Batch pose evaluation of the cross repair: B instances (t, fingers, wrist shift, grasp) -> arm joints, achieved
    shift, distances. The arm-joint and pose formulas are CrossSetModel's (build_q · hand_frame_pos · arm_dq ·
    mocap_pose); this class only calls them per instance and gathers. The decisions are BatchSolver's and
    cross_repair's."""

    def __init__(self, M, L, qpos, IK1, arm, base, lim, dqmax, n_dls, carry, gdist: GpuDist, distmax: float):
        self.M, self.L, self.qpos, self.IK1, self.arm, self.base = M, L, qpos, IK1, arm, np.asarray(base, float)
        self.lim, self.dqmax, self.n_dls, self.carry, self.G, self.distmax = lim, dqmax, n_dls, carry, gdist, distmax
        m = M.m
        self.gbody = np.asarray(m.geom_bodyid, int)
        self.gpos = np.asarray(m.geom_pos, float)
        self.gRl = _quat2mat(np.asarray(m.geom_quat, float))
        self.mb = np.where(m.body_mocapid >= 0)[0]
        self.mid = m.body_mocapid[self.mb]
        self.mocap0 = (np.array(M.d.mocap_pos, float), np.array(M.d.mocap_quat, float))

    def arm_at(self, ts, fq, shift):
        """Local arm-joint correction (n_dls damped-least-squares steps, joint limits, step cap) per instance with
        CrossSetModel's formulas. With shift None, or a side whose shift is < 1e-9, the IK joints stay.
        -> aq[s] (B,7) · achieved shift ach[s] (B,3)"""
        M, arm, base = self.M, self.arm, self.base
        B = len(ts)
        aq = {s: np.asarray(self.IK1[s], float)[ts].copy() for s in ("right", "left")}
        ach = {s: np.zeros((B, 3)) for s in ("right", "left")}
        if shift is None:
            return aq, ach
        for i, t in enumerate(ts):
            a_ = {s: np.array(self.IK1[s][t], float) for s in ("right", "left")}
            f_ = {s: fq[s][i] for s in ("right", "left")}
            for s in ("right", "left"):
                if np.linalg.norm(shift[s][i]) < 1e-9:
                    continue
                p0 = M.hand_frame_pos(M.build_q({k: (arm[k], a_[k]) for k in a_}, f_), s, base)
                for _ in range(self.n_dls):
                    q = M.build_q({k: (arm[k], a_[k]) for k in a_}, f_)
                    cur = M.hand_frame_pos(q, s, base)
                    dq = M.arm_dq(q, s, arm[s], (p0 + shift[s][i]) - cur)
                    a_[s] = np.clip(a_[s] + np.clip(dq, -self.dqmax, self.dqmax), self.lim[s][0], self.lim[s][1])
                ach[s][i] = M.hand_frame_pos(M.build_q({k: (arm[k], a_[k]) for k in a_}, f_), s, base) - p0
            for s in ("right", "left"):
                aq[s][i] = a_[s]
        return aq, ach

    def geoms(self, ts, aq, fq, ach=None, grasp=None):
        """The mocap (pos, quat) of B poses via CrossSetModel.mocap_pose (a holding hand moves its object by the
        achieved shift) and the geom world poses (B, ngeom, 3) · (B, ngeom, 3, 3) for the GPU filter. An object body
        without its side's block dies (the CPU set_pose path would keep the previous pose's value; the batch does not
        reproduce that)."""
        M, CC, L = self.M, self.CC_, self.L
        m = M.m
        B = len(ts)
        mpos = np.empty((B, m.nmocap, 3))
        mquat = np.empty((B, m.nmocap, 4))
        for b, t in enumerate(ts):
            _, obj = L.hand_blocks(self.qpos[t])
            if ach is not None and self.carry and grasp is not None:
                obj = {k: (None if v is None else np.array(v, float)) for k, v in obj.items()}
                for s in ("right", "left"):
                    if bool(grasp[s][b]) and obj.get(s) is not None:
                        obj[s][:3] = obj[s][:3] + CC.RZ_POS90 @ ach[s][b]
            for s in ("right", "left"):
                if M.obj_mocap.get(s) is not None and (obj.get(s) if obj else None) is None:
                    raise ValueError(
                        f"[XR_GPU] {s} has no object block (t={t}) — a side with an object body needs its block"
                    )
            mpos[b], mquat[b] = M.mocap_pose(
                {s: (self.arm[s], aq[s][b]) for s in ("right", "left")},
                {s: fq[s][b] for s in ("right", "left")},
                obj,
                self.base,
                prev=self.mocap0,
            )
        self._mocap = (mpos, mquat)
        bR = np.repeat(np.eye(3)[None, None], B, 0).repeat(m.nbody, 1)
        bp = np.zeros((B, m.nbody, 3))
        bp[:, self.mb] = mpos[:, self.mid]
        bR[:, self.mb] = _quat2mat(mquat[:, self.mid].reshape(-1, 4)).reshape(B, len(self.mb), 3, 3)
        gR = bR[:, self.gbody] @ self.gRl[None]
        gx = bp[:, self.gbody] + np.einsum("bgij,gj->bgi", bR[:, self.gbody], self.gpos)
        return gx, gR, (bR, bp)

    @property
    def CC_(self):
        from dexengine.placement.collision import cross_sets as CC

        return CC

    def eval(self, ts, fq, shift, grasp, want_pairs=True, sets=None, worst_sets=None):
        """Evaluate B instances: ts (B,) · fq[s] (B,nF) · shift None or {s: (B,3)} · grasp {s: (B,) bool} (meaningful
        only with a shift) -> aq, ach, mins, hmins, RowSet, (bR, bp)"""
        ts = np.asarray(ts, int)
        aq, ach = self.arm_at(ts, fq, shift)
        gx, gR, bpose = self.geoms(ts, aq, fq, ach if shift is not None else None, grasp)
        mins, hmins, rows = self.G.query(
            gx,
            gR,
            self.distmax,
            self._mocap,
            sets=sets,
            want_pairs=want_pairs,
            robust_zero=self.M.robust_zero,
            worst_sets=worst_sets,
        )
        return aq, ach, mins, hmins, rows, bpose
