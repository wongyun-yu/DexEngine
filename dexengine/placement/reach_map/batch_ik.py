"""Batched DLS arm IK on the GPU (pytorch_kinematics serial chain + damped least squares).

`ArmIK` is the solver of the judge, the bake's coarse path (`coarse_path.dp`), the reach-map builder and the
arm-flip repair: FK/Jacobian from pytorch_kinematics (validated against Pinocchio), weighted DLS with
lambda = 0.01 and joint-limit clipping, batched over targets. `_dls_fast` is the judge's fast path (same maths,
fewer launches, optional CUDA-graph replay).
"""

import pinocchio as pin  # noqa: F401 — imported before pytorch_kinematics (import order; the model comes from urdf_load)
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32


def log3_batch(R):
    """Batched SO(3) log: rotation matrices [N,3,3] → rotation vectors [N,3]."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos = torch.clamp((tr - 1) * 0.5, -1.0, 1.0)
    ang = torch.acos(cos)
    w = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    small = ang < 1e-5
    denom = torch.where(small, torch.ones_like(ang), 2 * torch.sin(ang))
    out = w * (ang / denom).unsqueeze(-1)
    out = torch.where(small.unsqueeze(-1), 0.5 * w, out)
    return out


class ArmIK:
    """Batched DLS IK of one arm: the serial chain from the URDF root to the EE frame, and the joint limits of the
    arm joints (from the Pinocchio model). URDF, EE frame and joint names come from `arm_ik.resolve_combo`."""

    def __init__(self, side, model, urdf, ee, joint_names):
        import pytorch_kinematics as pk  # noqa: PLC0415 — only here, so importers (judge.core) need not have it

        with open(urdf, "rb") as fh:
            self.chain = pk.build_serial_chain_from_urdf(fh.read(), ee).to(dtype=DT, device=DEV)
        self.n = len(self.chain.get_joint_parameter_names())
        qi = [model.joints[model.getJointId(n)].idx_q for n in joint_names]
        self.lo = torch.tensor([model.lowerPositionLimit[i] for i in qi], dtype=DT, device=DEV)
        self.hi = torch.tensor([model.upperPositionLimit[i] for i in qi], dtype=DT, device=DEV)
        self.mid = 0.5 * (self.lo + self.hi)
        self.rng_ = torch.clamp(self.hi - self.lo, min=1e-6)

    def fk(self, q):
        """EE transforms [N,4,4] in the chain base frame."""
        return self.chain.forward_kinematics(q).get_matrix()  # [N,4,4]

    def jac(self, q):
        """EE Jacobians [N,6,n] in the base frame, rows [linear(3); angular(3)]."""
        return self.chain.jacobian(q)  # [N,6,n] base frame, [lin(3);ang(3)]

    def _dls(self, T_tgt, iters, lam, null_alpha, q0):
        """One batched DLS-IK pass from seeds q0 [M,n] to targets [M,4,4]."""
        tgt_p = T_tgt[:, :3, 3]
        tgt_R = T_tgt[:, :3, :3]
        W = torch.diag(torch.tensor([lam**2] * 6, dtype=DT, device=DEV))
        eye = torch.eye(self.n, dtype=DT, device=DEV).unsqueeze(0)
        q = q0
        for _ in range(iters):
            T = self.fk(q)
            cur_p = T[:, :3, 3]
            cur_R = T[:, :3, :3]
            e6 = torch.cat([tgt_p - cur_p, log3_batch(torch.bmm(tgt_R, cur_R.transpose(1, 2)))], -1).unsqueeze(-1)
            J = self.jac(q)
            JT = J.transpose(1, 2)
            A = torch.bmm(J, JT) + W
            dq = torch.bmm(JT, torch.linalg.solve(A, e6))
            if null_alpha:  # optional nullspace posture bias (off by default: harms margin)
                Nproj = eye - torch.bmm(JT, torch.linalg.solve(A, J))
                dq = dq + null_alpha * torch.bmm(Nproj, (self.mid.unsqueeze(0) - q).unsqueeze(-1))
            q = torch.clamp(q + dq.squeeze(-1), self.lo, self.hi)
        T = self.fk(q)
        pe = torch.linalg.norm(tgt_p - T[:, :3, 3], dim=-1)
        red = torch.rad2deg(torch.linalg.norm(log3_batch(torch.bmm(tgt_R, T[:, :3, :3].transpose(1, 2))), dim=-1))
        margin = (torch.minimum(q - self.lo, self.hi - q) / self.rng_).min(dim=-1).values
        return q, pe, red, margin

    # ── judge GPU batch fast path ───────────────────────────────────────────
    # Same maths as _dls, fewer launches and no host syncs: (1) FK once per iteration —
    # `jacobian_tensor(all_transforms=)` reuses the transforms `fk()` computed (the stock
    # `fk()`/`jac()` pair runs `_fk_impl` twice and `fk()` syncs on `fi.item()`), (2) the
    # constant W/eye tensors are hoisted, (3) `linalg.solve_ex(check_errors=False)` avoids the
    # per-iteration info-check sync (same kernel as `linalg.solve`), (4) optionally the whole
    # pass is captured as a CUDA graph per (rows, iters, lam, null_alpha) shape and replayed.
    # The kernels are the ones `_dls` launches; on the **same row count** the outputs are bitwise identical.
    # A call runs on its row bucket's row count, and the batched solve/bmm kernels are shape dependent — so a
    # result may differ from `_dls` on the exact count (ulp-level, rarely a different IK branch); the contract
    # at judge level is functional (same accepted / unconfirmed / exhausted sets and plans at 4 dp).
    # `_dls` itself is unchanged (bake and reach-map users).
    def _dls_consts(self, lam):
        # One (W, eye) pair per lam, kept for the life of the IK: a captured graph reads them **by address**
        # (they live outside its pool), so they must never be freed while a graph of that lam exists — a
        # one-slot cache would free the old pair when lam changes and the clip-boundary empty_cache (judge)
        # would hand that address back to the driver under a live graph.
        cs = self.__dict__.setdefault("_dlsc", {})
        c = cs.get(lam)
        if c is None:
            W = torch.diag(torch.tensor([lam**2] * 6, dtype=DT, device=DEV))
            eye = torch.eye(self.n, dtype=DT, device=DEV).unsqueeze(0)
            c = cs[lam] = (W, eye)
        return c

    def _dls_body(self, T_tgt, iters, lam, null_alpha, q0, W, eye):
        ch = self.chain
        ee = ch._serial_eef_frame_idx  # private attribute: pytorch-kinematics is pinned (pyproject)
        tgt_p = T_tgt[:, :3, 3]
        tgt_R = T_tgt[:, :3, :3]
        q = q0
        for _ in range(iters):
            allT = ch.forward_kinematics_tensor(q)
            T = allT[ee]
            cur_p = T[:, :3, 3]
            cur_R = T[:, :3, :3]
            e6 = torch.cat([tgt_p - cur_p, log3_batch(torch.bmm(tgt_R, cur_R.transpose(1, 2)))], -1).unsqueeze(-1)
            J = ch.jacobian_tensor(q, all_transforms=allT)
            JT = J.transpose(1, 2)
            A = torch.bmm(J, JT) + W
            dq = torch.bmm(JT, torch.linalg.solve_ex(A, e6, check_errors=False)[0])
            if null_alpha:
                Nproj = eye - torch.bmm(JT, torch.linalg.solve_ex(A, J, check_errors=False)[0])
                dq = dq + null_alpha * torch.bmm(Nproj, (self.mid.unsqueeze(0) - q).unsqueeze(-1))
            q = torch.clamp(q + dq.squeeze(-1), self.lo, self.hi)
        T = ch.forward_kinematics_tensor(q)[ee]
        pe = torch.linalg.norm(tgt_p - T[:, :3, 3], dim=-1)
        red = torch.rad2deg(torch.linalg.norm(log3_batch(torch.bmm(tgt_R, T[:, :3, :3].transpose(1, 2))), dim=-1))
        margin = (torch.minimum(q - self.lo, self.hi - q) / self.rng_).min(dim=-1).values
        return q, pe, red, margin

    GRAPH_MIN_HITS = 3  # capture a shape on its 3rd use (warm-up + capture ≈ 3 eager passes)
    GRAPH_MAX = 64  # LRU cap on cached graphs per IK (each graph exec also holds driver memory outside torch)
    GRAPH_MIN_ROWS = 16  # smallest row bucket
    # Every _dls graph replayed on one stream is captured into **one shared memory pool per stream**
    # (graph_pool_handle). A private pool per graph keeps that graph's freed intermediates reserved for it alone, so
    # the pools add up with the cache size (≈ 37 MB a graph), none of it releasable by empty_cache.
    # Sharing is safe because nothing reads a graph's memory after another graph of the same pool ran: the graphs of
    # a pool replay one at a time on their stream and _dls_fast clones a replay's outputs before it returns; the static
    # inputs (T_in · q_in) and the constants (W · eye) live outside the pool. Each graph's live outputs stay
    # allocated, so a later capture never lands on them. Graphs of **different** streams may run at the same time
    # (arm_flip._track overlaps its two chains), so they never share a pool, nor a graph (the cache key holds the
    # stream). Kernels and shapes are unchanged, so no result changes.
    _GRAPH_POOLS: dict = {}

    @classmethod
    def _graph_pool(cls, stream_key):
        if stream_key not in cls._GRAPH_POOLS:
            cls._GRAPH_POOLS[stream_key] = torch.cuda.graph_pool_handle()
        return cls._GRAPH_POOLS[stream_key]

    @classmethod
    def _bucket(cls, n):
        """Row bucket of a call: the smallest of {2^k, 3·2^(k-1)} ≥ n (16, 24, 32, 48, 64, … 49152, 65536) —
        ≤ 50 % padding (33 % except right above a power of two), ≈ 25 buckets over 16..65536.
        A graph cache keyed by the exact row count would miss almost every call (the judge's row count — alive ×
        seeds, rescue sizes — changes constantly). Rows are padded to the bucket (row 0 repeated: a
        well-conditioned, valid target) and the outputs cut back to n."""
        b = cls.GRAPH_MIN_ROWS
        while b < n:
            b = b * 3 // 2 if (b & (b - 1)) == 0 else (b // 3) * 4
        return b

    @staticmethod
    def _pad_rows(T_tgt, q0, B):
        n = T_tgt.shape[0]
        if n == B:
            return T_tgt, q0
        return (torch.cat([T_tgt, T_tgt[:1].expand(B - n, -1, -1)]), torch.cat([q0, q0[:1].expand(B - n, -1)]))

    def _dls_fast(self, T_tgt, iters, lam, null_alpha, q0, use_graph=True):
        """`_dls` on the row bucket of the call, eager for the first GRAPH_MIN_HITS − 1 calls of a shape and then
        as a captured CUDA graph (LRU of GRAPH_MAX); eager on the CPU."""
        W, eye = self._dls_consts(lam)
        if not use_graph or DEV == "cpu":
            return self._dls_body(T_tgt, iters, lam, null_alpha, q0, W, eye)
        n = int(T_tgt.shape[0])
        B = self._bucket(n)
        # the stream the call runs on is part of the key: a graph's static inputs and outputs are one set of buffers,
        # so a graph (and its pool) serves one stream only — two streams replaying one graph at once would race
        sk = torch.cuda.current_stream().cuda_stream
        key = (B, int(iters), float(lam), float(null_alpha), sk)
        gc = self.__dict__.setdefault("_dls_graphs", {})
        hits = self.__dict__.setdefault("_dls_hits", {})
        ent = gc.get(key)
        if ent is None:
            hits[key] = hits.get(key, 0) + 1
            if hits[key] < self.GRAPH_MIN_HITS:
                # eager, but on the bucket shape too: a call's result depends on its bucket only, never on
                # whether the graph was captured yet (the batched solve/bmm kernels are shape dependent).
                self._dls_neager = getattr(self, "_dls_neager", 0) + 1
                T_in, q_in = self._pad_rows(T_tgt, q0, B)
                outs = self._dls_body(T_in, iters, lam, null_alpha, q_in, W, eye)
                return tuple(o[:n].clone() for o in outs)
            if len(gc) >= self.GRAPH_MAX:
                old = min(gc, key=lambda k: gc[k]["t"])
                del gc[old]
            T_in, q_in = self._pad_rows(T_tgt, q0, B)
            T_in = T_in.clone()
            q_in = q_in.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._dls_body(T_in, iters, lam, null_alpha, q_in, W, eye)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._graph_pool(sk)):
                outs = self._dls_body(T_in, iters, lam, null_alpha, q_in, W, eye)
            ent = {"g": g, "T_in": T_in, "q_in": q_in, "outs": outs, "t": 0}
            gc[key] = ent
            self._dls_ncap = getattr(self, "_dls_ncap", 0) + 1
        ent["t"] = self._dls_ncall = getattr(self, "_dls_ncall", 0) + 1
        # rows [n:B] keep their previous contents (valid targets/seeds from the capture or an earlier call);
        # their results are discarded below. The eviction clock `t` is a per-ArmIK call counter — the same
        # counter (a replay of a shape ticks it), so `min(t)` = least recently used.
        ent["T_in"][:n].copy_(T_tgt)
        ent["q_in"][:n].copy_(q0)
        ent["g"].replay()
        return tuple(o[:n].clone() for o in ent["outs"])
