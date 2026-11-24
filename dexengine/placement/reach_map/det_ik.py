"""Host-independent arm IK for the CPU bake: `ArmIK`'s damped least squares in float64 det_math.

Why: the bake's arm IK stage (coarse_path.dp, arm_flip) ran `ArmIK` — pytorch_kinematics FK/Jacobian, torch bmm and
linalg.solve in float32. On the CPU those kernels follow the instruction set of the host (oneDNN/MKL/OpenBLAS pick
AVX-512 or AVX2 code with different reduction orders), so the same cell gave different joint angles on different CPUs.
This class evaluates the same maths with det_math only (fixed-order element-wise float64, its own sin/cos/atan) and
DetFK (the pinocchio model's joint tree read once), so the result has the same bits on every x86-64 host:
  e6 = [p_tgt − p ; log3(R_tgt · Rᵀ)],  A = J Jᵀ + λ² I (6×6, J = LOCAL_WORLD_ALIGNED [linear; angular]),
  x = A⁻¹ e6 (Cholesky, det_math.solve_spd),  q ← clip(q + Jᵀ x, lo, hi)
— `ArmIK._dls` term by term (the nullspace posture term is not used by the bake: a nonzero null_alpha dies).
Interface = `ArmIK`'s as the bake uses it (n · lo · hi · mid · rng_ · fk · jac · _dls · _dls_fast), torch tensors in
and out (float64, CPU), so coarse_path and arm_flip call it unchanged. Which stage uses it: coarse_path.dp._arm_ik
(owner of the rule — the CPU bake). Agreement with pytorch_kinematics and bit goldens: tests/placement/test_det_ik.py.
"""

from __future__ import annotations

import numpy as np
import torch

from dexengine.placement.collision import det_math as D
from dexengine.placement.collision.det_fk import DetFK

ENGINE = "det-f64"
DT = torch.float64
_R2D = 57.295779513082320876798  # 180 / pi


def log3(R):
    """SO(3) log of (..., 3, 3) → rotation vectors (..., 3) — log3_batch's formula in det_math."""
    tr = (R[..., 0, 0] + R[..., 1, 1]) + R[..., 2, 2]
    ang = D.acos_rad((tr - 1.0) * 0.5)
    w = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    small = ang < 1e-5
    s, _c = D.sincos(ang)
    denom = np.where(small, 1.0, 2.0 * s)
    out = w * (ang / denom)[..., None]
    return np.where(small[..., None], 0.5 * w, out)


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64) if isinstance(x, torch.Tensor) else np.asarray(x, np.float64)


class DetArmIK:
    """`ArmIK` (the bake's use of it) in float64 det_math on the CPU."""

    engine = ENGINE

    def __init__(self, side, model, urdf, ee, joint_names):
        import pinocchio as pin  # noqa: PLC0415

        self.F = DetFK(model)
        if not model.existFrame(ee):
            raise ValueError(f"[DET_IK] end-effector frame {ee} is not in the model of {urdf}")
        self.fid = model.getFrameId(ee)
        self.jids = [model.getJointId(n) for n in joint_names]
        self.iq = [model.joints[j].idx_q for j in self.jids]
        self.q_full = np.asarray(pin.neutral(model), float)  # joints off the arm chain stay at neutral (no effect)
        self.n = len(joint_names)
        lo = np.array([model.lowerPositionLimit[i] for i in self.iq], float)
        hi = np.array([model.upperPositionLimit[i] for i in self.iq], float)
        self._lo, self._hi, self._rng = lo, hi, np.maximum(hi - lo, 1e-6)
        self.lo, self.hi = torch.tensor(lo, dtype=DT), torch.tensor(hi, dtype=DT)
        self.mid = 0.5 * (self.lo + self.hi)
        self.rng_ = torch.clamp(self.hi - self.lo, min=1e-6)
        # The DLS loop evaluates only the arm chain: the EE's support = a fixed prefix (joints off the arm, at neutral)
        # followed by the arm joints in order. The prefix pose is computed once with DetFK's own operations, so the
        # loop's pose and Jacobian have the same bits as DetFK.frames / frame_jac6 on the full configuration.
        pj = self.F.f_parent[self.fid]
        sup = self.F.support[pj]
        k0 = len(sup) - len(self.jids)
        if k0 < 0 or sup[k0:] != self.jids:
            raise ValueError(f"[DET_IK] {ee}: the arm joints {joint_names} are not the tail of its support chain")
        b0 = sup[k0 - 1] if k0 else 0  # 0 = the universe (identity), as DetFK.joints sets it
        oR, op = self.F.joints(self.q_full, upto={b0} if k0 else set())
        self._base = (oR[b0], op[b0])
        self._pj = pj

    def _full(self, q):
        Q = np.broadcast_to(self.q_full, q.shape[:-1] + (len(self.q_full),)).copy()
        Q[..., self.iq] = q
        return Q

    def _fk(self, q):
        R, p = self.F.frames(self._full(q), [self.fid])
        return R[..., 0, :, :], p[..., 0, :]

    def fk(self, q):
        """EE transforms [N, 4, 4] (float64)."""
        R, p = self._fk(_np(q))
        T = np.zeros(R.shape[:-2] + (4, 4))
        T[..., :3, :3], T[..., :3, 3], T[..., 3, 3] = R, p, 1.0
        return torch.from_numpy(T)

    def jac(self, q):
        """EE Jacobians [N, 6, n] in the base frame, rows [linear(3); angular(3)] (float64)."""
        q = _np(q)
        return torch.from_numpy(self.F.frame_jac6(self._full(q), self.fid, self.jids))

    def _chain(self, q):
        """Arm joint poses from the fixed prefix: oR (..., n, 3, 3), op (..., n, 3) — DetFK.joints' operations."""
        F = self.F
        bR, bp = self._base
        Rs, ps = [], []
        for k, i in enumerate(self.jids):
            s, c = D.sincos(q[..., k])
            Rl = D.matmul3(F.R0[i], D.rot_axis(F.axis[i], s, c))
            ps.append(bp + D.matvec3(bR, F.p0[i]))
            bR = D.matmul3(bR, Rl)
            bp = ps[-1]
            Rs.append(bR)
        return Rs, ps

    def pose_jac(self, q):
        """EE pose and Jacobian [linear; angular] (..., 6, n) from one pass over the arm chain."""
        F = self.F
        Rs, ps = self._chain(q)
        R = D.matmul3(Rs[-1], F.fR[self.fid])
        p = ps[-1] + D.matvec3(Rs[-1], F.fp[self.fid])
        cols = []
        for k, j in enumerate(self.jids):
            w = D.matvec3(Rs[k], F.axis[j])
            cols.append(np.concatenate([D.cross3(w, p - ps[k]), w], -1))
        return R, p, np.stack(cols, -1)

    def _pose(self, q):
        Rs, ps = self._chain(q)
        return D.matmul3(Rs[-1], self.F.fR[self.fid]), ps[-1] + D.matvec3(Rs[-1], self.F.fp[self.fid])

    def _dls_np(self, Tt, iters, lam, q):
        tp, tR = Tt[:, :3, 3], Tt[:, :3, :3]
        l2I = float(lam) ** 2 * np.eye(6)
        for _ in range(int(iters)):
            R, p, J = self.pose_jac(q)  # J (M, 6, n)
            e6 = np.concatenate([tp - p, log3(D.matmul3(tR, np.swapaxes(R, -1, -2)))], -1)
            # A = J Jᵀ + λ² I, summed over the joints in a fixed order (element-wise products · no BLAS)
            A = J[..., :, None, 0] * J[..., None, :, 0]
            for k in range(1, self.n):
                A = A + J[..., :, None, k] * J[..., None, :, k]
            A = A + l2I
            x = D.solve_spd(A, e6)
            dq = J[..., 0, :] * x[..., 0, None]
            for i in range(1, 6):
                dq = dq + J[..., i, :] * x[..., i, None]
            q = np.minimum(np.maximum(q + dq, self._lo), self._hi)
        R, p = self._pose(q)
        pe = D.norm(tp - p)
        red = D.norm(log3(D.matmul3(tR, np.swapaxes(R, -1, -2)))) * _R2D
        margin = (np.minimum(q - self._lo, self._hi - q) / self._rng).min(-1)
        return q, pe, red, margin

    def _dls(self, T_tgt, iters, lam, null_alpha, q0):
        """One batched DLS pass from seeds q0 [M, n] to targets [M, 4, 4] → (q, pos err m, rot err deg, margin)."""
        if null_alpha:
            raise ValueError("[DET_IK] null_alpha ≠ 0 (nullspace posture term) is not implemented on the det path")
        out = self._dls_np(_np(T_tgt), iters, lam, _np(q0).copy())
        return tuple(torch.from_numpy(np.ascontiguousarray(v)) for v in out)

    def _dls_fast(self, T_tgt, iters, lam, null_alpha, q0, use_graph=True):
        """The judge's fast path has no CPU variant of its own: the same maths as `_dls`."""
        return self._dls(T_tgt, iters, lam, null_alpha, q0)
