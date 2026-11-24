"""Host-independent forward kinematics of a pinocchio model (revolute joints) — the cross-repair decision path's FK.

pinocchio's own FK takes its sines and cosines from glibc ``sincos``, which is not the same function on every host
(2.35 ≠ 2.39 in 288 of 400,000 angles), and the cross-repair ruler turns that ulp into centimetres (det_math). This
class reads only the model's static data (joint tree, joint placements, joint axes, frame placements — parsed once by
pinocchio) and evaluates the same kinematics with det_math, batched over configurations:
  oMi = oMparent · jointPlacement_i · R(axis_i, q_i)   (revolute: no translation)
  oMf = oMi[parentJoint] · framePlacement
  frame Jacobian (LOCAL_WORLD_ALIGNED, linear rows) column of revolute joint j on the support of f:
      (oR_j · axis_j) × (p_f − p_j)
Supported joints: RX · RY · RZ · RevoluteUnaligned (every joint of the rby1 combo URDFs); any other kind raises — a new
joint type must be added here, not skipped. Agreement with pinocchio: test_det_fk (≤ 1e-12 on random q).
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.collision import det_math as D

_AXIS = {"JointModelRX": (1.0, 0.0, 0.0), "JointModelRY": (0.0, 1.0, 0.0), "JointModelRZ": (0.0, 0.0, 1.0)}


class DetFK:
    """Static data of a pinocchio Model → batched det_math FK. q is (..., nq) in the model's configuration order."""

    def __init__(self, model):
        nj = model.njoints
        self.nq = model.nq
        self.parent = [int(model.parents[i]) for i in range(nj)]
        self.R0 = np.array([np.asarray(model.jointPlacements[i].rotation, float) for i in range(nj)])
        self.p0 = np.array([np.asarray(model.jointPlacements[i].translation, float) for i in range(nj)])
        self.axis = np.zeros((nj, 3))
        self.iq = np.full(nj, -1)
        for i in range(1, nj):
            jm = model.joints[i]
            kind = jm.shortname()
            if kind in _AXIS:
                self.axis[i] = _AXIS[kind]
            elif kind == "JointModelRevoluteUnaligned":
                self.axis[i] = np.asarray(jm.extract().axis, float)
            else:
                raise ValueError(f"[DET_FK] joint {model.names[i]} is {kind} — only revolute joints are implemented")
            if jm.nq != 1:
                raise ValueError(f"[DET_FK] joint {model.names[i]} has nq {jm.nq}")
            self.iq[i] = int(jm.idx_q)
        self.f_parent = [int(f.parentJoint) for f in model.frames]
        self.fR = np.array([np.asarray(f.placement.rotation, float) for f in model.frames])
        self.fp = np.array([np.asarray(f.placement.translation, float) for f in model.frames])
        # support path (root → joint) of every joint
        self.support = []
        for i in range(nj):
            path, j = [], i
            while j > 0:
                path.append(j)
                j = self.parent[j]
            self.support.append(path[::-1])

    def joints(self, q, upto=None):
        """World poses of the joints (the frames of joint i): oR (..., nj, 3, 3), op (..., nj, 3). ``upto`` = a set of
        joint ids — only their support is evaluated (the others are left NaN)."""
        q = np.asarray(q, dtype=np.float64)
        lead = q.shape[:-1]
        nj = len(self.parent)
        need = range(1, nj) if upto is None else sorted({j for i in upto for j in self.support[i]})
        oR = np.full(lead + (nj, 3, 3), np.nan)
        op = np.full(lead + (nj, 3), np.nan)
        oR[..., 0, :, :] = np.eye(3)
        op[..., 0, :] = 0.0
        for i in need:
            s, c = D.sincos(q[..., self.iq[i]])
            Rl = D.matmul3(self.R0[i], D.rot_axis(self.axis[i], s, c))  # jointPlacement · R(axis, q)
            pa = self.parent[i]
            oR[..., i, :, :] = D.matmul3(oR[..., pa, :, :], Rl)
            op[..., i, :] = op[..., pa, :] + D.matvec3(oR[..., pa, :, :], self.p0[i])
        return oR, op

    def frames(self, q, fids):
        """World poses of the frames ``fids``: R (..., n, 3, 3), p (..., n, 3)."""
        fids = list(fids)
        oR, op = self.joints(q, upto={self.f_parent[f] for f in fids})
        pj = [self.f_parent[f] for f in fids]
        Rj, pjp = oR[..., pj, :, :], op[..., pj, :]
        R = D.matmul3(Rj, self.fR[fids])
        p = pjp + D.matvec3(Rj, self.fp[fids])
        return R, p

    def frame_pos(self, q, fid):
        """World position (..., 3) of one frame."""
        return self.frames(q, [fid])[1][..., 0, :]

    def frame_jac_lin(self, q, fid, joint_ids):
        """Linear rows (..., 3, k) of the LOCAL_WORLD_ALIGNED frame Jacobian for the revolute joints ``joint_ids``
        (in that column order); a joint off the frame's support gives a zero column."""
        pj = self.f_parent[fid]
        oR, op = self.joints(q, upto={pj})
        pf = op[..., pj, :] + D.matvec3(oR[..., pj, :, :], self.fp[fid])
        sup = set(self.support[pj])
        cols = []
        for j in joint_ids:
            if j in sup:
                w = D.matvec3(oR[..., j, :, :], self.axis[j])
                cols.append(D.cross3(w, pf - op[..., j, :]))
            else:
                cols.append(np.zeros(pf.shape))
        return np.stack(cols, -1)

    def frame_jac6(self, q, fid, joint_ids):
        """Rows [linear(3); angular(3)] (..., 6, k) of the LOCAL_WORLD_ALIGNED frame Jacobian (revolute joints)."""
        pj = self.f_parent[fid]
        oR, op = self.joints(q, upto={pj})
        pf = op[..., pj, :] + D.matvec3(oR[..., pj, :, :], self.fp[fid])
        sup = set(self.support[pj])
        cols = []
        for j in joint_ids:
            if j in sup:
                w = D.matvec3(oR[..., j, :, :], self.axis[j])
                cols.append(np.concatenate([D.cross3(w, pf - op[..., j, :]), w], -1))
            else:
                cols.append(np.zeros(pf.shape[:-1] + (6,)))
        return np.stack(cols, -1)
