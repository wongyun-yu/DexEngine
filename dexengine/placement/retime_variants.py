#!/usr/bin/env python3
"""Placement transform and retiming resampler of a combo tree (library, no CLI).

``BaseMap`` maps qpos[0:6] of one hand block to the world SE(3) of its base body, fitted from the tree's own scene
model; ``rigid_pl`` applies a planar placement (XY shift + yaw about a pivot) in world space to the qpos, qvel and
contact positions; ``resample_all`` resamples every channel of a tree onto a retiming time map (``retime.warp``).
The bake, the retiming and the augmentation call these same functions.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.placement.retime import warp as _rt

DT = _rt.DT


class BaseMap(_rt.RootSpelling):
    """qpos[0:6] of one hand block  <->  world SE(3) of its base body.

    Fitted numerically from the scene model so the axis convention (permuted/negated axes, base body quats) never has
    to be hand-derived: p_w = p0 + A @ q[:3];  R_w = R0 @ E(sign * q[3:6]) with E an Euler map from a 96-candidate
    family (12 orders x 8 sign patterns), asserted < 1e-8.
    """

    def __init__(self, model, qadr0: int):
        self.model = model
        self.qadr0 = qadr0
        data = mujoco.MjData(model)
        # body of the LAST base joint = the hand base body
        jids = [j for j in range(model.njnt) if qadr0 <= model.jnt_qposadr[j] < qadr0 + 6]
        assert len(jids) == 6, f"expected 6 base joints at qpos {qadr0}, got {len(jids)}"
        self.bid = int(model.jnt_bodyid[max(jids, key=lambda j: model.jnt_qposadr[j])])

        def fk(q6):
            data.qpos[:] = 0.0
            data.qpos[qadr0 : qadr0 + 6] = q6
            mujoco.mj_kinematics(model, data)
            q = data.xquat[self.bid].copy()  # wxyz
            return data.xpos[self.bid].copy(), Rot.from_quat([q[1], q[2], q[3], q[0]])

        self.p0, R0 = fk(np.zeros(6))
        self.R0 = R0
        # translation axes
        self.A = np.stack([fk(np.eye(6)[i])[0] - self.p0 for i in range(3)], axis=1)
        self.Ainv = np.linalg.inv(self.A)
        # rotation convention: R0^-1 R_w(q) == E(sign*q)
        rng = np.random.default_rng(0)
        samples = rng.uniform(-1.2, 1.2, size=(20, 3))
        actual = [(R0.inv() * fk(np.concatenate([np.zeros(3), s]))[1]).as_matrix() for s in samples]
        best = None
        for order in ("".join(p) for p in itertools.permutations("xyz")):
            for seq in (order, order.upper()):
                for sgn in itertools.product((1.0, -1.0), repeat=3):
                    err = max(
                        np.abs(Rot.from_euler(seq, np.asarray(sg) * s).as_matrix() - a).max()
                        for s, a, sg in ((s, a, sgn) for s, a in zip(samples, actual, strict=True))
                    )
                    if best is None or err < best[0]:
                        best = (err, seq, np.asarray(sgn))
        err, self.seq, self.sign = best
        assert err < 1e-8, f"no euler convention matched (best {best[1]}/{best[2]} err={err:.2e})"
        # alt_euler must be the same rotation on THIS model — a root that is not a Tait-Bryan chain dies here
        alt_err = max(
            np.abs((R0.inv() * fk(np.concatenate([np.zeros(3), self.alt_euler(s)]))[1]).as_matrix() - a).max()
            for s, a in zip(samples, actual, strict=True)
        )
        if not alt_err < 1e-8:
            raise AssertionError(
                f"alt_euler is not the same rotation on this root ({self.seq}/{self.sign} err={alt_err:.2e})"
            )

    def to_world(self, q6: np.ndarray):
        p = self.p0 + q6[:, :3] @ self.A.T
        R = self.R0 * Rot.from_euler(self.seq, q6[:, 3:6] * self.sign)
        return p, R

    def from_world(self, p: np.ndarray, R) -> np.ndarray:
        q = np.empty((len(p), 6))
        q[:, :3] = (p - self.p0) @ self.Ainv.T
        q[:, 3:6] = (self.R0.inv() * R).as_euler(self.seq) * self.sign
        return q

    # the root-angle convention: the columns as a rotation; its two spellings come from _rt.RootSpelling
    def rot(self, q3):
        """rotation columns (N, 3) -> world rotation of the base body."""
        return self.R0 * Rot.from_euler(self.seq, np.asarray(q3, dtype=np.float64) * self.sign)


def maps_for(src, hqd: int):
    """(right, left) BaseMap of a tree, read from its own scene — the convention owner of its two hand roots."""
    model = mujoco.MjModel.from_xml_path(str(Path(src) / "scene.xml"))
    return BaseMap(model, 0), BaseMap(model, hqd)


def rigid_pl(qpos, qvel, contact_pos, hqd, dx, dy, yaw_deg, pivot, maps, layout=None):
    """Apply the baseline placement transform to one robot's arrays.

    ``layout`` (a ``qpos_layout.Layout``, for articulated objects): the object 7-blocks and qvel 6-blocks are read from
    its declaration and the joint columns (``layout.artic_cols()``) pass through unchanged (a rigid transform does not
    move them). None (default) = the arithmetic "contiguous 7-blocks after the hands" layout."""
    Rz = Rot.from_euler("z", np.deg2rad(yaw_deg))
    Rzm = Rz.as_matrix()
    piv = np.array([pivot[0], pivot[1], 0.0])
    d3 = np.array([dx, dy, 0.0])

    def tf_p(p):  # (...,3)
        return (p - piv) @ Rzm.T + piv + d3

    q = qpos.copy()
    h2 = 2 * hqd
    D = qpos.shape[1]
    # objects: pos + left-composed quat. The object blocks are every 7-column group after the two hands (assuming
    # "at least two objects" would give a zero-width slice on a one-object tree). A width not divisible by 7 means
    # joint columns are mixed in — that dies instead of passing silently.
    if layout is None:
        if (D - h2) % 7 != 0:
            raise ValueError(
                f"rigid_pl: qpos width {D} - hands {h2} = {D - h2} is not a multiple of 7 (articulated "
                "object columns? pass layout)"
            )
        n_obj = (D - h2) // 7
        obj_slots = [(h2 + 7 * k, h2 + 7 * (k + 1)) for k in range(n_obj)]
        vel_slots = [h2 + 6 * k2 for k2 in range(n_obj)]
    else:
        if int(layout.nq) != D or int(layout.hand_qpos_dim) != hqd:
            raise ValueError(
                f"rigid_pl: layout (nq={layout.nq}, H={layout.hand_qpos_dim}) differs from the array (D={D}, H={hqd})"
            )
        obj_slots = [(s_, s_ + 7) for s_ in layout.object_starts()]
        vel_slots = [b[3] for b in layout.qvel_blocks() if b[0] == "object"]  # object qvel starts, as declared
        n_obj = len(obj_slots)
    qz = Rz.as_quat()  # xyzw
    qz_w = np.array([qz[3], qz[0], qz[1], qz[2]])
    for lo, _hi in obj_slots:
        q[:, lo : lo + 3] = tf_p(q[:, lo : lo + 3])
        # Baseline _pl quats are NORMALIZED (apply_xy_shift composed through
        # scipy; measured 2.2e-16). The raw-passthrough contract covers the
        # BASE retargets; from _pl onward the baseline itself normalizes, and
        # variant outputs must equal the baseline variants byte-for-byte.
        w = q[:, lo + 3 : lo + 7]  # wxyz
        w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
        # left-compose: q' = qz (x) q
        a1, b1, c1, d1 = qz_w
        a2, b2, c2, d2 = w[:, 0], w[:, 1], w[:, 2], w[:, 3]
        q[:, lo + 3 : lo + 7] = np.stack(
            [
                a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            ],
            axis=1,
        )
    # wrist bases via world map
    for base, m in ((0, maps[0]), (hqd, maps[1])):
        p_w, R_w = m.to_world(qpos[:, base : base + 6])
        q[:, base : base + 6] = m.from_world(tf_p(p_w), Rz * R_w)
    # velocities: rotate object lin+ang and wrist slide rates; euler rates as-is
    v = None
    if qvel is not None:
        v = qvel.copy()
        nv = v.shape[1]
        for base, m in ((0, maps[0]), (hqd, maps[1])):
            v[:, base : base + 3] = v[:, base : base + 3] @ (m.Ainv @ Rzm @ m.A).T
        for lo in vel_slots:  # object qvel = linear 3 + angular 3 per object (joint qvel passes)
            if lo + 6 <= nv:
                v[:, lo : lo + 3] = v[:, lo : lo + 3] @ Rzm.T
                v[:, lo + 3 : lo + 6] = v[:, lo + 3 : lo + 6] @ Rzm.T
    cp = None
    if contact_pos is not None:
        cp = tf_p(contact_pos.reshape(len(contact_pos), -1, 3)).reshape(contact_pos.shape)
    return q, v, cp


def resample_all(z, tau, s, hqd, layout=None, *, root_maps):
    """Resample every channel of a tree npz ``z`` onto the time map (tau, s) of the retiming.

    ``root_maps``: callable -> the hand roots' convention owners in slot order (``maps_for``) — see
    ``warp.root_branch_apply``. ``layout``: when given, the quaternion slots are read from the declared object blocks
    (+3:+7) and joint columns are scalars (linear interpolation); None = the contiguous 7-block arithmetic."""
    qpos = z["qpos"].astype(np.float64)
    D = qpos.shape[1]
    # 1e-9 corrects floating point: without stretch (tau = arange(T) * DT) tau[-1] / DT can come out as 235.99999...
    # and floor would drop a frame. It has no effect when there is real stretch.
    T_new = int(np.floor(tau[-1] / DT + 1e-9)) + 1
    new_t = np.arange(T_new) * DT
    qs = _rt._quat_slots(D, hqd) if layout is None else [(s_ + 3, s_ + 7) for s_ in layout.object_starts()]
    es = _rt._euler_slots(D, hqd)

    def tau_for(n):
        if n <= len(tau):
            return tau[:n]
        extra = np.arange(1, n - len(tau) + 1) * (s[-1] * DT)
        return np.concatenate([tau, tau[-1] + extra])

    out = dict(z)
    qpos_u = qpos.copy()
    for lo, hi in es:
        qpos_u[:, lo:hi] = np.unwrap(qpos_u[:, lo:hi], axis=0)
    out["qpos"] = _rt._resample(qpos_u, tau, new_t, qs)
    for lo, hi in es:
        out["qpos"][:, lo:hi] = (out["qpos"][:, lo:hi] + np.pi) % (2 * np.pi) - np.pi
    _rt.root_branch_apply(qpos, out["qpos"], tau, new_t, es, root_maps)  # hand root: far-off inserted frames only
    if "qvel" in z and z["qvel"] is not None:
        qvel = np.asarray(z["qvel"]).astype(np.float64)
        n = len(qvel)
        inv = 1.0 / np.concatenate([s, np.full(max(0, n - len(s)), s[-1])])[:n]
        out["qvel"] = _rt._resample(qvel * inv[:, None], tau_for(n), new_t, [])
    if "qpos_rollout" in z and np.asarray(z["qpos_rollout"]).ndim == 2:
        qr = np.asarray(z["qpos_rollout"]).astype(np.float64)
        out["qpos_rollout"] = _rt._resample(qr, tau_for(len(qr)), new_t, qs)
    if layout is not None:
        # every joint-value sidecar (sided and background — owner Layout.artic_npz_keys) follows the same time map. A
        # stated sidecar that is missing or not on qpos's frame axis dies: skipping it would leave it on the old axis.
        for k in layout.artic_npz_keys():
            sc = np.asarray(z[k]) if k in z else None
            if sc is None or sc.ndim != 2 or len(sc) != len(qpos):
                raise ValueError(
                    f"[ARTIC] joint sidecar {k!r} shape {None if sc is None else sc.shape} != (T={len(qpos)}, ·) — "
                    "it left qpos's time axis (a source written before the f0 pad carried sidecars: rerun f0)"
                )
            out[k] = _rt._resample(sc.astype(np.float64), tau, new_t, []).astype(sc.dtype)
    if "contact" in z:
        ck = np.asarray(z["contact"])
        tc = tau_for(len(ck))
        idx = np.clip(np.searchsorted(tc, new_t), 0, len(ck) - 1)
        out["contact"] = ck[idx]
        if "contact_pos" in z:
            cp = np.asarray(z["contact_pos"]).astype(np.float64)
            flat = cp.reshape(len(cp), -1)
            out["contact_pos"] = _rt._resample(flat, tau_for(len(cp)), new_t, []).reshape(T_new, *cp.shape[1:])
    return out
