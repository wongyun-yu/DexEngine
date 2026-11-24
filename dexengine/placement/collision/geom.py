"""Capsule geometry of the arm rulers (numpy): URDF capsule reading, segment distances, capsule end points and the
batched link forward kinematics. Stateless; callers import this module directly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np


def _rpy(rpy):
    r, p, y = (float(v) for v in str(rpy).split())
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def load_capsules(urdf_path) -> list[dict]:
    """The <capsule> colliders of the URDF, per link, with the vendor collision bitmasks (coltype ``ct``,
    colaffinity ``ca``). The axis is link z (URDF capsule convention)."""
    root = ET.parse(str(urdf_path)).getroot()
    caps = []
    for link in root.findall("link"):
        ln = link.get("name")
        for col in link.findall("collision"):
            g = col.find("geometry")
            if g is None:
                continue
            c = g.find("capsule")
            if c is None:
                continue
            o = col.find("origin")
            caps.append(
                {
                    "link": ln,
                    "r": float(c.get("radius")),
                    "L": float(c.get("length")),
                    "ct": int(c.get("coltype", "0")),
                    "ca": int(c.get("colaffinity", "0")),
                    "p": np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]),
                    "R": _rpy(o.get("rpy", "0 0 0") if o is not None else "0 0 0"),
                }
            )
    return caps


def _seg_seg_dist(p1, q1, p2, q2):
    """Shortest distance between two segments (batched: p*, q* are (...,3)) — Ericson, Real-Time Collision Detection
    §5.1.9 (ClosestPtSegmentSegment), vectorised."""
    d1, d2 = q1 - p1, q2 - p2
    r = p1 - p2
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1)
    b = np.sum(d1 * d2, -1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1), 0, 1), 0.0)
    t = np.clip((b * s + f) / np.where(e > 1e-12, e, 1), 0, 1)
    s = np.clip((b * t - c) / np.where(a > 1e-12, a, 1), 0, 1)
    w = (p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])
    return np.linalg.norm(w, axis=-1)


def capsule_endpoints(cap: dict, T_link: np.ndarray):
    """Link world transform (...,4,4) → the capsule's two endpoints (...,3)."""
    R = T_link[..., :3, :3]
    t = T_link[..., :3, 3]
    c = t + np.einsum("...ij,j->...i", R, cap["p"])
    axis = np.einsum("...ij,jk,k->...i", R, cap["R"], np.array([0.0, 0.0, 1.0]))
    h = cap["L"] / 2.0
    return c - axis * h, c + axis * h


def _rot_batch(axis, th):
    """Rotations about one axis by the angles th (T,) → (T, 3, 3) (Rodrigues)."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s = np.sin(th)[:, None, None]
    c = np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


def arm_link_transforms(ctx: dict, q_all) -> np.ndarray:
    """q_all (T,nj) → link world transforms (T,nj,4,4). Robot-local (base not included)."""
    q = np.asarray(q_all, float)
    T, nj = len(q), len(ctx["joints"])
    out = np.empty((T, nj, 4, 4))
    cur = np.broadcast_to(ctx["B0"], (T, 4, 4)).copy()
    for k in range(nj):
        if k > 0:
            cur = cur @ ctx["A"][k]
        M = np.zeros((T, 4, 4))
        M[:, :3, :3] = _rot_batch(ctx["axes"][k], q[:, k])
        M[:, 3, 3] = 1.0
        cur = cur @ M
        out[:, k] = cur
    return out
