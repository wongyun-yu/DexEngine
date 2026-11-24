#!/usr/bin/env python3
"""Fingertip keypoint ↔ object surface distance geometry; the base adapter (object assignment, interaction markers) and
contact.sidecar use **the same functions**.

ensure_surface(ds, obj)            meshes layer data/<ds>/meshes/<obj_id>/surface_points.npz + keypoints_fps2.npy
                                   (surface_parts.npz for articulated); only missing ones are built · producer =
                                       object_surface (ensure_rigid · ensure_parts)
tip_distances(tips, obj7, surf, art, hinge)  (T,5,3) world fingertips × (T,7) object pose (pos+quat wxyz) → dist (T,5) ·
witness_world (T,5,3) · n_pts
point_distances(pts, obj7, surf, art, hinge) the (T,3) single-point (wrist) version
contact_from_dist(dist)            thresholds and smoothing are owned by dexengine/data/contact/common.py alone
fit_articulation(A7, B7)           fits the relative motion of two rigid pose sequences to a screw axis →
rigid/revolute/prismatic/screw/free (oakink2 part pairs)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P  # noqa: E402
from dexengine.data import object_surface as OS  # noqa: E402  the only producer of surfaces and keypoints
from dexengine.data.contact import common as CC  # noqa: E402
from dexengine.data.contact import (
    fingerprint as SF,  # noqa: E402  owner of the surface file path rule (the sidecar fingerprint hashes the same file)
)

dl = P.data_layout
ROOT = P.main_root()
KEYPOINT_THRESHOLD_M = (
    CC.CONTACT_THRESHOLD_M + CC.KEYPOINT_PAD_M
)  # keypoint ↔ surface contact threshold (2 cm), owned by dexengine.data.contact.common


def ensure_surface(ds: str, obj: dict) -> Path:
    out = SF.surface_path(ROOT, ds, obj)
    src = ROOT / dl.object_path(ds, obj["mesh_dir"])  # stored = logical name → physical (owner data_layout)
    art = obj.get("articulation")
    if art:
        OS.ensure_parts(src, art["parts"], out, source=str(src.relative_to(ROOT)))
    else:
        OS.ensure_rigid(src / "visual.obj", out)
    return out


_TREES: dict[str, tuple] = {}


def _load_surf(surf: Path):
    k = str(surf)
    if k not in _TREES:
        z = np.load(surf)
        if "points" in z.files:
            _TREES[k] = (("all", z["points"], cKDTree(z["points"])),)
        else:
            _TREES[k] = (
                ("base", z["base_points"], cKDTree(z["base_points"])),
                ("moving", z["moving_points"], cKDTree(z["moving_points"])),
            )
    return _TREES[k]


def _axis_rot(axis, theta):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return Rot.from_rotvec(a[None, :] * np.asarray(theta, float)[:, None]).as_matrix()


def _query(pts_w: np.ndarray, obj7: np.ndarray, surf: Path, art: dict | None, hinge: np.ndarray | None):
    """pts_w (T,K,3) world → (dist (T,K), witness_world (T,K,3), n_pts)."""
    T, K = pts_w.shape[:2]
    R = Rot.from_quat(obj7[:, [4, 5, 6, 3]]).as_matrix()
    t = obj7[:, :3]
    Rt = np.transpose(R, (0, 2, 1))
    loc = np.einsum("tij,tkj->tki", Rt, pts_w - t[:, None, :])
    best_d = np.full((T, K), np.inf)
    best_w = np.zeros((T, K, 3))
    n = 0
    for name, pts, tree in _load_surf(surf):
        n += len(pts)
        q = loc
        if name == "moving":
            Rh = _axis_rot(art["axis"], hinge)
            pos = np.asarray(art["pos"], float)
            q = (
                np.einsum("tji,tkj->tki", Rh, loc - pos) + pos
            )  # query points are rotated back and placed against the resting surface
        d, i = tree.query(q.reshape(-1, 3))
        d = d.reshape(T, K)
        w = pts[i].reshape(T, K, 3)
        if name == "moving":
            w = np.einsum("tij,tkj->tki", Rh, w - pos) + pos
        m = d < best_d
        best_d = np.where(m, d, best_d)
        best_w = np.where(m[..., None], w, best_w)
    wit_w = np.einsum("tij,tkj->tki", R, best_w) + t[:, None, :]
    return best_d, wit_w, n


def tip_distances(tips_w, obj7, surf, art=None, hinge=None):
    return _query(np.asarray(tips_w, float), np.asarray(obj7, float), Path(surf), art, hinge)


def point_distances(pts_w, obj7, surf, art=None, hinge=None):
    d, w, n = _query(np.asarray(pts_w, float)[:, None, :], np.asarray(obj7, float), Path(surf), art, hinge)
    return d[:, 0], w[:, 0], n


def contact_from_dist(dist: np.ndarray) -> np.ndarray:
    """(T,K) → (T,K,1) bool; threshold (keypoint 2 cm) and smoothing from dexengine.data.contact.common."""
    return CC.smooth_contact_block((dist < KEYPOINT_THRESHOLD_M)[:, :, None])


# Joint fitting (6-DoF pose sequences of two parts).
ROT_MIN_DEG, TR_MIN_MM, RES_OK_MM = 3.0, 10.0, 10.0


def fit_articulation(A7: np.ndarray, B7: np.ndarray) -> dict:
    """Try to explain the relative motion of B seen in A's frame with one screw axis. The returned dict is written into
    task_info as is."""
    Ra = Rot.from_quat(A7[:, [4, 5, 6, 3]])
    Rb = Rot.from_quat(B7[:, [4, 5, 6, 3]])
    rel_R = Ra.inv() * Rb
    rel_t = Ra.inv().apply(B7[:, :3] - A7[:, :3])
    dR = rel_R[0].inv() * rel_R
    rv = dR.as_rotvec()
    ang = np.degrees(np.linalg.norm(rv, axis=1))
    dt = rel_t - rel_t[0]
    tr_range = float((dt.max(0) - dt.min(0)).max()) * 1000
    out = {
        "angle_range_deg": round(float(ang.max()), 2),
        "translation_range_mm": round(tr_range, 1),
        "n_frames": int(len(A7)),
        "frame": "part A(base) local",
    }
    if ang.max() < ROT_MIN_DEG:
        if tr_range < TR_MIN_MM:
            return {**out, "type": "rigid"}
        u, s, vt = np.linalg.svd(dt - dt.mean(0), full_matrices=False)
        lin = float(s[0] ** 2 / max((s**2).sum(), 1e-12))
        return {
            **out,
            "type": "prismatic" if lin > 0.95 else "free",
            "axis": vt[0].round(5).tolist(),
            "linearity": round(lin, 3),
        }
    sel = ang > max(ROT_MIN_DEG, 0.25 * ang.max())
    n = rv[sel] / np.linalg.norm(rv[sel], axis=1, keepdims=True)
    n = n * np.sign(n @ n.mean(0))[:, None]
    axis = n.mean(0)
    axis /= np.linalg.norm(axis)
    consist = float(np.abs(n @ axis).mean())
    theta = rv @ axis  # signed angle
    Rn = _axis_rot(axis, theta)  # (T,3,3)
    # (I − R(θ)) p + θ·d·axis = dt   → unknowns p(3), d(1)
    M = np.concatenate([np.eye(3)[None] - Rn, (theta[:, None] * axis[None, :])[:, :, None]], axis=2).reshape(-1, 4)
    sol, *_ = np.linalg.lstsq(M, dt.reshape(-1), rcond=None)
    p, pitch = sol[:3], sol[3]
    res = float(np.sqrt(np.mean(np.sum((M @ sol).reshape(-1, 3) - dt, axis=1) ** 2))) * 1000
    typ = "free" if (res > RES_OK_MM or consist < 0.9) else ("revolute" if abs(pitch) < 0.002 else "screw")
    return {
        **out,
        "type": typ,
        "axis": axis.round(5).tolist(),
        "point": p.round(4).tolist(),
        "pitch_m_per_rad": round(float(pitch), 5),
        "axis_consistency": round(consist, 3),
        "residual_mm": round(res, 2),
        "angle_min_deg": round(float(np.degrees(theta.min())), 2),
        "angle_max_deg": round(float(np.degrees(theta.max())), 2),
    }
