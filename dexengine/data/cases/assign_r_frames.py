"""Frame inputs of rule R (assign_r): hand points, object features and the continuous frame-layer quantities of
hand–object pairs. Pure functions only.

Same version as assign_r (dexengine.data.cases.assign_version.ASSIGN2_VERSION covers both modules). The meaning of the
rule is in the assign_r docstring; this module measures the continuous quantities behind its 'hand points' and 'frame
layer' (prox · touch · window net displacement Dobj · slip Drel · hand net displacement Dtip). Material points = the
hand points in the object (part) local frame (K1).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.data.cases import (
    # ASSIGN-1 measurement and shared thresholds (palm, fingertip count, small object, smoothing, joint drive)
    hand_object_assign as H,
)

TH_WIN_S = 0.5  # motion window (±0.25 s)
TH_PALM_ALPHA = 0.4  # palm proxy point
TH_JUMP_M = 0.04  # tracking jump (per frame); ±2 frames are dropped with it


def thresholds() -> dict:
    return {k: v for k, v in globals().items() if k.startswith("TH_")}


def _axis_rot(axis, theta):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    return Rot.from_rotvec(a[None, :] * np.asarray(theta, float)[:, None]).as_matrix()


def quat_wxyz_to_mat(q) -> np.ndarray:
    """(T,4) wxyz → (T,3,3); object pose (columns 3:7 of the object's 7 tree-qpos columns)."""
    return Rot.from_quat(np.asarray(q, float)[:, [1, 2, 3, 0]]).as_matrix()


def rot_deg(Rm) -> np.ndarray:
    """(T,3,3) rotations → angle (deg)."""
    return np.degrees(Rot.from_matrix(Rm).magnitude())


def hand_points(tips, wrist) -> np.ndarray:
    """Hand points (T,6,3) = 5 fingertips + the palm proxy."""
    tips, wrist = np.asarray(tips, float), np.asarray(wrist, float)
    return np.concatenate([tips, (wrist + TH_PALM_ALPHA * (tips.mean(1) - wrist))[:, None]], 1)


# Object layer.
def object_features(o7, fps: float, small: bool, art: dict | None, hinge, parts: dict | None) -> dict:
    """One object: pose, motion, jumps. o7 (T,7) = position + wxyz · parts = {"base"|"moving"|"all": (points, KD tree)}
    (articulated objects only)."""
    o7 = np.asarray(o7, float)
    po = o7[:, :3]
    Ro = quat_wxyz_to_mat(o7[:, 3:7])
    step = np.r_[0.0, np.linalg.norm(np.diff(po, axis=0), axis=1)]
    jump = step >= TH_JUMP_M
    if jump.any():
        jj = jump.copy()
        for k in (1, 2):
            jj[k:] |= jump[:-k]
            jj[:-k] |= jump[k:]
        jump = jj
    return {
        "po": po,
        "Ro": Ro,
        "jump": jump,
        "speed": H.smooth_velocity(po, fps)[0],
        "small": bool(small),
        "span": float((po.max(0) - po.min(0)).max()) if len(po) else 0.0,
        "art": art,
        "hinge": None if hinge is None else np.asarray(hinge, float),
        "parts": parts,
    }


def _q_part(parts, art, loc, hinge):
    """Object-local loc (T,K,3) → (on the moving part (T,K), part coordinates (T,K,3)), from the nearest surface
    sample's part."""
    T, K = loc.shape[:2]
    best = None
    for name, (_pts, tree) in parts.items():
        q = loc
        if name == "moving":
            Rh = _axis_rot(art["axis"], hinge)
            pos = np.asarray(art["pos"], float)
            q = np.einsum("tji,tkj->tki", Rh, loc - pos) + pos
        d, _ = tree.query(q.reshape(-1, 3))
        d = d.reshape(T, K)
        cur = (np.full((T, K), name == "moving"), q, d)
        if best is None:
            best = cur
        else:
            m = d < best[2]
            best = (np.where(m, cur[0], best[0]), np.where(m[..., None], cur[1], best[1]), np.where(m, cur[2], best[2]))
    return best[0], best[1]


def _part_world(m, mov, Ro, po, art, hinge, tt):
    x = m
    if art is not None:
        Rh = _axis_rot(art["axis"], hinge[tt])
        pos = np.asarray(art["pos"], float)
        xm = np.einsum("tij,tkj->tki", Rh, m - pos) + pos
        x = np.where(mov[..., None], xm, m)
    return np.einsum("tij,tkj->tki", Ro[tt], x) + po[tt][:, None, :]


def pair_features(P6, dist, palm_dist, ob: dict, fps: float, thr: float) -> dict:
    """Continuous frame-layer quantities of one hand × one object (material points = the hand points themselves, K1).
    dist (T,5) fingertip surface distance · palm_dist (T,) · thr = fingertip threshold."""
    P6 = np.asarray(P6, float)
    d, dp = np.asarray(dist, float), np.asarray(palm_dist, float)
    T = len(P6)
    near = d < thr
    palm = dp < H.TH_PALM_M
    prox = (near.sum(1) >= H.TH_GRASP_MIN_TIPS) | palm | (bool(ob["small"]) & near.any(1))
    touch = near.any(1) | palm
    C = np.concatenate([near, palm[:, None]], 1)
    nC = C.sum(1)
    Ro, po, art, hinge = ob["Ro"], ob["po"], ob["art"], ob["hinge"]
    loc = np.einsum("tji,tkj->tki", Ro, P6 - po[:, None, :])
    if art is not None:
        mov, mloc = _q_part(ob["parts"], art, loc, hinge)
    else:
        mov, mloc = np.zeros(loc.shape[:2], bool), loc
    h = max(1, int(round(TH_WIN_S * fps / 2)))
    t = np.arange(T)
    a = np.clip(t - h, 0, T - 1)
    b = np.clip(t + h, 0, T - 1)
    Dobj = np.linalg.norm(
        _part_world(mloc, mov, Ro, po, art, hinge, b) - _part_world(mloc, mov, Ro, po, art, hinge, a), axis=2
    )
    Dtip = np.linalg.norm(P6[b] - P6[a], axis=2)
    same = mov[b] == mov[a]
    Drel = np.where(same, np.linalg.norm(mloc[b] - mloc[a], axis=2), np.nan)
    span = (b - a) / fps
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.maximum(nC, 1)
        DobjC = np.where(nC > 0, (Dobj * C).sum(1) / den, np.nan)
        DrelC = np.where(nC > 0, (np.where(np.isfinite(Drel), Drel, 0) * C).sum(1) / den, np.nan)
        DtipC = np.where(nC > 0, (Dtip * C).sum(1) / den, np.nan)
    dmin = np.minimum(d.min(1), dp)
    drive_old = np.zeros(T, bool)
    if art is not None:
        drive_old = near.any(1) & (np.abs(np.gradient(np.asarray(hinge, float))) * fps > H.TH_ARTIC_DRIVE_RADPS)
    return {
        "prox": prox,
        "touch": touch,
        "DobjC": DobjC,
        "DrelC": DrelC,
        "DtipC": DtipC,
        "span": span,
        "dmin": dmin,
        "drive_old": drive_old,
    }
