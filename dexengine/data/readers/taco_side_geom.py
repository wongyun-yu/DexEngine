"""Geometry of the TACO side assignment (taco_side_assign.py): MANO hand vertices, object surfaces and
hand-object gaps, object motion and contact intervals."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexengine.data.objects import scene_requirement as SR
from dexengine.data.readers import mano_conventions as MC

FPS = 30.0  # TACO source frame rate (hand pkl frames = object pose frames; taco_convert compares the same T)


SURF_N = 4000  # object surface samples (vertices + face samples)


_MANO: dict = {}


def mano(side: str):
    if side not in _MANO:
        import smplx
        import torch

        from dexengine.core import capacity  # noqa: PLC0415 — threads per process: the owner
        from dexengine.core import paths as P

        torch.set_num_threads(capacity.THREADS)
        d, kw = str(P.mano_model_dir()), {"use_pca": False, "flat_hand_mean": True}
        _MANO[side] = m = smplx.MANO(d, is_rhand=(side == "right"), **kw)
        if side == "left":
            MC.negate_left("taco", m.shapedirs, smplx.MANO(d, **kw).shapedirs)
    return _MANO[side]


def hand_vertices(side: str, pose: np.ndarray, trans: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """(T, 778, 3) in source coordinates; joint 0 = hand_trans (the same convention as mano_lib.mano_forward with
    trans_is_wrist)."""
    import torch

    T = len(pose)
    with torch.no_grad():
        o = mano(side)(
            global_orient=torch.from_numpy(pose[:, :3].astype(np.float32)),
            hand_pose=torch.from_numpy(pose[:, 3:48].astype(np.float32)),
            betas=torch.from_numpy(np.repeat(shape.reshape(1, -1)[:, :10].astype(np.float32), T, 0)),
            transl=torch.zeros(T, 3),
        )
    return o.vertices.numpy().astype(np.float64) - o.joints[:, :1].numpy().astype(np.float64) + trans[:, None, :]


def surface(raw: Path, oid: str):
    import trimesh
    from scipy.spatial import cKDTree

    m = trimesh.load(raw / "object_models_released" / f"{oid}_cm.obj", force="mesh", process=False)
    v = np.asarray(m.vertices, float) * 0.01  # cm → m (the same unit as taco_convert_io prepare_mesh_for_taco)
    pts = np.concatenate([v, np.asarray(m.sample(SURF_N), float) * 0.01]) if len(m.faces) else v
    return cKDTree(pts)


def surface_gap(Vh: np.ndarray, traj: np.ndarray, tree, bound: float = np.inf) -> np.ndarray:
    """(T,) minimum distance (m) per frame between hand surface (vertices) and object surface samples, measured with the
    hand vertices moved into object coordinates. inf beyond bound (contact uses bound = 4·TOUCH_M for speed); the
    tie-break mean distance uses the same function without bound."""
    R, t = traj[:, :3, :3], traj[:, :3, 3]
    local = np.einsum("tji,tvj->tvi", R, Vh - t[:, None, :])  # Rᵀ (v − t)
    d, _ = tree.query(local.reshape(-1, 3), distance_upper_bound=bound)
    return d.reshape(len(Vh), -1).min(axis=1)


class LazyGap(dict):
    """{(hand, object): mean distance m}, measured without bound only when a tie-break asks (only tied sequences of the
    2,317 pay the cost)."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def __missing__(self, key):
        self[key] = v = float(self._fn(*key))
        return v


def obj_motion(tr: np.ndarray) -> dict:
    """Max displacement m · rotation deg from the first sample, and max step between samples m, of the trajectory (T, 4,
    4)."""
    p = tr[:, :3, 3]
    rel = np.einsum("tij,kj->tik", tr[:, :3, :3], tr[0, :3, :3])  # R_t · R_0^T
    cos = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "move_m": float(np.linalg.norm(p - p[0], axis=1).max()),
        "rot_deg": float(np.degrees(np.arccos(cos)).max()),
        "step_m": float(np.linalg.norm(np.diff(p, axis=0), axis=1).max()) if len(p) > 1 else 0.0,
    }


def moved(m: dict, touch: dict) -> bool:
    """scene_requirement's `moved` (thresholds from that module too)."""
    return (
        (m["move_m"] >= SR.MOVE_M or m["rot_deg"] >= SR.ROT_DEG)
        and SR.touched(touch, SR.TOUCH_MIN_S)
        and m["step_m"] < SR.JUMP_M
    )


def intervals(c: np.ndarray) -> list[list[float]]:
    """Runs of consecutive contact frames [[t0, t1]] (s), in the grasp_s form of grasp_runs.grasp_runs."""
    out, a = [], None
    for i, v in enumerate(list(c) + [False]):
        if v and a is None:
            a = i
        elif not v and a is not None:
            out.append([a / FPS, i / FPS])
            a = None
    return out
