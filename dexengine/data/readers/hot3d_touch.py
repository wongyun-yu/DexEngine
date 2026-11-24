"""HOT3D contact evidence for clip cutting: fingertip-to-object touch labels per frame and their windows.

Per hand, per frame the nearest object whose surface is within ``CONTACT_M`` of a fingertip is that frame's label;
runs of no label up to ``BRIDGE`` frames between the same object are bridged; maximal same-object runs are contact
windows, and overlapping windows of the two hands merge (``merge_windows``). The reader
(dexengine/data/readers/hot3d.py) turns the merged windows into clips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

CONTACT_M = 0.02
BRIDGE = 15
SURF_POINTS = 2000


def surface_points(glb: Path, n: int = SURF_POINTS) -> np.ndarray:
    """Surface samples with a **fixed seed** (different samples shift contact boundaries by a frame and change clip
    names; measured on e00295/e00296)."""
    m = trimesh.load(str(glb), force="mesh", process=False)
    if not len(m.faces):
        return np.asarray(m.vertices, float)[:n]
    try:
        pts = trimesh.sample.sample_surface(m, n, seed=20260916)[0]
    except TypeError:  # old trimesh without a seed argument → fix the global RNG
        st = np.random.get_state()
        np.random.seed(20260916)
        pts = trimesh.sample.sample_surface(m, n)[0]
        np.random.set_state(st)
    return np.asarray(pts, float)


def touch_labels(tips: np.ndarray, objs: dict[str, np.ndarray], surf: dict[str, np.ndarray]) -> list[str | None]:
    """Per frame: uid of the nearest object whose surface is within CONTACT_M of any fingertip, else None. Gaps ≤ BRIDGE
    bridged."""
    T = tips.shape[0]
    best = np.full(T, np.inf)
    lab = np.array([None] * T, dtype=object)
    for uid, pose in objs.items():
        pts = surf[uid]
        for t in range(T):
            if not np.isfinite(tips[t]).all():
                continue
            R = Rot.from_quat(pose[t, [4, 5, 6, 3]]).as_matrix()
            w = pts @ R.T + pose[t, :3]
            d = cKDTree(w).query(tips[t], k=1)[0].min()
            if d < CONTACT_M and d < best[t]:
                best[t] = d
                lab[t] = uid
    out = list(lab)
    # bridge: None runs ≤ BRIDGE between identical labels
    t = 0
    while t < T:
        if out[t] is None:
            u = t
            while u < T and out[u] is None:
                u += 1
            if t > 0 and u < T and out[t - 1] == out[u] and (u - t) <= BRIDGE:
                for k in range(t, u):
                    out[k] = out[u]
            t = u
        else:
            t += 1
    return out


def touch_windows(labels: list) -> list[tuple[int, int, str]]:
    out = []
    t = 0
    T = len(labels)
    while t < T:
        if labels[t] is None:
            t += 1
            continue
        u = t
        while u < T and labels[u] == labels[t]:
            u += 1
        out.append((t, u, labels[t]))
        t = u
    return out


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ws = sorted(windows)
    out = []
    for a, b in ws:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out
