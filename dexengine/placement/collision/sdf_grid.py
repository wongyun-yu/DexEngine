#!/usr/bin/env python3
"""Signed-distance grid of a union of convex hulls — build, disk cache and trilinear read (numpy · torch twin).

The torso filter (collision.torso) reads the static torso through it. Values: inside = min over hulls of the max plane
value (the exact depth of a convex hull); outside within ``shell_m`` = distance to the hulls' dense surface samples;
farther = the plane value (a lower bound). Outside the grid box the read is the grid's margin + the distance to the box
(a lower bound — a point is never 'far' by default). The margin and the shell are the caller's (torso: the mesh
ruler's NEAR_M shell · MARGIN_M, or the capsule proxy's wider arm_proxy.GRID_SHELL_M · GRID_MARGIN_M); the grid
dict carries its margin, so the read never assumes one. Hulls are dicts with ``v`` (vertices), ``eq`` (scipy ConvexHull
equations) and ``smp`` (dense surface samples).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from dexengine.core import capacity

GRID_M = 0.004  # grid spacing
MARGIN_M = 0.06  # grid margin around the hulls (the mesh ruler's)


def key_bytes(x) -> bytes:
    """The bytes the cache key hashes for one array: float64, C order, signed zeros made +0.0. The same hull reaches
    this key in the judge (Isaac python, numpy 1.26) and in the bake (retargeting python, numpy 2.3), and numpy 2.3
    gives -0.0 where numpy 1.26 gives +0.0 on some hull coordinates (measured on 4 torso hulls; values
    equal). Without the +0.0 the two keys differ and every lane rebuilds the grid. The grid values do not change."""
    return (np.ascontiguousarray(x, np.float64) + 0.0).tobytes()


def grid(hulls: list, shell_m: float, margin_m: float, cache_dir: Path) -> tuple[dict, str]:
    """(grid, marker text). Cached on disk under ``cache_dir`` keyed by the hull vertices and samples and the
    constants — a changed hull or constant never reads an old grid. Written atomically (temp + replace)."""
    hs = hashlib.sha256(repr((GRID_M, margin_m, shell_m)).encode())
    for h in hulls:
        for x in (h["v"], h["smp"]):
            hs.update(key_bytes(x))
    f = Path(cache_dir) / f"torso_grid_{hs.hexdigest()[:16]}.npz"
    if f.is_file():
        z = np.load(f)
        return {
            "lo": z["lo"],
            "h": float(z["h"]),
            "dims": z["dims"],
            "val": z["val"],
            "margin": margin_m,
        }, f"grid cache hit {f}"
    g = _build(hulls, shell_m, margin_m)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".{os.uname().nodename}.{os.getpid()}.tmp.npz")  # a shared volume: pods share pids
    np.savez(tmp, **{k: x for k, x in g.items() if k != "margin"})
    tmp.replace(f)
    return g, f"grid built {f}"


def _build(hulls: list, shell_m: float, margin_m: float) -> dict:
    from scipy.spatial import cKDTree

    lo = np.min([h["v"].min(0) for h in hulls], 0) - margin_m
    hi = np.max([h["v"].max(0) for h in hulls], 0) + margin_m
    dims = np.ceil((hi - lo) / GRID_M).astype(int) + 1
    axes = [lo[i] + GRID_M * np.arange(dims[i]) for i in range(3)]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    mp = np.full(len(X), np.inf)
    for i0 in range(0, len(X), 1 << 18):
        x = X[i0 : i0 + (1 << 18)]
        for h in hulls:
            mp[i0 : i0 + len(x)] = np.minimum(mp[i0 : i0 + len(x)], (x @ h["eq"][:, :3].T + h["eq"][:, 3]).max(-1))
    out = (mp > 0) & (mp < shell_m)
    val = mp.copy()
    val[out] = np.maximum(
        cKDTree(np.concatenate([h["smp"] for h in hulls])).query(X[out], workers=capacity.threads_here())[0], mp[out]
    )
    return {"lo": lo, "h": np.float64(GRID_M), "dims": dims, "val": val.reshape(*dims), "margin": margin_m}


def interp(g: dict, pw: np.ndarray) -> np.ndarray:
    """Trilinear read of the grid at world points pw (..., 3) (scipy map_coordinates, order 1 = trilinear)."""
    from scipy.ndimage import map_coordinates

    u = (pw - g["lo"]) / g["h"]
    ok = np.all((u >= 0) & (u < g["dims"] - 1), -1)
    out = map_coordinates(g["val"], u.reshape(-1, 3).T, order=1, mode="nearest").reshape(pw.shape[:-1])
    hi = g["lo"] + g["h"] * (g["dims"] - 1)
    return np.where(ok, out, g["margin"] + np.linalg.norm(np.maximum(np.maximum(g["lo"] - pw, pw - hi), 0.0), axis=-1))


def to_torch(g: dict, dev) -> dict:
    """The grid as torch float64 tensors on dev (the input of interp_t)."""
    import torch

    f64 = {"dtype": torch.float64, "device": dev}
    return {
        "lo": torch.tensor(g["lo"], **f64),
        "h": float(g["h"]),
        "dims": torch.tensor(g["dims"], dtype=torch.long, device=dev),
        "val": torch.tensor(g["val"], **f64),
        "margin": float(g["margin"]),
    }


def interp_t(g: dict, pw):
    """torch twin of interp (g from to_torch)."""
    import torch

    u = (pw - g["lo"]) / g["h"]
    i0 = torch.floor(u).long()
    ok = ((i0 >= 0) & (i0 <= g["dims"] - 2)).all(-1)
    i0 = torch.minimum(torch.clamp(i0, min=0), g["dims"] - 2)
    f, V = u - i0.to(u.dtype), g["val"]
    out = torch.zeros(pw.shape[:-1], dtype=pw.dtype, device=pw.device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (f[..., 0] if dx else 1 - f[..., 0])
                    * (f[..., 1] if dy else 1 - f[..., 1])
                    * (f[..., 2] if dz else 1 - f[..., 2])
                )
                out = out + w * V[i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz]
    hi = g["lo"] + g["h"] * (g["dims"] - 1)
    far = g["margin"] + torch.linalg.norm(torch.clamp(torch.maximum(g["lo"] - pw, pw - hi), min=0.0), dim=-1)
    return torch.where(ok, out, far)
