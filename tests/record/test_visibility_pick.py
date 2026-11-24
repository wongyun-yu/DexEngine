"""render.visibility.visible picks the depth pixels before the float cast: (inside, visible) equal the old whole-image
cast on random points, depths with NaN · inf · 0 · negative values, float32 views and float64 arrays, scalar and
per-point margins."""

from __future__ import annotations

import numpy as np
import pytest


def _old_visible(pc, depth, k: dict, margin):
    """The body before the change (the whole depth image cast to float64 per call)."""
    from dexengine.record.render import framing as CF

    pc = np.asarray(pc, float)
    ins = CF.inside(pc, k)
    vis = np.zeros(len(pc), bool)
    if ins.any():
        z = pc[ins, 2]
        u = np.clip((k["fx"] * pc[ins, 0] / z + k["cx"]).astype(int), 0, depth.shape[1] - 1)
        v = np.clip((k["fy"] * pc[ins, 1] / z + k["cy"]).astype(int), 0, depth.shape[0] - 1)
        d = np.asarray(depth, float)[v, u]
        m = np.broadcast_to(np.asarray(margin, float), (len(pc),))[ins]
        with np.errstate(invalid="ignore"):
            vis[ins] = ~np.isfinite(d) | (d <= 0) | (d >= z - m)
    return ins, vis


K = {
    "fx": 732.0785522460938,
    "fy": 732.0785522460938,
    "cx": 633.6272583007812,
    "cy": 356.186279296875,
    "width": 1280,
    "height": 720,
}


def _depth(rng, f64: bool):
    full = rng.uniform(0.05, 3.0, size=(800, 1400)).astype(np.float32)
    bad = rng.random(full.shape)
    full[bad < 0.05] = np.nan
    full[(bad >= 0.05) & (bad < 0.08)] = np.inf
    full[(bad >= 0.08) & (bad < 0.10)] = 0.0
    full[(bad >= 0.10) & (bad < 0.11)] = -1.0
    view = full[40:760, 60:1340]  # a non-contiguous crop like cameras.crop_ego
    return np.ascontiguousarray(view, np.float64) if f64 else view


@pytest.mark.parametrize("seed", range(30))
def test_visible_equals_whole_image_cast(seed):
    from dexengine.record.render.visibility import visible

    rng = np.random.default_rng(seed)
    n = int(rng.integers(0, 400))
    pc = np.c_[rng.uniform(-2, 2, n), rng.uniform(-1.2, 1.2, n), rng.uniform(-0.5, 3.5, n)]
    depth = _depth(rng, f64=bool(seed % 2))
    margin = 0.1 if seed % 3 == 0 else rng.uniform(0.0, 0.3, n)
    ins, vis = visible(pc, depth, K, margin)
    ins0, vis0 = _old_visible(pc, depth, K, margin)
    assert ins.dtype == ins0.dtype and vis.dtype == vis0.dtype
    assert np.array_equal(ins, ins0) and np.array_equal(vis, vis0)


def test_points_on_the_image_border_and_behind():
    from dexengine.record.render.visibility import visible

    rng = np.random.default_rng(7)
    depth = _depth(rng, f64=False)
    z = np.array([1.0, 1.0, 1.0, 1.0, -1.0, 0.0])
    x = (np.array([0.0, 1279.999, 0.0, 1279.999, 600.0, 600.0]) - K["cx"]) * np.abs(z) / K["fx"]
    y = (np.array([0.0, 0.0, 719.999, 719.999, 300.0, 300.0]) - K["cy"]) * np.abs(z) / K["fy"]
    pc = np.c_[x, y, z]
    for m in (0.0, 0.1, np.linspace(0, 0.5, len(pc))):
        a, b = visible(pc, depth, K, m), _old_visible(pc, depth, K, m)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
