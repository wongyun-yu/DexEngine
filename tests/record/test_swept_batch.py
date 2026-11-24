"""looks.augvis.swept.Swept.add (batched) gives the same height field bits and sphere count as the per-sphere loop it
replaced, over random desks, centres, scalar and per-sphere radii, spheres off the grid, above the cut height,
non-finite centres and repeated calls."""

from __future__ import annotations

import numpy as np
import pytest

from tests.imports import own_module

SW = own_module("dexengine.record.looks.augvis.swept")


def _old_add(sw, C, R) -> None:
    """Swept.add before batching (loop body verbatim)."""
    from dexengine.record.looks.augvis import catalog as AC

    C, R = np.asarray(C, float).reshape(-1, 3), np.broadcast_to(np.asarray(R, float), (len(C),))
    top = sw.d["top_z"] + AC.DISTRACT_MAX_H + SW.CLEAR_M + 0.05
    keep = (C[:, 2] - R) < top
    for c, r in zip(C[keep], R[keep], strict=True):
        ix = np.nonzero(np.abs(sw.x - c[0]) <= r)[0]
        iy = np.nonzero(np.abs(sw.y - c[1]) <= r)[0]
        if not len(ix) or not len(iy):
            continue
        dx = sw.x[ix][:, None] - c[0]
        dy = sw.y[iy][None, :] - c[1]
        inside = dx * dx + dy * dy <= r * r
        zlo = c[2] - np.sqrt(np.clip(r * r - dx * dx - dy * dy, 0, None))
        sub = sw.h[np.ix_(ix, iy)]
        sw.h[np.ix_(ix, iy)] = np.where(inside, np.minimum(sub, zlo), sub)
        sw.n += 1


def _desk(rng) -> dict:
    near = float(rng.uniform(0.1, 0.5))
    return {
        "near_x": near,
        "far_x": near + float(rng.uniform(0.3, 1.0)),
        "cy": float(rng.uniform(-0.3, 0.3)),
        "half_y": float(rng.uniform(0.2, 0.8)),
        "top_z": float(rng.uniform(0.6, 0.9)),
    }


def _centres(rng, d: dict, n: int):
    C = np.c_[
        rng.uniform(d["near_x"] - 0.3, d["far_x"] + 0.3, n),
        rng.uniform(d["cy"] - d["half_y"] - 0.3, d["cy"] + d["half_y"] + 0.3, n),
        rng.uniform(d["top_z"] - 0.2, d["top_z"] + 0.6, n),
    ]
    if n > 4 and rng.random() < 0.5:
        C[rng.integers(0, n, 3), rng.integers(0, 3, 3)] = [np.nan, np.inf, -np.inf]
    if n > 2 and rng.random() < 0.5:  # exact grid points and repeated centres (ties)
        C[0, :2] = d["near_x"], d["cy"]
        C[1] = C[0]
    return C


@pytest.mark.parametrize("seed", range(40))
def test_batch_equals_loop(seed):
    rng = np.random.default_rng(seed)
    d = _desk(rng)
    new, old = SW.Swept(d), SW.Swept(d)
    for _ in range(int(rng.integers(1, 6))):
        n = int(rng.integers(0, 900))
        C = _centres(rng, d, n)
        R = SW.R_LINK if rng.random() < 0.5 else rng.uniform(0.0, 0.25, n)
        new.add(C, R)
        _old_add(old, C, R)
        assert new.h.tobytes() == old.h.tobytes() and new.n == old.n


def test_single_point_and_empty():
    rng = np.random.default_rng(1)
    d = _desk(rng)
    new, old = SW.Swept(d), SW.Swept(d)
    for C, R in (
        ([[d["near_x"] + 0.05, d["cy"], d["top_z"]]], 0.07),
        (np.zeros((0, 3)), 0.07),
        ([[d["near_x"], d["cy"], d["top_z"] + 0.01]], [0.3]),
    ):
        new.add(C, R)
        _old_add(old, C, R)
        assert new.h.tobytes() == old.h.tobytes() and new.n == old.n


def test_signed_zero_tie_keeps_loop_order():
    """Two spheres giving +0.0 and -0.0 in the same cell: the loop keeps the first one met; so must the batch."""
    d = {"near_x": 0.0, "far_x": 0.1, "cy": 0.0, "half_y": 0.05, "top_z": 0.0}
    new, old = SW.Swept(d), SW.Swept(d)
    gx, gy = new.x[5], new.y[2]  # exact grid values: radius 0 still picks that cell
    C, R = np.array([[gx, gy, -0.0], [gx, gy, 0.0]]), np.array([0.0, 0.0])
    new.add(C, R)
    _old_add(old, C, R)
    assert new.h.tobytes() == old.h.tobytes() and new.n == old.n


def test_flat_centre_list_fails_like_the_loop():
    """A flat [x, y, z] (not [[x, y, z]]) broadcasts R to 3 entries against 1 centre: both versions raise."""
    d = {"near_x": 0.0, "far_x": 0.1, "cy": 0.0, "half_y": 0.05, "top_z": 0.0}
    with pytest.raises(IndexError):
        SW.Swept(d).add([0.05, 0.0, 0.0], 0.07)
    with pytest.raises(IndexError):
        _old_add(SW.Swept(d), [0.05, 0.0, 0.0], 0.07)


def test_radius_zero_on_grid_counts_the_cell():
    """dx² + dy² <= r² holds with equality at r = 0 on a grid point: the cell is stamped (not skipped)."""
    d = {"near_x": 0.0, "far_x": 0.1, "cy": 0.0, "half_y": 0.05, "top_z": 0.0}
    sw = SW.Swept(d)
    sw.add([[sw.x[3], sw.y[4], 0.25]], 0.0)
    assert sw.n == 1 and sw.h[3, 4] == 0.25 and np.isinf(np.delete(sw.h.ravel(), 3 * sw.h.shape[1] + 4)).all()


def test_cells_not_one_run_take_the_loop_path():
    """An unsorted grid (cells of a sphere not one run) goes through _add_one: still equal to the loop."""
    rng = np.random.default_rng(3)
    d = _desk(rng)
    new, old = SW.Swept(d), SW.Swept(d)
    perm = rng.permutation(len(new.x))
    new.x, old.x = new.x[perm], old.x[perm]
    C = _centres(rng, d, 500)
    new.add(C, 0.07)
    _old_add(old, C, 0.07)
    assert new.h.tobytes() == old.h.tobytes() and new.n == old.n
