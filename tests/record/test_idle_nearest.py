"""render.idle.nearest_body_mesh equals the full nearest-vertex search it replaced (name and float bits, ties too)."""

from __future__ import annotations

import numpy as np
import pytest

cKDTree = pytest.importorskip("scipy.spatial").cKDTree


def _full(trees: dict, H) -> tuple[str, float]:
    """The search idle.measure used before: every mesh in full, first mesh in order on a tie."""
    near = {n: float(tr.query(H)[0].min()) for n, tr in trees.items()}
    n_b = min(near, key=near.get)
    return n_b, near[n_b]


def _trees(rng, n_mesh: int, scale: float) -> dict:
    return {
        f"m{i}": cKDTree(rng.normal(size=(int(rng.integers(50, 3000)), 3)) * scale + rng.normal(size=3))
        for i in range(n_mesh)
    }


@pytest.mark.parametrize("seed", range(40))
def test_random_clouds_equal_full_search(seed):
    from dexengine.record.render.idle import nearest_body_mesh

    rng = np.random.default_rng(seed)
    trees = _trees(rng, int(rng.integers(1, 9)), float(rng.uniform(0.01, 0.5)))
    H = rng.normal(size=(int(rng.integers(1, 20000)), 3)) * rng.uniform(0.01, 0.3) + rng.normal(size=3) * 0.5
    assert nearest_body_mesh(trees, H) == _full(trees, H)


def test_tie_keeps_first_mesh_in_order():
    from dexengine.record.render.idle import nearest_body_mesh

    rng = np.random.default_rng(7)
    P = rng.normal(size=(500, 3))
    trees = {"far": cKDTree(P + 10.0), "a": cKDTree(P), "b": cKDTree(P.copy()), "c": cKDTree(P[::-1].copy())}
    H = rng.normal(size=(3000, 3)) + 0.05
    assert nearest_body_mesh(trees, H) == _full(trees, H)
    assert nearest_body_mesh(trees, H)[0] == "a"


def test_touching_and_single_point():
    from dexengine.record.render.idle import nearest_body_mesh

    rng = np.random.default_rng(3)
    P = rng.normal(size=(800, 3))
    trees = {"x": cKDTree(P + 0.3), "y": cKDTree(P)}
    H = np.concatenate([rng.normal(size=(1000, 3)) + 2.0, P[17:18]])  # one point sits on a vertex of y
    assert nearest_body_mesh(trees, H) == _full(trees, H) == ("y", 0.0)
    one = P[5:6] + 0.01
    assert nearest_body_mesh(trees, one) == _full(trees, one)
