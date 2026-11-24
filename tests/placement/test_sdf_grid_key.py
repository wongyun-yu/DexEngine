"""The torso SDF grid's disk cache key (dexengine.placement.collision.sdf_grid.key_bytes): the same hull values give the
same key whatever the sign of their zeros (the judge's numpy 1.26 and the bake's numpy 2.3 differ there)."""

import numpy as np

from dexengine.placement.collision import sdf_grid as SG


def _hulls(zero: float) -> list:
    v = np.array([[0.0, zero, 0.1], [0.2, 0.0, zero]])
    return [{"v": v, "smp": np.array([[zero, 0.3, 0.0]]), "eq": None}]


def _calls(monkeypatch, tmp_path):
    built = []

    def _build(hulls, shell_m, margin_m):
        built.append(1)
        return {"lo": np.zeros(3), "h": SG.GRID_M, "dims": np.ones(3, int), "val": np.zeros(1), "margin": margin_m}

    monkeypatch.setattr(SG, "_build", _build)
    first = SG.grid(_hulls(0.0), 0.01, 0.02, tmp_path)[1]
    second = SG.grid(_hulls(-0.0), 0.01, 0.02, tmp_path)[1]
    return first, second, len(built)


def test_signed_zeros_give_one_key():
    assert SG.key_bytes(np.array([0.0, 1.5])) == SG.key_bytes(np.array([-0.0, 1.5]))
    assert SG.key_bytes(np.array([0.0, 1.5])) != SG.key_bytes(np.array([0.0, 1.25]))  # other values: other key


def test_a_grid_built_with_plus_zero_is_read_with_minus_zero(monkeypatch, tmp_path):
    first, second, n = _calls(monkeypatch, tmp_path)
    assert first.startswith("grid built") and second.startswith("grid cache hit") and n == 1


def test_without_the_normalisation_the_keys_split(monkeypatch, tmp_path):
    """Negative: hashing the raw bytes (the old key) misses the cache for the same values with -0.0."""
    monkeypatch.setattr(SG, "key_bytes", lambda x: np.ascontiguousarray(x, np.float64).tobytes())
    first, second, n = _calls(monkeypatch, tmp_path)
    assert first.startswith("grid built") and second.startswith("grid built") and n == 2
