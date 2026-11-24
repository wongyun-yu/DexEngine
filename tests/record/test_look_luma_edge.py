"""The brightness gate decides on the luma it records (rounded to 2 places), so the re-render list (ask) and the
distance to the range (dist) agree at the edge. With the raw value, a luma 0.004 below the floor was listed while
dist() read the stored value as in range, and the redraw loop ran 0 tries.
"""

from __future__ import annotations

import numpy as np
import pytest

from dexengine.record.looks import rev as L


def _edge_image() -> tuple[np.ndarray, float]:
    floor = float(L.LOOK_REV["floor"])
    img = np.full((72, 128, 3), int(floor), np.uint8)  # mean luma = floor - 37/9216 ≈ floor - 0.004
    img.reshape(-1, 3)[:37] = int(floor) - 1
    return img, floor


@pytest.mark.parametrize("name", ["R", "D"])
def test_edge_luma_list_and_distance_agree(name):
    img, floor = _edge_image()
    assert float(floor).is_integer()
    look = dict(name=name, **(L.r_post(img, "t") if name == "R" else L.d_post(img, "t")))
    meta = {"look": look}
    assert look["measured"]["luma_mean"] == floor
    assert (L.ask(meta) is not None) == (L.dist(meta) > 0)
