"""Preview stills (store.preview._stills) give the same pixels as imageio.mimread(...)[j], and the depth colour map
(render.cameras.depth_colormap, bytes lut) the same bytes as the float formula it replaced (also checked on real
render episodes)."""

from __future__ import annotations

import numpy as np
import pytest

imageio = pytest.importorskip("imageio.v2")
pytest.importorskip("imageio_ffmpeg")
from matplotlib import colormaps  # noqa: E402

from dexengine.record.render import cameras as CC  # noqa: E402
from dexengine.record.store import preview as PV  # noqa: E402


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    rng = np.random.default_rng(0)
    frames = [np.clip(rng.normal(128, 60, (48, 64, 3)) + 3 * i, 0, 255).astype(np.uint8) for i in range(40)]
    p = tmp_path_factory.mktemp("v") / "v.mp4"
    imageio.mimwrite(p, frames, fps=30, codec="libx264", quality=None, ffmpeg_params=["-crf", "23"], macro_block_size=8)
    return p


@pytest.mark.parametrize("idx", [[0, 20, 39], [0, 1, 2], [38, 39], [5, 5, 17]])
def test_stills_equal_mimread(video, idx):
    full = imageio.mimread(str(video), memtest=False)
    got = PV._stills(video, idx)
    assert sorted(got) == sorted(set(idx))
    assert all(np.array_equal(got[j], np.asarray(full[j])) for j in idx)


def test_stills_missing_frame_dies(video):
    with pytest.raises(IndexError, match="the video gave"):
        PV._stills(video, [0, 40])


def test_depth_colormap_bytes_equal_float_formula():
    rng = np.random.default_rng(1)
    d = rng.uniform(-0.5, 3.0, (60, 80))
    d[::7, ::5] = np.nan
    d[3, :] = 0.0
    d[4, :] = np.inf
    rng_m = CC.DEPTH_VIZ_RANGE_M
    bad = ~np.isfinite(d) | (d <= 0)
    t = np.clip((np.where(bad, rng_m[1], d) - rng_m[0]) / (rng_m[1] - rng_m[0]), 0.0, 1.0)
    old = (colormaps["turbo"](t)[..., :3] * 255).astype(np.uint8)
    old[bad] = 0
    new = CC.depth_colormap(d)
    assert new.dtype == np.uint8 and np.array_equal(new, old)
