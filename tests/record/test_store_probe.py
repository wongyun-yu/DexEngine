"""store.lerobot video measurement. The frame count taken from the stats decode equals ffprobe -count_frames, and the
stats from that decode equal the count-then-decode path. The probe cache is keyed by content: a copy hits, a changed
file misses (also swept over real render videos)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from dexengine.record.store import lerobot as CL
from dexengine.record.store import store as ST

if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    pytest.skip("needs ffmpeg and ffprobe", allow_module_level=True)


def _mp4(path, n, size="64x48"):
    cmd = [CL.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30"]
    cmd += ["-frames:v", str(n), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return path


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(CL, "_PRE", {})


@pytest.mark.parametrize(("n", "size"), [(37, "64x48"), (130, "320x240")])  # 320 px: the auto-downsample branch
def test_decode_count_equals_count_frames(tmp_path, n, size):
    p = _mp4(tmp_path / "v.mp4", n, size)
    old = CL._ffprobe_stream(p, count=True)
    new = CL.measure_video(p)
    assert new == old
    assert new["frames"] == n
    h, w = new["height"], new["width"]
    entry = CL._PRE[CL._ckey(p)]
    assert (h, w) in entry  # the stats came from the probe's own decode
    want = CL._sampled_stats(CL._frames(CL.ffmpeg_bin(), p, h, w, CL.sample_indices(old["frames"])))
    assert list(entry[(h, w)]) == list(want)
    for k, v in want.items():
        np.testing.assert_array_equal(np.asarray(entry[(h, w)][k]), np.asarray(v))


def test_stated_count_mismatch_measures_the_count(tmp_path, monkeypatch):
    p = _mp4(tmp_path / "v.mp4", 37)
    real = CL._ffprobe_stream
    monkeypatch.setattr(CL, "_ffprobe_stream", lambda q, count: real(q, count) | ({} if count else {"frames": 36}))
    assert CL.measure_video(p)["frames"] == 37
    assert list(CL._PRE[CL._ckey(p)]) == ["probe"]  # no stats from a decode sampled for the wrong count
    r = ST._ffprobe(p)
    st = CL.video_stats(CL.ffmpeg_bin(), p, r["height"], r["width"], ST._ffprobe)
    assert st["count"] == [len(CL.sample_indices(37))]


@pytest.mark.parametrize("stated", [None, 25])  # 25 = the duplicating decode's own count
def test_duplicating_decode_measures_the_count(tmp_path, monkeypatch, stated):
    """20 frames with a 5-frame timestamp gap: ffmpeg's constant-rate output repeats 5 frames (25 written), while
    ffprobe -count_frames counts the 20 decoded ones. A stated count that matches the 25 must not be taken."""
    p = tmp_path / "gap.mp4"
    cmd = [CL.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=30", "-frames:v", "20"]
    cmd += ["-vf", "setpts='if(gte(N,10),PTS+5,PTS)'", "-fps_mode", "passthrough", "-c:v", "libx264"]
    subprocess.run([*cmd, "-pix_fmt", "yuv420p", str(p)], check=True)
    if stated is not None:
        real = CL._ffprobe_stream
        monkeypatch.setattr(
            CL, "_ffprobe_stream", lambda q, count: real(q, count) | ({} if count else {"frames": stated})
        )
    _, got, exact = CL._decode(CL.ffmpeg_bin(), p, 48, 64, [])
    assert (got, exact) == (25, False)
    r = CL.measure_video(p)
    assert r["frames"] == 20
    assert list(CL._PRE[CL._ckey(p)]) == ["probe"]
    st = CL.video_stats(CL.ffmpeg_bin(), p, r["height"], r["width"], ST._ffprobe)
    want = CL._sampled_stats(CL._frames(CL.ffmpeg_bin(), p, 48, 64, CL.sample_indices(20)))
    for k, v in want.items():
        np.testing.assert_array_equal(np.asarray(st[k]), np.asarray(v))


def test_probe_cache_keyed_by_content(tmp_path, monkeypatch):
    p = _mp4(tmp_path / "a.mp4", 20)
    decodes, counts = [], []
    real_decode, real_probe = CL._decode, CL._ffprobe_stream
    monkeypatch.setattr(CL, "_decode", lambda *a: decodes.append(a[1]) or real_decode(*a))
    monkeypatch.setattr(
        CL, "_ffprobe_stream", lambda q, count: (counts.append(q) if count else None) or real_probe(q, count)
    )
    first = CL.measure_video(p)
    cp = tmp_path / "copy" / "b.mp4"
    cp.parent.mkdir()
    shutil.copyfile(p, cp)
    assert ST._ffprobe(cp) == first  # a copy (other path and inode) hits
    CL.video_stats(CL.ffmpeg_bin(), cp, first["height"], first["width"], ST._ffprobe)
    assert (len(decodes), len(counts)) == (1, 0)
    first["frames"] = -1  # callers get copies
    assert ST._ffprobe(p)["frames"] == 20
    _mp4(p, 21)  # same path, changed content: a count-only caller pays one ffprobe -count_frames, no decode
    assert ST._ffprobe(p)["frames"] == 21
    assert (len(decodes), len(counts)) == (1, 1)
