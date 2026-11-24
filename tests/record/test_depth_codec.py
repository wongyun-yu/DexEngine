"""Depth codec (render.writer): thread rule, byte stability, the
priority and level of the compression command, and the CODEC record that lands in info.json."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from dexengine.record.render import writer as CW

pytestmark = pytest.mark.skipif(
    shutil.which("zstd") is None or shutil.which("nice") is None, reason="zstd and nice binaries are needed"
)


def _depth() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.integers(300, 1500, size=(6, 120, 160)) * (rng.random((6, 120, 160)) > 0.1)).astype(np.uint16)


def test_thread_rule_half_the_cores_capped():
    want = {1: 1, 2: 1, 4: 2, 8: 4, 16: 8, 64: 8}
    assert {n: CW.zstd_threads(n) for n in want} == want


def test_thread_count_does_not_change_bytes(tmp_path, monkeypatch):
    depth, got = _depth(), {}
    for t in (1, 8):
        monkeypatch.setattr(CW, "zstd_threads", lambda cores=None, t=t: t)
        p = tmp_path / f"t{t}{CW.EXT}"
        CW.write(p, depth)
        got[t] = p.read_bytes()
        assert np.array_equal(CW.read(p), depth)
    assert got[1] == got[8]


def test_command_is_niced_with_one_level(tmp_path, monkeypatch):
    seen, real = [], CW.subprocess.run

    def spy(argv, *a, **k):
        seen.append(list(argv))
        return real(argv, *a, **k)

    monkeypatch.setattr(CW.subprocess, "run", spy)
    CW.write(tmp_path / f"n{CW.EXT}", _depth())
    argv = seen[0]
    assert Path(argv[0]).name == "nice" and argv[1:3] == ["-n", str(CW.ZSTD_NICE)]
    assert Path(argv[3]).name == "zstd"
    assert [a for a in argv if a[:1] == "-" and a[1:].isdigit()] == [f"-{CW.ZSTD_LEVEL}"]
    assert CW.CODEC["compression"] == f"zstd -{CW.ZSTD_LEVEL}"
