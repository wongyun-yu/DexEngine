"""Render-stage thread pools: the OpenBLAS pool is bound to RENDER_BLAS_THREADS with its marker (session.kit), and the
store's pre-measurement pool is sized from the CPUs the process may really use (capacity.cpu_limit, the one reader)."""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from dexengine.core import capacity
from dexengine.record.session import kit as KX
from dexengine.record.store import lerobot as LR

# ---- OpenBLAS pool (kit.apply_render_blas / check_render_blas) ---------------------------------------------------

_CHILD = textwrap.dedent(
    """
    import os, sys
    import numpy as np
    from dexengine.record.session import kit as KX
    if not KX.openblas_libs():
        print("NO_OPENBLAS"); sys.exit(0)
    np.linalg.inv(np.eye(4) + 0.1)
    KX.apply_render_blas()
    np.linalg.inv(np.eye(4) + 0.1)
    KX.check_render_blas()
    if sys.argv[1] == "reset":
        KX._blas_threads(2)
        KX.check_render_blas()
    """
)


def _child(mode: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(Path(KX.__file__).parents[3]), OPENBLAS_NUM_THREADS="4")
    return subprocess.run([sys.executable, "-c", _CHILD, mode], capture_output=True, text=True, env=env, timeout=120)


def test_render_blas_marker_and_binding():
    r = _child("ok")
    if "NO_OPENBLAS" in r.stdout:
        pytest.skip("this interpreter's numpy loads no OpenBLAS")
    assert r.returncode == 0, r.stderr[-2000:]
    k = len(os.sched_getaffinity(0))
    m = re.search(rf"\[RENDER_THREADS\] openblas={KX.RENDER_BLAS_THREADS} affinity={k} libs=(\{{.*\}})", r.stdout)
    assert m, r.stdout
    libs = eval(m.group(1))  # noqa: S307 — our own marker, a dict literal
    assert libs and set(libs.values()) == {KX.RENDER_BLAS_THREADS}
    assert re.search(r"\[RENDER_THREADS\] end check openblas=1 libs=\{.+\}", r.stdout), r.stdout


def test_render_blas_end_check_catches_a_reset_pool():
    """Negative: a pool set back to 2 threads after the bind is caught by the end check."""
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("needs 2 CPUs")
    r = _child("reset")
    if "NO_OPENBLAS" in r.stdout:
        pytest.skip("this interpreter's numpy loads no OpenBLAS")
    assert r.returncode != 0 and "OpenBLAS threads at session end" in r.stderr, (r.stdout[-500:], r.stderr[-500:])


def test_render_blas_unknown_api_dies(monkeypatch):
    """Negative: a loaded OpenBLAS whose thread-count API is not in BLAS_API dies loudly (no silent skip)."""
    import numpy as np  # noqa: F401 — loads numpy's OpenBLAS

    if not KX.openblas_libs():
        pytest.skip("this interpreter's numpy loads no OpenBLAS")
    monkeypatch.setattr(KX, "BLAS_API", (("no_such_set_num_threads", "no_such_get_num_threads"),))
    with pytest.raises(SystemExit, match="no thread-count API"):
        KX._blas_threads(None)


def test_render_blas_api_covers_loaded_builds():
    """Every OpenBLAS this interpreter's numpy loads exposes one of the BLAS_API names (Kit numpy 1.26 64-bit build,
    numpy >= 2 scipy-openblas 64-bit build): read-back works without binding."""
    import numpy as np  # noqa: F401

    if not KX.openblas_libs():
        pytest.skip("this interpreter's numpy loads no OpenBLAS")
    got = KX._blas_threads(None)
    assert got and all(v >= 1 for v in got.values()), got


@pytest.mark.parametrize("n", [0, 10_000])
def test_render_blas_constant_outside_affinity_dies(monkeypatch, n):
    monkeypatch.setattr(KX, "RENDER_BLAS_THREADS", n)
    with pytest.raises(SystemExit, match="affinity core count"):
        KX.apply_render_blas()


# ---- pre-measurement pool (capacity.cpu_limit · capacity.plan) -------------------------------------------

#: (name, affinity CPUs, cgroup kind, own cgroup path, {relative dir: (file, text) | [(file, text)]}, expected pool).
#: The host core count (os.cpu_count) is planted as 64 in every case: reading it would give 6 and miss.
POOL_CASES = [
    ("affinity 8, no quota (v2 max)", 8, "v2", "/s", {"s": ("cpu.max", "max 100000")}, 2),
    ("affinity 32, quota 6 (v2)", 32, "v2", "/s", {"s": ("cpu.max", "600000 100000")}, 1),
    (
        "affinity 32, parent quota 8, child free (v2)",
        32,
        "v2",
        "/p/c",
        {"p": ("cpu.max", "800000 100000"), "p/c": ("cpu.max", "max 100000")},
        2,
    ),
    ("affinity 32, no quota (v2)", 32, "v2", "/s", {"s": ("cpu.max", "max 100000")}, 6),
    (
        "affinity 32, quota 6 (v1 cfs)",
        32,
        "v1",
        "/",
        {"cpu": [("cpu.cfs_quota_us", "600000"), ("cpu.cfs_period_us", "100000")]},
        1,
    ),
    ("affinity 32, quota 7.5 rounds up (v2)", 32, "v2", "/s", {"s": ("cpu.max", "750000 100000")}, 2),
    (
        "affinity 32, no quota (v1 -1)",
        32,
        "v1",
        "/",
        {"cpu": [("cpu.cfs_quota_us", "-1"), ("cpu.cfs_period_us", "100000")]},
        6,
    ),
]


def _pool_sizes(mod, td: Path) -> list[tuple[str, int, int, int]]:
    """Per case: (name, expected, usable CPUs // 4 capped, threads of the pool prewarm really built)."""
    out = []
    for k, (name, aff, kind, own, files, want) in enumerate(POOL_CASES):
        root = td / f"cg{k}"
        root.mkdir()
        if kind == "v2":
            (root / "cgroup.controllers").write_text("cpu\n")
        for rel, fs in files.items():
            (root / rel).mkdir(parents=True, exist_ok=True)
            for f, txt in fs if isinstance(fs, list) else [fs]:
                (root / rel / f).write_text(txt + "\n")
        proc = td / f"proc{k}"
        proc.write_text(f"0::{own}\n" if kind == "v2" else f"4:cpu,cpuacct:{own}\n")
        old = list(mod._PRE_POOL)
        mod._PRE_POOL.clear()
        try:
            with (
                mock.patch("os.sched_getaffinity", return_value=set(range(aff))),
                mock.patch("os.cpu_count", return_value=64),
                mock.patch.object(capacity.cpu_limit, "__kwdefaults__", {"proc_cgroup": str(proc), "root": root}),
                mock.patch.object(mod, "ffmpeg_bin", return_value="ffmpeg"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                got = max(1, min(capacity.cpu_limit()[0] // 4, capacity.NEEDS[mod.PREWARM_STEP].cap))
                mod.prewarm([])
                real = mod._PRE_POOL[0]._max_workers
            mod._PRE_POOL[0].shutdown()
        finally:
            mod._PRE_POOL[:] = old
        out.append((name, want, got, real))
    return out


def test_prewarm_pool_follows_the_usable_cpus(tmp_path):
    for name, want, got, real in _pool_sizes(LR, tmp_path):
        assert got == want == real, f"{name}: from cpu_limit {got}, real pool {real}, expected {want}"


def test_prewarm_pool_marker(tmp_path, capsys):
    old = list(LR._PRE_POOL)
    LR._PRE_POOL.clear()
    try:
        with mock.patch.object(LR, "ffmpeg_bin", return_value="ffmpeg"):
            LR.prewarm([])
        n = LR._PRE_POOL[0]._max_workers
        LR._PRE_POOL[0].shutdown()
    finally:
        LR._PRE_POOL[:] = old
    usable = capacity.cpu_limit()[0]
    assert f"{LR.MARK} pre-measurement pool {n} threads (usable CPUs {usable}" in capsys.readouterr().out


def test_prewarm_pool_negative_host_core_count(tmp_path):
    """Negative: a copy that sizes the pool from os.cpu_count() again is caught by the cases above."""
    src = Path(LR.__file__).read_text()
    bad = src.replace(
        "usable = capacity.cpu_limit()[0]",
        "usable = __import__('os').cpu_count() or 1",
    )
    assert bad != src
    mod = type(LR)("lerobot_planted")
    mod.__dict__["__file__"] = LR.__file__
    exec(compile(bad, LR.__file__, "exec"), mod.__dict__)  # noqa: S102 — the module's own source, one line changed
    rows = _pool_sizes(mod, tmp_path)
    assert [r for r in rows if not (r[2] == r[1] == r[3])]
