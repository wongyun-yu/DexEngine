"""runtime_env owns warp: kernels compile with TMPDIR set to a short per-process directory and the caller's TMPDIR comes
back afterwards, warp is imported and launched only through the owner, and the limit itself is measured against
NVRTC — including a real compile under a 172-character TMPDIR (a node job's scratch)."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

from dexengine.core import runtime_env as RE
from dexengine.core.repo import REPO_ROOT


def _dir_of_length(base, n: int) -> str:
    d = str(base) + "/" + "a" * (n - len(str(base)) - 1)
    assert len(d) == n
    os.makedirs(d, exist_ok=True)
    return d


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """A process that has not made its short directory yet (rooted in tmp_path, so the test leaves nothing in /tmp)."""
    monkeypatch.setattr(RE, "_TMP_DIR", [])
    monkeypatch.setattr(RE, "TMP_ROOT", str(tmp_path))
    return tmp_path


def test_the_scope_sets_a_short_dir_and_gives_the_callers_tmpdir_back(fresh, monkeypatch, capsys):
    long = "/" + "t" * 171
    monkeypatch.setenv("TMPDIR", long)
    with RE.nvrtc_scope() as d:
        assert os.environ["TMPDIR"] == d and len(d) <= RE.NVRTC_TMPDIR_MAX and os.path.isdir(d)
        assert os.path.basename(d).startswith(RE.TMP_PREFIX)
    assert os.environ["TMPDIR"] == long
    monkeypatch.delenv("TMPDIR")
    with RE.nvrtc_scope() as d2:
        assert d2 == d  # one directory per process
    assert "TMPDIR" not in os.environ  # unset before, unset after
    err = capsys.readouterr().err
    assert err.count("[XR_TMP] NVRTC temp dir") == 1 and "(original TMPDIR 172 chars)" in err


def test_the_scope_gives_tmpdir_back_when_the_compile_raises(fresh, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/job/scratch")
    with pytest.raises(RuntimeError), RE.nvrtc_scope():
        raise RuntimeError("kernel build failed")
    assert os.environ["TMPDIR"] == "/job/scratch"


def test_a_short_dir_over_the_limit_dies_with_length_and_limit(fresh, monkeypatch):
    root = _dir_of_length(fresh, RE.NVRTC_TMPDIR_MAX)
    monkeypatch.setattr(RE, "TMP_ROOT", root)
    with pytest.raises(SystemExit) as e:
        RE.short_tmp_dir()
    for part in ("characters", "at most 125", "EDG", "TMP_ROOT"):
        assert part in str(e.value)


def test_launch_runs_inside_the_scope(fresh, monkeypatch):
    seen = {}
    monkeypatch.setenv("TMPDIR", "/job/scratch")
    fake = type("W", (), {"launch": staticmethod(lambda *a, **k: seen.setdefault("tmp", os.environ["TMPDIR"]))})
    monkeypatch.setattr(RE, "warp", lambda: fake)
    RE.launch("kernel", dim=1)
    assert seen["tmp"] == RE.short_tmp_dir() and os.environ["TMPDIR"] == "/job/scratch"


def violations(src: str) -> list[str]:
    """Imports of warp and kernel launches (``<x>.launch(`` on warp: ``wp.launch``) outside the owner."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] == "warp" for a in n.names):
            out.append(f"{n.lineno}: import warp")
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "warp" and not n.level:
            out.append(f"{n.lineno}: from warp")
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "launch"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id in ("wp", "warp")
        ):
            out.append(f"{n.lineno}: {n.func.value.id}.launch")
    return out


def test_dexengine_imports_and_launches_warp_only_through_the_owner():
    bad = {}
    for p in sorted((REPO_ROOT / "dexengine").rglob("*.py")):
        if p.name == "runtime_env.py" and p.parent.name == "core":
            continue
        if v := violations(p.read_text(encoding="utf-8")):
            bad[str(p.relative_to(REPO_ROOT))] = v
    assert not bad, bad


@pytest.mark.parametrize(
    "src",
    [
        "import warp as wp\n",
        "def f():\n    import warp\n",
        "from warp import config\n",
        "import warp.sim\n",
        "def f(wp, k):\n    wp.launch(k, dim=1)\n",
    ],
)
def test_the_structural_check_catches_planted_imports_and_launches(src):
    assert violations(src)


_COMPILE = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, sys.argv[2])
    from dexengine.core import runtime_env as RE
    wp = RE.warp()
    from warp._src import build as B
    wp.config.quiet = True
    wp.config.kernel_cache_dir = sys.argv[1]
    wp.init()
    src = os.path.join(sys.argv[1], "k.cu")
    open(src, "w").write('extern "C" __global__ void k(float* a) { a[0] = 1.0f; }\\n')
    def build():
        try:
            B.build_cuda(src, 86, os.path.join(sys.argv[1], "k.ptx"))
            return "OK"
        except Exception:
            return "FAIL"
    if sys.argv[3] == "scope":
        with RE.nvrtc_scope():
            r = build()
    else:
        r = build()
    print(r, len(os.environ.get("TMPDIR", "")))
    """
)


def _compile(tmp_path, tmpdir: str, mode: str) -> tuple[list[str], str]:
    cache = tmp_path / f"cache{len(tmpdir)}{mode}"
    cache.mkdir()
    env = {**os.environ, "TMPDIR": tmpdir, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(
        [sys.executable, "-c", _COMPILE, str(cache), str(REPO_ROOT), mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return r.stdout.split(), r.stderr


needs_warp = pytest.mark.skipif(importlib.util.find_spec("warp") is None, reason="warp is not installed here")


@needs_warp
def test_the_limit_is_nvrtcs(tmp_path):
    """Oracle (measurement vs statement): NVRTC compiles a kernel with TMPDIR of exactly the limit and fails one past
    it (no GPU needed: NVRTC compiles on the CPU)."""
    base = tmp_path / "t"
    base.mkdir()
    if len(str(base)) > RE.NVRTC_TMPDIR_MAX - 2:
        pytest.skip(f"pytest's tmp_path is too long to build a {RE.NVRTC_TMPDIR_MAX}-character directory under it")
    got = {n: _compile(tmp_path, _dir_of_length(base, n), "bare")[0][0] for n in (125, 126)}
    assert got == {RE.NVRTC_TMPDIR_MAX: "OK", RE.NVRTC_TMPDIR_MAX + 1: "FAIL"}, got


@needs_warp
def test_a_node_jobs_172_character_tmpdir_compiles_inside_the_scope(tmp_path):
    """A real compile under a 172-character TMPDIR passes inside nvrtc_scope (marker [XR_TMP], the caller's TMPDIR is
    back afterwards) and fails without it (the negative: the scope is what makes it pass)."""
    base = tmp_path / "s"
    base.mkdir()
    long = _dir_of_length(base, 172)
    out, err = _compile(tmp_path, long, "scope")
    assert out == ["OK", "172"] and "[XR_TMP] NVRTC temp dir /tmp/xr." in err and "(original TMPDIR 172 chars)" in err
    assert _compile(tmp_path, long, "bare")[0] == ["FAIL", "172"]
