"""stdout / stderr redirection has one owner, dexengine/core/stdio.py.

Banned everywhere else in dexengine/: `contextlib.redirect_stdout` / `redirect_stderr` under any name (attribute,
from-import, alias, the name as a string) and assigning `sys.stdout` / `sys.stderr` (plain, tuple, an aliased `sys`,
`setattr(sys, …)` with a stream name or a computed name). The first swaps the process's stream for a block, so a
thread's block captures every other thread's prints (measured in the f0 prejudge's prep threads);
per-thread capture is `stdio.capture`, a process-level stream is `stdio.process_stream`. The rule is by owner module,
not by a list of allowed call sites.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from dexengine.core import stdio
from tests.imports import own_module

ROOT = Path(__file__).resolve().parents[2]
OWNER = "dexengine/core/stdio.py"
_REDIRECTS = {"redirect_stdout", "redirect_stderr"}
_STREAMS = {"stdout", "stderr"}


def _stream_targets(t, sys_names):
    if isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _stream_targets(e, sys_names)
    elif isinstance(t, ast.Starred):
        yield from _stream_targets(t.value, sys_names)
    elif isinstance(t, ast.Attribute) and t.attr in _STREAMS and isinstance(t.value, ast.Name):
        if t.value.id in sys_names:
            yield t


def _setattr_stream(n, sys_names) -> bool:
    if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" and len(n.args) >= 2):
        return False
    obj, name = n.args[0], n.args[1]
    named = isinstance(name, ast.Constant) and name.value in _STREAMS
    return isinstance(obj, ast.Name) and obj.id in sys_names and (named or not isinstance(name, ast.Constant))


def violations(src: str) -> list[str]:
    """Line-numbered redirections in one module's source."""
    tree = ast.parse(src)
    sys_names = {"sys"} | {
        a.asname for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names if a.name == "sys" and a.asname
    }
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in _REDIRECTS:
            out.append(f"{n.lineno}: .{n.attr}")
        elif isinstance(n, ast.ImportFrom) and any(a.name in _REDIRECTS for a in n.names):
            out.append(f"{n.lineno}: from {n.module} import redirect_*")
        elif isinstance(n, ast.Constant) and n.value in _REDIRECTS:
            out.append(f"{n.lineno}: {n.value!r} as a string")
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                out += [f"{n.lineno}: sys.{e.attr} =" for e in _stream_targets(t, sys_names)]
        elif _setattr_stream(n, sys_names):
            out.append(f"{n.lineno}: setattr(sys, {ast.unparse(n.args[1])}, …)")
    return out


def _library_files() -> list[str]:
    ls = subprocess.run(["git", "ls-files", "dexengine"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [f for f in ls.stdout.split() if f.endswith(".py") and f != OWNER and (ROOT / f).is_file()]


def test_no_redirection_outside_the_owner():
    found = {f: v for f in _library_files() if (v := violations((ROOT / f).read_text(encoding="utf-8")))}
    assert not found, (
        "stdout/stderr redirection outside dexengine/core/stdio.py — use stdio.capture (per thread) or "
        f"stdio.process_stream (process level): {found}"
    )


@pytest.mark.parametrize(
    "src",
    [
        "import contextlib\nwith contextlib.redirect_stdout(None):\n    pass\n",
        "import contextlib as _cl\nwith _cl.redirect_stderr(None):\n    pass\n",
        "from contextlib import redirect_stdout as r\n",
        "import contextlib\nf = getattr(contextlib, 'redirect_stdout')\n",
        "import sys\nsys.stdout = None\n",
        "import sys\nsys.stdout, sys.stderr = 1, 2\n",
        "import sys as _s\n_s.stderr = None\n",
        "import sys\nsetattr(sys, 'stdout', None)\n",
        "import sys\nname = 'stderr'\nsetattr(sys, name, None)\n",
    ],
)
def test_the_guard_catches_every_form(src):
    """Negative plants: every form of a redirection is caught."""
    assert violations(src)


def test_the_guard_passes_reads_and_docstrings():
    src = '"""contextlib.redirect_stdout in prose"""\nimport sys\nprint(1, file=sys.stderr)\nx = sys.stdout\n'
    assert violations(src) == []


def test_the_owner_itself_is_the_only_exemption():
    assert violations((ROOT / OWNER).read_text(encoding="utf-8"))  # it does redirect …
    assert OWNER not in _library_files()  # … and is the one file not scanned


def test_capture_is_per_thread_and_nests(capsys):
    """A thread's capture takes only that thread's prints; another thread's prints still reach the stream."""
    inside, release = threading.Event(), threading.Event()
    got = {}

    def quiet_thread():
        with stdio.capture() as outer:
            print("outer")
            with stdio.capture() as inner:
                print("inner")
            inside.set()
            release.wait(5)
        got.update(outer=outer.getvalue(), inner=inner.getvalue())

    t = threading.Thread(target=quiet_thread)
    t.start()
    inside.wait(5)
    print("shown")
    release.set()
    t.join()
    out = capsys.readouterr().out
    assert "shown" in out and "outer" not in out and "inner" not in out
    assert got == {"outer": "outer\n", "inner": "inner\n"}


def test_capture_takes_stderr_and_tracebacks():
    import traceback

    with stdio.capture(stderr=True) as buf:
        print("out")
        try:
            raise ValueError("boom")
        except ValueError:
            traceback.print_exc()
    assert "out" in buf.getvalue() and "ValueError: boom" in buf.getvalue()


def test_capture_onto_stderr_and_onto_itself_does_not_loop():
    """capture(sys.stderr) (stdout as diagnostics) and capture(sys.stdout) resolve to where that stream goes now."""
    err = io.StringIO()
    with stdio.capture(err, stderr=err):
        with stdio.capture(sys.stderr):
            print("to stderr")
        with stdio.capture(sys.stdout):
            print("to stdout")
    assert err.getvalue() == "to stderr\nto stdout\n"


def test_process_stream_sets_and_returns_the_old():
    rec = io.StringIO()
    old = stdio.process_stream("stdout", rec)
    try:
        print("x")
    finally:
        assert stdio.process_stream("stdout", old) is rec
    assert rec.getvalue() == "x\n"
    with pytest.raises(ValueError):
        stdio.process_stream("stdin", rec)


def test_gpu_engine_job_buffer_gets_prints_and_tracebacks():
    """The f0 GPU engine classifies a clip on its job buffer (shadow_tree.finish): a step's prints and its traceback
    must land there, and rc must follow the process convention."""
    G = own_module("dexengine.retarget.f0.gpu_engine")

    def gen():
        print("checked")
        yield "request"
        raise RuntimeError("failed late")

    job = G._Job()
    job.gen, job.buf, job.rc, job.done = gen(), io.StringIO(), None, False
    assert G._step(job, None, first=True) == "request"
    assert G._step(job, ("x", 0.0)) is None
    out = job.buf.getvalue()
    assert "checked" in out and "RuntimeError: failed late" in out and job.rc == 1 and job.done
