"""The stage-1 OpenBLAS pin (dexengine.core.blas_pin): every stage-1 entry applies it first, and it holds.

Entries are read from code, not listed by hand: the stage table's commands (STAGES Run argv and reuse), the table's
drivers (dexengine/data/stages.py and the dexengine/data/stage1 modules with a __main__ block), the retargeting entry
named in the table's first row, every ``-m dexengine.…`` child the retargeting drivers start, and the f0 GPU worker.
Rule (structure): in each entry's ``if __name__ == "__main__":`` block, ``blas_pin.apply(...)`` is the first statement
other than imports. Planted violations (the call removed, a statement put before it) must be caught.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest

from dexengine.core import blas_pin
from dexengine.core.repo import REPO_ROOT as REPO

_RETARGET_DRIVERS = (
    "dexengine/retarget/run.py",
    "dexengine/retarget/batch.py",
    "dexengine/retarget/reretarget.py",
    "dexengine/retarget/admit.py",
    *sorted(str(p.relative_to(REPO)) for p in (REPO / "dexengine/retarget/phases").glob("*.py")),
)


def _module_file(mod: str) -> str:
    base = REPO / mod.replace(".", "/")
    return str((base / "__main__.py" if base.is_dir() else base.with_suffix(".py")).relative_to(REPO))


def entries() -> list[str]:
    from dexengine.data.stage1 import table as T

    files = set()
    for row in T.STAGES:
        cmd = row[3]
        if isinstance(cmd, T.Run):
            files |= {a for a in (cmd.argv[:1] + cmd.reuse[:1]) if a.endswith(".py")}
        elif row[0].startswith("1 "):
            files |= set(re.findall(r"((?:scripts|dexengine)/[\w/]+\.py)", cmd))
    files.add("dexengine/data/stages.py")
    files |= {
        str(p.relative_to(REPO))
        for p in (REPO / "dexengine/data/stage1").glob("*.py")
        if 'if __name__ == "__main__":' in p.read_text()
    }
    for drv in _RETARGET_DRIVERS:
        src = (REPO / drv).read_text()
        if 'if __name__ == "__main__":' in src:
            files.add(drv)
        consts = dict(re.findall(r'^(_[A-Z_]+) = "(dexengine\.[\w.]+)"', src, re.M))
        for m in re.findall(r'"-m",\s*("dexengine\.[\w.]+"|_[A-Z_]+)', src):
            files.add(_module_file(consts.get(m, m.strip('"'))))
    if "dexengine/retarget/f0/gpu.py" in (REPO / "dexengine/retarget/f0/shadow_gpu.py").read_text():
        files.add("dexengine/retarget/f0/gpu.py")
    return sorted(files)


def violation(text: str) -> str | None:
    tree = ast.parse(text)
    mains = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    if len(mains) != 1:
        return f"{len(mains)} __main__ blocks"
    for st in mains[0].body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            continue
        ok = (
            isinstance(st, ast.Expr)
            and isinstance(st.value, ast.Call)
            and ast.unparse(st.value.func) == "blas_pin.apply"
        )
        return None if ok else f"first statement is not blas_pin.apply: {ast.unparse(st)[:80]}"
    return "no blas_pin.apply in the __main__ block"


def test_every_stage1_entry_applies_the_pin_first():
    es = entries()
    assert len(es) >= 20, es
    bad = {e: v for e in es if (v := violation((REPO / e).read_text()))}
    assert not bad, bad


def test_planted_violations_are_caught():
    text = (REPO / "dexengine/retarget/f0/prejudge.py").read_text()
    assert violation(text) is None
    assert violation(re.sub(r"\n\s*blas_pin\.apply\([^)]*\)\n", "\n", text)) is not None
    moved = text.replace('if __name__ == "__main__":\n', 'if __name__ == "__main__":\n    print("x")\n', 1)
    assert violation(moved) is not None


def _run(code: str, kernel: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "BLAS_PIN_REEXEC"}
    env.update(OPENBLAS_CORETYPE=kernel, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)


def test_apply_loads_the_pinned_kernel_under_another_forced_kernel():
    r = _run("from dexengine.core import blas_pin; blas_pin.apply('t')", "Prescott")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "=" + blas_pin.KERNEL + "/" in r.stdout and "[BLAS_PIN] t loaded" in r.stdout


def test_check_dies_on_another_loaded_kernel():
    """Negative: a process whose OpenBLAS loaded another kernel ('Katmai' is the name of the forced Prescott table in
    this build) must not pass the check."""
    r = _run("import numpy; from dexengine.core import blas_pin; blas_pin.check('t')", "Prescott")
    assert r.returncode != 0 and "✗" in (r.stdout + r.stderr)


@pytest.mark.parametrize("env", [{}, {"OPENBLAS_CORETYPE": "SkylakeX"}])
def test_pin_env_sets_the_kernel(env):
    assert blas_pin.pin_env(dict(env))["OPENBLAS_CORETYPE"] == blas_pin.KERNEL


def test_compute_env_names_host_packages_and_loaded_kernel():
    import numpy  # noqa: F401

    env = blas_pin.compute_env()
    assert env["host"] and env["python"] and env["cpu"]
    assert set(env["packages"]) == set(blas_pin.ENV_PACKAGES) and env["packages"]["numpy"] == numpy.__version__
    assert env["blas"] and all(f"={blas_pin.KERNEL}/" in b for b in env["blas"])


def test_f0_marker_records_the_env_outside_every_key(tmp_path, monkeypatch):
    """The shadow marker carries compute_env; the identity key (f0_key.KEY_FIELDS) and the input key ignore it."""
    import json
    import types

    from dexengine.placement.judge import f0_key
    from dexengine.retarget.f0 import shadow_tree

    monkeypatch.setenv("DEXENGINE_SCENE", "table")
    a = types.SimpleNamespace(hand="wuji2", dataset="hot3d", emb="bimanual", src_layer="base")
    d = tmp_path / "clip"
    d.mkdir()
    shadow_tree._write_marker(
        3,
        a,
        0,
        "clip",
        d,
        "right=1/2",
        {"input_parts": {}},
        "k",
        "prepend",
        tmp_path,
        "REPAIRED",
        {"right": tmp_path / "w" / "wrist_poses_right"},
    )
    mk = d / f0_key.MARKER
    m = json.loads(mk.read_text())
    assert m["compute_env"]["blas"] and m["input_key"] == "k"
    key = f0_key.key_of(mk)
    m["compute_env"] = {"host": "elsewhere"}
    mk.write_text(json.dumps(m))
    assert f0_key.key_of(mk) == key


def test_haswell_needs_avx2_and_fma(tmp_path):
    with pytest.raises(SystemExit, match="needs AVX2 · FMA — this CPU lacks avx2"):
        blas_pin.check_cpu("t", "Haswell", {"sse4_2", "fma"})
    with pytest.raises(SystemExit, match="lacks fma"):
        blas_pin.check_cpu("t", "Haswell", {"avx2"})
    blas_pin.check_cpu("t", "Haswell", {"avx2", "fma", "sse4_2"})
    with pytest.raises(SystemExit, match="no CPU requirement"):
        blas_pin.check_cpu("t", "Prescott", {"avx2", "fma"})
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nflags\t\t: fpu sse2 avx2 fma\n")
    assert {"avx2", "fma"} <= blas_pin.cpu_flags(str(info))
    info.write_text("processor\t: 0\n")
    with pytest.raises(SystemExit, match="no CPU flags"):
        blas_pin.cpu_flags(str(info))


def test_apply_dies_before_pinning_on_a_cpu_without_avx2(monkeypatch):
    monkeypatch.setattr(blas_pin, "cpu_flags", lambda path=blas_pin.CPUINFO: {"sse2"})
    monkeypatch.setattr(blas_pin, "pin_env", lambda env: pytest.fail("pinned on an unsupported CPU"))
    with pytest.raises(SystemExit, match="needs AVX2"):
        blas_pin.apply("t")


def test_this_cpu_runs_the_pinned_kernel():
    blas_pin.check_cpu("t")
