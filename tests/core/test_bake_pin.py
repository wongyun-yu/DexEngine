"""The bake pin profile (dexengine.core.blas_pin, profile "bake"): the cross repair runs on the same numbers on every
host — OpenBLAS kernel, numpy's AVX512 dispatch and the transcendental functions it calls.

Rules (structure — breaking one needs a code change, and each has a planted violation):
  1. Every process that runs the cross repair applies the bake profile before its work: the bake worker
     (bake/server.py, which runs chain.STAGES["xr"] in-process) and the cross repair's own __main__.
  2. The cross repair's import closure calls no transcendental function outside blas_pin.BAKE_MATH (numpy · math ·
     a non-integer ``**``): a new function must join the gated set, whose digest then proves it on every host.
  3. The gate dies when a gated function reads other values (digest ≠ the constant file) and when numpy's AVX512
     dispatch is still on.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys

import pytest

from dexengine.core import blas_pin
from dexengine.core.repo import REPO_ROOT as REPO

#: transcendental names: numpy ufunc (and math function) name -> gated name in BAKE_MATH
_NP_T = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "arctan2": "arctan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "arcsinh": "arcsinh",
    "arccosh": "arccosh",
    "arctanh": "arctanh",
    "exp": "exp",
    "exp2": "exp2",
    "expm1": "expm1",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "log1p": "log1p",
    "cbrt": "cbrt",
    "power": "power",
    "float_power": "power",
    "hypot": "hypot",
    "logaddexp": "logaddexp",
    "logaddexp2": "logaddexp2",
}
_MATH_T = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "cbrt",
    "pow",
    "hypot",
    "erf",
    "erfc",
    "gamma",
    "lgamma",
}


def _entry_files() -> dict[str, str]:
    from dexengine.placement.bake import chain

    xr = chain.STAGES["xr"]
    return {
        "dexengine/placement/bake/server.py": "server",
        str((REPO / (xr.replace(".", "/") + ".py")).relative_to(REPO)): "xr",
    }


def bake_violation(text: str) -> str | None:
    """None when the __main__ block applies the bake profile before any other work (a leading ``send`` branch that
    only returns the client's rc is allowed — the client does no numerical work)."""
    tree = ast.parse(text)
    mains = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"]
    if len(mains) != 1:
        return f"{len(mains)} __main__ blocks"
    for st in mains[0].body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(st, ast.If) and "client_main" in ast.unparse(st) and "main()" not in ast.unparse(st):
            continue
        call = st.value if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) else None
        if call is not None and ast.unparse(call.func) == "blas_pin.apply":
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            return None if kw.get("profile") == "'bake'" else f"blas_pin.apply without profile='bake': {kw}"
        return f"first statement is not blas_pin.apply(..., profile='bake'): {ast.unparse(st)[:80]}"
    return "no blas_pin.apply in the __main__ block"


def test_every_xr_process_applies_the_bake_pin_first():
    es = _entry_files()
    assert len(es) == 2, es
    bad = {e: v for e in es if (v := bake_violation((REPO / e).read_text()))}
    assert not bad, bad


def test_planted_entry_violations_are_caught():
    text = (REPO / "dexengine/placement/bake/server.py").read_text()
    assert bake_violation(text) is None
    gone = "\n".join(ln for ln in text.splitlines() if "blas_pin.apply(" not in ln)
    assert bake_violation(gone) is not None
    stage1 = text.replace('profile="bake"', 'profile="stage1"')
    assert bake_violation(stage1) is not None
    late = text.replace("    sys.exit(main())\n", "", 1).replace(
        '    blas_pin.apply("bake/server", profile="bake")',
        '    sys.exit(main())\n    blas_pin.apply("bake/server", profile="bake")',
    )
    assert bake_violation(late) is not None


def _closure(root_mod: str) -> list[str]:
    """Static import closure inside the repository (imports anywhere in a file, function bodies included)."""

    def mfile(m):
        p = REPO / m.replace(".", "/")
        for c in (p.with_suffix(".py"), p / "__init__.py"):
            if c.is_file():
                return c
        return None

    seen, todo = {}, [root_mod]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen[m] = f = mfile(m)
        if f is None:
            continue
        for n in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            todo += [x for x in names if x.startswith("dexengine") and mfile(x) is not None]
    return sorted(str(f.relative_to(REPO)) for f in seen.values() if f is not None)


def transcendental_calls(text: str) -> set[str]:
    """Gated names of the transcendental functions a module calls: ``np.<f>`` · ``numpy.<f>`` · ``math.<f>`` (→
    "math") · names imported from numpy or math · ``**`` with a non-integer exponent (→ "power")."""
    out = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            if n.value.id in ("np", "numpy") and n.attr in _NP_T:
                out.add(_NP_T[n.attr])
            elif n.value.id == "math" and n.attr in _MATH_T:
                out.add("math")
        elif isinstance(n, ast.ImportFrom) and n.module in ("numpy", "math"):
            for a in n.names:
                if n.module == "numpy" and a.name in _NP_T:
                    out.add(_NP_T[a.name])
                elif n.module == "math" and a.name in _MATH_T:
                    out.add("math")
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            r = n.right
            if not (isinstance(r, ast.Constant) and isinstance(r.value, int)):
                out.add("power")
    return out


def test_xr_closure_calls_only_gated_transcendental_functions():
    from dexengine.placement.bake import chain

    # blas_pin itself is the gate: its probe evaluates every function (math_digests) and is not the cross repair's work
    files = [f for f in _closure(chain.STAGES["xr"]) if f != "dexengine/core/blas_pin.py"]
    assert len(files) > 100, len(files)  # the closure was really walked
    used = {}
    for f in files:
        for name in transcendental_calls((REPO / f).read_text()):
            used.setdefault(name, []).append(f)
    extra = {k: v for k, v in used.items() if k not in blas_pin.BAKE_MATH}
    assert not extra, f"transcendental functions outside blas_pin.BAKE_MATH: {extra}"
    assert {"sin", "cos"} <= set(used)  # the scan sees what is there


@pytest.mark.parametrize(
    ("src", "want"),
    [
        ("import numpy as np\nnp.arctan(x)\n", {"arctan"}),
        ("import math\nmath.log1p(x)\n", {"math"}),
        ("from numpy import expm1\n", {"expm1"}),
        ("y = x ** 1.5\n", {"power"}),
        ("y = x ** 2\nz = np.sqrt(x)\n", set()),
    ],
)
def test_planted_transcendental_calls_are_seen(src, want):
    assert transcendental_calls(src) == want


def test_math_check_dies_on_another_digest():
    import json

    with open(blas_pin.MATH_REF) as f:
        ref = json.load(f)["profiles"]["bake"]
    # the positive half holds only where this test process reads the canonical values (no AVX512 dispatch here — the
    # pytest process is not pinned; the pinned process is test_bake_apply_pins_under_another_forced_kernel)
    if not blas_pin._avx512_dispatch_on():
        assert blas_pin.math_check("t", "bake", ref).startswith("math ")
    bad = {**ref, "digests": {**ref["digests"], "arccos": "0" * 16}}
    with pytest.raises(SystemExit, match="arccos"):
        blas_pin.math_check("t", "bake", bad)


def test_math_ref_names_its_machine_and_covers_the_gated_functions():
    import json

    with open(blas_pin.MATH_REF) as f:
        ref = json.load(f)["profiles"]["bake"]
    assert set(blas_pin.BAKE_MATH) <= set(ref["digests"])
    assert {"host", "cpu", "glibc", "numpy", "python"} <= set(ref["made_on"])


def _run(code: str, env_over: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("BLAS_PIN_REEXEC", "NPY_DISABLE_CPU_FEATURES")}
    env.update(PYTHONPATH=str(REPO), **env_over)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)


def _isaac_numpy() -> bool:
    import numpy

    return numpy.__version__ == "1.26.4"


@pytest.mark.skipif(not _isaac_numpy(), reason="the bake profile accepts the Isaac python's numpy 1.26.4 only")
def test_bake_apply_pins_under_another_forced_kernel():
    r = _run(
        "import numpy\nfrom dexengine.core import blas_pin\nblas_pin.apply('t', profile='bake')",
        {"OPENBLAS_CORETYPE": "Haswell"},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("[BLAS_PIN] t loaded "))
    assert "=Prescott/0.3.23.dev" in line and "numpy AVX512 dispatch off" in line and "math 11 = canonical" in line


@pytest.mark.skipif(not _isaac_numpy(), reason="the bake profile accepts the Isaac python's numpy 1.26.4 only")
def test_bake_check_dies_on_another_kernel():
    """Negative: numpy loaded under Haswell without apply → check must not pass."""
    r = _run(
        "import numpy\nfrom dexengine.core import blas_pin\nblas_pin.check('t', 'bake')",
        {"OPENBLAS_CORETYPE": "Haswell"},
    )
    assert r.returncode != 0 and "✗" in (r.stdout + r.stderr)


def test_dispatch_check_dies_when_avx512_reads_on(monkeypatch):
    """Negative for the dispatch proof (this CPU may have no AVX512): a process whose numpy reports an AVX512
    dispatch target on must not pass."""
    monkeypatch.setattr(blas_pin, "_avx512_dispatch_on", lambda: ["AVX512_SKX"])
    monkeypatch.setattr(blas_pin, "loaded", lambda: [("libopenblas64_p-r0-x.3.23.dev.so", "Prescott", "0.3.23.dev")])
    with pytest.raises(SystemExit, match="AVX512 dispatch still on"):
        blas_pin.check("t", "bake")


def test_pin_env_bake_sets_both_pins():
    e = blas_pin.pin_env({"OPENBLAS_CORETYPE": "SkylakeX"}, "bake")
    assert e["OPENBLAS_CORETYPE"] == "Prescott" and "AVX512_SKX" in e["NPY_DISABLE_CPU_FEATURES"]
    s1 = blas_pin.pin_env({"NPY_DISABLE_CPU_FEATURES": e["NPY_DISABLE_CPU_FEATURES"]}, "stage1")
    assert s1["NPY_DISABLE_CPU_FEATURES"].split()[-1] == "AVX512_SPR"  # each profile writes its own numpy's list


#: libm symbol -> gated name (sqrt is correctly rounded by IEEE 754 — not gated)
_LIBM = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "atan2": "arctan2",
    "exp": "exp",
    "expm1": "expm1",
    "log": "log",
    "log1p": "log1p",
    "pow": "power",
    "hypot": "hypot",
    "cbrt": "cbrt",
    "sincos": "sincos",
    "exp2": "exp2",
    "log2": "log2",
    "log10": "log10",
}


def libm_imports(so: str) -> set[str]:
    """Gated names of the libm functions a shared library imports (``nm -D --undefined-only``)."""
    out = subprocess.run(["nm", "-D", "--undefined-only", so], capture_output=True, text=True, check=True).stdout
    names = {ln.split()[-1].split("@")[0] for ln in out.splitlines() if ln.strip()}
    return {_LIBM[n] for n in names if n in _LIBM}


@pytest.mark.skipif(not _isaac_numpy(), reason="the bake process's mujoco (Isaac python)")
def test_mujoco_libm_imports_are_gated():
    """libmujoco's kinematics and distances run inside the cross repair; every libm function it imports is gated."""
    import glob
    import shutil

    import mujoco

    if shutil.which("nm") is None:
        pytest.skip("nm not available")
    sos = glob.glob(os.path.join(os.path.dirname(mujoco.__file__), "libmujoco.so*"))
    assert sos, "libmujoco not found"
    got = libm_imports(sos[0])
    assert {"sin", "cos"} <= got  # the reader sees what is there
    assert got <= set(blas_pin.BAKE_MATH), f"libmujoco imports ungated libm functions: {got - set(blas_pin.BAKE_MATH)}"


#: pinocchio functions that evaluate sin/cos of joint values (libm sincos — glibc 2.35 ≠ 2.39, not gateable by digest)
_PIN_KIN = {
    "forwardKinematics",
    "framesForwardKinematics",
    "computeJointJacobians",
    "computeFrameJacobian",
    "computeJointJacobian",
    "log",
    "log3",
    "log6",
    "exp",
    "exp3",
    "exp6",
    "integrate",
    "difference",
    "rnea",
    "aba",
    "crba",
}


def pin_kin_calls(text: str) -> list[tuple[str, str, str]]:
    """(enclosing function, pinocchio function, ast of the q argument) for every ``pin.<f>(...)`` / ``self.pin.<f>``
    kinematics call in a module."""
    out = []

    def visit(node, fn):
        for ch in ast.iter_child_nodes(node):
            f = ch.name if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(ch, ast.Call) and isinstance(ch.func, ast.Attribute) and ch.func.attr in _PIN_KIN:
                base = ast.unparse(ch.func.value)
                if base in ("pin", "pinocchio") or base.endswith(".pin"):  # pin · self.pin · s.pin …
                    q = ast.unparse(ch.args[2]) if len(ch.args) > 2 else ""
                    out.append((f, ch.func.attr, q))
            visit(ch, f)

    visit(ast.parse(text), "<module>")
    return out


def test_xr_closure_evaluates_pinocchio_kinematics_only_at_the_neutral_pose():
    """sincos is not gated (glibc 2.35 ≠ 2.39), so the cross repair may call pinocchio kinematics only where every
    angle is exactly 0: build_arm_ctx's neutral-pose FK. arm_ik's IK functions are in the closure (module imported
    for resolve_combo) but no other closure file may name them."""
    from dexengine.placement.bake import chain

    files = _closure(chain.STAGES["xr"])
    calls = {f: c for f in files if (c := pin_kin_calls((REPO / f).read_text()))}
    ik_defs = {"solve_ik_pose", "solve_ik_multistart"}
    bad = {}
    for f, cs in calls.items():
        for fn, name, q in cs:
            if f == "dexengine/placement/collision/core.py" and fn == "build_arm_ctx" and q == "q0":
                continue
            if f == "dexengine/placement/arm_ik.py" and fn in ik_defs:
                continue
            bad.setdefault(f, []).append((fn, name, q))
    assert not bad, f"pinocchio kinematics outside the neutral pose in the cross repair's closure: {bad}"
    core = (REPO / "dexengine/placement/collision/core.py").read_text()
    assert "q0 = pin.neutral(model)" in core  # the q0 the allowed call reads is the neutral pose
    users = [
        f for f in files if f != "dexengine/placement/arm_ik.py" and any(n in (REPO / f).read_text() for n in ik_defs)
    ]
    assert not users, f"arm_ik's pinocchio IK is reachable from the cross repair: {users}"


def test_planted_pinocchio_kinematics_is_seen():
    src = "import pinocchio as pin\ndef step(model, data, q):\n    pin.forwardKinematics(model, data, q)\n"
    assert pin_kin_calls(src) == [("step", "forwardKinematics", "q")]
    assert pin_kin_calls("def f(s, q):\n    s.pin.framesForwardKinematics(s.model, s.data, q)\n") == [
        ("f", "framesForwardKinematics", "q")
    ]
    assert pin_kin_calls("class A:\n    def f(self, q):\n        self.pin.log(x)\n") == [("f", "log", "")]
