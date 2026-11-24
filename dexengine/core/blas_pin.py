"""The stage-1 OpenBLAS kernel pin: the one owner of its value, its application and the proof of the kernel that
actually loaded. Standard library only, so an entry can import it before numpy.

Why: OpenBLAS DYNAMIC_ARCH picks a kernel per CPU, and stage-1 outputs differ in their last digits by kernel
(measured: the f0 unit tests give the golden bytes under Haswell and Zen and other bytes under Prescott and
SkylakeX; one kernel gives the same bytes on Arrow Lake, Xeon 4210 and EPYC 7313 at any thread count; the v0_equiv
harness under a forced Prescott kernel splits the IK rollout, collision and frame0 in the last digits). The canonical
stage 1 ran on Haswell.

Applied as the first statement of the ``if __name__ == "__main__":`` block of every stage-1 entry,
``blas_pin.apply("<name>")``:
  1. sets OPENBLAS_CORETYPE to the pinned kernel in the environment (children inherit it; another value is printed
     and overridden);
  2. if another kernel is already loaded in this process, re-executes the same command line with the pinned
     environment (OpenBLAS reads the environment when it loads); still another kernel after that dies;
  3. loads numpy (and scipy.linalg) and reads back the kernel and version that actually loaded; anything but the
     pinned ones dies (printing the environment is no proof). Marker ``[BLAS_PIN] <name> loaded …`` — a run without
     it is invalid.
Supported CPUs: x86-64 with AVX2 and FMA (the Haswell kernel; checked from /proc/cpuinfo before pinning, so a CPU
without them dies with a message instead of a SIGILL) and the retargeting venv's OpenBLAS. Anything else dies instead
of drifting. Entry list and checks: tests/core/test_blas_pin.py (every stage-1 entry calls it first; planted
violations are caught).

Profiles (PROFILES — one owner per process family; ``apply(tag)`` = the stage-1 profile above):
  stage1  the retargeting venv (numpy 2.3.5 · scipy 1.16.3): kernel Haswell, and numpy's AVX512 dispatch off (the
          9 AVX512 dispatch targets of numpy 2.3.5; proof as in bake 2 below). Measured before the pin: f0 on an
          AVX512 Xeon, dispatch on vs off, other bytes in 1 of 40 REPAIRED trees, A/A 0. The canonical stage 1 ran on a
          CPU without AVX512 (dispatch off by itself), so on such a CPU nothing changes. A numpy 1.26 child that
          inherits the variable warns (ImportWarning, hidden by default) that it has no AVX512_SPR; the bake profile
          writes its own list and proves it.
  bake    the stage-2 bake worker and the cross repair (Isaac python: numpy 1.26.4 · OpenBLAS 0.3.23.dev, scipy 1.17.1 ·
          scipy-openblas 0.3.30), applied by ``apply(tag, profile="bake")`` at dexengine/placement/bake/server.py and
          dexengine/placement/repair/cross_repair.py. Three pins, each proven after numpy loads:
          1. kernel Prescott — the kernel the canonical bytes were made with (OpenBLAS 0.3.23.dev does not know
             Arrow Lake and picks Prescott); one variable pins both libraries, scipy-openblas 0.3.30 then reads back
             'Katmai' (its name for that table);
          2. numpy's AVX512 dispatch off (NPY_DISABLE_CPU_FEATURES = the AVX512 dispatch targets of that numpy): on an
             AVX512 CPU the float64 transcendental ufuncs otherwise take SVML loops with other last bits; proof = every
             AVX512 target in numpy's __cpu_dispatch__ reads False in __cpu_features__;
          3. the transcendental functions the process uses read the canonical digests (math_digests ↔ the constant
             file math_digest_ref.json, which also names the machine · glibc · numpy that made them). With AVX512 off
             these are libm; glibc 2.35 and 2.39 differ in arctan · expm1 · log1p · sincos (measured) but not in
             the gated functions. A process that would read other values dies here (that is the point — never drifts).
          Which functions are gated is BAKE_MATH; tests/core/test_bake_pin.py fails when the cross repair's import
          closure calls a transcendental function outside it (new function → add it here → the digest proves it on
          every host).
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys

#: numpy 1.26.4's AVX512 dispatch targets (``__cpu_dispatch__``; only dispatch targets can be switched off).
_NPY126_AVX512 = "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
#: numpy 2.3.5's (the retargeting venv): numpy 1.26.4's and AVX512_SPR.
_NPY235_AVX512 = _NPY126_AVX512 + " AVX512_SPR"
#: transcendental functions the bake profile proves by digest: what the cross repair's import closure calls (numpy ·
#: math — one digest "math" · ``**`` with a non-integer exponent = libm pow "power") and the libm functions libmujoco
#: imports (its kinematics and distances run in the cross repair: sin cos acos asin atan2 exp log pow tan; sqrt is
#: correctly rounded everywhere). Not gated: sincos — glibc 2.35 ≠ 2.39 — which only pinocchio imports; the cross
#: repair calls pinocchio for the URDF parse and the neutral-pose FK alone (sincos(0) is exact; DetFK for other q).
BAKE_MATH = ("sin", "cos", "tan", "arcsin", "arccos", "arctan2", "hypot", "exp", "log", "power", "math")
#: profile -> kernel · accepted OpenBLAS builds (file-name prefix -> (version, kernel read back)) · numpy dispatch
#: targets switched off · gated transcendental functions. Unknown builds die.
PROFILES = {
    "stage1": {
        "kernel": "Haswell",
        "libs": {"libscipy_openblas64_": ("0.3.30", "Haswell"), "libscipy_openblas-": ("0.3.29.dev", "Haswell")},
        "npy_off": _NPY235_AVX512,
        "math": (),
    },
    "bake": {
        "kernel": "Prescott",
        "libs": {"libopenblas64_p": ("0.3.23.dev", "Prescott"), "libscipy_openblas-": ("0.3.30", "Katmai")},
        "npy_off": _NPY126_AVX512,
        "math": BAKE_MATH,
    },
}
KERNEL = PROFILES["stage1"]["kernel"]
# Loaded OpenBLAS (file-name prefix) -> version: the retargeting venv (numpy 2.3.5 · scipy 1.16.3). Unknown dies.
VERSIONS = {k: v for k, (v, _) in PROFILES["stage1"]["libs"].items()}
_REEXEC = "BLAS_PIN_REEXEC"
MATH_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "math_digest_ref.json")
#: CPU features (/proc/cpuinfo flags) a kernel needs. OpenBLAS does not refuse a forced kernel the CPU lacks: it runs
#: instructions the CPU does not have (measured: SkylakeX forced on a CPU without AVX-512 dies by SIGILL).
#: Prescott (the bake profile) needs SSE3, which /proc/cpuinfo names "pni".
NEEDS = {"Haswell": ("avx2", "fma"), "SkylakeX": ("avx512f",), "Prescott": ("pni",)}
CPUINFO = "/proc/cpuinfo"


def _sym(lib, base):
    for pre in ("scipy_", ""):
        for suf in ("64_", ""):
            f = getattr(lib, f"{pre}{base}{suf}", None)
            if f is not None:
                f.restype = ctypes.c_char_p
                return f
    raise SystemExit(f"[BLAS_PIN] ✗ {lib._name}: no {base} symbol — the kernel cannot be proven")


def loaded() -> list[tuple[str, str, str]]:
    """(file name, kernel, version) of every OpenBLAS loaded in this process."""
    paths = []
    with open("/proc/self/maps") as f:
        for ln in f:
            parts = ln.split(maxsplit=5)
            p = parts[5].strip() if len(parts) == 6 else ""
            b = os.path.basename(p)
            if "openblas" in b and ".so" in b and p not in paths:
                paths.append(p)
    out = []
    for p in paths:
        lib = ctypes.CDLL(p)
        core = _sym(lib, "openblas_get_corename")().decode()
        cfg = _sym(lib, "openblas_get_config")().decode().split()
        out.append((os.path.basename(p), core, cfg[1] if len(cfg) > 1 else "?"))
    return out


def cpu_flags(path: str = CPUINFO) -> set[str]:
    """The CPU feature flags of this machine (first `flags` line of /proc/cpuinfo); none found dies."""
    with open(path) as f:
        for line in f:
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    raise SystemExit(f"[BLAS_PIN] ✗ no CPU flags in {path} — cannot prove this CPU runs the pinned kernel")


def check_cpu(tag: str, kernel: str = KERNEL, flags: set[str] | None = None) -> None:
    """Die with a message (not a SIGILL) when this CPU lacks a feature the pinned kernel needs."""
    need = NEEDS.get(kernel)
    if need is None:
        raise SystemExit(f"[BLAS_PIN] ✗ {tag}: no CPU requirement is stated for the {kernel} kernel (blas_pin.NEEDS)")
    have = cpu_flags() if flags is None else flags
    if missing := [x for x in need if x not in have]:
        raise SystemExit(
            f"[BLAS_PIN] ✗ {tag}: the {kernel} kernel needs {' · '.join(x.upper() for x in need)} — "
            f"this CPU lacks {', '.join(missing)}"
        )


def pin_env(env: dict, profile: str = "stage1") -> dict:
    """Pin the profile's kernel (and numpy dispatch) in a child environment (another value is overridden; the child's
    apply proves what loaded)."""
    P = PROFILES[profile]
    env["OPENBLAS_CORETYPE"] = P["kernel"]
    if P["npy_off"]:
        env["NPY_DISABLE_CPU_FEATURES"] = P["npy_off"]
    return env


def _avx512_dispatch_on() -> list[str]:
    """The AVX512 dispatch targets numpy would use in this process (empty = off or absent on this CPU)."""
    try:
        from numpy._core import _multiarray_umath as U
    except ImportError:
        from numpy.core import _multiarray_umath as U
    return [t for t in U.__cpu_dispatch__ if t.startswith("AVX512") and U.__cpu_features__.get(t)]


def apply(tag: str, profile: str = "stage1") -> str:
    """First statement of an entry: run on the profile's pins (re-executing if needed) and prove what loaded.
    Returns the marker line."""
    P = PROFILES[profile]
    kern = P["kernel"]
    check_cpu(tag, kern)
    have = os.environ.get("OPENBLAS_CORETYPE")
    if have not in (None, "", kern):
        print(f"[BLAS_PIN] {tag}: env OPENBLAS_CORETYPE={have} → {kern} (owner blas_pin)", flush=True)
    npy = os.environ.get("NPY_DISABLE_CPU_FEATURES")
    if npy not in (None, "", P["npy_off"]):
        print(f"[BLAS_PIN] {tag}: env NPY_DISABLE_CPU_FEATURES={npy} → {P['npy_off']} (owner blas_pin)", flush=True)
    late_npy = bool(P["npy_off"]) and "numpy" in sys.modules and bool(_avx512_dispatch_on())
    pin_env(os.environ, profile)
    if late_npy or any(core != P["libs"].get(_prefix(P, name), (None, kern))[1] for name, core, _ in loaded()):
        if os.environ.get(_REEXEC) == "1":
            raise SystemExit(f"[BLAS_PIN] ✗ {tag}: still not pinned after re-executing: {loaded()}")
        os.environ[_REEXEC] = "1"
        print(f"[BLAS_PIN] {tag}: loaded before the pin {loaded()} → re-executing ({profile})", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], dict(os.environ))
    os.environ.pop(_REEXEC, None)
    import numpy  # noqa: F401  (load numpy's OpenBLAS now, to prove it)

    with contextlib.suppress(ImportError):
        import scipy.linalg  # noqa: F401  (and scipy's)
    return check(tag, profile)


def _prefix(P: dict, name: str) -> str | None:
    return next((k for k in P["libs"] if name.startswith(k)), None)


def check(tag: str, profile: str = "stage1") -> str:
    """Read the kernel and version of every loaded OpenBLAS (and, per profile, numpy's dispatch state and the
    transcendental digests) and die unless they are the pinned ones. Prints and returns the marker line."""
    P = PROFILES[profile]
    libs = loaded()
    if not libs:
        raise SystemExit(f"[BLAS_PIN] ✗ {tag}: no OpenBLAS loaded — the kernel cannot be proven")
    for name, core, ver in libs:
        want = P["libs"].get(_prefix(P, name))
        if want is None or (ver, core) != want:
            raise SystemExit(
                f"[BLAS_PIN] ✗ {tag}: {name} kernel {core} · version {ver} — profile {profile} accepts "
                f"{P['libs']} (owner blas_pin · supported CPUs = x86-64 with AVX2)"
            )
    libs_s = " ".join(f"{n.split('-')[0]}={c}/{v}" for n, c, v in libs)
    extra = ""
    if P["npy_off"]:
        on = _avx512_dispatch_on()
        if on:
            raise SystemExit(
                f"[BLAS_PIN] ✗ {tag}: numpy AVX512 dispatch still on {on} — NPY_DISABLE_CPU_FEATURES "
                "must be set before numpy loads (owner blas_pin)"
            )
        extra += " · numpy AVX512 dispatch off"
    if P["math"]:
        extra += " · " + math_check(tag, profile)
    line = f"[BLAS_PIN] {tag} loaded {libs_s}{extra} (owner blas_pin · profile {profile})"
    print(line, flush=True)
    return line


def math_digests(names=None) -> dict:
    """Digests (sha256[:16]) of the float64 transcendental functions on fixed probe inputs built with exact
    arithmetic (fmod of multiples — no libm), 2^19 points per numpy function and 2^15 for the math module."""
    import hashlib
    import math

    import numpy as np

    n = 1 << 19
    i = np.arange(n, dtype=np.float64)
    a = ((i * 0.6180339887498949) % 1.0 - 0.5) * 40.0  # angles in [-20, 20) rad
    b = ((i * 0.7548776662466927) % 1.0 - 0.5) * 6.0
    u = ((i * 0.5698402909980532) % 1.0) * 2.0 - 1.0  # [-1, 1)
    p = ((i * 0.4142135623730951) % 1.0) * 50.0 + 1e-3  # (0, 50]

    def _math():
        m = 1 << 15
        rows = [
            (math.sin(x), math.cos(x), math.acos(y), math.atan2(x, y), math.exp(y), math.hypot(x, y))
            for x, y in zip(a[:m].tolist(), u[:m].tolist(), strict=True)
        ]
        return np.array(rows)

    F = {
        "sin": lambda: np.sin(a),
        "cos": lambda: np.cos(a),
        "tan": lambda: np.tan(b),
        "arcsin": lambda: np.arcsin(u),
        "arccos": lambda: np.arccos(u),
        "arctan": lambda: np.arctan(a),
        "arctan2": lambda: np.arctan2(a, b),
        "hypot": lambda: np.hypot(a, b),
        "exp": lambda: np.exp(b),
        "expm1": lambda: np.expm1(b),
        "log": lambda: np.log(p),
        "log1p": lambda: np.log1p(p),
        "power": lambda: np.power(p, 1.37),
        "cbrt": lambda: np.cbrt(a),
        "math": _math,
    }
    return {k: hashlib.sha256(np.ascontiguousarray(F[k]()).tobytes()).hexdigest()[:16] for k in (names or F)}


def math_check(tag: str, profile: str, ref: dict | None = None) -> str:
    """Compare the profile's gated digests with the constant file (or ``ref`` — tests); die naming each function
    that reads other values on this host."""
    import json

    if ref is None:
        with open(MATH_REF) as f:
            ref = json.load(f)["profiles"][profile]
    names = PROFILES[profile]["math"]
    got = math_digests(names)
    bad = [k for k in names if got[k] != ref["digests"].get(k)]
    if bad:
        raise SystemExit(
            f"[BLAS_PIN] ✗ {tag}: transcendental functions read other values on this host: "
            + ", ".join(f"{k} {got[k]}≠{ref['digests'].get(k)}" for k in bad)
            + f" (canonical made on {ref['made_on']} · {MATH_REF})"
        )
    return f"math {len(names)} = canonical ({ref['made_on']['glibc']} · numpy {ref['made_on']['numpy']})"


def forward(result, tag: str) -> None:
    """Copy the [BLAS_PIN] lines a child (a subprocess run with captured output) printed into this log. A successful
    child without the load-proof line dies (its run did not prove its kernel)."""
    out = result.stdout
    if result.returncode != 0 or out is None:
        return
    out = out.decode(errors="replace") if isinstance(out, bytes) else out
    lines = [ln for ln in out.splitlines() if ln.startswith("[BLAS_PIN]")]
    for ln in lines:
        print(ln, flush=True)
    if not any(" loaded " in ln for ln in lines):
        raise SystemExit(f"[BLAS_PIN] ✗ {tag}: the child printed no load proof — its run is invalid")


#: packages whose versions decide f0's numbers (recorded per shadow by compute_env, never part of a key)
ENV_PACKAGES = ("numpy", "scipy", "mujoco", "warp-lang", "pin", "trimesh")


def _version(metadata, name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def compute_env() -> dict:
    """Where and with what a stage-1 output was computed: host, CPU model, python, the ENV_PACKAGES versions and the
    loaded OpenBLAS kernel/version (read back, as in check). A record — never part of a cache or identity key."""
    import platform
    import socket
    from importlib import metadata

    cpu = "?"
    with contextlib.suppress(OSError), open("/proc/cpuinfo") as f:
        cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")), "?")
    pk = {name: _version(metadata, name) for name in ENV_PACKAGES}
    return {
        "host": socket.gethostname(),
        "cpu": cpu,
        "python": platform.python_version(),
        "packages": pk,
        "blas": [f"{n.split('-')[0]}={c}/{v}" for n, c, v in loaded()],
    }
