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
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys

KERNEL = "Haswell"
# Loaded OpenBLAS (file-name prefix) -> version: the retargeting venv (numpy 2.3.5 · scipy 1.16.3). Unknown dies.
VERSIONS = {"libscipy_openblas64_": "0.3.30", "libscipy_openblas-": "0.3.29.dev"}
_REEXEC = "BLAS_PIN_REEXEC"
#: CPU features (/proc/cpuinfo flags) a kernel needs. OpenBLAS does not refuse a forced kernel the CPU lacks: it runs
#: instructions the CPU does not have (measured: SkylakeX forced on a CPU without AVX-512 dies by SIGILL).
NEEDS = {"Haswell": ("avx2", "fma"), "SkylakeX": ("avx512f",)}
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
    raise SystemExit(f"[BLAS_PIN] ✗ no CPU flags in {path} — cannot prove this CPU runs the {KERNEL} kernel")


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


def pin_env(env: dict) -> dict:
    """Pin the kernel in a child environment (another value is overridden; the child's apply proves what loaded)."""
    env["OPENBLAS_CORETYPE"] = KERNEL
    return env


def apply(tag: str) -> str:
    """First statement of a stage-1 entry: run on the pinned kernel (re-executing if needed) and prove what loaded.
    Returns the marker line."""
    check_cpu(tag)
    have = os.environ.get("OPENBLAS_CORETYPE")
    if have not in (None, "", KERNEL):
        print(f"[BLAS_PIN] {tag}: env OPENBLAS_CORETYPE={have} → {KERNEL} (owner blas_pin)", flush=True)
    pin_env(os.environ)
    if any(core != KERNEL for _, core, _ in loaded()):
        if os.environ.get(_REEXEC) == "1":
            raise SystemExit(f"[BLAS_PIN] ✗ {tag}: still not {KERNEL} after re-executing: {loaded()}")
        os.environ[_REEXEC] = "1"
        print(f"[BLAS_PIN] {tag}: another kernel is loaded {loaded()} → re-executing with {KERNEL}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], dict(os.environ))
    os.environ.pop(_REEXEC, None)
    import numpy  # noqa: F401  (load numpy's OpenBLAS now, to prove it)

    with contextlib.suppress(ImportError):
        import scipy.linalg  # noqa: F401  (and scipy's)
    return check(tag)


def check(tag: str) -> str:
    """Read the kernel and version of every loaded OpenBLAS and die unless they are the pinned ones. Prints and
    returns the marker line."""
    libs = loaded()
    if not libs:
        raise SystemExit(f"[BLAS_PIN] ✗ {tag}: no OpenBLAS loaded — the kernel cannot be proven")
    for name, core, ver in libs:
        want = next((v for k, v in VERSIONS.items() if name.startswith(k)), None)
        if core != KERNEL or ver != want:
            raise SystemExit(
                f"[BLAS_PIN] ✗ {tag}: {name} kernel {core} · version {ver} — pinned {KERNEL} · version {want} "
                "(owner blas_pin · supported CPUs = x86-64 with AVX2 · the retargeting venv)"
            )
    libs_s = " ".join(f"{n.split('-')[0]}={c}/{v}" for n, c, v in libs)
    line = f"[BLAS_PIN] {tag} loaded {libs_s} (owner blas_pin)"
    print(line, flush=True)
    return line


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
