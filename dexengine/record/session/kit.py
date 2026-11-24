"""Kit extension settings of the Isaac record session: registry downloads and folder watch off, only the pre-filled
shared extension folder.

Why: every job has an empty XDG_CACHE_HOME (one per job to avoid shader-cache races; kept), so Kit downloaded the
extension archives again at boot and hung without limit waiting on the download (two GPUs of the 8-GPU machine:
main futex, download thread do_poll). With the registry off there is no network dependency and no shared-cache race.
Value owners: the two settings = KIT_SETTINGS (the same ones the cluster jobs put into their kit copy; no copy here) ·
the extension folder = $HOME/OV_EXTS/v2 (a site's cluster jobs fill the same folder). They are passed with
AppLauncher's --kit_args (the canonical entry is unchanged; only the record session entry calls this; outside the
stage-1 and stage-2 cache fingerprints).
Markers: before the launch ``[KIT_EXT] registry=off …`` (apply) · after the app is up ``[KIT_EXT] applied {…}``
(check_applied reads the carb settings and dies if they are not off). A session without both lines is invalid.

Render-stage worker thread cap (RENDER_THREADS; stage-2 render and render_clip only; stage-1 record (physics)
untouched):
Why: Kit's default carb.tasking workers = the core count (24 and 32 on the two machines measured) and as many TBB
workers start in every session; idle, they spin in sched_yield (measured on the 64-core machine: one render session ≈
220k involuntary carb switches/s, CPU 160-270 % of which carb 80-175 %). With 2 sessions per GPU x 8 = 16 sessions, ≈
1,000 workers share 64 cores: load 280-318, step 61-69 -> 80-141 ms. Pinning one session to 4 cores at the same time
gave the same step as its neighbours (105.6 ms, 161 %): the real work fits in 4 cores. The render is a kinematic replay,
independent of the trajectory. Canary (24-core machine, 1 clip 5 episodes, the same stage-1 trajectory, two uncapped +
one capped): depth and head_view sha all equal; mp4 differs even between the two uncapped runs (RTX RGB is not
deterministic per run): mean pixel |Δ| 0.9-1.6, pixels |Δ| > 8 0.08-0.52 %, so capped vs uncapped = uncapped vs uncapped
(within the noise).
Markers: ``[KIT_THREADS] kit_args='…'`` (apply_render_threads) · after the app is up ``[KIT_THREADS] applied {…} ·
carb.tasking threads n`` (check_render_threads dies when a setting != the cap or the carb.tasking threads in
/proc/self/task exceed the cap).

Render-stage OpenBLAS pool (RENDER_BLAS_THREADS; the same render stages only; stage-1 record untouched):
Why: isaacsim's simulation_app.py overwrites OPENBLAS_NUM_THREADS with min(os.cpu_count(), limit_cpu_threads 32) when
the app starts (a user value is ignored; os.cpu_count ignores the affinity: 32 and 24 on the two machines measured). The
OpenBLAS 0.3.23 of Kit's numpy 1.26.0 (pip_prebundle numpy.libs/libopenblas64_p) runs even a 4x4 np.linalg.inv as dgesv
-> dgetrs_N_parallel -> exec_blas and wakes every helper (the render rig's camera binding, Isaac's cloner, ...); a woken
helper then spins on sched_yield for ~0.1 s (thread_timeout). On the 64-core machine with 15-18 render workers (about 10
per 32-core node): the 'python' threads took sys 83 s per 5 s and 15 M involuntary switches/s; the hot pool = 31 threads
per process (+ the caller = 32), about 1 core each, 70 % sys. OMP/MKL/OPENBLAS env caps did nothing because the app
overwrites OPENBLAS. The render stages only run small BLAS calls, so the helpers have nothing to gain.
Applied after the app is up (inside check_render_threads: apply_render_blas): every loaded OpenBLAS gets
openblas_set_num_threads(constant), a later one reads env OPENBLAS_NUM_THREADS at load. The constant is at most the
worker's affinity core count. Output-neutral on the measured clip (depth, head_view, parquet, forces, dataset depth
byte-equal; RGB inside the base-vs-base band).
Markers: ``[RENDER_THREADS] openblas=<n> affinity=<k> libs={…}`` (dies when a read-back value != n or no OpenBLAS is
loaded) · at the end of the session check_render_blas (persist_kit.finish_render) reads every loaded OpenBLAS again,
late-loaded ones included.

RTX driver check off (render stages only; env COLLECT_RTX_SKIP_DRIVER_CHECK, default off = argv unchanged):
Why: on an RTX 3090 with driver 535.104 Kit printed '[gpu.foundation.plugin] … driver is unsupported' and
'rtx driver verification failed', rebuilt the RTX viewport forever and produced 0 frames (render_clip 600 s stall).
With 1 the Kit setting /rtx/verifyDriverVersion/enabled=false is added through --kit_args; adoption is decided by a
frame comparison with the 5090 render (only that lane turns it on).
Markers: before the launch ``[RTX_DRIVER_CHECK] off host=<host> driver=<version>`` (apply_driver_check) · after the
app is up ``[RTX_DRIVER_CHECK] applied {…}`` (check_driver_check dies if it was requested and the setting is not false).
Values other than 0 and 1 die.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

from dexengine.core import capacity

MARK = "[KIT_EXT]"
SETTING_PATH = "/app/extensions/"  # carb path of the kit file's [settings.app.extensions]
TMARK = "[KIT_THREADS]"
#: render-stage worker cap: 64 cores / 16 sessions = 4 cores per session (measured work < 4 cores) · carb and
#: TBB 8 each = twice that as margin
RENDER_THREADS = 8
#: carb settings that carry the cap (path -> value): carb.tasking workers · omni.tbb global cap
RENDER_THREAD_SETTINGS = (
    ("/plugins/carb.tasking.plugin/threadCount", RENDER_THREADS),
    ("/plugins/omni.tbb.globalcontrol/maxThreadCount", RENDER_THREADS),
)
BMARK = "[RENDER_THREADS]"
#: render-stage OpenBLAS threads (the caller included) = the owner's threads per worker (capacity.THREADS = 1: no helper
#: threads — module doc 'Render-stage OpenBLAS pool'); the number lives in capacity only
RENDER_BLAS_THREADS = capacity.THREADS
#: OpenBLAS thread-count API (set, get): Kit numpy's 64-bit-int build · scipy-openblas (numpy >= 2 64-bit, Kit scipy
#: 32-bit) · the plain one (opencv)
BLAS_API = (
    ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
    ("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
    ("scipy_openblas_set_num_threads", "scipy_openblas_get_num_threads"),
    ("openblas_set_num_threads", "openblas_get_num_threads"),
)


#: Kit app settings the sessions force (extension registry and file watcher off).
KIT_SETTINGS = (("registryEnabled", "false"), ("fsWatcherEnabled", "false"))
OV_EXTS = Path(".local/share/ov/data/exts")  # under $HOME: the Kit registry extensions (v2)


def settings() -> tuple[tuple[str, str], ...]:
    return KIT_SETTINGS


def ext_folder() -> Path:
    return Path(os.environ["HOME"]) / OV_EXTS / "v2"


DMARK = "[RTX_DRIVER_CHECK]"
DRIVER_ENV = "COLLECT_RTX_SKIP_DRIVER_CHECK"
DRIVER_SETTING = "/rtx/verifyDriverVersion/enabled"


def driver_check_off() -> bool:
    """The env's off request (default off); values other than 0, 1 and empty die."""
    v = os.environ.get(DRIVER_ENV, "")
    if v not in ("", "0", "1"):
        raise SystemExit(f"{DMARK} ✗ {DRIVER_ENV}={v!r}: 0 or 1 only")
    return v == "1"


def apply_driver_check(argv: list[str]) -> list[str]:
    """With an off request, a copy of argv with --/rtx/verifyDriverVersion/enabled=false added; otherwise argv
    unchanged (contract in the module doc)."""
    if not driver_check_off():
        return argv
    import socket
    import subprocess

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
    )
    drv = (r.stdout.split() or ["?"])[0]
    print(f"{DMARK} off host={socket.gethostname()} driver={drv} kit_args='--{DRIVER_SETTING}=false'", flush=True)
    return _add_kit_args(argv, f"--{DRIVER_SETTING}=false")


def check_driver_check() -> str:
    """After the app is up: with an off request, a carb setting that is not false dies. -> marker line (empty without
    a request)."""
    if not driver_check_off():
        return ""
    import carb

    got = carb.settings.get_settings().get(DRIVER_SETTING)
    if got not in (False, "false", 0):
        raise SystemExit(f"{DMARK} ✗ {DRIVER_SETTING}={got!r}: the off request did not take")
    line = f"{DMARK} applied {{{DRIVER_SETTING!r}: {got!r}}}"
    print(line, flush=True)
    return line


def kit_flags() -> str:
    return " ".join(f"--{SETTING_PATH}{k}={v}" for k, v in settings())


def apply(argv: list[str]) -> list[str]:
    """Copy of argv with the --kit_args settings added (appended to an existing value); an empty extension folder dies
    (a Kit with the registry off reads only this folder)."""
    f = ext_folder()
    exts = sorted(p.name for p in f.iterdir()) if f.is_dir() else []
    if not exts:
        raise SystemExit(f"{MARK} ✗ extension folder {f} is empty: a Kit with the registry off reads only this folder")
    flags = kit_flags()
    out = _add_kit_args(argv, flags)
    sha = hashlib.sha256("\n".join(exts).encode()).hexdigest()[:12]
    print(
        f"{MARK} registry=off fsWatcher=off folder={f} exts={len(exts)} list_sha={sha} kit_args='{flags}'", flush=True
    )
    return out


def check_applied(_mod=None) -> str:
    """After the app is up (post-import hook): read the carb settings and check both are off, else die. Returns the
    marker line (the hook proof prints it again: when gymnasium is first imported during Kit startup, the line printed
    here does not reach the session log)."""
    import carb

    s = carb.settings.get_settings()
    got = {k: s.get(SETTING_PATH + k) for k, _v in settings()}
    if bad := {k: v for k, v in got.items() if v not in (False, "false", 0)}:
        raise SystemExit(f"{MARK} ✗ Kit settings did not take {bad}: with the registry on the download can hang")
    line = f"{MARK} applied {got}"
    print(line, flush=True)
    return line


def _add_kit_args(argv: list[str], flags: str) -> list[str]:
    out = list(argv)
    if "--kit_args" in out:
        i = out.index("--kit_args") + 1
        out[i] = f"{out[i]} {flags}".strip()
    else:
        out += ["--kit_args", flags]
    return out


def apply_render_threads(argv: list[str]) -> list[str]:
    """Copy of the render-stage argv with the worker cap (RENDER_THREAD_SETTINGS) added as --kit_args (module doc)."""
    flags = " ".join(f"--{k}={v}" for k, v in RENDER_THREAD_SETTINGS)
    print(f"{TMARK} kit_args='{flags}'", flush=True)
    return _add_kit_args(argv, flags)


def carb_tasking_threads() -> int:
    """Number of threads in /proc/self/task whose name starts with carb.tasking (how many workers really started)."""
    n = 0
    for t in Path("/proc/self/task").iterdir():
        with contextlib.suppress(OSError):
            n += (t / "comm").read_text().startswith("carb.tasking")
    return n


def check_render_threads() -> str:
    """After the app is up: a setting != the cap, or more live carb.tasking threads than the cap, dies; then the
    OpenBLAS pool is bound (apply_render_blas). -> marker."""
    import carb

    s = carb.settings.get_settings()
    got = {k: s.get(k) for k, _v in RENDER_THREAD_SETTINGS}
    if bad := {k: v for k, v in got.items() if v != dict(RENDER_THREAD_SETTINGS)[k]}:
        raise SystemExit(f"{TMARK} ✗ worker cap did not take {bad} (cap {RENDER_THREADS})")
    n = carb_tasking_threads()
    if not 0 < n <= RENDER_THREADS:
        raise SystemExit(f"{TMARK} ✗ carb.tasking threads {n}: must be at most {RENDER_THREADS} (setting {got})")
    line = f"{TMARK} applied {got} · carb.tasking threads {n} · {apply_render_blas()}"
    print(line, flush=True)
    return line


def openblas_libs() -> list[str]:
    """Paths of the OpenBLAS shared libraries loaded in this process (/proc/self/maps; sorted, no duplicates)."""
    out = set()
    with open("/proc/self/maps") as f:
        for ln in f:
            p = ln.rstrip("\n").split(maxsplit=5)
            if len(p) == 6 and "openblas" in os.path.basename(p[5]).lower():
                out.add(p[5])
    return sorted(out)


def _blas_threads(n: int | None) -> dict[str, int]:
    """Per loaded OpenBLAS: bind it to n (when given) and read the thread count back -> {file name: value}. Opens only
    libraries already loaded (RTLD_NOLOAD)."""
    import ctypes

    got = {}
    for p in openblas_libs():
        lib = ctypes.CDLL(p, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        api = next(((s, g) for s, g in BLAS_API if hasattr(lib, s) and hasattr(lib, g)), None)
        if api is None:
            raise SystemExit(f"{BMARK} ✗ {p}: no thread-count API (known names {BLAS_API})")
        if n is not None:
            getattr(lib, api[0])(ctypes.c_int(n))
        got[os.path.basename(p)] = int(getattr(lib, api[1])())
    return got


def apply_render_blas() -> str:
    """After the app is up (render stages): bind every loaded OpenBLAS to RENDER_BLAS_THREADS and later ones through
    the env (module doc). A read-back value != the constant, or no OpenBLAS at all, dies. -> marker."""
    k = len(os.sched_getaffinity(0))
    if not 0 < RENDER_BLAS_THREADS <= k:
        raise SystemExit(f"{BMARK} ✗ constant {RENDER_BLAS_THREADS}: must be 1 to the affinity core count {k}")
    os.environ["OPENBLAS_NUM_THREADS"] = str(RENDER_BLAS_THREADS)
    got = _blas_threads(RENDER_BLAS_THREADS)
    if not got or any(v != RENDER_BLAS_THREADS for v in got.values()):
        raise SystemExit(f"{BMARK} ✗ OpenBLAS threads not bound {got} (constant {RENDER_BLAS_THREADS})")
    line = f"{BMARK} openblas={RENDER_BLAS_THREADS} affinity={k} libs={got}"
    print(line, flush=True)
    return line


def check_render_blas() -> str:
    """End of the session: read every loaded OpenBLAS again (late-loaded ones included); a value != the constant dies
    (one that ignored the env). -> marker."""
    got = _blas_threads(None)
    if any(v != RENDER_BLAS_THREADS for v in got.values()):
        raise SystemExit(f"{BMARK} ✗ OpenBLAS threads at session end {got} (constant {RENDER_BLAS_THREADS})")
    line = f"{BMARK} end check openblas={RENDER_BLAS_THREADS} libs={got}"
    print(line, flush=True)
    return line
