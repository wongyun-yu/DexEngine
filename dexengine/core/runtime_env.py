"""The one owner of warp in dexengine and of the directory NVRTC compiles its kernels in.

NVRTC (the CUDA compiler warp links in statically) writes its temporary files under TMPDIR, and its C++ front end
(EDG) refuses a TMPDIR longer than NVRTC_TMPDIR_MAX characters: ``Catastrophic error: name of directory for temporary
files is too long``, which warp reports as ``CUDA kernel build failed with error code 6``. Measured with warp 1.11.0
(the Isaac environment) and 1.11.0.dev20251120 (the retarget venv): 125 characters compile, 126 fail, whatever the
program name; a trailing "/" counts. A job's scratch TMPDIR is often longer (a node job's is ~172 characters), and a
kernel cache hit skips NVRTC, so the failure shows only on a machine with a cold cache.

``nvrtc_scope()`` wraps every call that can compile a kernel (``launch``: warp compiles a module on its first launch):
meanwhile TMPDIR is this process's own short directory (``short_tmp_dir``: mkdtemp /tmp/xr.XXXXXXXX, made on first use,
removed at process exit — two jobs on one machine never share it); afterwards the caller's TMPDIR is back, so the other
steps' large temporary files stay where the job put them. The scope dies only when that short directory is itself over
the limit. Marker ``[XR_TMP] NVRTC temp dir <dir> (original TMPDIR <n> chars)`` once per process, on stderr (an f0 GPU
worker's stdout carries its results).
``check()`` (``python dexengine/core/runtime_env.py --check``; the augment bake's machine gate record.augment.xr_deps
runs it in the bake's interpreter) compiles a probe with NVRTC inside the scope and, with a CUDA device, launches it
on a fresh kernel cache: a machine whose conditions NVRTC refuses fails before any bake.
``warp()`` is the only way dexengine imports warp; tests/core/test_runtime_env.py checks structurally that warp is
imported nowhere else and that dexengine launches a kernel only through ``launch()`` (inside the scope).
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import sys
import tempfile

TMP_MARK = "[XR_TMP]"
#: longest TMPDIR NVRTC's front end accepts (characters, measured — module doc); NVRTC reads TMPDIR, else /tmp
NVRTC_TMPDIR_MAX = 125
TMP_ROOT = "/tmp"  # root of the short directory (present on every machine and pod)
TMP_PREFIX = "xr."
_TMP_DIR: list[str] = []


def short_tmp_dir() -> str:
    """This process's short NVRTC temporary directory — created on the first call (unique name), removed at exit; dies
    when it is longer than NVRTC_TMPDIR_MAX (a TMP_ROOT that long)."""
    if not _TMP_DIR:
        d = tempfile.mkdtemp(prefix=TMP_PREFIX, dir=TMP_ROOT)
        atexit.register(shutil.rmtree, d, True)
        if len(d) > NVRTC_TMPDIR_MAX:
            raise SystemExit(
                f"{TMP_MARK} ✗ the short NVRTC directory {d!r} is {len(d)} characters; NVRTC (the CUDA compiler in "
                f"warp) accepts at most {NVRTC_TMPDIR_MAX} — its front end (EDG) stops with 'name of directory for "
                f"temporary files is too long'. TMP_ROOT ({TMP_ROOT!r}) must be a short directory."
            )
        _TMP_DIR.append(d)
        print(
            f"{TMP_MARK} NVRTC temp dir {d} (original TMPDIR {len(os.environ.get('TMPDIR', ''))} chars)",
            file=sys.stderr,
            flush=True,
        )
    return _TMP_DIR[0]


@contextlib.contextmanager
def nvrtc_scope():
    """Wrap a call that can compile a Warp kernel (a launch): TMPDIR = short_tmp_dir() meanwhile, the caller's after."""
    d = short_tmp_dir()
    old = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = d
    try:
        yield d
    finally:
        if old is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old


def launch(*args, **kwargs):
    """``wp.launch`` inside nvrtc_scope() — the only launch of a Warp kernel in dexengine (a launch compiles its module
    the first time, and again when the module changed)."""
    with nvrtc_scope():
        return warp().launch(*args, **kwargs)


def warp():
    """The warp module (the only import of warp in dexengine)."""
    import warp as wp  # noqa: PLC0415 — the one import of warp

    return wp


def check() -> str:
    """Compile a probe kernel with NVRTC inside nvrtc_scope (PTX only: no GPU needed) and, with a CUDA device visible,
    launch a Warp kernel on a fresh kernel cache and read its value back -> the marker line; raises when warp cannot."""
    wp = warp()
    from warp._src import build as B  # noqa: PLC0415 — NVRTC itself (the compile a launch does on a cold cache)

    wp.config.quiet = True
    with nvrtc_scope() as d:
        work = tempfile.mkdtemp(prefix="kc.", dir=d)
        wp.config.kernel_cache_dir = work  # fresh cache: NVRTC really runs
        wp.init()
        src = os.path.join(work, "probe.cu")
        with open(src, "w", encoding="utf-8") as f:
            f.write('extern "C" __global__ void probe(float* a) { a[0] = 1.0f; }\n')
        B.build_cuda(src, 75, os.path.join(work, "probe.ptx"))
        dev = "cuda:0" if wp.is_cuda_available() else ""
        if dev:
            a = wp.zeros(1, dtype=float, device=dev)
            launch(_probe_kernel(wp), dim=1, inputs=[a], device=dev)
            wp.synchronize()
            if float(a.numpy()[0]) != 1.0:
                raise RuntimeError(f"{TMP_MARK} probe kernel wrote {a.numpy()[0]}, not 1.0")
    return (
        f"{TMP_MARK} CHECK OK nvrtc compile in {d} · launch {dev or 'not run (no CUDA device)'} · caller TMPDIR "
        f"{len(os.environ.get('TMPDIR', ''))} chars"
    )


def _probe_kernel(wp):
    @wp.kernel
    def _xr_tmp_probe(a: wp.array(dtype=float)):
        a[wp.tid()] = 1.0

    return _xr_tmp_probe


if __name__ == "__main__":
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("usage: python dexengine/core/runtime_env.py --check")
    print(check(), flush=True)
