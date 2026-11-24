"""The machine gate of the cross repair (cross_repair -> xr_gpu) the augment bake runs, before a round: in the bake's
own interpreter (bake.isaac_py) and environment (the job's TMPDIR, the round's GPU) the cross-repair dependencies
(XR_DEPS) must import and a Warp kernel must really compile and run (``runtime_env.py --check``: NVRTC inside
nvrtc_scope, on a fresh kernel cache). Without it a missing mujoco_warp, or a compiler that refuses the machine's
conditions, failed every bake of the round and the canonical cell (pv00) went missing. Called by record.chain
.after_store; a failure is the bundle's failure. Marker (the proof it ran): ``[AUG_XR_DEPS] cross-repair deps … OK ·
[XR_TMP] CHECK OK …``. Not part of the augment cache fingerprint: it decides no output.
"""

from __future__ import annotations

import os
import subprocess

from dexengine.core.repo import REPO_ROOT

MARK = "[AUG_XR_DEPS]"
XR_DEPS = ("mujoco_warp",)
CHECK_TIMEOUT_S = 180
CHECK = REPO_ROOT / "dexengine" / "core" / "runtime_env.py"  # run by path: it imports nothing from dexengine
_FIND = (
    "import importlib.util as u, sys; m = [n for n in sys.argv[1:] if u.find_spec(n) is None]; "
    "print(' '.join(m)); sys.exit(3 if m else 0)"
)


def xr_preflight(gpu: str) -> str:
    """Dependencies, then the compile check, in the bake's interpreter with CUDA_VISIBLE_DEVICES=gpu -> the marker line;
    dies with what failed."""
    from dexengine.record.augment import bake as AB  # noqa: PLC0415 — owner of the bake's interpreter

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    py = AB.isaac_py(env)
    r = subprocess.run([py, "-c", _FIND, *XR_DEPS], capture_output=True, text=True, env=env, timeout=CHECK_TIMEOUT_S)
    if r.returncode:
        raise SystemExit(
            f"{MARK} ✗ cross-repair dependencies {r.stdout.strip() or XR_DEPS} are missing in the bake's Isaac python "
            f"{py} — install them into that machine's env (rc {r.returncode} {r.stderr.strip()[-160:]})"
        )
    c = subprocess.run([py, str(CHECK), "--check"], capture_output=True, text=True, env=env, timeout=CHECK_TIMEOUT_S)
    out = (c.stdout + c.stderr).strip().splitlines()
    ok = [ln for ln in out if "CHECK OK" in ln]
    if c.returncode or not ok:
        raise SystemExit(
            f"{MARK} ✗ Warp kernel compile check failed rc {c.returncode} · TMPDIR "
            f"{len(env.get('TMPDIR', ''))} chars — {_reason(out)}"
        )
    return f"{MARK} cross-repair deps {' · '.join(XR_DEPS)} OK · {ok[-1]} · {py}"


def _reason(out: list[str]) -> str:
    """The failure's own lines: NVRTC's reason first, then the last exception line (else the last two lines)."""
    nvrtc = [ln for ln in out if "Catastrophic" in ln or "too long" in ln][-1:]
    exc = [ln for ln in out if ln.startswith(("Exception", "RuntimeError", "ModuleNotFoundError", "Error"))][-1:]
    return " | ".join(nvrtc + exc or out[-2:])[-300:]
