"""Session entry imports must not load numpy before Kit starts (the [PRE_APP] guard of dexengine.rl.preapp kills a
session that did)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRE_APP = (
    "dexengine.record.session.entry",
    "dexengine.record.session.kit",
    "dexengine.record.session.persist_kit",
    "dexengine.record.session.persist_guard",
    "dexengine.record.render.episode",
    "dexengine.record.render.episode_frames",
    "dexengine.record.render.rig",
    "dexengine.record.render.clip_session",
    "dexengine.record.render.clip_phases",
    "dexengine.record.scene.extras_apply",
)


def numpy_loaded(mods) -> bool:
    code = "import sys\n" + "".join(f"import {m}\n" for m in mods) + "print('numpy' in sys.modules)"
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    return r.stdout.strip() != "False"


def test_session_modules_do_not_load_numpy():
    assert not numpy_loaded(PRE_APP)


@pytest.mark.parametrize("extra", ["dexengine.record.render.framing", "dexengine.record.render.fov"])
def test_numpy_module_is_detected(extra):
    assert numpy_loaded((*PRE_APP, extra))
