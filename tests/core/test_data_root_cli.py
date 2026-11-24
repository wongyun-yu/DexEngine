"""`--data-root` on every stage entry point: declared through config.add_data_root_arg, applied by
config.apply_data_root after parsing, and effective because no entry module reads the data root at import time."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.core import config as C

REPO = Path(__file__).resolve().parents[2]
STAGE4_TOOLS = ("drive", "bundle", "merge", "scene-layout", "look-mirror", "assets", "text")
#: python -m entry modules (and their files) that take the flag.
MODULES = (
    "dexengine.retarget.run",
    "dexengine.retarget.reretarget",
    "dexengine.data.intake",
    "dexengine.data.read_raw.__main__",
    "dexengine.data.stages",
    "dexengine.record.driver",
    "dexengine.record.bundle",
    "dexengine.record.store.merge",
    "dexengine.record.store.scene_layout",
    "dexengine.record.looks.look_d",
    "dexengine.record.looks.augvis.assets",
    "dexengine.record.text.pipeline",
)
SCRIPTS = (
    "scripts/stage1_retarget.py",
    "scripts/stage3_rl.py",
    "scripts/build_robot_usd.py",
)


def _env(**kw):
    return {**os.environ, "PYTHONPATH": str(REPO), "DEXENGINE_ARM": "rby1", "DEXENGINE_HAND": "sharpa_wave", **kw}


def _run(argv, **kw):
    return subprocess.run(argv, cwd=REPO, capture_output=True, text=True, timeout=180, env=_env(**kw))


def _help_cases():
    py = sys.executable
    yield from ([py, s, "--help"] for s in SCRIPTS)
    yield from ([py, "scripts/stage4_record.py", t, "--help"] for t in STAGE4_TOOLS)
    for m in MODULES:
        yield (
            [py, "dexengine/data/stages.py", "--help"]
            if m == "dexengine.data.stages"
            else [py, "-m", m.removesuffix(".__main__"), "--help"]
        )


@pytest.mark.parametrize("argv", list(_help_cases()), ids=lambda a: " ".join(a[1:-1]))
def test_help_lists_data_root(argv):
    r = _run(argv)
    assert r.returncode == 0, r.stderr[-300:]
    assert "--data-root" in r.stdout


def test_shell_stages_take_data_root():
    r = _run(["bash", "scripts/stage2_ground.sh", "--help"])
    assert r.returncode == 0 and "--data-root" in r.stdout
    assert "--data-root|--main-root)" in (REPO / "scripts/_lib.sh").read_text(encoding="utf-8")


def _calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]


def test_every_entry_declares_and_applies_through_the_owner():
    files = [REPO / s for s in SCRIPTS] + [REPO / (m.replace(".", "/") + ".py") for m in MODULES]
    for f in files:
        calls = _calls(f)
        assert {"add_data_root_arg", "apply_data_root"} <= {c.func.attr for c in calls}, f
        own = [
            c
            for c in calls
            if c.func.attr == "add_argument"
            and c.args
            and isinstance(c.args[0], ast.Constant)
            and c.args[0].value == "--data-root"
        ]
        assert not own, f"{f}: declares --data-root itself instead of config.add_data_root_arg"
    assert "take_data_root" in {c.func.attr for c in _calls(REPO / "scripts/stage4_record.py")}


_PROBE = r"""
import importlib, sys
import dexengine.core.layout as L
import dexengine.core.layout.roots as R
reads = []
orig = R.configured_data_root
def probe():
    reads.append(cur[0])
    return orig()
cur = [""]
R.configured_data_root = probe
L.configured_data_root = probe
for m in sys.argv[1:]:
    cur[0] = m
    importlib.import_module(m)
print("READS", sorted(set(reads)))
"""


def test_entry_modules_read_no_data_root_at_import():
    """A module that reads the data root while it is imported keeps that value: the flag, applied after parsing,
    would not reach it."""
    r = _run([sys.executable, "-c", _PROBE, *MODULES])
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.strip().splitlines()[-1] == "READS []"


def test_take_data_root_passes_the_rest(tmp_path, monkeypatch):
    monkeypatch.delenv("DEXENGINE_MAIN_ROOT", raising=False)
    rest = C.take_data_root(["--dataset", "x", "--data-root", str(tmp_path), "--set", "a=b", "--data", "y"])
    assert rest == ["--dataset", "x", "--set", "a=b", "--data", "y"]  # no abbreviation is taken for the flag
    assert os.environ["DEXENGINE_MAIN_ROOT"] == str(tmp_path)
    assert C.take_data_root(["--help"]) == ["--help"]
