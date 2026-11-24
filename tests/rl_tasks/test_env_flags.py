"""Every switch read goes through env_flags / switches, and no converted switch is read from the environment."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from dexengine.rl.tasks import env_flags, switches

ROOT = Path(__file__).resolve().parents[2]
SCOPE = [ROOT / "dexengine" / "rl" / p for p in ("tasks", "env_setup", "assets")]
READERS = ("env_flags.py", "switches.py")
# Direct accesses that are not package switches: the extra-slot ledger restores a checkpoint's recorded extra-slot
# switch (name owned by the data package); the per-side reset reads the torch.distributed rank.
ALLOWED_DIRECT = {
    ("dexengine/rl/env_setup/extra_slots_ledger.py", "env"),
    ("dexengine/rl/tasks/env_side_reset.py", "'LOCAL_RANK'"),
}
# Stdlib-only modules run with the system python3 (no package on sys.path); their entry must not read a switch.
STDLIB_SCRIPTS = {"dexengine/rl/tasks/scene/collision_presets.py": ["--what", "list"]}
# Files that run on pinned old trees (experiments/README.md): they spell the switches with the old prefix on purpose
# (experiments.workspace.LEGACY_ENV_PREFIX).
PINNED = ("experiments/scoring/k_sweep/", "experiments/ablations/rl_arm/")
# Launcher-side reads of a switch's old name (outside the trainer process): the value is a launch input that the RL
# launcher reads from its inherited environment as a recorded-environment alias and writes into the switches file.
LAUNCH_INPUTS: set = set()


def _module_consts(tree):
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = n.value.value
    return out


def _env_reads(paths):
    """(file, line, name) of os.environ / os.getenv reads, and of env_flags.get / switches.text reads."""
    direct, via = [], []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        consts = _module_consts(tree)
        rel = str(p.relative_to(ROOT))
        for n in ast.walk(tree):
            key = None
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.args:
                base = ast.unparse(n.func.value)
                if n.func.attr in ("get", "getenv") and (base.endswith("environ") or base == "os"):
                    key = n.args[0]
                elif (n.func.attr, base.split(".")[-1].lstrip("_")) in (("get", "env_flags"), ("text", "switches")):
                    a = n.args[0]
                    name = a.value if isinstance(a, ast.Constant) else consts.get(getattr(a, "id", ""))
                    via.append((rel, n.lineno, name))
            elif isinstance(n, ast.Subscript) and ast.unparse(n.value).endswith("environ"):
                key = n.slice
            if key is not None:
                name = (
                    key.value if isinstance(key, ast.Constant) else consts.get(getattr(key, "id", ""), ast.unparse(key))
                )
                direct.append((rel, n.lineno, name, ast.unparse(key)))
    return direct, via


def _scope_files():
    for d in SCOPE:
        yield from (p for p in d.rglob("*.py") if p.name not in READERS)


def test_no_direct_environment_reads_in_the_package():
    direct, _ = _env_reads(_scope_files())
    assert [d[:3] for d in direct if (d[0], d[3]) not in ALLOWED_DIRECT] == []


def test_no_converted_switch_is_read_from_the_environment():
    """Across dexengine/ and the current-tree experiments, no code reads a switch by its old environment name."""
    files = [p for p in (ROOT / "dexengine").rglob("*.py") if p.name not in READERS]
    files += [p for p in (ROOT / "experiments").rglob("*.py") if not str(p.relative_to(ROOT)).startswith(PINNED)]
    direct, _ = _env_reads(files)
    assert [d[:3] for d in direct if d[2] in switches.FIELD_OF and (d[0], d[2]) not in LAUNCH_INPUTS] == []


def test_every_read_name_is_registered():
    from dexengine.core import config

    _, via = _env_reads(_scope_files())
    assert via, "no env_flags reads found"
    assert [v for v in via if v[2] not in switches.FIELD_OF and v[2] not in config.ENV_VARS] == []


def test_no_variable_is_stated_twice():
    """core.config.ENV_VARS states the process-boundary variables; a switch is not one of them."""
    from dexengine.core import config

    assert sorted(set(switches.FIELD_OF) & set(config.ENV_VARS)) == []
    assert switches.SWITCHES_ENV in config.ENV_VARS


@pytest.mark.parametrize("rel", sorted(STDLIB_SCRIPTS))
def test_stdlib_script_runs_without_the_package(rel):
    """-I: no cwd, no PYTHONPATH, no user site on sys.path."""
    r = subprocess.run(
        [sys.executable, "-I", str(ROOT / rel), *STDLIB_SCRIPTS[rel]],
        capture_output=True,
        text=True,
        timeout=60,
        cwd="/",
    )
    assert r.returncode == 0, r.stderr[-500:]


def test_get_semantics_without_a_file(monkeypatch):
    """A process without a switches file reads the old names from its environment, as before."""
    monkeypatch.delenv("DEXENGINE_ENVZ", raising=False)
    assert env_flags.get("DEXENGINE_ENVZ") == "1"
    assert env_flags.get("DEXENGINE_ARM_KP") is None
    assert env_flags.get("DEXENGINE_ENVZ", None) is None
    monkeypatch.setenv("DEXENGINE_ENVZ", "")
    assert env_flags.get("DEXENGINE_ENVZ") == ""
    with pytest.raises(KeyError, match="neither a switch"):
        env_flags.get("DEXENGINE_NOT_A_SWITCH")
    monkeypatch.delenv("DEXENGINE_ARM", raising=False)
    assert env_flags.get("DEXENGINE_ARM") == ""
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    assert env_flags.get("DEXENGINE_ARM") == "rby1"
