"""Every third-party package that code run by the retargeting interpreter (RETARGET_PY) imports is in pyproject's
`retarget` extra, so a venv built from that extra does not stop mid-pipeline ("cannot import warp" at step 7, "No module
named pxr" at step 4).

What runs under that interpreter: the stage-1 table's steps whose slot is `retarget`, the scripts and modules named by
step 1's command (retargeting), the f0 GPU worker that step 7 starts by path, and every `-m dexengine.…` child those
start. Their import closure follows every dexengine import (module level and inside functions, parent packages
included) and `importlib.import_module("<literal>")`; each third-party name maps to its distribution through
IMPORT_DIST (an unmapped name fails, so a new dependency is added there and to the extra together)."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import tomllib

from dexengine.data.stage1 import table as T

REPO = Path(__file__).resolve().parents[2]
#: Started by path (not `-m`) by step 7's shadow_gpu with the prejudge's `--python` (default: the retarget interpreter).
F0_GPU_WORKER = "dexengine/retarget/f0/gpu.py"
#: Import name -> distribution name.
IMPORT_DIST = {
    "coacd": "coacd",
    "loguru": "loguru",
    "mujoco": "mujoco",
    "mujoco_warp": "mujoco-warp",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "pinocchio": "pin",
    "pxr": "usd-core",
    "pyarrow": "pyarrow",
    "pymeshlab": "pymeshlab",
    "scipy": "scipy",
    "smplx": "smplx",
    "threadpoolctl": "threadpoolctl",
    "torch": "torch",
    "trimesh": "trimesh",
    "tyro": "tyro",
    "warp": "warp-lang",
    "yaml": "pyyaml",
}


def _file(module: str) -> Path | None:
    """Source file of a dexengine module or package, None when the name is not one (e.g. an imported symbol)."""
    base = REPO / module.replace(".", "/")
    for p in (base.with_suffix(".py"), base / "__init__.py"):
        if p.is_file():
            return p
    return None


def _module_of(path: Path) -> str:
    """Dotted module of a repository file (a package __init__ keeps its `.__init__` part)."""
    return ".".join(path.relative_to(REPO).with_suffix("").parts)


def _absolute(node: ast.ImportFrom, module: str) -> str:
    if not node.level:
        return node.module or ""
    return ".".join(module.split(".")[: -node.level] + ([node.module] if node.module else []))


def _calls_import_module(n: ast.AST) -> str | None:
    if isinstance(n, ast.Call) and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
        f = n.func
        if (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")) == "import_module":
            return n.args[0].value
    return None


def _children(n: ast.AST) -> list[str]:
    """`"-m", "dexengine.…"` pairs inside a list or tuple display (a child process of the same interpreter)."""
    if not isinstance(n, (ast.List, ast.Tuple)):
        return []
    el = n.elts
    return [
        b.value
        for a, b in zip(el, el[1:], strict=False)
        if isinstance(a, ast.Constant) and a.value == "-m" and isinstance(b, ast.Constant)
        if isinstance(b.value, str) and b.value.startswith("dexengine.")
    ]


def references(src: str, module: str) -> tuple[set[str], set[str]]:
    """(every imported name, `-m` children) of one source; relative imports resolve against `module`."""
    names, kids = set(), set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            m = _absolute(n, module)
            names |= {m} | {f"{m}.{a.name}" for a in n.names}
        elif _calls_import_module(n):
            names.add(_calls_import_module(n))
        kids |= set(_children(n))
    return names, kids


def closure(entries: list[Path]) -> dict[str, set[str]]:
    """{third-party top-level name: {files importing it}} over the import closure of `entries` (module doc)."""
    todo, seen, third = list(entries), set(), {}
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        names, kids = references(f.read_text(encoding="utf-8"), _module_of(f))
        for k in kids:
            todo += [p for p in (_file(k), _file(f"{k}.__main__")) if p]
        for name in names:
            parts = name.split(".")
            if parts[0] == "dexengine":
                todo += [p for i in range(1, len(parts) + 1) if (p := _file(".".join(parts[:i])))]
            elif parts[0] not in sys.stdlib_module_names and parts[0] != "__future__":
                third.setdefault(parts[0], set()).add(str(f.relative_to(REPO)))
    return third


def retarget_entries() -> list[Path]:
    """Entry files run by the retargeting interpreter (module doc), read from the stage-1 table."""
    out = [REPO / F0_GPU_WORKER]
    for name, _cell, slot, cmd in T.STAGES:
        if slot == "retarget":
            if not isinstance(cmd, T.Run):
                raise AssertionError(f"step {name!r} runs under RETARGET_PY but is not a Run; read its entry here")
            out.append(REPO / cmd.argv[0])
    (step1,) = [cmd for name, _cell, _slot, cmd in T.STAGES if T.step_id((name,)) == "1"]
    words = step1.split()
    out += [REPO / w for w in words if w.endswith(".py")]
    out += [p for i, w in enumerate(words[:-1]) if w == "-m" for p in (_file(words[i + 1]),) if p]
    missing = [str(p) for p in out if not p.is_file()]
    assert not missing, f"entries that do not exist: {missing}"
    return out


def extra_dists(pyproject_text: str, extra: str) -> set[str]:
    """Normalized distribution names of one optional-dependency group."""
    reqs = tomllib.loads(pyproject_text)["project"]["optional-dependencies"][extra]
    return {re.split(r"[\s\[<>=!~@;]", r, maxsplit=1)[0].lower().replace("_", "-") for r in reqs}


def missing(third: dict[str, set[str]], dists: set[str]) -> dict[str, list[str]]:
    """{distribution: [files]} for third-party imports whose distribution is not in `dists`; an import name without
    an IMPORT_DIST row raises KeyError naming it."""
    unknown = sorted(set(third) - set(IMPORT_DIST))
    if unknown:
        raise KeyError(f"imports without an IMPORT_DIST row: {unknown} (e.g. {sorted(third[unknown[0]])[:3]})")
    return {IMPORT_DIST[n]: sorted(fs)[:3] for n, fs in sorted(third.items()) if IMPORT_DIST[n] not in dists}


def test_retarget_interpreter_imports_are_in_the_retarget_extra():
    entries = retarget_entries()
    assert REPO / "dexengine/data/object_roles.py" in entries and REPO / F0_GPU_WORKER in entries
    third = closure(entries)
    assert {"pxr", "warp"} <= set(third), "step 4 (pxr) and the f0 GPU worker (warp) are in the closure"
    dists = extra_dists((REPO / "pyproject.toml").read_text(), "retarget")
    assert not missing(third, dists), "imports of the retargeting interpreter missing from the retarget extra"


def test_closure_follows_function_imports_children_and_import_module(tmp_path):
    src = (
        "import importlib\nimport numpy\n\n\ndef f():\n    import warp as wp\n"
        "    from mujoco_warp._src import types\n    importlib.import_module('threadpoolctl')\n"
        "    return ['py', '-m', 'dexengine.retarget.decompose']\n"
    )
    names, kids = references(src, "x.y")
    assert {"numpy", "warp", "mujoco_warp._src", "threadpoolctl"} <= names and kids == {"dexengine.retarget.decompose"}
    third = {"numpy": {"a.py"}, "warp": {"a.py"}, "mujoco_warp": {"a.py"}}
    pyproject = '[project.optional-dependencies]\nretarget = ["numpy==2.3.5", "warp-lang==1.11.0"]\n'
    assert missing(third, extra_dists(pyproject, "retarget")) == {"mujoco-warp": ["a.py"]}
    with pytest.raises(KeyError, match="cupy"):
        missing({"cupy": {"b.py"}}, set())
