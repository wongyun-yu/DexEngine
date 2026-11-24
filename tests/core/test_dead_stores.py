"""No function keeps a dead local store.

A local name assigned in a function and read nowhere in it (nor in a scope nested inside it) is a dead store. A
function split leaves one behind when an assignment the caller reads moves into the helper: the caller then reads an
unbound or stale name. ruff's F841 misses these here, because its default dummy-variable rule exempts every
`_name` local and this code names most locals that way. Only `_` itself is a throwaway name.

Counted: plain `name = …` / `name: T = …` targets in a function body. Not counted: tuple unpacking, `global` /
`nonlocal` names, functions that call `locals()`, `vars()`, `exec` or `eval`, and a line marked `# noqa: F841` (a
store kept on purpose, such as a lock held until the function returns). An augmented assignment reads its name.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

from tests.core.references_paths import SITE

ROOT = Path(__file__).resolve().parents[2]
SCOPES = ("dexengine", "experiments", SITE, "scripts", "tests")
_INTROSPECT = {"locals", "vars", "exec", "eval"}


def _own_nodes(fn: ast.AST):
    """Nodes of fn's own body, not of nested functions, lambdas or classes."""
    todo = list(ast.iter_child_nodes(fn))
    while todo:
        n = todo.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            todo.extend(ast.iter_child_nodes(n))


def dead_stores(src: str, path: str = "<src>") -> list[str]:
    """`path:line name` for each local assigned in a function and read nowhere in it."""
    out, lines = [], src.splitlines()
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        own = list(_own_nodes(fn))
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _INTROSPECT for n in own):
            continue
        declared = {name for n in own if isinstance(n, (ast.Global, ast.Nonlocal)) for name in n.names}
        stores = {}
        for n in own:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
            for t in targets:
                if isinstance(t, ast.Name) and t.id != "_" and t.id not in declared:
                    stores.setdefault(t.id, t.lineno)
        loaded = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Store)}
        loaded |= {n.target.id for n in ast.walk(fn) if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)}
        out += [
            f"{path}:{line} {name}"
            for name, line in stores.items()
            if name not in loaded and "noqa: F841" not in lines[line - 1]
        ]
    return out


def _files() -> list[Path]:
    r = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / f for f in r.stdout.split() if f.split("/")[0] in SCOPES]


def test_no_dead_local_stores():
    found = [x for f in _files() for x in dead_stores(f.read_text(encoding="utf-8"), str(f.relative_to(ROOT)))]
    assert not found, "dead local stores (assigned, never read):\n" + "\n".join(found)


def test_a_split_that_strands_an_assignment_is_caught():
    """The shape of the left-keypoint split: the helper assigns the flag, the caller reads its own unbound name."""
    src = (
        "def caller(x):\n"
        "    paths = helper(x)\n"
        "    return paths, _flag\n"
        "def helper(x):\n"
        "    _flag = 0\n"
        "    return [x]\n"
    )
    assert dead_stores(src) == ["<src>:5 _flag"]
    fixed = src.replace("    _flag = 0\n    return [x]\n", "    return [x]\n").replace(
        "    paths = helper(x)\n", "    _flag = 0\n    paths = helper(x)\n"
    )
    assert dead_stores(fixed) == []
