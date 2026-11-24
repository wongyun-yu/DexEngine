"""The datasets whose reader calls a rule's owner function (scope source `READER_CALL_SCOPE` of the rule registry).

A rule whose mechanism is one reader function (its `owner`, `path:function`) applies to every dataset whose reader
module calls that function, so a new reader that calls it falls in the rule's range without a name being written
anywhere. A dataset's reader is the `module` of its row in `dexengine.data.readers.READERS`; the module source is read
with ast and never executed. A call is a call node that names the function:

- by its bare name inside the owner's own module,
- through the owner module bound by an import (`from <package> import <module>` · `import <package>.<module> as m`)
  as `<name>.<function>(…)`,
- through the function imported by name (`from <owner module> import <function> [as f]`) as `f(…)`.

A reader that reaches the function only through another module (a helper calling it) is not seen here;
tests/core/test_rule_reader_scope.py fails when any dexengine module that calls an owner is not a reader row in the
rule's range.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

#: `Rule.scope_from` value: the datasets whose reader module calls the rule's owner function.
READER_CALL_SCOPE = "readers:calls-owner"


def owner_function(owner: str) -> tuple[str, str]:
    """`<file>.py:<function>` -> (dotted module, function), e.g. the D12-g owner -> ("dexengine.data.readers.hot3d",
    "_fill_gt_gaps"). Anything else raises ValueError."""
    path, sep, fn = owner.rpartition(":")
    if not (sep and path.endswith(".py") and fn.isidentifier()):
        raise ValueError(f"owner {owner!r} is not `<file>.py:<function>` — a reader-call scope needs one function")
    return path[: -len(".py")].replace("/", "."), fn


def _absolute(node: ast.ImportFrom, module: str) -> str:
    """The absolute module an ImportFrom reads from, inside the file of `module` (not a package __init__)."""
    if not node.level:
        return node.module or ""
    base = module.split(".")[: -node.level]
    return ".".join(base + ([node.module] if node.module else []))


def _bindings(tree: ast.Module, module: str, owner_mod: str, fn: str) -> tuple[set, set]:
    """(names bound to the owner module, names bound to the owner function) by the imports of one module."""
    mods, fns = set(), {fn} if module == owner_mod else set()
    pkg, _, leaf = owner_mod.rpartition(".")
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.asname for a in n.names if a.name == owner_mod and a.asname}
        elif isinstance(n, ast.ImportFrom):
            src = _absolute(n, module)
            mods |= {a.asname or a.name for a in n.names if src == pkg and a.name == leaf}
            fns |= {a.asname or a.name for a in n.names if src == owner_mod and a.name == fn}
    return mods, fns


def calls_owner(src: str, module: str, owner: str) -> bool:
    """Does the source of `module` call the owner function (module doc for what a call is)?"""
    owner_mod, fn = owner_function(owner)
    tree = ast.parse(src)
    mods, fns = _bindings(tree, module, owner_mod, fn)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Name) and f.id in fns:
            return True
        if isinstance(f, ast.Attribute) and f.attr == fn and isinstance(f.value, ast.Name) and f.value.id in mods:
            return True
    return False


def module_source(module: str) -> str:
    """Source text of an importable module, found without executing it; a module that cannot be found raises."""
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        raise ValueError(f"reader module {module!r} has no Python source (find_spec: {spec})")
    return Path(spec.origin).read_text(encoding="utf-8")


def reader_callers(owner: str, readers: dict) -> tuple:
    """Sorted datasets whose reader row (`readers`: dataset -> row with `.module`) calls the owner function."""
    return tuple(sorted(ds for ds, row in readers.items() if calls_owner(module_source(row.module), row.module, owner)))


def reader_scope(owner: str) -> tuple:
    """The reader-call scope of an owner over the registered readers (`dexengine.data.readers.READERS`)."""
    from dexengine.data.readers import READERS  # noqa: PLC0415 — the reader registry (no reader module is imported)

    return reader_callers(owner, READERS)
