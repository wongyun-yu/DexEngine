"""Every repository import in dexengine.rl (lazy ones inside functions included) names a module file that exists and,
for `from M import N`, a submodule or a top-level name of M. Read from the sources only (nothing is imported), so a
renamed or moved module fails here instead of at the first judge or render that reaches the lazy import."""

from __future__ import annotations

import ast
from pathlib import Path

from dexengine.core.repo import REPO_ROOT
from tests.core.references_paths import SITE

ROOTS = ("dexengine", "experiments", SITE)
SKIP = ("tasks", "env_setup", "assets")  # dexengine/rl parts owned elsewhere


def _file_of(mod: str) -> Path | None:
    """The module's file, its package folder (a namespace package has no __init__.py), or None."""
    base = Path(REPO_ROOT, *mod.split("."))
    for f in (base.with_suffix(".py"), base / "__init__.py"):
        if f.is_file():
            return f
    return base if base.is_dir() else None


def _top_names(f: Path) -> set[str]:
    out: set[str] = set()
    if f.is_dir():
        return out
    for x in ast.parse(f.read_text(encoding="utf-8")).body:
        if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(x.name)
        elif isinstance(x, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for tg in x.targets if isinstance(x, ast.Assign) else [x.target]:
                out |= {e.id for e in ast.walk(tg) if isinstance(e, ast.Name)}
        elif isinstance(x, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in x.names}
        elif isinstance(x, (ast.If, ast.Try)):
            out |= {n.id for n in ast.walk(x) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    return out


def _imports(p: Path):
    pkg = list(p.relative_to(REPO_ROOT).with_suffix("").parts[:-1])
    for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom):
            mod = ".".join((pkg[: len(pkg) - n.level + 1] if n.level else []) + ([n.module] if n.module else []))
            yield n.lineno, mod, [a.name for a in n.names if a.name != "*"]
        elif isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, a.name, []


def test_repository_imports_resolve():
    bad = []
    for p in sorted(Path(REPO_ROOT, "dexengine/rl").rglob("*.py")):
        if p.relative_to(REPO_ROOT).parts[2] in SKIP:
            continue
        for line, mod, names in _imports(p):
            if mod.split(".")[0] not in ROOTS:
                continue
            f = _file_of(mod)
            if f is None:
                bad.append(f"{p.relative_to(REPO_ROOT)}:{line} no module {mod}")
                continue
            defined = None
            for nm in names:
                if _file_of(f"{mod}.{nm}"):
                    continue
                defined = _top_names(f) if defined is None else defined
                if nm not in defined and "__getattr__" not in defined:
                    bad.append(f"{p.relative_to(REPO_ROOT)}:{line} {mod} has no {nm}")
    assert not bad, "\n".join(bad)
