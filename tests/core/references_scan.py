"""Static cross-reference scan of the repo tree, used by tests/core/test_references.py (the rules are stated there).

Print every finding of the current tree, before exemptions, as `file<TAB>line<TAB>key<TAB>message` (the key is
what the exemption lists in tests/core/ match)::

    python -m tests.core.references_scan
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from tests.core.references_paths import SITE, TEXT_EXT, missing_paths, with_parents

ROOT = Path(__file__).resolve().parents[2]

PKG_ROOTS = ("dexengine", "experiments", SITE, "tests", "scripts")
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


# ---------------------------------------------------------------- module index


def modname(rel: str) -> str:
    parts = rel[:-3].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def scope_nodes(node: ast.AST):
    """Nodes evaluated in `node`'s own scope: nested scopes are yielded but not entered (their decorators, defaults
    and bases are entered: they run in this scope); comprehensions are entered only for their walrus targets."""
    for child in ast.iter_child_nodes(node):
        yield child
        if isinstance(child, SCOPES):
            outer = [
                *getattr(child, "decorator_list", []),
                *getattr(child, "bases", []),
                *getattr(child, "keywords", []),
            ]
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                outer += [*child.args.defaults, *[d for d in child.args.kw_defaults if d is not None]]
            for o in outer:
                yield o
                yield from scope_nodes(o)
        elif isinstance(child, COMPS):
            for sub in ast.walk(child):
                if isinstance(sub, ast.NamedExpr):
                    yield sub
                    yield from scope_nodes(sub.value)
        else:
            yield from scope_nodes(child)


def bound_names(node: ast.AST) -> set[str]:
    """Names a statement or expression binds in the scope it runs in."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names if a.name != "*"}
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return {node.id}
    if isinstance(node, ast.NamedExpr):
        return {node.target.id}
    if isinstance(node, ast.ExceptHandler) and node.name:
        return {node.name}
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return {node.name}
    if isinstance(node, ast.MatchMapping) and node.rest:
        return {node.rest}
    return set()


class Index:
    """Top-level names of every repo module, by dotted name."""

    def __init__(self, sources: dict[str, str]):
        self.trees: dict[str, tuple[str, ast.Module]] = {}
        for rel, src in sources.items():
            if rel.endswith(".py") and rel.split("/")[0] in PKG_ROOTS:
                try:
                    self.trees[modname(rel)] = (rel, ast.parse(src))
                except SyntaxError:
                    continue
        self.nodes = {m: list(ast.walk(tree)) for m, (_, tree) in self.trees.items()}  # one walk per module
        self.names: dict[str, set[str]] = {}
        self.opaque: set[str] = set()
        self.alias: dict[str, str] = {}
        self.stars: dict[str, list[str]] = {}
        for m, (rel, tree) in self.trees.items():
            self._scan(m, rel, tree)
        for m in list(self.trees):  # namespace packages: every parent directory is importable
            parts = m.split(".")
            for k in range(1, len(parts)):
                self.names.setdefault(".".join(parts[:k]), set())
        for m in list(self.stars):
            self._expand_stars(m, set())
        self._scopes: dict[str, _Scopes] = {}
        self._accessors: dict[tuple[str, str], str | None] = {}

    def scopes(self, m: str) -> _Scopes:
        if m not in self._scopes:
            rel, tree = self.trees[m]
            self._scopes[m] = _Scopes(self, m, rel, tree)
        return self._scopes[m]

    def bound_module(self, m: str, name: str) -> str | None:
        """The repo module that module-level name `m.name` is bound to by imports only (`A.B.x`), else None."""
        m = self.alias.get(m, m)
        if m not in self.trees or m in self.opaque:
            return None
        binds = self.scopes(m).root.binds.get(name) or [None]
        if all(isinstance(b, str) for b in binds) and len(set(binds)) == 1:
            return self.alias.get(binds[0], binds[0])
        return None

    def accessor(self, m: str, name: str) -> str | None:
        """The repo module that module-level function `m.name` returns on every path, else None (also for re-exports
        of such a function through `from ... import`)."""
        m = self.alias.get(m, m)
        key = (m, name)
        if key not in self._accessors:
            self._accessors[key] = None  # recursion guard
            if m in self.trees:
                sc = self.scopes(m)
                self._accessors[key] = sc.called_module(sc.root.binds.get(name) or [None])
        return self._accessors[key]

    def _scan(self, m: str, rel: str, tree: ast.Module) -> None:
        names = self._scope_names(m, rel, tree)
        names |= self._global_names(m)
        if m in self.alias:
            self._resolve_alias(m, rel)
        self.names[m] = names

    def _scope_names(self, m: str, rel: str, tree: ast.Module) -> set[str]:
        """Names bound in the module scope; records star imports and marks a module with `__getattr__` opaque."""
        names: set[str] = set()
        for n in scope_nodes(tree):
            names |= bound_names(n)
            if isinstance(n, ast.FunctionDef) and n.name == "__getattr__":
                self.opaque.add(m)
            if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names):
                src = resolve_from(m, rel, n)
                if src in self.trees or (src and src.split(".")[0] in PKG_ROOTS):
                    self.stars.setdefault(m, []).append(src)
                else:
                    self.opaque.add(m)
        return names

    def _global_names(self, m: str) -> set[str]:
        """Names declared `global` anywhere in the module; marks dynamic namespace writes opaque and records a
        `sys.modules[__name__] = x` alias."""
        names: set[str] = set()
        for n in self.nodes[m]:
            if isinstance(n, ast.Global):
                names.update(n.names)
            elif _writes_module_namespace(n):
                self.opaque.add(m)
            elif (
                isinstance(n, ast.Assign)
                and any(_is_sys_modules_self(t) for t in n.targets)
                and isinstance(n.value, ast.Name)
            ):
                self.alias[m] = n.value.id  # resolved to a module by _resolve_alias, through this module's imports
        return names

    def _resolve_alias(self, m: str, rel: str) -> None:
        """Turn the alias name of `m` into the module it was imported as; an unresolvable alias makes `m` opaque."""
        target = None
        for n in self.nodes[m]:
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    if (a.asname or a.name) == self.alias[m]:
                        target = f"{resolve_from(m, rel, n)}.{a.name}"
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.asname == self.alias[m]:
                        target = a.name
        if target is None:
            self.opaque.add(m)
            del self.alias[m]
        else:
            self.alias[m] = target

    def _expand_stars(self, m: str, seen: set[str]) -> None:
        if m in seen:
            return
        seen.add(m)
        for src in self.stars.pop(m, []):
            self._expand_stars(src, seen)
            src = self.alias.get(src, src)
            if src in self.opaque or src not in self.names:
                self.opaque.add(m)
                continue
            self.names[m] |= {x for x in self.names[src] if not x.startswith("_")} | self._all_of(src)

    def _all_of(self, m: str) -> set[str]:
        if m not in self.trees:
            return set()
        for n in self.trees[m][1].body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets):
                try:
                    return set(ast.literal_eval(n.value))
                except ValueError:
                    return set()
        return set()

    def is_module(self, m: str | None) -> bool:
        return m is not None and m in self.names

    def has(self, m: str, attr: str) -> bool:
        m = self.alias.get(m, m)
        return m in self.opaque or attr in self.names.get(m, set()) or f"{m}.{attr}" in self.names


def _is_sys_modules_self(t: ast.AST) -> bool:
    return (
        isinstance(t, ast.Subscript)
        and isinstance(t.value, ast.Attribute)
        and t.value.attr == "modules"
        and isinstance(t.slice, ast.Name)
        and t.slice.id == "__name__"
    )


def _writes_module_namespace(n: ast.AST) -> bool:
    """globals()[k] = v, globals().update(...), setattr(sys.modules[__name__], ...): names unknowable statically."""
    if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
        return ast.unparse(n.value) == "globals()"
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("update", "setdefault"):
        return ast.unparse(n.func.value) == "globals()"
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" and n.args:
        return ast.unparse(n.args[0]) in ("sys.modules[__name__]", "globals()")
    return False


def resolve_from(cur: str, rel: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    base = cur.split(".")
    if not rel.endswith("__init__.py"):
        base = base[:-1]
    if node.level > 1:
        base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else [])) or None


# ---------------------------------------------------------------- imports and aliases


def import_problems(idx: Index) -> list[tuple[str, int, str, str]]:
    """(file, line, key, message); key = `module:name` (the exemption key)."""
    out = []
    for m, (rel, tree) in idx.trees.items():
        for n in idx.nodes[m]:
            if isinstance(n, ast.ImportFrom):
                src = resolve_from(m, rel, n)
                if src is None:
                    continue
                if not idx.is_module(src):
                    if n.level or src.split(".")[0] in PKG_ROOTS:
                        out.append((rel, n.lineno, f"{src}:", f"from {src} import ... - no such repo module"))
                    continue
                out.extend(
                    (rel, n.lineno, f"{src}:{a.name}", f"from {src} import {a.name} - not defined there")
                    for a in n.names
                    if a.name != "*" and not idx.has(src, a.name)
                )
            elif isinstance(n, ast.Import):
                out.extend(
                    (rel, n.lineno, f"{a.name}:", f"import {a.name} - no such repo module")
                    for a in n.names
                    if a.name.split(".")[0] in PKG_ROOTS and not idx.is_module(a.name)
                )
            elif isinstance(n, ast.Call):
                lit = import_module_target(n)
                if lit and lit.split(".")[0] in PKG_ROOTS and not idx.is_module(lit):
                    out.append((rel, n.lineno, f"{lit}:", f"import_module({lit!r}) - no such repo module"))
        out += _alias_problems(idx, m, rel, tree)
    return out


class _Scope:
    def __init__(self, node: ast.AST, parent: _Scope | None):
        self.node, self.parent = node, parent
        #: name -> per binding: the bound repo module, a function definition (maybe an accessor), or None
        self.binds: dict[str, list] = {}
        self.declared: set[str] = set()  # global / nonlocal: looked up outside

    def add(self, name: str, mod) -> None:
        self.binds.setdefault(name, []).append(mod)


class _Scopes:
    """Every scope of one module with its bindings, and the scope each Name node is evaluated in."""

    def __init__(self, idx: Index, m: str, rel: str, tree: ast.Module):
        self.idx, self.m, self.rel = idx, m, rel
        self.home: dict[int, _Scope] = {}  # id(Name node) -> scope it is evaluated in
        self._acc: dict[int, str | None] = {}  # id(function definition) -> the repo module it returns
        self.root = self._build(tree, None)

    def _build(self, node: ast.AST, parent: _Scope | None) -> _Scope:
        sc = _Scope(node, parent)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                if arg is not None:
                    sc.add(arg.arg, None)
        roots = node.generators if isinstance(node, COMPS) else [node]
        for root in roots:
            for n in scope_nodes(root) if root is node else [root, *scope_nodes(root)]:
                self._visit(n, sc)
        if isinstance(node, COMPS):
            for g in node.generators:
                for t in ast.walk(g.target):
                    if isinstance(t, ast.Name):
                        sc.add(t.id, None)
            for part in ("elt", "key", "value"):
                expr = getattr(node, part, None)
                for n in [expr, *scope_nodes(expr)] if expr is not None else []:
                    if isinstance(n, ast.Name):
                        self.home[id(n)] = sc
                    if isinstance(n, (*SCOPES, *COMPS)):
                        self._build(n, sc)
        return sc

    def _visit(self, n: ast.AST, sc: _Scope) -> None:
        if isinstance(n, ast.Name):
            self.home[id(n)] = sc
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            sc.declared.update(n.names)
        elif isinstance(n, ast.NamedExpr):
            target = sc
            while isinstance(target.node, COMPS):
                target = target.parent
            target.add(n.target.id, None)
        elif isinstance(n, ast.Import):
            for a in n.names:
                name, mod = (a.asname, a.name) if a.asname else (a.name.split(".")[0],) * 2
                sc.add(name, mod if self.idx.is_module(mod) else None)
        elif isinstance(n, ast.ImportFrom):
            src = resolve_from(self.m, self.rel, n)
            for a in n.names:
                if a.name != "*":
                    sub = f"{src}.{a.name}"
                    if self.idx.is_module(sub):
                        sc.add(a.asname or a.name, sub)
                    else:  # maybe an accessor function of a repo module, resolved when called
                        sc.add(a.asname or a.name, (src, a.name) if src in self.idx.trees else None)
        else:
            fn = isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            for name in bound_names(n):
                sc.add(name, n if fn else None)
            if isinstance(n, (*SCOPES, *COMPS)):
                self._build(n, sc)

    @staticmethod
    def lookup(name: str, sc: _Scope | None) -> list[str | None] | None:
        """Bindings of `name` seen from `sc` (Python rules: a class scope is invisible to nested scopes)."""
        first = sc
        while sc is not None:
            if name in sc.declared:
                return None  # global / nonlocal: bound elsewhere, skipped
            if (sc is first or not isinstance(sc.node, ast.ClassDef)) and name in sc.binds:
                return sc.binds[name]
            sc = sc.parent
        return None

    def _one(self, expr: ast.Name, kind: type) -> list | None:
        """The bindings of a name when all of them are of `kind` (str = module, FunctionDef = function)."""
        mods = self.lookup(expr.id, self.home[id(expr)]) if id(expr) in self.home else None
        return mods if mods and all(isinstance(x, kind) for x in mods) else None

    def accessor(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        """The repo module a function returns on every path (`def _m(): from x import y as M; return M`), else None."""
        if id(fn) not in self._acc:
            self._acc[id(fn)] = None  # recursion guard
            rets = [n for n in scope_nodes(fn) if isinstance(n, ast.Return)]
            mods = {self.module_of(r.value) if r.value is not None else None for r in rets}
            self._acc[id(fn)] = mods.pop() if len(mods) == 1 else None
        return self._acc[id(fn)]

    def module_of(self, expr: ast.AST) -> str | None:
        """The repo module `expr` certainly is: a name, a dotted chain, an accessor call or import_module("<lit>")."""
        if isinstance(expr, ast.Name):
            mods = self._one(expr, str)
            return self.idx.alias.get(mods[0], mods[0]) if mods and len(set(mods)) == 1 else None
        if isinstance(expr, ast.Attribute):
            base = self.module_of(expr.value)
            sub = f"{base}.{expr.attr}" if base else None
            if self.idx.is_module(sub):
                return self.idx.alias.get(sub, sub)
            return self.idx.bound_module(base, expr.attr) if base else None  # A.B: B a module that A imports
        if not isinstance(expr, ast.Call):
            return None
        lit = import_module_target(expr)
        if lit is not None:
            if ast.unparse(expr.func) == "__import__":
                lit = lit.split(".")[0]  # __import__("a.b") returns the top package a
            return self.idx.alias.get(lit, lit) if self.idx.is_module(lit) else None
        if isinstance(expr.func, ast.Name) and id(expr.func) in self.home:
            return self.called_module(self.lookup(expr.func.id, self.home[id(expr.func)]) or [None])
        return None

    def called_module(self, binds: list) -> str | None:
        """The module a call of a name with these bindings returns: every binding is a function definition here or a
        function imported from a repo module, all accessors of one module."""
        mods = set()
        for b in binds:
            if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mods.add(self.accessor(b))
            elif isinstance(b, tuple):
                mods.add(self.idx.accessor(*b))
            else:
                return None
        return mods.pop() if len(mods) == 1 else None


def import_module_target(call: ast.Call) -> str | None:
    """The literal absolute target of `importlib.import_module("x")` / `import_module("x")` / `__import__("x")`."""
    if ast.unparse(call.func) not in ("importlib.import_module", "import_module", "__import__"):
        return None
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return None if call.args[0].value.startswith(".") else call.args[0].value
    return None


def _alias_problems(idx: Index, m: str, rel: str, tree: ast.Module) -> list[tuple[str, int, str, str]]:
    scopes = idx.scopes(m)
    stored_attrs = {
        (ast.unparse(n.value), n.attr)
        for n in idx.nodes[m]
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, (ast.Store, ast.Del))
    }
    out = []
    for n in idx.nodes[m]:
        if not isinstance(n, ast.Attribute) or n.attr.startswith("__"):
            continue
        mod = scopes.module_of(n.value)
        if mod and not idx.has(mod, n.attr) and (ast.unparse(n.value), n.attr) not in stored_attrs:
            text = f"{ast.unparse(n.value)}.{n.attr} - {mod} has no {n.attr}"
            out.append((rel, n.lineno, f"{mod}:{n.attr}", text))
    return out


# ---------------------------------------------------------------- the tree


def tree_files(root: Path = ROOT) -> list[str]:
    """Tracked and untracked-but-not-ignored files that exist (LookupError outside a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise LookupError(f"not a git checkout: the file list of {root} is unknown") from e
    return [f for f in out.splitlines() if f and (root / f).is_file()]


def tree_problems(root: Path = ROOT) -> list[tuple[str, int, str, str]]:
    """(file, line, key, message) over the whole tree, before exemptions."""
    files = tree_files(root)
    texts = {}
    for f in files:
        if f.endswith(TEXT_EXT):
            try:
                texts[f] = (root / f).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    present = with_parents(set(files))
    rows = [(f, i, p, f"path {p} does not exist") for f, i, p in missing_paths(texts, present)]
    return rows + import_problems(Index(texts))


def problems_of(sources: dict[str, str]) -> list[tuple[str, int, str, str]]:
    """The same scan over an in-memory tree {relative path: text} (every key counts as present)."""
    present = with_parents(set(sources))
    rows = [(f, i, p, f"path {p} does not exist") for f, i, p in missing_paths(sources, present)]
    return rows + import_problems(Index(sources))


if __name__ == "__main__":
    for row in sorted(tree_problems()):
        sys.stdout.write("\t".join(map(str, row)) + "\n")
