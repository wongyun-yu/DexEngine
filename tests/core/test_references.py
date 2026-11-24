"""Cross-references inside the repo stay valid.

1. A repo-relative file path quoted in a text file (`dexengine/<package>/<module>.py`, `scripts/<name>.sh`,
   `assets/<dir>/<name>.json`, ...) names a file or directory of this tree.
2. `from <repo module> import name` names something that module defines (or a submodule), and an imported repo
   module exists.
3. `<alias>.attr`, where `alias` is bound to a repo module by an import, names something that module defines. An alias
   is resolved in its own scope (function, class, lambda, comprehension) by Python's lookup rules; a name bound to
   anything other than one repo module in that scope is ambiguous and skipped. Chains follow submodules and
   module-level imports of the module (`A.B.name`: B a module that A's module imports).
4. `_acc().attr`, where `_acc` is an accessor (a function that imports a repo module in its body and returns it on
   every path, here or imported from another repo module), names something that module defines; so does
   `importlib.import_module("<repo module>").attr`, and a literal `import_module` target exists. Function-local
   imports are checked like module-level ones (2). A `sys.modules[__name__] = X` shim counts as X everywhere.

Two lists exempt rows. Both key a row by (file, reference), never by line number:
- `REFERENCES_EXTERNAL` (permanent): the path resolves in another tree (an external checkout, a pinned commit, a raw
  dataset root) or is a value stored on disk that readers compare. A row that matches nothing fails the test.
- `REFERENCES_PENDING` (temporary): stale rows not yet fixed (files pinned as cache keys among them).
  Empty it as the rows are fixed; a row that matches nothing only warns.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests.core.references_scan import ROOT, problems_of, tree_problems

REFERENCES_EXTERNAL = Path(__file__).with_name("references_external.tsv")
#: The same exemptions for the files of the deployment layer (read only where that layer is present).
REFERENCES_EXTERNAL_SITE = ROOT / "tests" / "site" / "references_external.tsv"
REFERENCES_PENDING = Path(__file__).with_name("references_pending.tsv")
#: Reference targets the deployment layer keeps (read only where that layer is present).
SITE_TARGETS = ROOT / "tests" / "site" / "references_site_targets.txt"


def read_rows(path: Path) -> dict[tuple[str, str], str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            f, ref, *rest = line.split("\t")
            rows[(f, ref)] = "\t".join(rest)
    return rows


def site_targets() -> set[str]:
    if not SITE_TARGETS.is_file():
        return set()
    return {ln for ln in SITE_TARGETS.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")}


def external_rows() -> dict[tuple[str, str], str]:
    """The permanent exemptions: this list, and the deployment layer's list where that layer is present."""
    rows = read_rows(REFERENCES_EXTERNAL)
    if REFERENCES_EXTERNAL_SITE.is_file():
        rows |= read_rows(REFERENCES_EXTERNAL_SITE)
    return rows


@pytest.fixture(scope="module")
def problems():
    try:
        return tree_problems(ROOT)
    except LookupError as e:
        pytest.skip(str(e))


def _fmt(rows) -> str:
    return "\n".join(f"{f}:{i}: {msg}" for f, i, _, msg in sorted(rows)[:80])


def test_references_resolve(problems):
    exempt = set(external_rows()) | set(read_rows(REFERENCES_PENDING))
    bad = [r for r in problems if (r[0], r[2]) not in exempt and r[2] not in site_targets()]
    assert not bad, (
        f"{len(bad)} dangling references. Fix them. Only a reference that resolves in another tree (external "
        f"checkout, pinned commit, raw data root) or a stored value goes to {REFERENCES_EXTERNAL.name}; a stale "
        f"reference owned by unfinished work goes to {REFERENCES_PENDING.name} (rows = python -m "
        f"tests.core.references_scan):\n{_fmt(bad)}"
    )


def test_external_rows_still_match(problems):
    hit = {(r[0], r[2]) for r in problems}
    stale = sorted(set(external_rows()) - hit)
    assert not stale, f"{REFERENCES_EXTERNAL.name} rows that match nothing (remove them): {stale}"


def test_pending_rows_still_match(problems):
    hit = {(r[0], r[2]) for r in problems}
    stale = sorted(set(read_rows(REFERENCES_PENDING)) - hit)
    if stale:
        warnings.warn(f"{REFERENCES_PENDING.name}: {len(stale)} rows are fixed, remove them: {stale}", stacklevel=1)


# ---------------------------------------------------------------- the checker itself (synthetic trees)

_PKG = "dexengine"
_GONE = "/".join([_PKG, "nowhere", "gone" + ".py"])  # built by parts: this file must not name it literally


def test_negative_bad_path_and_bad_import_are_caught():
    src = {
        f"{_PKG}/__init__.py": "",
        f"{_PKG}/a.py": "X = 1\n\ndef f():\n    return 2\n",
        f"{_PKG}/real.py": "Q = 3\n",
        f"{_PKG}/shim.py": f"import sys\nfrom {_PKG} import real as _r\nsys.modules[__name__] = _r\n",
        f"{_PKG}/b.py": f'from {_PKG}.a import X, Y\nRUN = "{_GONE}"\n',
        f"{_PKG}/c.py": f"from {_PKG} import a as A\n\ndef g():\n    return A.f() + A.Z\n",
        f"{_PKG}/d.py": f"import {_PKG}.zz\n",
        # function-local import, accessors (local, imported, through the sys.modules shim), import_module literal
        f"{_PKG}/e.py": (
            "import importlib\n"
            "def lazy():\n"
            f"    from {_PKG}.a import W\n"
            "    return W\n"
            "def _a():\n"
            f"    from {_PKG} import a as M\n"
            "    return M\n"
            "def _s():\n"
            f"    from {_PKG} import shim as S\n"
            "    return S\n"
            "def use():\n"
            f"    return _a().X + _a().V + _s().Q + _s().P + importlib.import_module('{_PKG}.nope').R\n"
        ),
        f"{_PKG}/f.py": f"from {_PKG}.e import _a\n\ndef use():\n    return _a().U\n",
        f"{_PKG}/g.py": f"from {_PKG} import a as AA\n",  # A.B.name: B imported by A's module
        f"{_PKG}/h.py": f"from {_PKG} import g as G\n\ndef use():\n    return G.AA.X + G.AA.T\n",
    }
    keys = {(r[0], r[2]) for r in problems_of(src)}
    assert keys == {
        (f"{_PKG}/b.py", f"{_PKG}.a:Y"),
        (f"{_PKG}/b.py", _GONE),
        (f"{_PKG}/c.py", f"{_PKG}.a:Z"),
        (f"{_PKG}/d.py", f"{_PKG}.zz:"),
        (f"{_PKG}/e.py", f"{_PKG}.a:W"),
        (f"{_PKG}/e.py", f"{_PKG}.a:V"),
        (f"{_PKG}/e.py", f"{_PKG}.real:P"),
        (f"{_PKG}/e.py", f"{_PKG}.nope:"),
        (f"{_PKG}/f.py", f"{_PKG}.a:U"),
        (f"{_PKG}/h.py", f"{_PKG}.a:T"),
    }


def test_known_false_positives_stay_quiet():
    src = {
        f"{_PKG}/__init__.py": "",
        f"{_PKG}/rules.py": "STEPS = [1]\nif _miss := [s for s in STEPS if s > 9]:\n    raise SystemExit(_miss)\n",
        f"{_PKG}/m1.py": "A = 1\n",
        f"{_PKG}/m2.py": "B = 2\n",
        f"{_PKG}/real.py": "Q = 3\n",
        f"{_PKG}/shim.py": f"import sys\nfrom {_PKG} import real as _r\nsys.modules[__name__] = _r\n",
        f"{_PKG}/use.py": (
            f"from {_PKG}.rules import _miss\n"
            f"from {_PKG}.shim import Q\n"
            "def f():\n"
            f"    from {_PKG} import m1 as m\n"
            "    return m.A\n"
            "def g():\n"
            f"    from {_PKG} import m2 as m\n"
            "    return m.B\n"
            "def h(m):\n"
            "    return m.C\n"
            "def k():\n"
            f"    from {_PKG} import m1 as M\n"
            "    M = None\n"
            "    return M.Z\n"
            "def either(x):\n"  # returns one of two modules: not an accessor
            f"    from {_PKG} import m1, m2\n"
            "    if x:\n"
            "        return m1\n"
            "    return m2\n"
            "def u():\n"
            "    return either(1).A + either(0).B\n"
            "def scoped():\n"
            "    def either():\n"  # a local function of the same name shadows the module-level one
            "        return 0\n"
            "    return either().real\n"
        ),
    }
    assert problems_of(src) == []
