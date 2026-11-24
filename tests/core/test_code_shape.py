"""No function is longer than 80 lines, except the ones listed in EXCEPTIONS with their reason.

Length is what `ast` reports for a def: from the `def` line to its last line, decorators excluded; a nested function
is measured on its own and also counts inside its parent. Every tracked .py file is scanned. An exception that is no
longer over the limit, or no longer exists, fails as well, so the list can only shrink.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAX_LINES = 80

#: (file, function) -> why it is still longer than MAX_LINES.
EXCEPTIONS: dict[tuple[str, str], str] = {}


def long_functions(src: str, path: str = "<src>") -> dict[tuple[str, str], int]:
    """(path, function name) -> line count, for every function longer than MAX_LINES."""
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = node.end_lineno - node.lineno + 1
            if n > MAX_LINES:
                out[(path, node.name)] = max(n, out.get((path, node.name), 0))
    return out


def _tracked_python() -> list[str]:
    r = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True, check=True)
    return r.stdout.split()


def test_no_function_over_80_lines_outside_the_exceptions():
    found = {}
    for f in _tracked_python():
        found.update(long_functions((ROOT / f).read_text(encoding="utf-8"), f))
    new = sorted(f"{p} {name} ({n} lines)" for (p, name), n in found.items() if (p, name) not in EXCEPTIONS)
    assert not new, f"functions over {MAX_LINES} lines:\n" + "\n".join(new)
    stale = sorted(f"{p} {name}" for p, name in EXCEPTIONS if (p, name) not in found)
    assert not stale, "exceptions no longer over the limit (remove them from EXCEPTIONS):\n" + "\n".join(stale)


def test_the_limit_is_80_lines_and_nested_functions_count():
    body = "".join(f"    x{i} = {i}\n" for i in range(79))
    assert long_functions(f"def f():\n{body}") == {}
    assert long_functions(f"def f():\n{body}    x = 0\n") == {("<src>", "f"): 81}
    inner = "".join(f"        y{i} = {i}\n" for i in range(80))
    assert set(long_functions(f"def outer():\n    def inner():\n{inner}")) == {("<src>", "outer"), ("<src>", "inner")}
