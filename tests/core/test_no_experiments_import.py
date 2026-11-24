"""The release package depends on neither the paper code nor a deployment layer: no module of ``dexengine`` imports
``experiments`` or the deployment layer (artifact_paths.OPS_DIR), at module level or inside a function, and no string
names one of their modules for a dynamic import. Experiment variants and a deployment reach the release code through
the plugin hooks
(``dexengine.rl.tasks.hooks``, ``dexengine.record.hooks``) or by passing their rules in (``dexengine.rl.manifest``)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from tests.core.references_paths import SITE

PKG = Path(__file__).resolve().parents[2] / "dexengine"
FORBIDDEN = ("experiments", SITE)
DOTTED = re.compile(r"^({})(\.\w+)+$".format("|".join(FORBIDDEN)))


def forbidden_uses(source: str) -> list[tuple[int, str]]:
    """(line, what) of every import of a forbidden top-level package and every string that is one of its module
    paths (an ``importlib.import_module`` argument or a registry entry)."""
    out = []
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import):
            out += [(n.lineno, f"import {a.name}") for a in n.names if a.name.split(".")[0] in FORBIDDEN]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.split(".")[0] in FORBIDDEN:
            out.append((n.lineno, f"from {n.module} import ..."))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and DOTTED.match(n.value):
            out.append((n.lineno, f"module name {n.value!r}"))
    return out


def test_scanner_catches_each_form():
    src = (
        "import experiments.scoring\n"
        "def f():\n    from experiments.ablations import x\n"
        "M = 'experiments.ablations.placement.place_strategy'\n"
        "from dexengine.core import floating\n"
        "doc = 'see experiments/README.md'\n"
        "from ops.layer import campaign_manifest as CM\n"
        "P = 'ops.layer.record.plugin'\n"
        "from dexengine.core import ops_helpers\n"
    )
    assert sorted(ln for ln, _ in forbidden_uses(src)) == [1, 3, 4, 7, 8]


def test_release_package_does_not_import_paper_code():
    bad = [
        f"{p.relative_to(PKG.parent)}:{ln} {what}"
        for p in sorted(PKG.rglob("*.py"))
        for ln, what in forbidden_uses(p.read_text(encoding="utf-8"))
    ]
    assert bad == []
