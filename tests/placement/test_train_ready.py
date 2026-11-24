"""train_ready (the trainability gate) — its entry runs, every owner attribute it reads exists, and the row rules.

The attribute check is the guard earlier versions lacked: train_ready was deleted while ops still ran it, and
then the owner symbols only it read (static_desk.assess · input_clip.base_part) were deleted as dead.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "dexengine/placement/train_ready"

#: Owner modules imported lazily inside functions (alias → module), checked like the top-level ones.
LAZY = {
    "C": "dexengine.placement.coords.core",
    "SD": "dexengine.placement.scene.static_desk",
    "PR": "dexengine.placement.scene.place_rule",
    "COL": "dexengine.placement.collision.idle_arm",
    "TC": "dexengine.data.candidates",
    "artic_ranges": "dexengine.data.artic.ranges",
}


def _aliases(tree: ast.AST) -> dict[str, str]:
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("dexengine"):
            for a in n.names:
                out[a.asname or a.name] = f"{n.module}.{a.name}"
    return out


def test_every_owner_attribute_exists():
    missing = []
    for f in sorted(PKG.glob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        aliases = {**LAZY, **_aliases(tree)}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                mod = importlib.import_module(aliases[n.value.id])
                if not hasattr(mod, n.attr):
                    missing.append(f"{f.name}: {aliases[n.value.id]}.{n.attr}")
    assert not missing, missing


def test_entry_help_runs():
    r = subprocess.run(
        [sys.executable, "-m", "dexengine.placement.train_ready", "--help"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert "--stale-out" in r.stdout and "--json-clips" in r.stdout


def test_row_rules():
    from dexengine.placement import fold_rules
    from dexengine.placement.ledger_states import JUDGED_STATUSES
    from dexengine.placement.train_ready import assets as A

    assert A.t5j_scope("ADOPTED") and all(A.t5j_scope(s) for s in JUDGED_STATUSES)
    assert not A.t5j_scope("NO_CANDIDATE")
    excluded, route = {("d", "x")}, {("d", "r")}
    assert A.first_pass_out("d", "x", True, excluded, route)  # a real exclusion drops it on every axis
    assert A.first_pass_out("d", "r", False, excluded, route)  # a route clip is out on the canonical axis
    assert not A.first_pass_out("d", "r", True, excluded, route)  # and is the variant axis' target
    assert A.is_carry_note(fold_rules.CARRY_NOTE_HEAD + "wuji2:same)") and not A.is_carry_note("")
    assert A.roles_of({"objects": [{"name": "b", "role": "main", "side": "right"}, {"name": "a"}]}) == [
        ("a", None, None),
        ("b", "main", "right"),
    ]
