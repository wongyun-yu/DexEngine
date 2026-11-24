"""app.bake_kit — the robot USD bake tools start Kit on the plain kit (no RTX renderer).

urdf_to_usd · contact_report_bake started without --headless took the full app kit (isaaclab.python.kit), which
segfaults on some cluster nodes; with headless they bake there. The rule lives in dexengine/rl/app.py (the owner of
the no-render kit); the tools call it before AppLauncher."""

import argparse
import ast
from pathlib import Path

import pytest

from dexengine.rl import app as TK

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ("dexengine/rl/assets/urdf_to_usd.py", "dexengine/rl/assets/contact_report_bake.py")


def test_bake_kit_sets_headless_and_marks(capsys):
    a = argparse.Namespace(headless=False, enable_cameras=False)
    TK.bake_kit(a, "urdf_to_usd")
    assert a.headless is True
    assert TK.BAKE_MARK in capsys.readouterr().out


def test_bake_kit_refuses_cameras():
    with pytest.raises(ValueError, match="never renders"):
        TK.bake_kit(argparse.Namespace(headless=False, enable_cameras=True), "contact_report_bake")


@pytest.mark.parametrize("rel", TOOLS)
def test_tool_calls_bake_kit_before_applauncher(rel):
    """Structure, not a word search: a module-level call to ``*.bake_kit(...)`` precedes the first ``AppLauncher(``
    call — removing the call or moving it after the launch fails here."""
    tree = ast.parse((ROOT / rel).read_text())
    first = {}
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name in ("bake_kit", "AppLauncher"):
                    first.setdefault(name, node.lineno)
    assert "bake_kit" in first, f"{rel}: no bake_kit call"
    assert first["bake_kit"] < first["AppLauncher"], f"{rel}: bake_kit after AppLauncher"
