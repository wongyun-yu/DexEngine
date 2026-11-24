"""The f0 repair split: every module whose code reads the global HC is rebound by share_hand_cfg, so main_steps'
per-run hand (--hand/--arm) reaches all of them."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.imports import own_module

F0 = Path(__file__).resolve().parents[2] / "dexengine/retarget/f0"


def _reads_global_hc(path: Path) -> bool:
    """True when a function or method in the file loads the name HC without binding it locally."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        local |= {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        local -= {name for n in ast.walk(fn) if isinstance(n, ast.Global) for name in n.names}
        if "HC" not in local and any(
            isinstance(n, ast.Name) and n.id == "HC" and isinstance(n.ctx, ast.Load) for n in ast.walk(fn)
        ):
            return True
    return False


def _declared_readers() -> set[str]:
    """repair_cfg.HC_READERS read from the source (importing repair_cfg needs the active hand and arm)."""
    tree = ast.parse((F0 / "repair_cfg.py").read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "HC_READERS" for t in n.targets):
            return set(ast.literal_eval(n.value))
    raise AssertionError("repair_cfg.HC_READERS not found")


def test_reader_scan_flags_global_reads_only(tmp_path):
    reads = tmp_path / "reads.py"
    reads.write_text("def f():\n    return HC.name\n")
    local = tmp_path / "local.py"
    local.write_text("def f(HC):\n    return HC.name\n\n\ndef g():\n    HC = 1\n    return HC\n")
    rebinds = tmp_path / "rebinds.py"
    rebinds.write_text("def f():\n    global HC\n    HC = 2\n    return HC\n")
    assert _reads_global_hc(reads) and _reads_global_hc(rebinds)
    assert not _reads_global_hc(local)


def test_hc_readers_list_matches_the_code():
    readers = {f"dexengine.retarget.f0.{p.stem}" for p in sorted(F0.glob("*.py")) if _reads_global_hc(p)}
    readers.discard("dexengine.retarget.f0.repair")  # the entry rebinds its own HC (global HC in main_steps)
    assert readers == _declared_readers()


def test_share_hand_cfg_rebinds_every_reader(monkeypatch):
    monkeypatch.setenv("DEXENGINE_HAND", "sharpa_wave")  # repair_cfg resolves the active hand and arm at import
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    C = own_module("dexengine.retarget.f0.repair_cfg")
    import importlib

    mods = [importlib.import_module(n) for n in C.HC_READERS]
    for m in mods:
        monkeypatch.setattr(m, "HC", m.HC)  # restored after the test
    sentinel = object()
    C.share_hand_cfg(sentinel)
    assert all(m.HC is sentinel for m in mods)
