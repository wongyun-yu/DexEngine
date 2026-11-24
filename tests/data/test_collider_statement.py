"""The collider shape of a slot has one owner (collider_shape.shape_of / row_shape): no other module maps a kind or a
row to TRIANGLES / PARTS (structure scan, with planted violations). One clip's collider statement has one fingerprint
(collider_shape.statement_fp) that the env build (train, judge and render) prints in `[TRI_COLL] statement` and stage
2f writes in its ledger: same input, same value; a changed shape or object changes it; both callers are wired."""

from __future__ import annotations

import ast

import pytest

from dexengine.core.repo import REPO_ROOT
from dexengine.data.objects import collider_shape as CS

OWNER = "dexengine/data/objects/collider_shape.py"
SHAPE_NAMES = {"TRIANGLES", "PARTS"}


def _is_shape(n) -> bool:
    return (isinstance(n, ast.Attribute) and n.attr in SHAPE_NAMES) or (isinstance(n, ast.Name) and n.id in SHAPE_NAMES)


def _literal_shape(n: ast.Compare) -> bool:
    """A comparison with the shape strings written out: 'triangles' anywhere, 'parts' only against a shape or collider
    value (other modules use the word 'parts' for other things, e.g. case kinds)."""
    ops = [n.left, *n.comparators]
    lits = {c.value for c in ops if isinstance(c, ast.Constant)}
    named = any(
        w in ast.unparse(c).lower() for c in ops if not isinstance(c, ast.Constant) for w in ("shape", "collider")
    )
    return CS.TRIANGLES in lits or (CS.PARTS in lits and named)


def violations(src: str) -> list[str]:
    """Shape decisions outside the owner: TRIANGLE_KINDS read, a conditional or table that yields TRIANGLES / PARTS,
    a shape returned as a constant, or the shape strings written as literals in a comparison."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if (isinstance(n, ast.Attribute) and n.attr == "TRIANGLE_KINDS") or (
            isinstance(n, ast.Name) and n.id == "TRIANGLE_KINDS"
        ):
            out.append(f"{n.lineno}: TRIANGLE_KINDS read")
        elif isinstance(n, ast.IfExp) and (_is_shape(n.body) or _is_shape(n.orelse)):
            out.append(f"{n.lineno}: conditional shape {ast.unparse(n)[:80]}")
        elif isinstance(n, ast.Dict) and any(_is_shape(v) for v in n.values):
            out.append(f"{n.lineno}: shape table")
        elif isinstance(n, ast.Return) and n.value is not None and _is_shape(n.value):
            out.append(f"{n.lineno}: shape returned as a constant")
        elif isinstance(n, ast.Compare) and _literal_shape(n):
            out.append(f"{n.lineno}: shape literal compared")
    return out


def test_no_module_but_the_owner_decides_a_collider_shape():
    bad = {}
    for p in sorted((REPO_ROOT / "dexengine").rglob("*.py")):
        rel = str(p.relative_to(REPO_ROOT))
        if rel != OWNER and (v := violations(p.read_text(encoding="utf-8"))):
            bad[rel] = v
    assert not bad, bad


@pytest.mark.parametrize(
    "src",
    [
        "shape = _cs.TRIANGLES if row['kind'] == 'receiver' else _cs.PARTS",
        "TABLE = {'receiver': CS.TRIANGLES, 'moving': CS.PARTS}",
        "def f(k):\n    return CS.TRIANGLES",
        "ok = row_shape == 'triangles'",
        "ok = r['shape'] == 'parts'",
        "kinds = CS.TRIANGLE_KINDS",
    ],
)
def test_a_planted_shape_decision_is_caught(src):
    assert violations(src)  # negative: each form of a second owner is found


def test_reading_the_statement_is_not_a_decision():
    assert not violations("tri = [s for s, c in cells if c == TRIANGLES]\nshape = CS.row_shape(ds, row or {})")
    assert not violations("case = 'BI_PARTS_PLUS' if 'parts' in kinds else 'BI_SHARED_PLUS'")  # another 'parts'


ROWS = [
    {"name": "extra_object_0", "kind": "receiver", "static": True, "obj_id": "bowl"},
    {"name": "extra_object_1", "kind": "moving", "static": False, "obj_id": "cup"},
]


def test_the_fingerprint_is_stable_and_order_free(monkeypatch):
    monkeypatch.setattr(CS, "on", lambda ds: True)
    a = CS.statement_fp("oakink2", ROWS)
    assert a.startswith("ts-") and a == CS.statement_fp("oakink2", ROWS[::-1])


def test_the_fingerprint_follows_the_shape_and_the_objects(monkeypatch):
    monkeypatch.setattr(CS, "on", lambda ds: True)
    a = CS.statement_fp("oakink2", ROWS)
    monkeypatch.setattr(CS, "on", lambda ds: False)  # the rule off: the receiver stands as parts
    assert CS.statement_fp("oakink2", ROWS) != a
    monkeypatch.setattr(CS, "on", lambda ds: True)
    other = [dict(ROWS[0], obj_id="plate"), ROWS[1]]
    assert CS.statement_fp("oakink2", other) != a  # negative: another object is another statement


def _calls(path: str, func: str, name: str) -> bool:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    return any(isinstance(n, ast.Call) and ast.unparse(n.func).endswith(name) for n in ast.walk(fn))


@pytest.mark.parametrize(
    "path,func",
    [
        ("dexengine/rl/env_setup/extra_slots_plan.py", "_plan_clip"),
        ("dexengine/placement/repair/hand_clear_stage.py", "tree_rows"),
    ],
)
def test_env_and_stage_2f_both_fingerprint_the_slot_rows(path, func):
    assert _calls(path, func, "statement_fp") and (
        _calls(path, func, "slot_rows") or _calls(path, func, "slot_rows_of")
    )


def callers(name: str, root=REPO_ROOT) -> set[str]:
    """{path::function} of every function under dexengine/ that calls `name` (attribute or alias import resolved)."""
    out = set()
    for p in sorted((root / "dexengine").rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        alias = {
            a.asname or a.name
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)
            for a in n.names
            if a.name == name
        }
        alias |= {name}
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for c in ast.walk(fn):
                if isinstance(c, ast.Call) and ast.unparse(c.func).split(".")[-1] in alias and fn.name != name:
                    out.add(f"{p.relative_to(root)}::{fn.name}")
    return out


def test_one_slot_builder_for_train_judge_and_render():
    """Train (rl/train overrides), judge and render (rl/eval cfg overrides; the record session runs rl.cli) build the
    slots through one dispatcher: build_scene_supports -> build_extra_slots -> the statement print."""
    assert callers("build_extra_slots") == {"dexengine/rl/env_setup/supports.py::build_scene_supports"}
    got = {c.split("::")[0] for c in callers("build_scene_supports")}
    assert got == {"dexengine/rl/train/overrides_scene.py", "dexengine/rl/eval/cfg_overrides.py"}, got
    runner = (REPO_ROOT / "dexengine/record/session/runner.py").read_text(encoding="utf-8")
    assert 'run_module("dexengine.rl.cli"' in runner


def test_a_second_slot_builder_is_caught(tmp_path):
    pkg, mod = tmp_path / "dexengine", "rogue"
    pkg.mkdir()
    (pkg / f"{mod}.py").write_text("from x import build_extra_slots as b\n\ndef f(cfg):\n    b(cfg, None, 'r')\n")
    want = {f"{pkg.name}/{mod}.py::f"}
    assert callers("build_extra_slots", tmp_path) == want  # negative: an aliased second caller is found
