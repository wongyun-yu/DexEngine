"""Log and exit texts another program reads, pinned writer -> reader.

record/driver_spec.bundle_rc labels a collect failure rc 12 when its root line carries core.errors.INPUT_MISSING; the
task-side exits that it must classify raise InputMissingError. scripts/rl/judge_sidecar.sh compares the
collision_presets.collider_line output between training and judge logs. Each pair is pinned here.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = [ROOT / "dexengine" / "rl" / p for p in ("tasks", "env_setup", "assets")]

#: (file, function) of every exit in scope that the collect classification reads as "input missing" (rc 12).
INPUT_MISSING_EXITS = {
    ("dexengine/rl/env_setup/extra_slots_ledger.py", "_by_clip_name"),
    ("dexengine/rl/env_setup/extra_slots_ledger.py", "_ledger_compare"),
    ("dexengine/rl/env_setup/extra_slots_ledger.py", "_authored_line"),
    ("dexengine/rl/env_setup/extra_slots_ledger.py", "extra_slots_ledger_check"),
    ("dexengine/rl/env_setup/extra_slots_ledger.py", "_scene_without_ledger"),
    ("dexengine/rl/env_setup/objtrack.py", "auto_passive_sides"),
    ("dexengine/rl/tasks/mdp/actions/contract.py", "_abduction_regex"),
    ("dexengine/rl/tasks/mdp/actions/contract.py", "_apply_effort_anneal"),
    ("dexengine/rl/tasks/scene/all_obj_collision.py", "assert_all_obj_ready"),
}


def _text(node) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    if isinstance(node, ast.BinOp):
        return _text(node.left) + _text(node.right)
    return "{}"


def _input_missing_raisers():
    for d in SCOPE:
        for p in sorted(d.rglob("*.py")):
            for fn in ast.walk(ast.parse(p.read_text())):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(n, ast.Raise)
                    and isinstance(n.exc, ast.Call)
                    and getattr(n.exc.func, "id", "") == "InputMissingError"
                    for n in ast.walk(fn)
                ):
                    yield str(p.relative_to(ROOT)), fn.name


def test_input_missing_exits_keep_their_classification():
    got = set(_input_missing_raisers())
    assert got == INPUT_MISSING_EXITS, (
        f"collect classification changed — missing {sorted(INPUT_MISSING_EXITS - got)}, "
        f"new {sorted(got - INPUT_MISSING_EXITS)}"
    )


def test_input_missing_error_text_is_what_bundle_rc_reads():
    from dexengine.core.errors import INPUT_MISSING, InputMissingError

    e = InputMissingError("[EXTRA_SLOTS] ctx: scene check — the tree of clip c is not in /s")
    assert isinstance(e, SystemExit) and str(e.code).startswith(f"[EXTRA_SLOTS] {INPUT_MISSING} ")
    # the bundle-side reading of this text is pinned in tests/record/test_record_units.py (log -> reason -> rc 12)


def _sidecar_collider_payload(log: str) -> str:
    """judge_sidecar.sh's own coll_payload function (used for the training and the judge log) on one log text."""
    import tempfile

    src = (ROOT / "scripts/rl/judge_sidecar.sh").read_text()
    m = re.search(r"^coll_payload\(\) \{.*?^\}", src, re.S | re.M)
    assert m and 'xcj=$(coll_payload "$log")' in src, "judge_sidecar.sh no longer compares the collider lines"
    with tempfile.NamedTemporaryFile("w", suffix=".log") as f:
        f.write(log)
        f.flush()
        return subprocess.run(
            ["bash", "-c", m.group(0) + '\ncoll_payload "$1"', "_", f.name], capture_output=True, text=True
        ).stdout.strip()


def test_collider_line_is_compared_by_the_sidecar(monkeypatch):
    """collision_authoring prints collision_presets.collider_line; the sidecar compares its statement part between the
    training and the judge log (counts dropped)."""
    from dexengine.rl.tasks.scene import collision_presets as CP

    kinds = ["moving", "touched", "receiver"]
    train = CP.collider_line(
        kinds,
        {"moving": 2, "touched": 0, "receiver": 1, "empty": 0},
        {"moving": 2, "touched": 1, "receiver": 1, "empty": 3},
    )
    judge = CP.collider_line(
        kinds,
        {"moving": 4, "touched": 0, "receiver": 2, "empty": 0},
        {"moving": 4, "touched": 2, "receiver": 2, "empty": 6},
    )
    assert train.startswith(CP.COLLIDER_MARK)
    pt, pj = _sidecar_collider_payload(f"x\n{train}\n"), _sidecar_collider_payload(f"{judge}\n")
    assert pt and pt == pj
    monkeypatch.setattr(CP, "extra_has_collider", lambda k: k == "touched")  # a different statement
    other = CP.collider_line(
        kinds,
        {"moving": 2, "touched": 0, "receiver": 1, "empty": 0},
        {"moving": 2, "touched": 1, "receiver": 1, "empty": 3},
    )
    assert _sidecar_collider_payload(other + "\n") != pt


def test_held_side_values_match_the_stored_episode_meta():
    """held_side goes into the collected episode meta (record.store.meta); stored values 'unknown' and 'none'."""
    src = (ROOT / "dexengine/rl/tasks/scene/extra_slots.py").read_text()
    assert 'self.held_side[c][k] = "unknown"' in src and '"+".join(sorted(h)) or "none"' in src


def test_runtime_missing_rows_keep_rc12(tmp_path, monkeypatch):
    """Exits whose text said 'missing' only through a filled-in value (no ledger row · no checkpoint) stay rc 12."""
    import pytest

    from dexengine.core.errors import is_input_missing
    from dexengine.rl import acceptance as ACC

    rows = [
        {"axis": "a", "clip": "c1", "state": "pending", "ckpt_rel": "x"},
        {
            "axis": "a",
            "clip": "c2",
            "state": "canon",
            "ckpt_rel": "missing.pt",
            "campaign": "k",
            "bundle": "b",
            "run": "r",
        },
    ]
    monkeypatch.setattr(ACC, "read_ledger", lambda path: rows)
    for clips, want in (([("a", "c1")], False), ([("a", "c1"), ("a", "c0")], True), ([("a", "c2")], True)):
        with pytest.raises(SystemExit) as e:
            ACC.use_rows(tmp_path / "l.tsv", clips, tmp_path)
        assert is_input_missing(str(e.value.code)) is want, (clips, e.value.code)
