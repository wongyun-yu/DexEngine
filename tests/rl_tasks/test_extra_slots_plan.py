"""The Kit-free extra-slot plan and the trainer preflight (CPU): the plan imports no Isaac Lab, Omniverse, USD or torch;
the scene build and the preflight call the same plan, spawn-list and off-branch functions; the preflight raises the
scene build's error on a refused clip (PART_UNMERGED) and on a spawn list that differs from the checkpoint's ledger;
the trainer entry runs it after every pre-Kit check and before AppLauncher, and ends there."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from dexengine.rl.cli_entry import preflight as PF
from dexengine.rl.env_setup import extra_slots as ES
from dexengine.rl.env_setup import extra_slots_ledger as EL
from dexengine.rl.env_setup import extra_slots_plan as XP

REPO = Path(__file__).resolve().parents[2]
BLOCKED = ("isaaclab", "isaacsim", "omni", "carb", "pxr", "torch", "warp")
PART_UNMERGED = "[EXTRA_SLOTS] PART_UNMERGED c1: main right_object(o1) is a body part in the spec but stays apart"


def _fresh_ledger(monkeypatch) -> None:
    """Fresh process state of the ledger module; DEXENGINE_EXTRA_SLOTS unset (restored afterwards)."""
    for name in ("_EXTRA_STATE", "_EXTRA_LEDGER_SEEN", "_EXTRA_NO_LEDGER", "_EXTRA_OFF_BUILT"):
        monkeypatch.setattr(EL, name, {} if name == "_EXTRA_STATE" else [])
    monkeypatch.setenv(EL.extra_slots_owner().ENV, "")
    monkeypatch.setenv("DEXENGINE_DATASET", "hot3d")


def _tree(root: Path, *clips: str) -> Path:
    """A tree folder whose clips have no extra objects (the rule reads task_info only)."""
    for c in clips:
        (root / c).mkdir(parents=True)
        (root / c / "task_info.json").write_text(json.dumps({"dataset_name": "hot3d", "objects": []}))
    return root


def _args(seq_dir, sequence="c1", task="train", **kw):
    base = {"resume_from": None, "resume": None, "checkpoint": None}
    return types.SimpleNamespace(task=task, sequence_dir=str(seq_dir), sequence=sequence, **(base | kw))


# ── imports ────────────────────────────────────────────────────────────────────────────────────────────────────────

_PROBE = """
import importlib.abc, json, sys
tried = []
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in {blocked!r}:
            tried.append(name)
            raise ImportError("blocked " + name)
sys.meta_path.insert(0, Block())
from dexengine.rl.env_setup import extra_slots_plan as XP
plan = XP.plan_rows(sys.argv[1], "c1")
state = XP.spawn_state(plan, plan.rule())
seen = sorted(set(tried) | {{m for m in sys.modules if m.partition(".")[0] in {blocked!r}}})
try:
    import {probe}
except ImportError:
    pass
print(json.dumps({{"clips": list(state["clips"]), "heavy": seen, "probe": sorted(set(tried))}}))
"""


def _probe(tmp_path, probe: str) -> dict:
    code = _PROBE.format(blocked=BLOCKED, probe=probe)
    env = {"PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "DEXENGINE_DATASET": "hot3d"}
    env["DEXENGINE_MAIN_ROOT"] = str(tmp_path)
    r = subprocess.run(
        [sys.executable, "-c", code, str(_tree(tmp_path / "t", "c1"))],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_plan_imports_and_runs_without_isaac_or_torch(tmp_path):
    out = _probe(tmp_path, "dexengine.rl.tasks.scene.extra_slots")  # a module that imports torch at its top
    assert out["clips"] == ["c1"] and out["heavy"] == []
    assert "torch" in out["probe"]  # negative: the watch sees a torch import (a plan that tried one would fail)


def test_the_preflight_module_is_stdlib_only_at_import():
    code = (
        "import sys, dexengine.rl.cli_entry.preflight; print(sorted(k for k in ('numpy', 'torch') if k in sys.modules))"
    )
    env = {"PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "[]", r.stderr[-2000:]


# ── one owner: the scene build and the preflight call the same functions ──────────────────────────────────────────


class _Called(Exception):
    pass


def _fake_isaac(monkeypatch):
    """Stand-ins for the Isaac imports of `_plan_extra_slot_rows` (this interpreter has no Isaac Lab)."""
    sensors = types.ModuleType("isaaclab.sensors")
    sensors.ContactSensorCfg = object
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", sensors)
    monkeypatch.setitem(sys.modules, "dexengine.rl.tasks.mdp.terms", types.ModuleType("dexengine.rl.tasks.mdp.terms"))


def test_scene_build_and_preflight_plan_through_one_function(monkeypatch, tmp_path):
    _fresh_ledger(monkeypatch)
    seen = []

    def plan_rows(seq_dir, sequence):
        seen.append((seq_dir, sequence))
        raise _Called

    monkeypatch.setattr(XP, "plan_rows", plan_rows)
    _fake_isaac(monkeypatch)
    args = _args(tmp_path, "a,b")
    with pytest.raises(_Called):
        ES._plan_extra_slot_rows(types.SimpleNamespace(), args)
    with pytest.raises(_Called):
        PF.run(args)
    assert seen == [(str(tmp_path), "a,b")] * 2


def test_scene_build_and_preflight_share_the_off_branch(monkeypatch, tmp_path):
    _fresh_ledger(monkeypatch)
    seen = []
    monkeypatch.setattr(XP, "slots_off", lambda args_cli, ctx: seen.append(ctx))
    monkeypatch.setenv(EL.extra_slots_owner().ENV, "0")
    ES.build_extra_slots(None, _args(tmp_path), "train_cfg_overrides")
    assert PF.run(_args(tmp_path)).startswith(PF.PASS)
    assert seen == ["train_cfg_overrides", "preflight train"]


def _calls(func: ast.FunctionDef, name: str) -> bool:
    return any(isinstance(n, ast.Call) and ast.unparse(n.func) == name for n in ast.walk(func))


def _func(src: str, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize(
    "path,func",
    [("dexengine/rl/env_setup/extra_slots.py", "build_extra_slots"), ("dexengine/rl/cli_entry/preflight.py", "run")],
)
def test_both_record_the_spawn_list_through_one_function(path, func):
    src = (REPO / path).read_text(encoding="utf-8")
    assert _calls(_func(src, func), "XP.record_spawn_list")
    broken = src.replace("XP.record_spawn_list(", "XP.spawn_state(")
    assert not _calls(_func(broken, func), "XP.record_spawn_list")  # negative: a second path is seen


# ── the preflight refuses what the scene build refuses ─────────────────────────────────────────────────────────────


def test_preflight_raises_the_part_unmerged_error(monkeypatch, tmp_path):
    _fresh_ledger(monkeypatch)

    def slot_rows(tree, ti):
        raise ValueError(PART_UNMERGED)

    monkeypatch.setattr(XP._xs, "slot_rows", slot_rows)
    with pytest.raises(ValueError) as e:
        PF.run(_args(_tree(tmp_path / "t", "c1")))
    assert str(e.value) == PART_UNMERGED


def _run_with_ledger(tmp_path, clip_line: str) -> str:
    seq = _tree(tmp_path / "t", "c1")
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    ck = run / "checkpoints" / "agent_1.pt"
    ck.touch()
    rule = XP.SlotPlan(["c1"], dsets={"hot3d"}).rule()
    (run / EL.EXTRA_SLOTS_LEDGER).write_text(f"1\nrule\t{rule}\n{clip_line}\n")
    return PF.run(_args(seq, resume_from=str(ck)))


def test_preflight_passes_on_the_training_spawn_list(monkeypatch, tmp_path):
    _fresh_ledger(monkeypatch)
    sig = EL.extra_slot_signature([])  # a clip without extras
    out = _run_with_ledger(tmp_path, f"clip\tc1\t{sig}")
    assert out.startswith(PF.PASS) and "extra slots on · clips 1" in out
    assert EL._EXTRA_LEDGER_SEEN and EL._EXTRA_STATE["clips"] == {"c1": sig}


def test_preflight_raises_on_a_spawn_list_that_differs_from_training(monkeypatch, tmp_path):
    _fresh_ledger(monkeypatch)
    with pytest.raises(SystemExit, match=r"spawn lists of 1/1 clips differ from training"):
        _run_with_ledger(tmp_path, "clip\tc1\t1/0/0:000000000000")


def test_a_checkpoint_not_written_yet_is_not_read_as_trained_off(monkeypatch, tmp_path):
    """A dry run ahead of its checkpoint: the plan runs under the default switch (on) and nothing is written from a
    ledger that cannot exist. Read as a ledger-less checkpoint it would have stated slots off and skipped the plan."""
    _fresh_ledger(monkeypatch)
    ck = tmp_path / "run" / "checkpoints" / "agent_1.pt"
    out = PF.run(_args(_tree(tmp_path / "t", "c1"), resume_from=str(ck)))
    assert "extra slots on · clips 1" in out and f"checkpoint {ck} absent — ledger not compared" in out
    assert os.environ[EL.extra_slots_owner().ENV] == "" and not EL._EXTRA_LEDGER_SEEN and not EL._EXTRA_NO_LEDGER
    ck.parent.mkdir(parents=True)
    ck.touch()  # negative: the same path with a file and no ledger is a checkpoint trained with slots off
    _fresh_ledger(monkeypatch)
    out = PF.run(_args(tmp_path / "t", resume_from=str(ck)))
    assert "extra slots off" in out and os.environ[EL.extra_slots_owner().ENV] == "0"


def test_preflight_follows_the_checkpoint_of_each_task(tmp_path):
    ck = str(tmp_path / "r" / "checkpoints" / "a.pt")
    assert EL.ledger_checkpoint(_args(tmp_path, resume_from=ck)) == ck
    assert EL.ledger_checkpoint(_args(tmp_path, resume=ck)) == ck
    assert EL.ledger_checkpoint(_args(tmp_path, task="eval", checkpoint="x.pt")) == os.path.abspath("x.pt")
    assert EL.ledger_checkpoint(_args(tmp_path, task="eval")) is None


@pytest.mark.parametrize("value,want", [("", False), ("0", False), ("1", True), (" 1 ", True)])
def test_flag_values(value, want):
    assert PF.requested({PF.ENV: value}) is want
    assert PF.requested({}) is False


def test_an_unknown_flag_value_fails():
    with pytest.raises(SystemExit, match="only 0|1"):
        PF.requested({PF.ENV: "yes"})


# ── the trainer entry ──────────────────────────────────────────────────────────────────────────────────────────────


def _entry_order(src: str) -> list[str]:
    """Module-level statements of the entry around the preflight, as tags in source order."""
    tags = []
    for n in ast.parse(src).body:
        text = ast.unparse(n)
        if text == "_preapp.assert_clean()":
            tags.append("assert_clean")
        elif isinstance(n, ast.If) and ast.unparse(n.test) == "_preflight.requested()":
            body = [ast.unparse(s) for s in n.body]
            tags.append("preflight" if body == ["print(_preflight.run(args_cli), flush=True)", "sys.exit(0)"] else "?")
        elif text == "app_launcher = AppLauncher(args_cli)":
            tags.append("app_launcher")
    return tags


def test_the_entry_runs_the_preflight_last_before_kit_and_ends():
    src = (REPO / "dexengine/rl/cli.py").read_text(encoding="utf-8")
    assert _entry_order(src) == ["assert_clean", "preflight", "app_launcher"]
    body = [ast.unparse(n) for n in ast.parse(src).body]
    i = body.index("app_launcher = AppLauncher(args_cli)")
    assert body[i - 1].startswith("if _preflight.requested():")  # nothing between the preflight and Kit
    # negatives: a preflight after Kit, or one that falls through to AppLauncher, is seen
    pre = src.index("if _preflight.requested():")
    block = src[pre : src.index("app_launcher = AppLauncher(args_cli)")]
    moved = src.replace(block, "").replace(
        "simulation_app = app_launcher.app\n", "simulation_app = app_launcher.app\n" + block
    )
    assert _entry_order(moved) == ["assert_clean", "app_launcher", "preflight"]
    assert _entry_order(src.replace("    sys.exit(0)\n\napp_launcher", "\napp_launcher")) == [
        "assert_clean",
        "?",
        "app_launcher",
    ]
