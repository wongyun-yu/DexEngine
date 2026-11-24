"""Experiment callers hand RL switches to the launcher as `--set field=value`, never as environment."""

from __future__ import annotations

import ast
from pathlib import Path

from dexengine.core.repo import REPO_ROOT

REAL_KIT = REPO_ROOT / "experiments/real_kit/v4"


def test_real_kit_passes_all_obj_collision_as_a_set_arg():
    for name in ("run.py", "batch.py"):
        assert "--all-obj-collision" in (REAL_KIT / name).read_text(encoding="utf-8")
    tree = ast.parse((REAL_KIT / "run.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "AC.rollout"]
    assert calls and all(any(k.arg == "launch_args" for k in c.keywords) for c in calls)


def test_record_dry_run_puts_launch_args_after_the_run_name(tmp_path, monkeypatch):
    from dexengine.record import bundle

    seen = {}

    class R:
        returncode = 0

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return R()

    monkeypatch.setattr(bundle.subprocess, "run", fake_run)
    bundle.dryrun(Path(tmp_path), ["SIDE=both"], "a1", tmp_path / "d.out", ("--set", "all_obj_collision=1"))
    assert seen["cmd"][-4:] == ["scripts/stage3_rl.sh", "a1", "--set", "all_obj_collision=1"]
    bundle.dryrun(Path(tmp_path), ["SIDE=both"], "a1", tmp_path / "d.out")
    assert seen["cmd"][-2:] == ["scripts/stage3_rl.sh", "a1"]
