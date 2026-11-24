"""PhysX GPU buffer overflow = dropped contacts = an invalid run: launch/process counts the lines of every finished
process ([PHYSX_OVERFLOW]) and turns its result into PHYSX_OVERFLOW_RC; the bimanual training path fails such a run
even with a final checkpoint, never resumes it, and counts every segment log of the run. The collision stack size is
one rule (physx_buffers.collision_stack, by env count) written once by rl/cli and read back on the stage
([PHYSX_BUFFERS])."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from dexengine.core.repo import REPO_ROOT
from dexengine.rl.launch import process, training
from dexengine.rl.tasks.scene import physx_buffers as PB

LINE = (
    "2026-11-20T01:50:16Z [454,032ms] [Error] [omni.physx.plugin] PhysX error: "
    "PxGpuDynamicsMemoryConfig::collisionStackSize buffer overflow detected, please increase its size to at least "
    "71808448 in the scene desc! Contacts have been dropped.\n"
)


def _fn(path: str, name: str, tree: ast.AST | None = None) -> ast.FunctionDef:
    tree = tree or ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_overflow_lines_make_the_process_invalid(tmp_path, capsys):
    log = tmp_path / "r_train.log"
    log.write_text("  2%|1 | 1/64\n" + LINE + LINE.replace("collisionStackSize", "maxRigidContactCount") + LINE)
    assert process.physx_overflow(log) == {
        "PxGpuDynamicsMemoryConfig::collisionStackSize": 2,
        "PxGpuDynamicsMemoryConfig::maxRigidContactCount": 1,
    }
    assert process.overflow_verdict(log, 0) == process.PHYSX_OVERFLOW_RC
    out = capsys.readouterr().out
    assert out.startswith("[PHYSX_OVERFLOW] 3 lines") and "run INVALID" in out


def test_a_clean_log_keeps_the_trainer_rc(tmp_path, capsys):
    log = tmp_path / "r_train.log"
    log.write_text(
        "[Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setLinearVelocity: Body must be "
        "non-kinematic!\n  Training Complete\n"
    )
    assert process.physx_overflow(log) == {}  # negative: another PhysX error is not an overflow
    assert process.overflow_verdict(log, 0) == 0 and process.overflow_verdict(log, 8) == 8
    assert process.physx_overflow(tmp_path / "missing.log") == {}
    assert "[PHYSX_OVERFLOW] 0 lines {}" in capsys.readouterr().out


def _returns_verdict(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Return)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", "") == "overflow_verdict"
        for n in ast.walk(fn)
    )


def test_every_process_result_goes_through_the_verdict():
    start = _fn("dexengine/rl/launch/process.py", "start")
    assert _returns_verdict(start)
    start.body = [s for s in start.body if "overflow_verdict" not in ast.unparse(s)]
    assert not _returns_verdict(start)  # negative: the check removed is caught


# ---- the bimanual training path on fakes (no trainer): what it does with an overflowed run


def _bi(tmp_path, monkeypatch, *, rc: int, logs: dict, ckpt: bool):
    """Run training._train_bi with a fake trainer of result `rc` that writes `logs` ({name: text}) into the refbase
    folder and (if `ckpt`) the final checkpoint. Returns (rc, summary statuses, _auto_resume calls)."""
    ref, ck = tmp_path / "ref", tmp_path / "ck" / "agent_64.pt"
    ref.mkdir()
    ck.parent.mkdir()
    rows, resumed = [], []

    def start(_launch, _root, _dry):
        for name, text in logs.items():
            (ref / name).write_text(text)
        if ckpt:
            ck.touch()
        return rc

    req = SimpleNamespace(run="R1", root=tmp_path, run_tree=tmp_path, dry_run=False, gpu="0", path="bi")
    plan = SimpleNamespace(req=req, values={"NE": "4096", "MI": "2", "MB": "16"}, preset="p", env={})
    monkeypatch.setattr(training.PL, "final_ckpt", lambda _p: ck)
    for name in ("check_coll_preset", "write_coll_preset", "write_switches_record"):
        monkeypatch.setattr(training.records, name, lambda *a, **k: None)
    monkeypatch.setattr(training.modes, "train_call", lambda *a, **k: None)
    monkeypatch.setattr(training.process, "resolve_call", lambda *a, **k: None)
    monkeypatch.setattr(training.process, "start", start)
    monkeypatch.setattr(training, "_row", lambda _p, status, _s, tail="": rows.append((status, tail)))
    monkeypatch.setattr(training, "_auto_resume", lambda *a: resumed.append(a))
    return training._train_bi(plan, ref, 0), rows, resumed


def test_an_overflowed_training_fails_even_with_a_checkpoint_and_is_not_resumed(tmp_path, monkeypatch):
    rc, rows, resumed = _bi(tmp_path, monkeypatch, rc=process.PHYSX_OVERFLOW_RC, logs={"R1_train.log": LINE}, ckpt=True)
    assert rc == process.PHYSX_OVERFLOW_RC and rows == [("TRAIN_FAIL", "PHYSX_OVERFLOW")] and not resumed


def test_an_overflow_in_a_renamed_dead_segment_fails_the_run(tmp_path, monkeypatch):
    logs = {"R1_train.log": "  Training Complete\n", "R1_train_seg0.log": LINE, "R1_R40_train.log": ""}
    rc, rows, _ = _bi(tmp_path, monkeypatch, rc=0, logs=logs, ckpt=True)
    assert rc == process.PHYSX_OVERFLOW_RC and rows == [("TRAIN_FAIL", "PHYSX_OVERFLOW")]


def test_a_clean_training_goes_to_the_judge(tmp_path, monkeypatch):
    logs = {"R1_train.log": "  Training Complete\n", "R1_train_seg0.log": "CUDA out of memory\n"}
    rc, rows, _ = _bi(tmp_path, monkeypatch, rc=0, logs=logs, ckpt=True)
    assert rc == 0 and [r[0] for r in rows] == ["SIDECAR"]  # negative: no overflow line, no failure


# ---- the collision stack rule and its one writer


@pytest.mark.parametrize("ne,log2", [(1, 26), (8, 26), (256, 26), (257, 27), (1024, 28), (4096, 30), (8192, 31)])
def test_collision_stack_follows_the_env_count(ne, log2):
    assert PB.collision_stack(ne) == 2**log2
    assert PB.collision_stack(4096) >= 4 * PB.PHYSX_COLLISION_STACK_PEAK_B
    assert PB.ISAAC_COLLISION_STACK == 2**26  # Isaac Lab's PhysxCfg default: NE <= 256 keep it (byte-equal)


def test_collision_stack_refuses_no_envs():
    with pytest.raises(ValueError):
        PB.collision_stack(0)


def _wired(src: str) -> bool:
    """`_construct_env` assigns cfg.sim.physx.gpu_collision_stack_size from collision_stack(num_envs)."""
    fn = _fn("", "_construct_env", ast.parse(src))
    return any(
        isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", "") == "collision_stack"
        and ast.unparse(n.value.args[0]) == "num_envs"
        and any(ast.unparse(t) == "cfg.sim.physx.gpu_collision_stack_size" for t in n.targets)
        for n in ast.walk(fn)
    )


def test_cli_writes_the_collision_stack_and_nothing_else_does():
    src = (REPO_ROOT / "dexengine/rl/cli.py").read_text(encoding="utf-8")
    assert _wired(src)
    tree = ast.parse(src)
    fn = _fn("", "_construct_env", tree)
    fn.body = [s for s in fn.body if "collision_stack(" not in ast.unparse(s)]
    assert not _wired(ast.unparse(tree))  # negative: the wiring removed is caught
    writers = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in (REPO_ROOT / "dexengine").rglob("*.py")
        if "gpu_collision_stack_size =" in p.read_text(encoding="utf-8")
    )
    assert writers == ["dexengine/rl/cli.py"], writers


def _stage(stack: int, ne: int):
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/physicsScene", "PhysicsScene")
    cfg = SimpleNamespace(**{f: 2**20 for _n, _a, f in PB.BUFFERS})
    cfg.gpu_collision_stack_size = stack
    for _n, a, f in PB.BUFFERS:
        prim.CreateAttribute(a, Sdf.ValueTypeNames.UInt).Set(getattr(cfg, f))
    return stage, prim, cfg


def test_buffers_marker_reads_the_stage_against_cfg_and_rule():
    stage, prim, cfg = _stage(2**30, 4096)
    s = PB.line(stage, cfg, "/physicsScene", 4096)
    assert s.startswith("[PHYSX_BUFFERS] collision_stack 2^26 → 2^30 at NE 4096 (2^30 x NE/4096, floor 2^26")
    assert "136,653,960 B" in s and "contact 2^20" in s
    prim.GetAttribute("physxScene:gpuCollisionStackSize").Set(2**26)
    with pytest.raises(RuntimeError, match="differs from cfg"):
        PB.line(stage, cfg, "/physicsScene", 4096)  # negative: the stage did not get the statement


def test_buffers_marker_refuses_a_cfg_that_skipped_the_rule():
    stage, _prim, cfg = _stage(2**26, 4096)  # stage = cfg, but not the rule's value at NE 4096
    with pytest.raises(RuntimeError, match="did not go through rl/cli"):
        PB.line(stage, cfg, "/physicsScene", 4096)
    assert "→ 2^26 at NE 8" in PB.line(stage, cfg, "/physicsScene", 8)  # the same values are right at NE 8


def test_buffers_marker_is_wired_after_load():
    fn = _fn("dexengine/rl/tasks/env_load_steps.py", "audit_collision_and_preset")
    assert "print_line(self)" in ast.unparse(fn)
    assert Path(PB.__file__).name == "physx_buffers.py"
