"""Structure invariants of the RL task package that hold without Isaac Lab."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCOPE = [REPO / "dexengine/rl/tasks", REPO / "dexengine/rl/env_setup", REPO / "dexengine/rl/assets"]

#: Modules other packages import in a plain interpreter: they must never pull Isaac / Kit / gymnasium.
ISAAC_FREE = (
    "dexengine.rl.tasks.hooks",
    "dexengine.rl.tasks.cfg.constants",
    "dexengine.rl.tasks.scene.collision_presets",
    "dexengine.rl.tasks.scene.extra_slots",
    "dexengine.rl.tasks.scene.object_pin",
    "dexengine.rl.tasks.scene.split_artic",
    "dexengine.rl.tasks.scene.spawn_hold",
    "dexengine.rl.tasks.mdp.trajectory.loader",
)
HEAVY = ("isaaclab", "omni", "carb", "pxr", "gymnasium")


def _py_files():
    for root in SCOPE:
        yield from sorted(root.rglob("*.py"))


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module


def test_release_code_never_imports_experiments():
    bad = [f"{p.relative_to(REPO)}: {m}" for p in _py_files() for m in _imports(p) if m.split(".")[0] == "experiments"]
    assert not bad, bad


def test_package_inits_are_empty():
    # Isaac-free leaves live under these packages: an import in an __init__ would drag Isaac into them.
    inits = [p for p in _py_files() if p.name == "__init__.py" and p.read_text(encoding="utf-8").strip()]
    assert not inits, inits


def _module_defines(mod: str, name: str) -> bool:
    path = REPO / (mod.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return True
    return False


def test_registry_entry_points_resolve_statically():
    tree = ast.parse((REPO / "dexengine/rl/tasks/registry.py").read_text(encoding="utf-8"))
    consts = {
        n.targets[0].id: n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
    }
    cfgs = next(
        n.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_ENV_CFGS"
    )
    entries = [consts["ENV_ENTRY_POINT"]] + [v.value for v in cfgs.values]
    for entry in entries:
        mod, name = entry.split(":")
        assert _module_defines(mod, name), entry
    ids = [consts[k] for k in ("RIGHT_UNI", "LEFT_UNI", "BIMANUAL")]
    assert ids == [
        "DexEngine-ArmResidual-RightUni-v0",
        "DexEngine-ArmResidual-LeftUni-v0",
        "DexEngine-ArmResidual-Bimanual-v0",
    ]
    assert (REPO / "configs/rl/skrl_ppo.yaml").is_file()


@pytest.mark.parametrize("module", ISAAC_FREE)
def test_leaf_imports_without_isaac(module):
    code = f"import sys, {module}; print(sorted(k for k in sys.modules if k.split('.')[0] in {HEAVY!r}))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={"PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_camera_constants_keep_their_values():
    from dexengine.rl.tasks.cfg import constants as C

    assert C.CAMERA_CLIPPING_RANGE == (0.05, 10.0)
    assert C.CAMERA_OFFSET_POS == (0.0, 0.005, 0.015)
    from dexengine.core.arms import ARM_REGISTRY

    rby1 = ARM_REGISTRY["rby1"]  # arm and head drives and the head pitch moved to the arm spec (values kept)
    assert (rby1.arm_drive.kp, rby1.arm_drive.kd, rby1.head_drive.kp, rby1.head_drive.kd) == (800.0, 80.0, 400.0, 80.0)
    assert (rby1.head_drive.effort_limit, rby1.head_drive.velocity_limit, rby1.head_drive.pitch_deg) == (
        10.0,
        10.0,
        30.0,
    )
    assert C.DESK_HEIGHT == 0.76 and C.ROBOT_DEFAULT_POS == (-0.65, -0.15, 0.30)


#: Top-level symbols that other programs read from these files by path (AST or source text, without importing them).
#: Moving one of them breaks the reader silently, so a split must keep it here or repoint the readers in the same
# change.
READ_BY_PATH = {
    "dexengine/rl/tasks/cfg/residual.py": [
        "DEFAULT_OBJECT_MASS_KG"
    ],  # experiments/baselines/common prep_mass_0p2, method_cmds
    "dexengine/rl/env_setup/build.py": [
        "AERIAL_DISC_HEIGHT",  # experiments/baselines/common/prep_start_a.py
        "_apply_shared_object",
    ],  # core rules registry (rule D6-a)
    "dexengine/rl/tasks/mdp/rewards/task.py": ["RewardObjArtic"],  # core rules registry (rule D1-c)
    "dexengine/rl/tasks/env.py": ["ArmResidualTrainEnv"],  # core rules registry (rule D32)
    "dexengine/rl/tasks/scene/collision_presets.py": ["EXTRA_KIND_COLLIDE", "SUPPORT_COLLIDE"],  # core rules registry
}


@pytest.mark.parametrize("rel", sorted(READ_BY_PATH))
def test_symbols_read_by_path_stay_top_level(rel):
    tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
    have = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            have.add(n.name)
        elif isinstance(n, ast.Assign):
            have |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    assert sorted(set(READ_BY_PATH[rel]) - have) == []
