"""v0 keeps the floor's own MDL.

render_unify removed the walnut MDL the scene floor carried (every render binds the render look's floor over it).
The A/B on one SharpA clip showed the removal alone moves oblique-view renders (floor-joint texture detail, about
0.5 dB time-averaged PSNR in side · side_front), so v0 — the code that made the published data — keeps it and its
renders match that data; v0.1 drops it. The floor is read from the scene source (no Isaac import).
"""

from __future__ import annotations

import ast

from dexengine.core.repo import REPO_ROOT

SCENE = REPO_ROOT / "dexengine/rl/tasks/cfg/scene.py"
HAND_TRACKING = REPO_ROOT / "dexengine/rl/tasks/cfg/hand_tracking.py"
V0_FLOOR = {
    "func": "sim_utils.MdlFileCfg",
    "mdl_path": "FLOOR_MDL",
    "project_uvw": "True",
    "texture_scale": "(0.5, 0.5)",
}


def _floor_material(src: str) -> dict | None:
    """The keywords of the floor's own visual material in TacoEvalSceneCfg (None when the floor has none)."""
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "TacoEvalSceneCfg")
    for st in cls.body:
        targets = st.targets if isinstance(st, ast.Assign) else [st.target] if isinstance(st, ast.AnnAssign) else []
        if any(getattr(t, "id", None) == "floor" for t in targets):
            spawn = next(k.value for k in st.value.keywords if k.arg == "spawn")
            vm = next((k.value for k in spawn.keywords if k.arg == "visual_material"), None)
            if vm is None:
                return None
            return {"func": ast.unparse(vm.func), **{k.arg: ast.unparse(k.value) for k in vm.keywords}}
    raise AssertionError("TacoEvalSceneCfg has no floor")


def _floor_mdl_value(src: str) -> str:
    """The FLOOR_MDL assignment of hand_tracking.py, as source text."""
    for st in ast.parse(src).body:
        if isinstance(st, ast.Assign) and any(getattr(t, "id", None) == "FLOOR_MDL" for t in st.targets):
            return ast.unparse(st.value)
    raise AssertionError("hand_tracking.py has no FLOOR_MDL")


def test_v0_floor_keeps_its_own_mdl():
    assert _floor_material(SCENE.read_text(encoding="utf-8")) == V0_FLOOR
    assert _floor_mdl_value(HAND_TRACKING.read_text(encoding="utf-8")).endswith(
        "/Materials/Base/Wood/Walnut_Planks.mdl'"
    )


def test_the_floor_rule_sees_the_v01_floor():
    """Negative: the floor of v0.1 (no material of its own) and another MDL both fail the rule."""
    src = SCENE.read_text(encoding="utf-8")
    block = """        spawn=sim_utils.MeshCuboidCfg(
            size=(20.0, 20.0, 0.01),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=FLOOR_MDL,
                project_uvw=True,
                texture_scale=(0.5, 0.5),
            ),
        ),"""
    assert src.count(block) == 1
    v01 = src.replace(block, "        spawn=sim_utils.MeshCuboidCfg(size=(20.0, 20.0, 0.01)),")
    assert _floor_material(v01) is None
    other = src.replace("texture_scale=(0.5, 0.5)", "texture_scale=(1.0, 1.0)")
    assert _floor_material(other) != V0_FLOOR
