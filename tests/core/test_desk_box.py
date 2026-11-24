"""The one visual desk (Willow) keeps its scaled asset and vertex stretch; the box desk is gone (render_unify)."""

from __future__ import annotations

import ast
from types import SimpleNamespace as NS

import pytest

from dexengine.core import desk as DK
from dexengine.core.repo import REPO_ROOT
from dexengine.record.render import desk_stretch as DS

SCENE_PY = REPO_ROOT / "dexengine/rl/tasks/cfg/scene.py"
CUBOIDS = ("desk_collision_top", "desk_leg_0", "desk_leg_1", "desk_leg_2", "desk_leg_3")


def _scene_cuboid_sizes() -> dict:
    """The `spawn=…(size=…)` of the desk cuboids in the scene cfg class, evaluated with the desk constants (read
    from the source: the scene module needs Isaac Lab)."""
    names = {k: getattr(DK, k) for k in dir(DK) if k.startswith("DESK_")}
    out = {}
    for node in ast.walk(ast.parse(SCENE_PY.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", "") in CUBOIDS:
            spawn = next(k.value for k in node.value.keywords if k.arg == "spawn")
            size = next(k.value for k in spawn.keywords if k.arg == "size")
            out[node.targets[0].id] = tuple(eval(compile(ast.Expression(size), "scene.py", "eval"), {}, names))
    assert sorted(out) == sorted(CUBOIDS), out
    return out


def _scene(spawn) -> NS:
    """A scene cfg stand-in: the five cuboids with the scene cfg's sizes, positions to be written by the fitting."""
    sizes = _scene_cuboid_sizes()
    cfg = NS(desk=NS(init_state=NS(pos=(9.0, 9.0, 9.0)), spawn=spawn))
    for key in CUBOIDS:
        setattr(cfg, key, NS(init_state=NS(pos=(9.0, 9.0, 9.0)), spawn=NS(size=sizes[key])))
    return cfg


def test_willow_desk_keeps_its_scaled_asset():
    cfg = _scene(NS(usd_path="/x/Willow.usd", scale=None))
    msg = DK.apply_desk_top(cfg, DK.DESK_TOP_BY_WORKSPACE["tabletop"])
    assert "(top, legs×4, visual)" in msg
    assert cfg.desk.spawn.scale == pytest.approx((DK.DESK_VISUAL_UNIT,) * 3)


def test_willow_stretch_plan_is_unchanged():
    assert DS.stretch_plan(0.02) == {
        "delta_m": -0.02,
        "z0_m": DS.STRETCH_Z0_M,
        "z1_m": DS.STRETCH_Z1_M,
        "span_canon_m": round(DS.STRETCH_Z1_M - DS.STRETCH_Z0_M, 6),
        "span_m": round(DS.STRETCH_Z1_M - DS.STRETCH_Z0_M - 0.02, 6),
        "min_span_m": DS.STRETCH_MIN_SPAN_M,
        "gap_m": DS.STRETCH_GAP_M,
    }


def test_stretch_plan_and_marker(monkeypatch):
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")  # the body chain is the active arm's skeleton (no default arm)
    monkeypatch.setattr(DS.B, "solve", lambda ch, dz, none_ok=False: None)  # no torso solution
    body = DS.decide(-0.03, False, ch={})
    assert body["method"] == "desk_stretch" and DS.world_moved(body)
    sp = body["desk_stretch"]
    assert "desk" not in sp and sp["span_m"] == pytest.approx(sp["span_canon_m"] + 0.03)
    meta = dict(body, visual={"top_visual_m": [0.76, 0.79], "collider": {"top_m": 0.79}, "reread_max_err_m": 0.0})
    assert f"cuts z0 {DS.STRETCH_Z0_M}" in DS.mark(meta)
    with pytest.raises(SystemExit, match="not clamped"):
        DS.stretch_plan(0.70)
