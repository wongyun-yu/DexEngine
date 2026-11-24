"""S-dyn-pairs collision groups (CPU, pxr only). On a stage with Isaac Lab's env groups and the robot-off groups, the
pair groups of a pair_off clip cut only right object <-> left object of that env: each object still collides with the
robot, the robot-off slot, its env and the global group, an env whose clip is not pair_off keeps the pair, nothing
crosses envs. Planted defects fail the read-back, the POSTINIT gate and the audit (removing the pair-off fails)."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from dexengine.rl.tasks.scene import collision_bodies as CB  # noqa: E402
from dexengine.rl.tasks.scene import collision_dyn_pairs as DG  # noqa: E402
from dexengine.rl.tasks.scene import collision_groups as CG  # noqa: E402

ENVS = ["/World/envs/env_0", "/World/envs/env_1"]
GROUND = "/World/ground"
LEAVES = ("Robot/link_a", "Robot/link_b", "extra_object_0/body", "object_right/body", "object_left/body", "desk")


def _body(stage, path):
    UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))


def _scene(monkeypatch, off, static=True):
    """Two envs as Isaac Lab's cloner groups them (inverted filter, global group), cfg statement `off` per clip."""
    stage = Usd.Stage.CreateInMemory()
    for env in ENVS:
        for leaf in LEAVES:
            _body(stage, f"{env}/{leaf}")
    _body(stage, GROUND)
    UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim().CreateAttribute(
        "physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool
    ).Set(True)
    UsdGeom.Scope.Define(stage, CG.ROOT)
    lay, glob = stage.GetRootLayer(), f"{CG.ROOT}/global_group"
    CG._group_spec(lay, glob, [GROUND], [glob])
    for i, env in enumerate(ENVS):
        g = f"{CG.ROOT}/group{i}"
        CG._group_spec(lay, g, [env], [g, glob])
        lay.GetPrimAtPath(glob).relationships[CG.FILT].targetPathList.Append(g)

    class Scene:
        AUTHORS_ROBOT_OFF_GROUPS = True

    sc = Scene()
    sc.stage, sc.env_prim_paths = stage, list(ENVS)
    sc.cfg = types.SimpleNamespace(replicate_physics=False)
    cfg = types.SimpleNamespace(trajectory=types.SimpleNamespace(static_clip_assignment=static))
    setattr(cfg, DG.CFG_KEY, off)
    monkeypatch.setattr(
        CB, "object_prim_paths", lambda s, r: {"right": f"{r}/object_right", "left": f"{r}/object_left"}
    )
    return sc, cfg


def _author(sc, cfg, robot_off=True):
    """The filter_collisions order: robot-off groups first, then the pair groups."""
    if robot_off:
        for env in ENVS:
            CG.plan(sc, env, [f"{env}/extra_object_0"])
    n = DG.plan(sc, cfg)
    CG.author_pending(sc)
    return n, DG.author_pending(sc)


def _cut(sc, env):
    return DG.cut(sc.stage, [f"{env}/object_right/body"], [f"{env}/object_left/body"])


def test_only_the_pair_is_cut_in_the_pair_off_env(monkeypatch, capsys):
    sc, cfg = _scene(monkeypatch, [True, False])
    n, info = _author(sc, cfg)
    assert n == 1 and info == {"groups": 2, "envs": 1} and "read-back ok" in capsys.readouterr().out
    DG.assert_done(sc)
    CG.assert_done(sc)
    ix = CG.GroupIndex(sc.stage)
    e0, e1 = ENVS
    a, b = f"{e0}/object_right/body", f"{e0}/object_left/body"
    assert ix.blocked(a, b) and _cut(sc, e0) == 1
    for o in (a, b):
        for other in (f"{e0}/Robot/link_a", f"{e0}/extra_object_0/body", f"{e0}/desk", GROUND):
            assert not ix.blocked(o, other), (o, other)
        for other in (f"{e1}/object_right/body", f"{e1}/Robot/link_a", f"{e1}/desk"):
            assert ix.blocked(o, other), (o, other)
    assert ix.blocked(f"{e0}/Robot/link_a", f"{e0}/extra_object_0/body")  # robot-off kept
    assert _cut(sc, e1) == 0 and not ix.blocked(f"{e1}/object_right/body", f"{e1}/object_left/body")


def test_pair_groups_without_robot_off_groups(monkeypatch):
    sc, cfg = _scene(monkeypatch, [True])
    _author(sc, cfg, robot_off=False)
    ix = CG.GroupIndex(sc.stage)
    for env in ENVS:
        assert _cut(sc, env) == 1 and not ix.blocked(f"{env}/object_right/body", f"{env}/Robot/link_a")


def test_no_pair_off_clip_authors_nothing(monkeypatch):
    sc, cfg = _scene(monkeypatch, [False, False])
    assert _author(sc, cfg) == (0, None)
    assert all(_cut(sc, env) == 0 for env in ENVS)


def test_several_clips_need_the_static_assignment(monkeypatch):
    sc, cfg = _scene(monkeypatch, [True, False], static=False)
    with pytest.raises(RuntimeError, match="static clip assignment"):
        DG.plan(sc, cfg)


def test_a_pending_plan_fails_at_postinit(monkeypatch):
    sc, cfg = _scene(monkeypatch, [True])
    DG.plan(sc, cfg)
    with pytest.raises(RuntimeError, match="still wait"):
        DG.assert_done(sc)


def test_planted_defects_fail_the_read_back(monkeypatch):
    """The two pair groups listing each other (pair on) and an object body left in its env group."""
    sc, cfg = _scene(monkeypatch, [True, False])
    _author(sc, cfg)
    rows = DG._rows(sc.stage, [(ENVS[0], f"{ENVS[0]}/object_right", f"{ENVS[0]}/object_left")])
    DG.readback(sc.stage, rows)
    lay = sc.stage.GetRootLayer()
    ga, gb = (f"{CG.ROOT}/{nm}0" for nm in DG.GROUPS)
    lay.GetPrimAtPath(ga).relationships[CG.FILT].targetPathList.Append(gb)
    with pytest.raises(RuntimeError, match="still collide"):
        DG.readback(sc.stage, rows)
    lay.GetPrimAtPath(ga).relationships[CG.FILT].targetPathList.Erase(gb)
    lay.GetPrimAtPath(f"{CG.ROOT}/group0").relationships[CG.EXC].targetPathList.Erase(f"{ENVS[0]}/object_right")
    with pytest.raises(RuntimeError, match="alone"):
        DG.readback(sc.stage, rows)


def test_the_read_back_fails_on_an_extra_partner_or_a_moved_body(monkeypatch):
    """A pair group that also reaches env_1's group, and another body of the env that left its group."""
    sc, cfg = _scene(monkeypatch, [True, False])
    for env in ENVS:
        CG.plan(sc, env, [f"{env}/extra_object_0"])
    CG.author_pending(sc)
    todo = [(ENVS[0], f"{ENVS[0]}/object_right", f"{ENVS[0]}/object_left")]
    rows = DG._rows(sc.stage, todo)
    before = DG._others(sc.stage, rows)
    DG.plan(sc, cfg)
    DG.author_pending(sc)
    DG.readback(sc.stage, rows, before)
    lay = sc.stage.GetRootLayer()
    ga = f"{CG.ROOT}/{DG.GROUPS[0]}0"
    lay.GetPrimAtPath(ga).relationships[CG.FILT].targetPathList.Append(f"{CG.ROOT}/group1")
    with pytest.raises(RuntimeError, match="collides with"):
        DG.readback(sc.stage, rows, before)
    lay.GetPrimAtPath(ga).relationships[CG.FILT].targetPathList.Erase(f"{CG.ROOT}/group1")
    lay.GetPrimAtPath(f"{CG.ROOT}/group0").relationships[CG.EXC].targetPathList.Append(f"{ENVS[0]}/desk")
    with pytest.raises(RuntimeError, match="other bodies changed groups"):
        DG.readback(sc.stage, rows, before)


def test_a_stage_without_the_inverted_filter_cannot_author(monkeypatch):
    sc, cfg = _scene(monkeypatch, [True])
    sc.stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:invertCollisionGroupFilter").Set(False)
    with pytest.raises(RuntimeError, match="not inverted"):
        DG.author(sc.stage, [(ENVS[0], f"{ENVS[0]}/object_right", f"{ENVS[0]}/object_left")])


def test_authoring_twice_fails(monkeypatch):
    sc, cfg = _scene(monkeypatch, [True])
    _author(sc, cfg)
    with pytest.raises(RuntimeError, match="authored twice"):
        DG.author(sc.stage, [(ENVS[0], f"{ENVS[0]}/object_right", f"{ENVS[0]}/object_left")])


def _audit_rows(sc, off):
    return [{"dyn_pair_off": o, "dyn_cut": (_cut(sc, env), 1)} for env, o in zip(ENVS, off, strict=True)]


def test_the_audit_compares_the_measured_cut_with_the_statement(monkeypatch):
    sc, cfg = _scene(monkeypatch, [True, False])
    _author(sc, cfg)
    assert DG.audit_line(_audit_rows(sc, [True, False]), [0, 1]).endswith("= statement")
    with pytest.raises(RuntimeError, match="differs from the statement"):
        DG.audit_line(_audit_rows(sc, [True, True]), [0, 1])  # env_1 stated off, its pair still collides


def test_removing_the_pair_off_fails_the_audit(monkeypatch):
    sc, cfg = _scene(monkeypatch, [False, False])
    _author(sc, cfg)  # the statement says pair_off for env_0 but nothing cut it
    with pytest.raises(RuntimeError, match="env_0 pair_off cut 0/1"):
        DG.audit_line(_audit_rows(sc, [True, False]), [0, 1])


def _tree(root, name):
    d = root / name / "0"
    d.mkdir(parents=True)
    objs = [
        {"name": "right_object", "side": "right"},
        {"name": "left_object", "side": "left"},
        {"name": "x", "side": "left"},
    ]
    (root / name / "task_info.json").write_text(json.dumps({"dataset_name": "hot3d", "objects": objs}))
    np.savez(d / "trajectory_kinematic.npz", qpos=np.zeros((3, 14)))


def test_the_scene_statement_is_one_bool_per_loader_clip(tmp_path, monkeypatch, capsys):
    from dexengine.data.objects import dyn_pairs as DP
    from dexengine.rl.env_setup import dyn_pairs as ED

    for n in ("a_pl_rt", "b_pl_rt"):
        _tree(tmp_path, n)
    monkeypatch.setattr(
        DP,
        "off_pairs",
        lambda tree, ti, q: [{"a": "right_object", "b": "left_object", "f0_m": 0.003}] if "a_" in tree else [],
    )
    args = types.SimpleNamespace(sequence="a_pl_rt,b_pl_rt", sequence_dir=str(tmp_path))
    cfg = types.SimpleNamespace()
    ED.build_dyn_pairs(cfg, args, "test")
    assert getattr(cfg, DG.CFG_KEY) == [True, False]
    assert "[DYN_PAIRS] test: clips 2 · pair_off 1 · a_pl_rt: right_object×left_object" in capsys.readouterr().out
    monkeypatch.setattr(
        DP, "off_pairs", lambda tree, ti, q: [{"a": "a", "b": "b", "f0_m": 0.003}, {"a": "c", "b": "d", "f0_m": 0.003}]
    )
    with pytest.raises(ValueError, match="one per clip"):  # dyn_pairs.scene_pair, the one owner of the scene's pair
        ED.build_dyn_pairs(cfg, args, "test")
    monkeypatch.setattr(DP, "off_pairs", lambda tree, ti, q: [{"a": "left_object", "b": "x", "f0_m": 0.003}])
    with pytest.raises(ValueError, match="right object against the left"):
        ED.build_dyn_pairs(cfg, args, "test")
