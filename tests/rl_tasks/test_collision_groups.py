"""Robot-off collision groups (CPU, pxr only). The stage carries Isaac Lab's env groups as its cloner writes them
(inverted filter, one group per env, optional global group); the robot-off groups go on top. Measured on the stage:
only robot <-> robot-off is cut inside an env, nothing reaches across envs, and planted defects fail the read-back,
the plan or the POSTINIT gate. The audit reads the same cut from the stage."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from dexengine.rl.tasks.scene import collision_groups as CG  # noqa: E402
from dexengine.rl.tasks.scene import extra_slot_audit as xa  # noqa: E402

ENVS = ["/World/envs/env_0", "/World/envs/env_1"]
GROUND = "/World/ground"


def _body(stage, path):
    UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
    return path


def _clone_groups(stage, global_paths):
    """Isaac Lab's env groups as the cloner writes them (isaacsim.core.cloner filter_collisions)."""
    UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim().CreateAttribute(
        "physxScene:invertCollisionGroupFilter", Sdf.ValueTypeNames.Bool
    ).Set(True)
    UsdGeom.Scope.Define(stage, CG.ROOT)
    lay = stage.GetRootLayer()
    glob = None
    if global_paths:
        glob = CG.ROOT + "/global_group"
        CG._group_spec(lay, glob, global_paths, [glob])
    for i, env in enumerate(ENVS):
        g = f"{CG.ROOT}/group{i}"
        CG._group_spec(lay, g, [env], [g, *([glob] if glob else [])])
        if glob:
            lay.GetPrimAtPath(glob).relationships[CG.FILT].targetPathList.Append(g)


def _scene(with_global=True):
    stage = Usd.Stage.CreateInMemory()
    for env in ENVS:
        for p in ("Robot/link_a", "Robot/link_b", "RobotL/link_c", "extra_object_0/body", "object_right"):
            _body(stage, f"{env}/{p}")
    _body(stage, GROUND)
    _clone_groups(stage, [GROUND] if with_global else [])

    class Scene:
        AUTHORS_ROBOT_OFF_GROUPS = True

    sc = Scene()
    sc.stage, sc.cfg = stage, types.SimpleNamespace(replicate_physics=False)
    return sc


def _plan_all(sc):
    for env in ENVS:
        CG.plan(sc, env, [f"{env}/extra_object_0"])


@pytest.mark.parametrize("with_global", [True, False])
def test_only_robot_vs_robot_off_is_cut_and_nothing_crosses_envs(with_global, capsys):
    sc = _scene(with_global)
    _plan_all(sc)
    info = CG.author_pending(sc)
    assert info == {"groups": 4, "envs": 2, "roots": 2, "global": with_global}
    assert "read-back ok" in capsys.readouterr().out
    CG.assert_done(sc)
    ix = CG.GroupIndex(sc.stage)
    e0, e1 = ENVS
    robot0 = [f"{e0}/Robot/link_a", f"{e0}/Robot/link_b", f"{e0}/RobotL/link_c"]
    slot0, obj0 = f"{e0}/extra_object_0/body", f"{e0}/object_right"
    assert all(ix.blocked(r, slot0) for r in robot0)
    assert not any(ix.blocked(r, obj0) for r in robot0)
    assert not ix.blocked(slot0, obj0)
    assert not ix.blocked(robot0[0], robot0[2])
    for a in [*robot0, slot0, obj0]:
        for b in (f"{e1}/Robot/link_a", f"{e1}/extra_object_0/body", f"{e1}/object_right"):
            assert ix.blocked(a, b), (a, b)
        assert not ix.blocked(a, GROUND)


def test_the_audit_reads_the_cut_from_the_stage():
    sc = _scene()
    e0 = ENVS[0]
    slots, robot, obj = [f"{e0}/extra_object_0"], {f"{e0}/Robot/link_a"}, {f"{e0}/object_right"}
    before = xa.rows(slots, ["touched"], robot, obj, xa.group_pairs(sc.stage, slots, robot | obj))
    assert before == {"RE": "touched:ON", "OE": "touched:ON"}
    _plan_all(sc)
    CG.author_pending(sc)
    after = xa.rows(slots, ["touched"], robot, obj, xa.group_pairs(sc.stage, slots, robot | obj))
    assert xa.norm(after["RE"]) == "touched:OFF" and after["OE"] == "touched:ON"


def test_a_scene_without_the_authoring_point_or_with_replicated_physics_cannot_plan():
    sc = _scene()
    with pytest.raises(RuntimeError, match="no group authoring point"):
        CG.plan(types.SimpleNamespace(stage=sc.stage, cfg=sc.cfg), ENVS[0], ["x"])
    sc.cfg.replicate_physics = True
    with pytest.raises(RuntimeError, match="replicate_physics=True"):
        CG.plan(sc, ENVS[0], ["x"])


def test_a_pending_plan_fails_at_postinit():
    sc = _scene()
    _plan_all(sc)
    with pytest.raises(RuntimeError, match="still wait"):
        CG.assert_done(sc)


def test_groups_before_the_env_groups_or_under_an_undefined_parent_fail():
    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(RuntimeError, match="not defined"):
        CG.author(stage, [(ENVS[0], [f"{ENVS[0]}/extra_object_0"])])
    stage.OverridePrim(CG.ROOT)
    with pytest.raises(RuntimeError, match="not defined"):
        CG.author(stage, [(ENVS[0], [f"{ENVS[0]}/extra_object_0"])])


def test_authoring_twice_fails():
    sc = _scene()
    _plan_all(sc)
    CG.author_pending(sc)
    with pytest.raises(RuntimeError, match="authored twice"):
        CG.author(sc.stage, [(ENVS[0], [f"{ENVS[0]}/extra_object_0"])])


def test_planted_defects_fail_the_read_back():
    """A slot left in its env group (two groups) and a robot group that lists the robot-off group (pair on)."""
    sc = _scene()
    _plan_all(sc)
    CG.author_pending(sc)
    todo = [(e, [f"{e}/extra_object_0"]) for e in ENVS]
    CG.verify(sc.stage, todo)
    rel = sc.stage.GetPrimAtPath(f"{CG.ROOT}/group0").GetRelationship(CG.EXC)
    rel.RemoveTarget(Sdf.Path(f"{ENVS[0]}/extra_object_0"))
    with pytest.raises(RuntimeError, match="robot-off bodies 1 · not in"):
        CG.verify(sc.stage, todo)
    rel.AddTarget(Sdf.Path(f"{ENVS[0]}/extra_object_0"))
    CG.verify(sc.stage, todo)
    rg = sc.stage.GetPrimAtPath(f"{CG.ROOT}/{CG.ROBOT_GROUP}1").GetRelationship(CG.FILT)
    rg.AddTarget(Sdf.Path(f"{CG.ROOT}/{CG.OFF_GROUP}1"))
    with pytest.raises(RuntimeError, match="robot<->robot-off True"):
        CG.verify(sc.stage, todo)


def test_the_truth_table_rd_row_reads_the_disc_cut_from_the_stage():
    from dexengine.rl.tasks.scene.collision_check import _group_cut_rd

    sc = _scene()
    e0 = ENVS[0]
    disc = _body(sc.stage, f"{e0}/support_disc_0/body")
    robot = {f"{e0}/Robot/link_a", f"{e0}/Robot/link_b", f"{e0}/RobotL/link_c"}
    assert _group_cut_rd(sc.stage, robot, {disc}, set()) == 0
    for env in ENVS:
        CG.plan(sc, env, [f"{env}/support_disc_0"] if env == e0 else [f"{env}/extra_object_0"])
    CG.author_pending(sc)
    pairs: set = set()
    assert _group_cut_rd(sc.stage, robot, {disc}, pairs) == 3 and len(pairs) == 3


def _cfg_scene(sc, **slots):
    """The scene's cfg statement of slot prims (entity name -> prim under each env), as the audits read it."""
    sc.env_regex_ns = "/World/envs/env_.*"
    sc.cfg = types.SimpleNamespace(
        replicate_physics=False,
        **{k: types.SimpleNamespace(prim_path="{ENV_REGEX_NS}/" + v) for k, v in slots.items()},
    )
    return sc


def _filter(stage, src, dst):
    UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(src)).CreateFilteredPairsRel().AddTarget(Sdf.Path(dst))


def test_a_filtered_pairs_robot_off_slot_fails_the_extra_slot_audit():
    sc = _cfg_scene(_scene(), extra_obj_0="extra_object_0")
    e0 = ENVS[0]
    robot, obj = {f"{e0}/Robot/link_a", f"{e0}/RobotL/link_c"}, {f"{e0}/object_right"}
    _plan_all(sc)
    CG.author_pending(sc)
    _filter(sc.stage, f"{e0}/extra_object_0/body", f"{e0}/object_right")  # slot <-> object stays a FilteredPairs cut
    got = xa.fields(sc, e0, ["touched"], robot, obj)["extra"]
    assert xa.norm(got["RE"]) == "touched:OFF" and xa.norm(got["OE"]) == "touched:OFF"
    _filter(sc.stage, f"{e0}/Robot/link_a", f"{e0}/extra_object_0/body")
    with pytest.raises(RuntimeError, match="robot <-> extra slot cut by FilteredPairs"):
        xa.fields(sc, e0, ["touched"], robot, obj)


def test_a_filtered_pairs_robot_off_disc_fails_the_truth_table():
    from dexengine.rl.tasks.scene.collision_check import _collect_collision_state

    sc = _cfg_scene(_scene(), support_disc_0="support_disc_0")
    e0 = ENVS[0]
    for env in ENVS:
        _body(sc.stage, f"{env}/support_disc_0/body")
        CG.plan(sc, env, [f"{env}/support_disc_0"])
    CG.author_pending(sc)
    cls = {f"{e0}/Robot/link_a": "RA", f"{e0}/RobotL/link_c": "LA", f"{e0}/support_disc_0": "DISC"}
    filt = _collect_collision_state(sc, e0, cls, sc.stage)[3]
    assert filt["RD"] == 2
    _filter(sc.stage, f"{e0}/support_disc_0/body", f"{e0}/RobotL/link_c")
    with pytest.raises(RuntimeError, match="robot <-> support disc cut by FilteredPairs"):
        _collect_collision_state(sc, e0, cls, sc.stage)
