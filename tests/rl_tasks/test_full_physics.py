"""Full physics (CPU): the switch's collision tables (training default unchanged), the ledger check that passes only
the switch's own differences, and the manifest cells (delivered and training values)."""

from __future__ import annotations

from dexengine.rl.env_setup import extra_slots_ledger as EL
from dexengine.rl.tasks.scene import collision_presets as CP


def test_every_kind_collides_with_the_robot_under_full_physics(run_dataset):
    run_dataset(robot_static=False)
    for k in (*CP.EXTRA_KIND_COLLIDE, *CP.EXTRA_SCOPED_KIND_COLLIDE):
        t = CP.extra_collide(k, full=True)
        assert t["hand"] == t["arm"] == "ON"
        assert t["objects"] == ("OFF" if k in CP.FULL_PHYSICS_OBJECTS_OFF else "ON")
    assert CP.extra_collide("touched", full=False) == CP.EXTRA_TOUCHED_COLLIDE  # off = the tables as they are


def test_the_ledger_passes_only_the_switchs_differences(monkeypatch, run_dataset):
    run_dataset()
    monkeypatch.setattr(CP, "full_physics", lambda: True)
    ts = EL.train_scene_owner()
    trained = {"collide": {"touched": ts.partners(CP.extra_collide("touched", full=False))}}
    assert EL._full_physics_explained(CP, ["touched"], trained, "p", "ctx") == []
    other = {"collide": {"touched": ts.partners({"hand": "OFF", "arm": "OFF", "objects": "OFF"})}}
    assert EL._full_physics_explained(CP, ["touched"], other, "p", "ctx") == ["touched"]
    monkeypatch.setattr(CP, "full_physics", lambda: False)
    assert EL._full_physics_explained(CP, ["touched"], trained, "p", "ctx") == ["touched"]


def test_training_keeps_the_tables_and_the_switch_is_off_by_default(run_dataset):
    from dexengine.rl.tasks import switches

    run_dataset()
    assert switches.RLSwitches().full_physics is False
    for k in CP.EXTRA_KIND_COLLIDE:
        assert CP.extra_collide(k) == CP.extra_collide(k, full=False)


def test_the_manifest_cells_carry_delivered_and_training_values(run_dataset):
    from dexengine.record.scene import manifest as MF

    run_dataset(robot_static=False)
    r = {
        "name": "plate",
        "kind": "touched",
        "touch_gate": {"why": "true-shape overlap", "pairs": {"cup": {"depth_m": 0.008}}},
    }
    robot, objs = MF._slot_cells(r, CP, "hot3d")
    assert robot["delivered"] == "ON" and robot["training"] == CP.EXTRA_TOUCHED_COLLIDE["hand"]
    assert objs["delivered"] == "OFF" and objs["why"].startswith("S-touch-gate") and "0.008" in objs["why"]
    robot, objs = MF._slot_cells(dict(r, kind="envelope"), CP, "hot3d")
    assert robot["delivered"] == objs["delivered"] == "ON" and robot["training"] == CP.EXTRA_ENVELOPE_COLLIDE["hand"]
