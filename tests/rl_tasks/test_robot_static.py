"""S-robot-static (CPU): where the rule is on for the run's dataset the robot collides with the still kinds in
training; a dataset that opts out keeps the tables; the run's dataset is required (no default); the rule version and
the manifest name it."""

from __future__ import annotations

import pytest

from dexengine.data.objects import scene_rules as SC
from dexengine.rl.tasks.scene import collision_presets as CP

TABLE = {**CP.EXTRA_KIND_COLLIDE, **CP.EXTRA_SCOPED_KIND_COLLIDE}


def test_still_kinds_collide_with_the_robot_where_the_rule_is_on(run_dataset):
    run_dataset()
    for k, t in TABLE.items():
        got = CP.extra_collide(k, full=False)
        if k in CP.ROBOT_STATIC_KINDS:
            assert got == {**t, "hand": "ON", "arm": "ON"} and CP.extra_has_collider(k)
        else:
            assert got == t  # moving · carried_attached keep their own cells
    assert CP.extra_collide("touched", full=True) == {"hand": "ON", "arm": "ON", "objects": "OFF"}  # full: unchanged
    assert "touched(hand ON · arm ON · objects OFF)" in CP.extra_collide_text()  # the run-log marker states it


def test_an_opted_out_dataset_keeps_the_tables(run_dataset):
    run_dataset(robot_static=False)
    assert all(CP.extra_collide(k, full=False) == t for k, t in TABLE.items())
    assert not CP.extra_has_collider("touched")  # all partners OFF: no collider stands
    assert CP.extra_collide("receiver", full=False, ds="taco")["hand"] == "ON"  # a named dataset is read for itself


def test_the_run_dataset_is_required(run_dataset):
    """Negative: without DEXENGINE_DATASET (cleared by the shared setup) the still kinds cannot be stated."""
    with pytest.raises(ValueError, match="DEXENGINE_DATASET"):
        CP.extra_collide("receiver")
    assert CP.extra_collide("moving") == CP.EXTRA_COLLIDE  # kinds outside the rule need no dataset
    assert CP.extra_collide("receiver", ds="hot3d")["arm"] == "ON"


def test_the_rule_version_and_the_manifest_name_it(run_dataset):
    from dexengine.record.scene import manifest as MF

    run_dataset()
    assert SC.VERSIONS[SC.ROBOT_RULE] in SC.rule_version("base", ["hot3d"])
    robot, _objs = MF._slot_cells({"name": "rack", "kind": "receiver"}, CP, "hot3d")
    assert robot["training"] == "ON" and "S-robot-static" in robot["why"]
    run_dataset(robot_static=False)
    assert SC.VERSIONS[SC.ROBOT_RULE] not in SC.rule_version("base", ["hot3d"])
    robot, _objs = MF._slot_cells({"name": "rack", "kind": "receiver"}, CP, "hot3d")
    assert robot["training"] == "OFF" and "S-robot-static" not in robot["why"]
