"""The ledger's check of the collision authoring (CPU): with a slot of every kind of the collision table (base kinds,
scene-rule kinds and an empty place), the robot-filtered count the authoring's own rule gives (collision_presets
.slot_robot_off) passes the check, under the training tables and under full physics."""

from __future__ import annotations

import pytest

from dexengine.rl.env_setup import extra_slots_ledger as EL
from dexengine.rl.env_setup import extra_slots_plan as XP
from dexengine.rl.tasks.scene import collision_presets as CP

ALL_KINDS = (*CP.EXTRA_KIND_COLLIDE, *CP.EXTRA_SCOPED_KIND_COLLIDE)


def _authored(tables: dict) -> dict:
    """What the authoring returns for one env with one slot per kind and one empty place."""
    kinds = dict.fromkeys(tables, 1) | {"empty": 1}
    off = sum(CP.slot_robot_off(t) for t in tables.values())  # the empty place stands with collision off, no filter
    return {"slots": len(kinds), "envs": 1, "pairs": len(tables), "robot_off": off, "kinds": kinds}


@pytest.mark.parametrize("robot_static", [True, False])
@pytest.mark.parametrize("full", [False, True])
def test_every_kind_of_the_table_passes_the_check(monkeypatch, run_dataset, full, robot_static):
    run_dataset(robot_static=robot_static)
    tables = {k: CP.extra_collide(k, full=full) for k in ALL_KINDS}
    assert set(CP.extra_kinds_in_play(list(CP.EXTRA_SCOPED_KIND_COLLIDE))) == set(ALL_KINDS)
    monkeypatch.setattr(EL, "_EXTRA_STATE", {"clips": {}, **XP.collide_state(tables)})
    line = EL._authored_line(_authored(tables))
    assert f"robot_off={_authored(tables)['robot_off']}" in line
    if not full:
        robot_off = {k for k, t in tables.items() if t["hand"] == "OFF"}
        assert robot_off == (set() if robot_static else set(CP.ROBOT_STATIC_KINDS))


def test_a_wrong_count_fails(monkeypatch, run_dataset):
    """Negative: the base-kinds-only count (receiver and touched, scene-rule kinds left out) is refused (a dataset
    out of S-robot-static, where the still kinds are filtered from the robot)."""
    run_dataset(robot_static=False)
    tables = {k: CP.extra_collide(k, full=False) for k in ALL_KINDS}
    monkeypatch.setattr(EL, "_EXTRA_STATE", {"clips": {}, **XP.collide_state(tables)})
    a = _authored(tables)
    base_only = sum(CP.slot_robot_off(tables[k]) for k in CP.EXTRA_KIND_COLLIDE)
    assert base_only != a["robot_off"]
    with pytest.raises(SystemExit):
        EL._authored_line(dict(a, robot_off=base_only))
