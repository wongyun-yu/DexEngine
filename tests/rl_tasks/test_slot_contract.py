"""The env -> clip contract of the fixed slots (objects._slot_entity): a slot with two or more clip assets spawns per
env and keeps each env on its clip, whatever multi_object says (a uni run whose clips share one object)."""

import sys
import types

import pytest

from dexengine.rl.env_setup import objects as O


class _Cfg(types.SimpleNamespace):
    """Stand-in for the IsaacLab cfg classes: keeps the keyword arguments it is built with."""

    InitialStateCfg = types.SimpleNamespace


@pytest.fixture
def isaaclab_stub(monkeypatch):
    """The two IsaacLab modules _slot_entity imports, as stand-ins (no Kit)."""
    assets = types.ModuleType("isaaclab.assets")
    assets.RigidObjectCfg = _Cfg
    wrappers = types.ModuleType("isaaclab.sim.spawners.wrappers")
    wrappers.MultiAssetSpawnerCfg = _Cfg
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.sim.spawners.wrappers", wrappers)


def _cfg():
    """A scene cfg as IsaacLab leaves it for a uni run whose clips share one object (multi_object off)."""
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(replicate_physics=True),
        trajectory=types.SimpleNamespace(static_clip_assignment=False),
    )


def test_two_clip_assets_hold_the_contract(isaaclab_stub, capsys):
    cfg = _cfg()
    O._slot_entity(cfg, "support_disc_0", "SupportDisc0", ["clip0.usd", "clip1.usd"])
    assert cfg.scene.replicate_physics is False and cfg.trajectory.static_clip_assignment is True
    assert f"{O.SLOT_CONTRACT_MARK} support_disc_0: 2 clip assets" in capsys.readouterr().out
    slot = cfg.scene.support_disc_0
    assert slot.spawn.assets_cfg == ["clip0.usd", "clip1.usd"] and slot.spawn.random_choice is False


def test_one_clip_asset_leaves_the_scene_as_it_is(isaaclab_stub, capsys):
    cfg = _cfg()
    O._slot_entity(cfg, "extra_obj_0", "ExtraObj0", ["clip0.usd"], contact=True)
    assert cfg.scene.replicate_physics is True and cfg.trajectory.static_clip_assignment is False
    assert O.SLOT_CONTRACT_MARK not in capsys.readouterr().out
    assert cfg.scene.extra_obj_0.spawn.activate_contact_sensors is True
