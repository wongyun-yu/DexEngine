"""Rule S-carried runtime on a fake env (CPU torch): every carried slot follows the **simulated** carrier at its stated
relative pose at every frame of the clip; other slots keep their reference targets."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest

from tests.imports import own_module

torch = pytest.importorskip("torch")
EC = own_module("dexengine.rl.tasks.scene.extra_carry")


def _qz(deg):
    a = math.radians(deg) / 2
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def _body(pos, quat):
    t = lambda v: torch.tensor([v], dtype=torch.float32)  # noqa: E731
    return NS(data=NS(root_pos_w=t(pos), root_quat_w=t(quat)))


REL = [0.03, 0.0, 0.02, *_qz(0)]


def _env(carrier, role="main"):
    cell = {"carrier_role": role, "carrier_side": "right", "carrier_slot": 1 if role == "slot" else None, "rel": REL}
    cfg = NS(extra_slot_carry=[[cell, None]], shared_object=False)
    return NS(cfg=cfg, device="cpu", scene={"object_right": carrier})


def _ref():
    pos = torch.tensor([[[9.0, 9.0, 9.0], [5.0, 5.0, 5.0]]])  # world-recorded targets (must not be used)
    quat = torch.tensor([[[1.0, 0, 0, 0], [1.0, 0, 0, 0]]])
    return pos, quat


def test_a_carried_slot_follows_the_simulated_carrier_at_every_frame():
    carrier = _body([1.0, 2.0, 0.8], _qz(90))
    env = _env(carrier)
    st = EC.CarryState(env, 2, ["clip0"])
    ids, ci = torch.tensor([0]), torch.tensor([0])
    pos, quat = _ref()
    p, q = st.apply(env, ids, ci, pos, quat)
    assert torch.allclose(p[0, 0], torch.tensor([1.0, 2.03, 0.82]), atol=1e-5)  # Rz(90°)·(0.03,0,0.02) + carrier
    assert torch.allclose(q[0, 0], torch.tensor(_qz(90)), atol=1e-6)
    assert torch.equal(p[0, 1], pos[0, 1])  # other slot: reference as is
    carrier.data.root_pos_w += torch.tensor([[0.1, 0.0, 0.0]])  # the policy moves the carrier
    p2, _ = st.apply(env, ids, ci, pos, quat)
    assert torch.allclose(p2[0, 0] - p[0, 0], torch.tensor([0.1, 0.0, 0.0]), atol=1e-6)
    assert not hasattr(st, "lo") and not hasattr(st, "left")  # no interval and no gate: attached at every frame


def test_a_carried_slot_on_a_slot_carrier_uses_that_slot_target():
    env = _env(_body([0, 0, 0], _qz(0)), role="slot")
    st = EC.CarryState(env, 2, ["clip0"])
    pos, quat = _ref()
    p, _ = st.apply(env, torch.tensor([0]), torch.tensor([0]), pos, quat)
    assert torch.allclose(p[0, 0], torch.tensor([5.03, 5.0, 5.02]), atol=1e-6)


def test_a_carrier_slot_outside_the_clip_or_itself_fails():
    env = _env(_body([0, 0, 0], _qz(0)), role="slot")
    env.cfg.extra_slot_carry[0][0]["carrier_slot"] = 0
    with pytest.raises(RuntimeError, match="carrier slot 0 outside"):
        EC.CarryState(env, 2, ["clip0"])


def test_the_statement_has_no_stand_mode():
    """The build writes carrier and relative pose only; a dynamic or interval field has no reader left."""
    XC = own_module("dexengine.rl.env_setup.extra_carry")
    cfg = NS()
    carry = {"carrier": "tray", "carrier_role": "main", "carrier_side": "right", "start": 3, "end": 9, "rel": REL}
    XC.carry_statement(cfg, [[{"name": "bread", "kind": "carried_attached", "carry": carry}, None]], 2)
    assert cfg.extra_slot_carry == [
        [{"carrier_role": "main", "carrier_side": "right", "carrier_slot": None, "rel": REL}, None]
    ]
    empty = NS()
    XC.carry_statement(empty, [[{"name": "cup", "kind": "moving"}, None]], 2)
    assert not hasattr(empty, "extra_slot_carry")


def test_training_scene_check_sees_the_carried_kind():
    """train_scene.changed: a ledger-less checkpoint on a clip with a carried slot is a changed scene; with a ledger,
    the carried table is compared only for a clip that stands that kind, and only when both sides state it."""
    TS = own_module("dexengine.rl.launch.train_scene")
    kinds = ("moving", "touched", "receiver")
    hits = {"touched", "receiver", "carried_attached"}
    now = {"moving": "m", "touched": "t", "receiver": "r", "carried_attached": "a2"}
    assert TS.changed(None, "clip", ["carried_attached"], "1/0/0:x", hits, now, kinds)  # collider now, none trained
    assert TS.changed(None, "clip", ["moving"], "1/0/0:x", hits, now, kinds) == ""  # moving: no collider
    led = {
        "rule": "R",
        "clips": {"clip": "1/0/0:x"},
        "collide": {"moving": "m", "touched": "t", "receiver": "r", "carried_attached": "a1"},
    }
    assert "carried_attached" in TS.changed(led, "clip", ["carried_attached"], "1/0/0:x", hits, now, kinds)
    assert TS.changed(led, "clip", ["moving"], "1/0/0:x", hits, now, kinds) == ""  # clip without a carried slot
    old = dict(led, collide={"moving": "m", "touched": "t", "receiver": "r"})  # ledger without carried kinds
    assert TS.changed(old, "clip", ["moving"], "1/0/0:x", hits, now, kinds) == ""
