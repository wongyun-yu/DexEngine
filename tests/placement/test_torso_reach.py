"""The arm reach of the torso decomposition (placement/collision/torso_split): a statement (ARM_REACH) checked by an
oracle that measures the hands whose URDF meshes are present. A checkout with no vendor stage of the other hands
reads the statement and goes on; a statement the measured hands contradict dies; an arm without a statement, or a
stated hand that is not registered, dies."""

from __future__ import annotations

from pathlib import Path

import pytest

from dexengine.placement.collision import sim_collider as SC
from dexengine.placement.collision import torso_split as TS

R, HAND = TS.ARM_REACH["rby1"]
HANDS = ["dg5f_s", "sharpa_wave", "wuji2", "xhand_pro"]
#: Reach (m) of each rby1 hand, as _chain_reach measured them on the full data root.
MEASURED = {"dg5f_s": 0.8957295891665912, "sharpa_wave": 0.9147757031724574, "wuji2": 0.8732825993252886, HAND: R}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(TS, "_REACH", {})
    monkeypatch.setattr(SC, "arm_hands", lambda arm: list(HANDS))


def _reach(present: dict):
    """_chain_reach over `present` (hand -> m); every other hand's vendor mesh is missing."""

    def f(arm, hand):
        if hand not in present:
            raise TS.MeshAbsent(hand, Path("x.urdf"), Path(f"/no/{hand}/mesh.obj"))
        return present[hand]

    return f


def test_the_statement_is_the_largest_reach_of_the_registered_hands():
    assert max(MEASURED.values()) == R and max(MEASURED, key=MEASURED.get) == HAND


def test_a_public_tree_without_the_other_vendor_meshes_reads_the_statement(monkeypatch, capsys):
    """Only sharpa_wave's meshes are present (a fresh checkout): the statement is read, the oracle measured 1/4."""
    monkeypatch.setattr(TS, "_chain_reach", _reach({"sharpa_wave": MEASURED["sharpa_wave"]}))
    assert TS.reach_stated("rby1") == (R, HAND)
    out = capsys.readouterr().out
    assert "[TORSO_REACH] rby1: stated" in out and "oracle measured 1/4 hands" in out
    assert "not measured (meshes absent) ['dg5f_s', 'wuji2', 'xhand_pro']" in out


def test_no_vendor_mesh_at_all_still_reads_the_statement(monkeypatch, capsys):
    monkeypatch.setattr(TS, "_chain_reach", _reach({}))
    assert TS.reach_stated("rby1") == (R, HAND) and "oracle measured 0/4 hands" in capsys.readouterr().out


def test_every_hand_measured_agrees_with_the_statement(monkeypatch):
    monkeypatch.setattr(TS, "_chain_reach", _reach(MEASURED))
    assert TS.reach_stated("rby1") == (R, HAND)


@pytest.mark.parametrize("shift_m", [1e-3, -1e-3])
def test_a_changed_statement_dies_on_the_oracle(monkeypatch, shift_m):
    """The statement moved by 1 mm either way: the stated hand measures otherwise (or a hand reaches farther)."""
    monkeypatch.setitem(TS.ARM_REACH, "rby1", (R + shift_m, HAND))
    monkeypatch.setattr(TS, "_chain_reach", _reach(MEASURED))
    with pytest.raises(ValueError, match="reach statement .* is not what the hands measure"):
        TS.reach_stated("rby1")


def test_a_measured_hand_beyond_the_statement_dies_with_the_stated_hand_absent(monkeypatch):
    """The stated hand's meshes are absent, but a measured hand reaches farther than the statement: stale."""
    monkeypatch.setattr(TS, "_chain_reach", _reach({"sharpa_wave": R + 1e-3}))
    with pytest.raises(ValueError, match="is not what the hands measure"):
        TS.reach_stated("rby1")


def test_an_arm_without_a_statement_dies(monkeypatch):
    monkeypatch.setattr(TS, "ARM_REACH", {})
    with pytest.raises(SystemExit, match="no reach statement"):
        TS.reach_stated("rby1")


def test_a_stated_hand_that_is_not_registered_dies(monkeypatch):
    monkeypatch.setitem(TS.ARM_REACH, "rby1", (R, "no_hand"))
    with pytest.raises(ValueError, match="not registered"):
        TS.reach_stated("rby1")


def test_arm_reach_reads_the_statement_once_per_arm(monkeypatch):
    """arm_reach (the ruler's input) returns the stated radius and hand; the oracle runs once per arm and process."""
    calls = []
    monkeypatch.setattr(TS, "_chain_reach", lambda arm, hand: calls.append(hand) or MEASURED[hand])
    urdf = SC.combo_urdf("rby1", "sharpa_wave")
    for _ in range(2):
        _S, r, h = TS.arm_reach("rby1", urdf)
        assert (r, h) == (R, HAND)
    assert sorted(calls) == HANDS


def test_the_reach_reads_a_missing_mesh_as_absent(monkeypatch):
    """On the real URDFs with the harness's empty data root (conftest: no vendor stage): the hands whose
    meshes the tree holds are measured, the others are absent, and the statement is read where the old code died on
    the first missing mesh (step 2e's torso). The oracle against every hand runs on a full data root in every chain
    process (reach_stated); the harness never reads the live data tree. A hand whose vendor source is not set is not
    selected (sim_collider.arm_hands) and is never opened."""
    from dexengine.robots import vendor  # noqa: PLC0415

    un = vendor.unavailable_hands()
    sel = [h for h in _registered("rby1") if h not in un]
    monkeypatch.setattr(SC, "arm_hands", lambda arm: list(sel))
    got, absent = TS.reach_oracle("rby1", sel)
    if not got:
        pytest.skip("no hand's meshes resolve in this tree")
    assert set(got) | set(absent) == set(sel)
    assert all(v <= R + TS.REACH_ORACLE_TOL_M for v in got.values())
    assert TS.reach_stated("rby1") == (R, HAND)


def _registered(arm):
    from dexengine.core import paths as P  # noqa: PLC0415 — the real registry (the fixture stubs arm_hands)

    return [h for a, h, _v in (P.split_combo(c, P.ARM_IDS, P.HAND_IDS) for c in P.combo_ids()) if a == arm]
