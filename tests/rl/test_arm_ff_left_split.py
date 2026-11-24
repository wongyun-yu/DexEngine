"""Arm feed-forward of a split articulation's left term (rl.tasks.mdp.actions.residual_ik_feedforward), CPU only.

The left term of the split articulation used to skip the velocity lead, on the belief that its joint ids index the
shared articulation. The left ids and the left all-frame IK cache are published under the left names and found on the
left articulation (mdp/events), so the left term takes the lead like the right one. What it must never do is borrow
the legacy right slot, which belongs to the other articulation; a one-hand build still reads that slot."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.imports import own_module

FF = own_module("dexengine.rl.tasks.mdp.actions.residual_ik_feedforward")
LEFT_IDS, RIGHT_IDS = [0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]


class _Robot:
    """The term's articulation: every arm joint regex matches, and the gravity query is absent (eff_ff = None)."""

    def find_joints(self, exprs):
        return list(range(7)), [f"j{i}" for i in range(7)]

    @property
    def root_physx_view(self):
        raise RuntimeError("no PhysX view on CPU")


def _term(env, side="left"):
    t = FF.ArmFeedforwardMixin()
    t._inner = SimpleNamespace(_asset=_Robot())
    t._env, t._side_rb = env, side
    t._ff_arm_all_ids = t._ff_vel_ids = None
    t._ff_warned = False
    seen = {}
    # (cache source, side, ..., cache, lengths, eff_ff, robot): the cache is the fourth argument from the end
    t._apply_arm_feedforward_effort = lambda *a: seen.update(src=a[0], cache=a[-4])
    return t, seen


@pytest.fixture
def split(monkeypatch):
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    monkeypatch.setenv("DEXENGINE_BI_SPLIT_ARTIC", "1")


@pytest.fixture
def one_hand(monkeypatch):
    monkeypatch.setenv("DEXENGINE_ARM", "rby1")
    monkeypatch.delenv("DEXENGINE_BI_SPLIT_ARTIC", raising=False)


def test_the_split_left_term_takes_the_lead_from_its_own_cache(split, capsys):
    env = SimpleNamespace(
        _arm_ik_joint_ids_left=LEFT_IDS,
        _arm_ik_joint_ids_right=RIGHT_IDS,
        _arm_ik_cache_all_left="left cache",
        _arm_ik_cache_all_len_left="left len",
        _arm_ik_cache_all_right="right cache",
        _arm_ik_cache_all_len_right="right len",
    )
    t, seen = _term(env)
    t._apply_arm_feedforward()
    assert t._ff_vel_ids == LEFT_IDS and seen == {"src": "_arm_ik_cache_all_left", "cache": "left cache"}
    assert "CANNOT engage" not in capsys.readouterr().out


def test_the_split_left_term_never_borrows_the_right_slot(split, capsys):
    env = SimpleNamespace(
        _arm_ik_joint_ids_right=RIGHT_IDS, _arm_ik_cache_all_right="right cache", _arm_ik_cache_all_len_right="n"
    )
    t, seen = _term(env)
    t._apply_arm_feedforward()
    assert t._ff_vel_ids is None and seen["cache"] is None
    assert "CANNOT engage for side='left'" in capsys.readouterr().out


def test_a_one_hand_left_build_still_reads_the_legacy_slot(one_hand, capsys):
    env = SimpleNamespace(
        _arm_ik_joint_ids_right=LEFT_IDS, _arm_ik_cache_all_right="active cache", _arm_ik_cache_all_len_right="n"
    )
    t, seen = _term(env)
    t._apply_arm_feedforward()
    assert t._ff_vel_ids == LEFT_IDS and seen["cache"] == "active cache"
    assert "CANNOT engage" not in capsys.readouterr().out
