"""Rule 1d's main pairs (two main objects): separated by one rigid mover within the cap and kept colliding, or left
to S-dyn-pairs' pair-off. The bodies, rows and the tree fixture are test_f0_sep.py's."""

from __future__ import annotations

import hashlib
import types
from pathlib import Path

import numpy as np
import pytest

from dexengine.placement.coords import core as COORDS
from dexengine.retarget.f0 import repair_sep as S
from tests.retarget import test_f0_sep as T

tree = T.tree  # test_f0_sep's tree fixture, found by name in this module too
_main, _rows, _still = T._main, T._rows, T._still


def _files(d: Path) -> dict:
    return {
        str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(d.rglob("*")) if f.is_file()
    }


def _main_pair(tree, gap_mm=3.0, art=(), cup_at=None):
    """Two main cubes, cup 3 mm into cube's +x face (or at cup_at) at frame 0; cube leaves along +x at the end."""
    cube, cup = _main("cube", 0), _main("cup", 7, side="left")
    for it in (it for it in (cube, cup) if it.name in art):
        it.art = object()
    q = _rows((0, 0, 0), cup_at or (0.04 - gap_mm / 1000.0, 0, 0))
    return cube, cup, q, tree.put([cube, cup], q)


def test_a_main_pair_separates_and_keeps_its_collision(tree):
    """Two main objects 3 mm into each other: one rigid main object moves out (the smaller move, then name order:
    cube), the pair is keyed as a main pair, and S-dyn-pairs reads it as freed (its collision stays on)."""
    cube, cup, q, pl = _main_pair(tree)
    assert pl["verdict"] == S.SEPARATE and set(pl["moves"]) == {"cube"}
    (p,) = pl["pairs"]
    assert (p["kind"], p["mover"], p["partner"], p["verdict"]) == ("main", "cube", "cup", S.SEPARATE)
    assert p["move_mm"] == pytest.approx(2.5) and p["after_mm"] <= S.TOL_M * 1e3
    assert S.freed_main_pairs(pl) == {frozenset(("cube", "cup"))} and ">cup:main:" in S.key_line(pl)
    q2 = S.applied(tree.path, q)
    poses = {"cube": S.XG.pose_mats(q2, 0), "cup": S.XG.pose_mats(q2, 7)}
    assert S.depth(cube, cup, q2, poses, 0) <= S.TOL_M and np.array_equal(q2[:, 7:], q[:, 7:])
    from dexengine.data.objects import dyn_pairs as DP

    assert DP.separated(tree.path, q) == {frozenset(("cube", "cup"))}


def test_without_the_main_pairs_the_tree_is_untouched(tree, monkeypatch):
    """Negative: with the main pairs taken out of the plan, the test above fails."""
    monkeypatch.setattr(S, "_main_pairs", lambda *a: [])
    with pytest.raises(AssertionError):
        test_a_main_pair_separates_and_keeps_its_collision(tree)


@pytest.mark.parametrize(
    ("gap_mm", "art", "why"), [(6.0, (), "deeper than 5 mm"), (3.0, ("cube", "cup"), "both articulated")]
)
def test_a_main_pair_the_move_does_not_free_stays_off(tree, gap_mm, art, why):
    """Deeper than the cap, or both articulated: KEEP_OFF (S-dyn-pairs keeps it off) — the tree is clean, nothing
    moves, no key, no hold, the tree's files as they were."""
    (tree.path / "0").mkdir()
    np.savez(tree.path / "0/trajectory_kinematic.npz", qpos=_rows((0, 0, 0), (0.04 - gap_mm / 1000.0, 0, 0)))
    _c, _u, q, pl = _main_pair(tree, gap_mm=gap_mm, art=art)
    before = _files(tree.path)
    (p,) = pl["pairs"]
    assert p["verdict"] == S.KEEP_OFF and p["why"] == why and pl["verdict"] == S.CLEAN
    assert S.applied(tree.path, q) is q and S.key_line(pl) == "" and S.moved(pl) is None and S.held_status(pl) is None
    assert S.freed_main_pairs(pl) == set() and _files(tree.path) == before


def test_the_rigid_main_object_is_the_mover(tree):
    """cube articulated: cup moves (an articulated main object is never moved)."""
    assert set(_main_pair(tree, art=("cube",))[3]["moves"]) == {"cup"}


def test_the_mover_that_makes_no_new_overlap_moves(tree, monkeypatch):
    """cup 3 mm into the top of cube, cube on the desk: cube (name order first) would go down into the desk; cup goes
    up."""
    monkeypatch.setattr(COORDS, "desk_height", lambda: -0.0195)
    pl = _main_pair(tree, cup_at=(0, 0, 0.04 - 0.003))[3]
    assert pl["verdict"] == S.SEPARATE and set(pl["moves"]) == {"cup"} and pl["moves"]["cup"]["rows"][0][2] > 0


def test_without_the_new_overlap_choice_the_desk_mover_is_taken(tree, monkeypatch):
    """Negative: the per-mover new-overlap check blind (its first call), cube goes into the desk and the tree's own
    new-overlap check holds it."""
    real, calls = S._new_overlaps, []
    monkeypatch.setattr(S, "_new_overlaps", lambda *a: [] if not calls.append(1) and len(calls) == 1 else real(*a))
    with pytest.raises(AssertionError):
        test_the_mover_that_makes_no_new_overlap_moves(tree, monkeypatch)


def test_a_main_receiver_pair_still_separates(tree):
    """The same 3 mm against a receiver (kinematic, collision on): the cube moves out."""
    cube, box = _main("cube", 0), _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, 0.0), 7)
    pl = tree.put([cube, box], _rows((0, 0, 0.02 - 0.003), (0, 0, 0)))
    assert pl["verdict"] == S.SEPARATE and set(pl["moves"]) == {"cube"} and pl["pairs"][0]["after_mm"] <= S.TOL_M * 1e3


def test_a_pair_that_moves_together_is_not_this_rules(tree):
    """The same 3 mm against a receiver with the cube at rest on it the whole clip (moves together,
    dyn_pairs.together): no 1d pair, nothing moves; S-dyn-pairs makes it one body or drops it."""
    cube, box = _main("cube", 0), _still("box", (-0.05, -0.05, -0.04), (0.05, 0.05, 0.0), 7)
    q = _rows((0, 0, 0.02 - 0.003), (0, 0, 0), leave=False)
    pl = tree.put([cube, box], q)
    assert pl["verdict"] == S.CLEAN and pl["pairs"] == [] and S.applied(tree.path, q) is q


def test_together_routing_removed_breaks_the_together_test(tree, monkeypatch):
    """Negative: with the together routing gone, the pair at rest separates and the test above fails."""
    monkeypatch.setattr(S.DP, "together", lambda q, a, b, pair: None)
    with pytest.raises(AssertionError):
        test_a_pair_that_moves_together_is_not_this_rules(tree)


def test_the_main_objects_are_dynamic_in_training():
    """The owner of the main objects' rigid props spawns them dynamic (kinematic_enabled False): `dynamic` reads it."""
    from dexengine.rl.env_setup import object_mass as M

    su = types.SimpleNamespace(RigidBodyPropertiesCfg=lambda **k: k)
    assert M.object_rigid_props(su)["kinematic_enabled"] is False and S.dynamic(_main("cube", 0))
