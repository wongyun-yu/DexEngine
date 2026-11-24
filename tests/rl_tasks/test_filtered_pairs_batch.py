"""collision_bodies.author_filtered_pairs (reads first, writes under one Sdf.ChangeBlock) authors the same layer and
returns the same pairs as the one-edit-at-a-time writer it replaced, kept here as the reference (also checked on
the real robot USDs: 4 hands x 256 envs, layer sha256 equal)."""

from __future__ import annotations

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from dexengine.rl.tasks.scene.collision_bodies import author_filtered_pairs


def _reference(stage, src_paths, target_paths):
    pairs, n = {}, 0
    for s in src_paths:
        api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(s))
        rel = api.GetFilteredPairsRel() or api.CreateFilteredPairsRel()
        existing = set(rel.GetTargets())
        for t in target_paths:
            if t not in existing:
                rel.AddTarget(t)
                pairs.setdefault(s, []).append(t)
                n += 1
    return pairs, n


def _stage():
    st = Usd.Stage.CreateInMemory()
    for i in range(6):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(st, f"/W/link{i}").GetPrim())
    for k in range(3):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(st, f"/W/disc{k}").GetPrim())
    return st


CASES = [
    ([f"/W/link{i}" for i in range(6)], [f"/W/disc{k}" for k in range(3)]),  # strings (the callers)
    (["/W/link0", "/W/link1", "/W/link0"], ["/W/disc0", "/W/disc1"]),  # a source twice
    ([f"/W/link{i}" for i in range(3)], [Sdf.Path("/W/disc0"), Sdf.Path("/W/disc2")]),  # Sdf.Path targets
    (["/W/link2"], ["/W/disc1", "/W/disc1"]),  # a target twice
]


@pytest.mark.parametrize("srcs, tgts", CASES)
def test_same_layer_and_pairs_as_one_edit_at_a_time(srcs, tgts):
    a, b = _stage(), _stage()
    for pre in ([], [(["/W/link0"], [Sdf.Path("/W/disc0")])]):  # with and without earlier filters
        for st in (a, b):
            for s, t in pre:
                _reference(st, s, t)
        got, want = author_filtered_pairs(a, srcs, tgts), _reference(b, srcs, tgts)
        assert got == want
        assert a.GetRootLayer().ExportToString() == b.GetRootLayer().ExportToString()
        a, b = _stage(), _stage()


def test_read_back_after_the_call_sees_every_target():
    st = _stage()
    srcs, tgts = [f"/W/link{i}" for i in range(6)], [f"/W/disc{k}" for k in range(3)]
    author_filtered_pairs(st, srcs, tgts)
    for s in srcs:
        rel = UsdPhysics.FilteredPairsAPI(st.GetPrimAtPath(s)).GetFilteredPairsRel()
        assert {p.pathString for p in rel.GetTargets()} == set(tgts)
